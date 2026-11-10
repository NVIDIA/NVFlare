# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pytest

from nvflare.apis.dataset_config_spec import DatasetConfigError
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.dataset_config.api import parse_study_config
from nvflare.app_common.fdn.catalog.common.payload import (
    CATALOG_JOB_META,
    CATALOG_READ_KEY,
    TASK_NAME,
    ReasonCode,
    ResultStatus,
    build_task_payload,
    parse_result,
)
from nvflare.app_common.fdn.catalog.common.schema import ColumnInfo, Lineage, TableInfo
from nvflare.app_common.fdn.catalog.site.connector import CatalogConnector, ConnectorError
from nvflare.app_common.fdn.catalog.site.executor import CatalogExecutor
from tests.unit_test.app_common.fdn.catalog.helpers import LIMITS, make_catalog

MODULE = "tests.unit_test.app_common.fdn.catalog.site.executor_test"


class FakeConnector(CatalogConnector):
    """Behavior is chosen by the dataset name."""

    source = "fake"
    closed = 0
    secrets = []
    durations = []

    def __init__(self, **args):
        self.args = args

    def read_catalog(self, dataset, secret, limits):
        FakeConnector.durations.append(limits.max_duration_s)
        name = dataset.name
        if name.startswith("corrupt"):
            raise ConnectorError("unreadable")
        if name.startswith("budget"):
            raise ConnectorError("too big", budget=True)
        if name.startswith("boom"):
            raise RuntimeError("secret path /x/y")
        if name.startswith("swapped"):
            return make_catalog(dataset="other-name")
        if name.startswith("invalid"):
            t = TableInfo("t", (ColumnInfo("a", "int"),))
            return make_catalog(dataset=name, tables=(t, t))
        if name.startswith("secret"):
            FakeConnector.secrets.append(secret("password"))
        return make_catalog(
            dataset=name,
            operations=("cohort.count", "cohort.sum"),
            lineage=(Lineage("good", None), Lineage("private", None)),
        )

    def close(self):
        FakeConnector.closed += 1


class BrokenConnector(CatalogConnector):
    source = "broken"

    def __init__(self, **args):
        raise RuntimeError("secret")


SHARE = {"share": True, "operations": ["cohort.count"]}
SETTINGS = {"connectors": {"fake": {"class": f"{MODULE}.FakeConnector"}}, "installed_operations": ["cohort.count"]}


def dataset(type_="fake", rules=SHARE, **extra):
    entry = {"type": type_, "uri": "file:///data/study1/x", **extra}
    if rules is not None:
        entry["usage"] = {"catalog": rules}
    return entry


def config_doc(datasets, settings=SETTINGS):
    doc = {"format_version": 1, "studies": {"study1": {"datasets": datasets}}}
    if settings is not None:
        doc["usage"] = {"catalog": settings}
    return doc


class ConfiguredExecutor(CatalogExecutor):
    def __init__(self, doc):
        super().__init__()
        self.doc = doc

    def load_config(self, fl_ctx):
        if isinstance(self.doc, Exception):
            raise self.doc
        return parse_study_config(self.doc, "study1", "datasets.yaml")


def run(doc, datasets=None, task_name=TASK_NAME, meta=None, abort=None, time_budget_s=None):
    executor = ConfiguredExecutor(doc)
    ctx = FLContext()
    ctx.set_prop(
        FLContextKey.JOB_META,
        meta if meta is not None else {"study": "study1", CATALOG_JOB_META: True},
        private=False,
        sticky=False,
    )
    shareable = Shareable()
    shareable[CATALOG_READ_KEY] = build_task_payload(datasets, time_budget_s)
    return executor.execute(task_name, shareable, ctx, abort or Signal())


def result(reply):
    assert reply.get_return_code() == ReturnCode.OK
    return parse_result(reply[CATALOG_READ_KEY], None, LIMITS)


@pytest.fixture(autouse=True)
def reset():
    FakeConnector.closed = 0
    FakeConnector.secrets = []
    FakeConnector.durations = []


class TestAdmission:
    def test_unknown_task(self):
        assert run(config_doc({}), task_name="other").get_return_code() == ReturnCode.TASK_UNKNOWN

    def test_job_without_the_catalog_marker_is_refused(self):
        reply = run(config_doc({"good": dataset()}), meta={"study": "study1"})
        assert reply.get_return_code() == ReturnCode.TASK_UNSUPPORTED

    def test_abort_returns_task_aborted(self):
        abort = Signal()
        abort.trigger(True)
        assert run(config_doc({"good": dataset()}), abort=abort).get_return_code() == ReturnCode.TASK_ABORTED


class TestConfiguration:
    def test_unusable_configuration_fails_the_read(self):
        parsed = result(run(DatasetConfigError("bad file")))
        assert parsed.status == ResultStatus.FAILED and parsed.reason_code == ReasonCode.CONFIG_UNAVAILABLE

    def test_invalid_site_settings_fail_the_read(self):
        parsed = result(run(config_doc({"good": dataset()}, settings={"limits": {"max_records": 0}})))
        assert parsed.reason_code == ReasonCode.CONFIG_UNAVAILABLE and parsed.catalogs == ()

    def test_invalid_rules_of_a_shared_dataset_make_it_unavailable(self):
        bad = dataset(rules={"share": True, "exclude_tables": "t"})
        parsed = result(run(config_doc({"good": dataset(), "bad": bad})))
        assert [c.dataset for c in parsed.catalogs] == ["good"] and parsed.unavailable == ("bad",)

    @pytest.mark.parametrize(
        "rules",
        [
            {"share": False, "exclude_tables": "t"},
            {"share": "yes"},
            {"exclude_tables": ["t"]},
            ["share"],
            {"share": False, "exclude_columns": {"secret_table": "secret_column"}},
            {"share": True, "exclude_columns": {"secret_table": "secret_column"}},
            {"share": True, "secret_table": ["x"]},
            {"share": True, "exclude_metadata": ["secret_field"]},
        ],
    )
    def test_invalid_rules_of_a_dataset_not_explicitly_shared_never_name_it(self, rules, caplog):
        reply = run(config_doc({"good": dataset(), "private-x": dataset(rules=rules)}))
        parsed = result(reply)
        assert [c.dataset for c in parsed.catalogs] == ["good"]
        assert parsed.unavailable == (("private-x",) if isinstance(rules, dict) and rules.get("share") is True else ())
        if parsed.unavailable == ():
            assert "private-x" not in str(reply[CATALOG_READ_KEY]) and "private-x" not in caplog.text
        for secret in ("secret_table", "secret_column", "secret_field"):
            assert secret not in caplog.text and secret not in str(reply[CATALOG_READ_KEY])

    def test_no_datasets_is_an_ok_empty_read(self):
        parsed = result(run(config_doc({})))
        assert parsed.status == ResultStatus.OK and parsed.catalogs == ()


class TestSharing:
    def test_unshared_datasets_are_omitted_by_name(self):
        doc = config_doc(
            {"good": dataset(), "private": dataset(rules={"share": False}), "norules": dataset(rules=None)}
        )
        reply = run(doc)
        parsed = result(reply)
        assert [c.dataset for c in parsed.catalogs] == ["good"]
        text = str(reply[CATALOG_READ_KEY])
        assert "private" not in text and "norules" not in text

    def test_lineage_to_an_unshared_dataset_is_removed(self):
        doc = config_doc({"good": dataset(), "private": dataset(rules={"share": False})})
        catalog = result(run(doc)).catalogs[0]
        assert [x.dataset for x in catalog.lineage] == ["good"]

    def test_rules_are_applied(self):
        rules = {**SHARE, "exclude_tables": ["visit"], "exclude_metadata": ["description"]}
        catalog = result(run(config_doc({"good": dataset(rules=rules)}))).catalogs[0]
        assert [t.name for t in catalog.tables] == ["person"]
        assert catalog.operations == ("cohort.count",)

    def test_selection_reads_only_listed_shared_datasets(self):
        doc = config_doc({"good": dataset(), "good-b": dataset(), "private": dataset(rules={"share": False})})
        parsed = result(run(doc, datasets=["good-b", "private", "unknown"]))
        assert [c.dataset for c in parsed.catalogs] == ["good-b"]
        assert parsed.unavailable == () and parsed.skipped == ()


class TestFailures:
    def test_one_bad_dataset_does_not_affect_the_others(self):
        doc = config_doc(
            {
                "good": dataset(),
                "corrupt-b": dataset(),
                "boom-c": dataset(),
                "swapped-d": dataset(),
                "invalid-e": dataset(),
            }
        )
        parsed = result(run(doc))
        assert [c.dataset for c in parsed.catalogs] == ["good"]
        assert set(parsed.unavailable) == {"corrupt-b", "boom-c", "swapped-d", "invalid-e"}

    def test_budget_stop_is_skipped(self):
        parsed = result(run(config_doc({"budget-a": dataset(), "good": dataset()})))
        assert parsed.skipped == ("budget-a",) and [c.dataset for c in parsed.catalogs] == ["good"]

    def test_exhausted_record_budget_skips_the_rest(self):
        settings = {**SETTINGS, "limits": {"max_records": 12}}
        parsed = result(run(config_doc({"good": dataset(), "good-b": dataset()}, settings=settings)))
        assert [c.dataset for c in parsed.catalogs] == ["good"] and parsed.skipped == ("good-b",)

    def test_connectors_are_closed(self):
        result(run(config_doc({"good": dataset(), "good-b": dataset()})))
        assert FakeConnector.closed == 1


class TestDeadline:
    def test_the_task_time_budget_bounds_every_read(self):
        result(run(config_doc({"good": dataset(), "good-b": dataset()}), time_budget_s=5))
        assert FakeConnector.durations and all(d <= 5 for d in FakeConnector.durations)

    def test_without_a_budget_the_site_limit_applies(self):
        result(run(config_doc({"good": dataset()})))
        assert 5 < FakeConnector.durations[0] <= 60

    def test_the_shorter_site_limit_wins(self):
        settings = {**SETTINGS, "limits": {"max_duration_s": 3}}
        result(run(config_doc({"good": dataset()}, settings=settings), time_budget_s=30))
        assert FakeConnector.durations[0] <= 3

    def test_an_exhausted_budget_skips_and_keeps_completed_catalogs(self, monkeypatch):
        from nvflare.app_common.fdn.catalog.site import executor as executor_module

        clock = iter([0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 9.0])
        monkeypatch.setattr(executor_module.time, "monotonic", lambda: next(clock, 9.0))
        parsed = result(run(config_doc({"good": dataset(), "good-b": dataset()}), time_budget_s=8))
        assert [c.dataset for c in parsed.catalogs] == ["good"] and parsed.skipped == ("good-b",)


class TestConnectorSelection:
    def test_type_without_a_connector_is_unavailable(self):
        assert result(run(config_doc({"x": dataset(type_="no-such-type")}))).unavailable == ("x",)

    def test_rules_may_name_another_connector(self):
        doc = config_doc({"x": dataset(type_="parquet", rules={**SHARE, "connector": "fake"})})
        assert [c.dataset for c in result(run(doc)).catalogs] == ["x"]

    def test_connector_that_cannot_be_built_is_unavailable(self):
        settings = {"connectors": {"fake": {"class": f"{MODULE}.BrokenConnector"}}}
        assert result(run(config_doc({"x": dataset()}, settings=settings))).unavailable == ("x",)

    def test_secrets_are_read_through_the_dataset(self, monkeypatch):
        monkeypatch.setenv("PW", "s3cret")
        doc = config_doc({"secret-a": dataset(secrets={"password": "PW"})})
        assert [c.dataset for c in result(run(doc)).catalogs] == ["secret-a"]
        assert FakeConnector.secrets == ["s3cret"]

    def test_unset_secret_makes_the_dataset_unavailable(self, monkeypatch):
        monkeypatch.delenv("PW", raising=False)
        doc = config_doc({"secret-a": dataset(secrets={"password": "PW"})})
        assert result(run(doc)).unavailable == ("secret-a",)
