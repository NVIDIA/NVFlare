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
from nvflare.app_common.fdn.catalog.site import executor as executor_module
from nvflare.app_common.fdn.catalog.site.executor import CatalogExecutor
from nvflare.app_common.fdn.catalog.site.source_error import CatalogSourceError
from tests.unit_test.app_common.fdn.catalog.helpers import LIMITS, contract_entry


class FakeReader:
    def __init__(self):
        self.secrets = []
        self.durations = []

    def __call__(self, dataset, secret, budget, *, site):
        budget.request()
        self.durations.append(budget.remaining())
        name = dataset.name
        if name.startswith("corrupt") or dataset.type != "fake":
            raise CatalogSourceError("unreadable")
        if name.startswith("budget"):
            raise CatalogSourceError("too big", budget=True)
        if name.startswith("boom"):
            raise RuntimeError("secret path /x/y")
        if name.startswith("invalid"):
            return {}
        if name.startswith("secret"):
            self.secrets.append(secret("password"))
        return contract_entry(name)["contract"]


SETTINGS = {}


def dataset(type_="fake", rules=None, **extra):
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
def reader(monkeypatch):
    fake = FakeReader()
    monkeypatch.setattr(executor_module, "run_catalog_discovery", fake)
    return fake


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

    @pytest.mark.parametrize(
        "rules",
        [
            {"share": False},
            {"share": True},
            {"exclude_columns": {"secret_table": ["secret_column"]}},
            {"connector": "fake"},
            {"operations": ["cohort.count"]},
            {"contract_uri": "file:///a"},
        ],
    )
    @pytest.mark.parametrize("selection", [None, ["good"]])
    def test_obsolete_policy_fails_the_study_without_disclosing_identifiers(self, rules, selection, caplog):
        reply = run(config_doc({"good": dataset(), "private": dataset(rules=rules)}), datasets=selection)
        parsed = result(reply)
        assert parsed.status == ResultStatus.FAILED and parsed.reason_code == ReasonCode.CONFIG_UNAVAILABLE
        assert parsed.catalogs == parsed.unavailable == parsed.skipped == ()
        assert "obsolete" in caplog.text
        for private in ("private", "secret_table", "secret_column"):
            assert private not in caplog.text and private not in str(reply[CATALOG_READ_KEY])

    @pytest.mark.parametrize("rules", [False, True, []])
    def test_malformed_rules_refuse_the_study_before_discovery(self, rules, reader):
        parsed = result(run(config_doc({"good": dataset(), "private": dataset(rules=rules)})))
        assert parsed.reason_code == ReasonCode.CONFIG_UNAVAILABLE
        assert not parsed.catalogs and not reader.durations

    def test_no_datasets_is_an_ok_empty_read(self):
        parsed = result(run(config_doc({})))
        assert parsed.status == ResultStatus.OK and parsed.catalogs == ()


class TestSourceMetadata:
    def test_all_study_datasets_are_read_without_publication_rules(self):
        doc = config_doc({"good": dataset(), "other": dataset()})
        assert [c["dataset"] for c in result(run(doc)).catalogs] == ["good", "other"]

    def test_selection_reads_only_requested_study_datasets(self):
        doc = config_doc({"good": dataset(), "good-b": dataset()})
        parsed = result(run(doc, datasets=["good-b", "unknown"]))
        assert [c["dataset"] for c in parsed.catalogs] == ["good-b"]
        assert parsed.unavailable == () and parsed.skipped == ()


class TestFailures:
    def test_one_bad_dataset_does_not_affect_the_others(self):
        doc = config_doc(
            {
                "good": dataset(),
                "corrupt-b": dataset(),
                "boom-c": dataset(),
                "invalid-e": dataset(),
            }
        )
        parsed = result(run(doc))
        assert [c["dataset"] for c in parsed.catalogs] == ["good"]
        assert set(parsed.unavailable) == {"corrupt-b", "boom-c", "invalid-e"}

    def test_revoked_source_access_does_not_reuse_a_previous_schema(self, monkeypatch):
        doc = config_doc({"good": dataset()})
        assert result(run(doc)).catalogs

        def denied(*args, **kwargs):
            raise CatalogSourceError("source unavailable")

        monkeypatch.setattr(executor_module, "run_catalog_discovery", denied)
        latest = result(run(doc))
        assert latest.catalogs == () and latest.unavailable == ("good",)

    def test_budget_stop_is_skipped(self):
        parsed = result(run(config_doc({"budget-a": dataset(), "good": dataset()})))
        assert parsed.skipped == ("budget-a",) and [c["dataset"] for c in parsed.catalogs] == ["good"]

    @pytest.mark.parametrize("first,status", [("good", "catalogs"), ("corrupt", "unavailable")])
    def test_call_limit_is_shared_even_when_a_read_fails(self, first, status):
        parsed = result(run(config_doc({first: dataset(), "second": dataset()}, settings={"limits": {"max_calls": 1}})))
        assert getattr(parsed, status) and parsed.skipped == ("second",)

    def test_exhausted_record_budget_skips_the_rest(self):
        settings = {**SETTINGS, "limits": {"max_records": 3}}
        parsed = result(run(config_doc({"good": dataset(), "good-b": dataset()}, settings=settings)))
        assert [c["dataset"] for c in parsed.catalogs] == ["good"] and parsed.skipped == ("good-b",)


class TestDeadline:
    def test_the_task_time_budget_bounds_every_read(self, reader):
        result(run(config_doc({"good": dataset(), "good-b": dataset()}), time_budget_s=5))
        assert reader.durations and all(d <= 5 for d in reader.durations)

    def test_without_a_budget_the_site_limit_applies(self, reader):
        result(run(config_doc({"good": dataset()})))
        assert 5 < reader.durations[0] <= 60

    def test_the_shorter_site_limit_wins(self, reader):
        settings = {**SETTINGS, "limits": {"max_duration_s": 3}}
        result(run(config_doc({"good": dataset()}, settings=settings), time_budget_s=30))
        assert reader.durations[0] <= 3

    def test_an_exhausted_budget_skips_and_keeps_completed_catalogs(self, monkeypatch):
        from nvflare.app_opt.fdn.catalog import _metadata
        from tests.timing_utils import ManualClock

        clock = ManualClock()
        monkeypatch.setattr(_metadata, "time", clock)
        reader = FakeReader()

        def read(dataset, secret, budget, *, site):
            if dataset.name == "good-b":
                clock.advance(9)
            return reader(dataset, secret, budget, site=site)

        monkeypatch.setattr(executor_module, "run_catalog_discovery", read)
        parsed = result(run(config_doc({"good": dataset(), "good-b": dataset()}), time_budget_s=8))
        assert [c["dataset"] for c in parsed.catalogs] == ["good"] and parsed.skipped == ("good-b",)


class TestSourceSelection:
    def test_unsupported_source_type_is_unavailable(self):
        assert result(run(config_doc({"x": dataset(type_="no-such-type")}))).unavailable == ("x",)

    def test_secrets_are_read_through_the_dataset(self, monkeypatch, reader):
        monkeypatch.setenv("PW", "s3cret")
        doc = config_doc({"secret-a": dataset(secrets={"password": "PW"})})
        assert [c["dataset"] for c in result(run(doc)).catalogs] == ["secret-a"]
        assert reader.secrets == ["s3cret"]

    def test_unset_secret_makes_the_dataset_unavailable(self, monkeypatch):
        monkeypatch.delenv("PW", raising=False)
        doc = config_doc({"secret-a": dataset(secrets={"password": "PW"})})
        assert result(run(doc)).unavailable == ("secret-a",)


def test_site_reserves_space_for_statuses_when_reply_size_is_limited(monkeypatch):
    from nvflare.app_common.fdn.catalog.common.payload import json_size

    selected = ["first", "second"]
    doc = config_doc({n: dataset() for n in selected})
    full = run(doc)[CATALOG_READ_KEY]
    limit = json_size(full) - 1
    monkeypatch.setattr(executor_module, "MAX_SITE_REPLY_BYTES", limit)
    reply = run(doc)[CATALOG_READ_KEY]
    assert json_size(reply) <= limit
    assert [e["dataset"] for e in reply["catalogs"]] == ["first"]
    assert reply["skipped"] == ["second"]
