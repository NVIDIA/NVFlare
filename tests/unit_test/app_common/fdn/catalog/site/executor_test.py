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

import json
from types import SimpleNamespace

import pytest

from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.fdn.catalog.common.payload import (
    CATALOG_READ_KEY,
    ReasonCode,
    ResultStatus,
    Selection,
    build_task_payload,
    parse_result,
)
from nvflare.app_common.fdn.catalog.common.schema import ColumnInfo, TableInfo
from nvflare.app_common.fdn.catalog.site import executor as executor_module
from nvflare.app_common.fdn.catalog.site.connector import CatalogConnector, ConnectorError
from nvflare.app_common.fdn.catalog.site.executor import CatalogExecutor, MountResolver
from nvflare.app_common.fdn.catalog.site.reader import CatalogReader
from tests.unit_test.app_common.fdn.catalog.helpers import make_catalog, make_large_catalog


class FakeConnector(CatalogConnector):
    source = "fake"
    closed = False

    def read_catalog(self, dataset, limits, options=None):
        if dataset.startswith("corrupt"):
            raise ConnectorError("unreadable")
        if dataset.startswith("budget"):
            raise ConnectorError("too big", budget=True)
        if dataset.startswith("boom"):
            raise RuntimeError("secret path /x/y")
        if dataset.startswith("swapped"):
            return make_catalog(dataset="other-name")
        if dataset.startswith("invalid"):
            t = TableInfo("t", (ColumnInfo("a", "int"),))
            return make_catalog(dataset=dataset, tables=(t, t))
        return make_catalog(dataset=dataset, operations=("cohort.count", "cohort.sum"))

    def close(self):
        FakeConnector.closed = True


class BrokenConnector(CatalogConnector):
    source = "broken"

    def __init__(self, mounts):
        raise RuntimeError("secret")


class RemoteConnector(CatalogConnector):
    """A source with no mounts: the site's binding gives each dataset its options."""

    source = "remote"
    seen = []

    def read_catalog(self, dataset, limits, options=None):
        RemoteConnector.seen.append((dataset, options))
        return make_catalog(dataset=dataset, display_name="from remote", operations=("cohort.count",))


class RaisingConnector(RemoteConnector):
    def read_catalog(self, dataset, limits, options=None):
        raise RuntimeError("secret endpoint")


BROKEN = "tests.unit_test.app_common.fdn.catalog.site.executor_test.BrokenConnector"
CONNECTOR = "tests.unit_test.app_common.fdn.catalog.site.executor_test.FakeConnector"
REMOTE = "tests.unit_test.app_common.fdn.catalog.site.executor_test.RemoteConnector"
RAISING = "tests.unit_test.app_common.fdn.catalog.site.executor_test.RaisingConnector"


@pytest.fixture
def env(tmp_path):
    mounts = tmp_path / "data" / "study1"
    policy = tmp_path / "policy"
    policy.mkdir()
    for name in ("good", "corrupt-b", "good-c", "budget-d", "boom-e", "invalid-f", "hidden"):
        (mounts / name).mkdir(parents=True)
    return tmp_path


def write_policy(env, datasets, study="study1"):
    directory = env / "policy" / study
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "catalog_policy.json").write_text(json.dumps({"format_version": 1, "datasets": datasets}))


def run(
    env,
    datasets=None,
    previous=(),
    connector=CONNECTOR,
    limits=None,
    abort=None,
    task_name="catalog.read",
    site_config=None,
    reader=None,
):
    config = {
        "format_version": 1,
        "connectors": {"files": {"class": connector}},
        "default_connector": "files",
        "limits": limits or {},
        "installed_operations": ["cohort.count"],
    }
    if site_config is not None:
        config = site_config
    (env / "policy" / "catalog_site.json").write_text(json.dumps(config))
    executor = CatalogExecutor(reader_id="reader" if reader is not None else None)
    executor.policy_root = env / "policy"
    executor.resolver = MountResolver(str(env / "data"))
    ctx = FLContext()
    if reader is not None:
        ctx.set_prop(
            ReservedKey.ENGINE,
            SimpleNamespace(get_component=lambda _: reader, get_workspace=lambda: None),
            private=True,
            sticky=False,
        )
    ctx.set_prop(FLContextKey.JOB_META, {"study": "study1"}, private=False, sticky=False)
    shareable = Shareable()
    shareable[CATALOG_READ_KEY] = build_task_payload(datasets, list(previous))
    abort = abort or Signal()
    reply = executor.execute(task_name, shareable, ctx, abort)
    return reply


def result(reply, datasets=None, previous=()):
    return parse_result(reply[CATALOG_READ_KEY], Selection(None if datasets is None else tuple(datasets)))


SHARE = {"share": True, "operations": ["cohort.count"]}


def bound(connectors, studies, default="files"):
    return {
        "format_version": 1,
        "connectors": connectors,
        "default_connector": default,
        "studies": studies,
        "installed_operations": ["cohort.count"],
    }


FILES = {"files": {"class": CONNECTOR}}
LAKE = {"lake": {"class": REMOTE}}
LAKE_DS = {"study1": {"remote-a": {"connector": "lake", "options": {"schema": "onc"}}}}


class TestDatasetsOfDifferentSources:
    def test_a_dataset_bound_to_a_remote_connector_needs_no_mount(self, env):
        write_policy(env, {"good": SHARE, "remote-a": SHARE})
        parsed = result(run(env, site_config=bound({**FILES, **LAKE}, LAKE_DS)))
        by_name = {c.dataset: c for c in parsed.catalogs}
        assert set(by_name) == {"good", "remote-a"} and by_name["remote-a"].display_name == "from remote"

    def test_the_dataset_options_reach_its_connector_and_a_mounted_dataset_has_none(self, env):
        RemoteConnector.seen.clear()
        write_policy(env, {"remote-a": SHARE})
        run(env, site_config=bound({**FILES, **LAKE}, LAKE_DS))
        assert RemoteConnector.seen == [("remote-a", {"schema": "onc"})]

    def test_one_study_can_mix_several_source_types(self, env):
        write_policy(env, {"good": SHARE, "remote-a": SHARE, "remote-b": SHARE})
        studies = {
            "study1": {
                "remote-a": {"connector": "lake", "options": {"schema": "a"}},
                "remote-b": {"connector": "claims", "options": {"table": "b"}},
            }
        }
        connectors = {**FILES, **LAKE, "claims": {"class": REMOTE}}
        parsed = result(run(env, site_config=bound(connectors, studies)))
        assert {c.dataset for c in parsed.catalogs} == {"good", "remote-a", "remote-b"}

    def test_a_binding_belongs_to_its_study(self, env):
        write_policy(env, {"remote-a": SHARE})
        elsewhere = {"other": LAKE_DS["study1"]}
        parsed = result(run(env, ["remote-a"], site_config=bound({**FILES, **LAKE}, elsewhere)))
        assert not parsed.catalogs and parsed.unavailable == ("remote-a",)

    def test_a_binding_beats_the_default_connector_for_a_mounted_dataset(self, env):
        write_policy(env, {"good": SHARE})
        studies = {"study1": {"good": {"connector": "lake"}}}
        parsed = result(run(env, ["good"], site_config=bound({**FILES, **LAKE}, studies)))
        assert parsed.catalogs[0].display_name == "from remote"

    def test_a_connector_that_cannot_be_built_costs_only_its_own_datasets(self, env):
        write_policy(env, {"good": SHARE, "remote-a": SHARE})
        reply = run(env, site_config=bound({**FILES, "lake": {"class": BROKEN}}, LAKE_DS))
        parsed = result(reply)
        assert parsed.status == ResultStatus.OK and parsed.reason_code == ReasonCode.INTERNAL
        assert [c.dataset for c in parsed.catalogs] == ["good"] and parsed.unavailable == ("remote-a",)
        assert "secret" not in json.dumps(reply[CATALOG_READ_KEY])

    def test_a_dataset_whose_connector_raises_is_unavailable_and_the_rest_are_read(self, env):
        write_policy(env, {"good": SHARE, "remote-a": SHARE})
        reply = run(env, site_config=bound({**FILES, "lake": {"class": RAISING}}, LAKE_DS))
        parsed = result(reply)
        assert [c.dataset for c in parsed.catalogs] == ["good"] and parsed.unavailable == ("remote-a",)
        assert "secret" not in json.dumps(reply[CATALOG_READ_KEY])

    def test_when_every_connector_fails_every_indexed_name_is_withheld(self, env):
        write_policy(env, {"good": SHARE})
        config = bound({"files": {"class": BROKEN}}, {})
        parsed = result(run(env, ["good"], previous=["good"], site_config=config))
        assert parsed.status == ResultStatus.FAILED and parsed.withheld == ("good",)

    def test_every_connector_is_closed(self, env):
        write_policy(env, {"good": SHARE})
        FakeConnector.closed = False
        run(env, ["good"], site_config=bound({**FILES, **LAKE}, LAKE_DS))
        assert FakeConnector.closed

    def test_a_bound_dataset_without_a_policy_rule_is_never_reported(self, env):
        write_policy(env, {"good": SHARE})  # remote-a is bound but has no rule
        parsed = result(run(env, site_config=bound({**FILES, **LAKE}, LAKE_DS)))
        assert [c.dataset for c in parsed.catalogs] == ["good"] and not parsed.withheld and not parsed.unavailable


class TestExecute:
    def test_unknown_task(self, env):
        assert run(env, task_name="other").get_return_code() == ReturnCode.TASK_UNKNOWN

    def test_one_bad_dataset_does_not_block_the_others(self, env):
        write_policy(env, {n: SHARE for n in ("good", "corrupt-b", "good-c")})
        reply = run(env, ["good", "corrupt-b", "good-c"])
        assert reply.get_return_code() == ReturnCode.OK
        parsed = result(reply)
        assert [c.dataset for c in parsed.catalogs] == ["good", "good-c"]
        assert parsed.unavailable == ("corrupt-b",)
        assert FakeConnector.closed

    def test_policy_filters_operations(self, env):
        write_policy(env, {"good": SHARE})
        parsed = result(run(env, ["good"]))
        assert parsed.catalogs[0].operations == ("cohort.count",)

    def test_only_previously_indexed_unshared_names_are_reported_withheld(self, env):
        write_policy(env, {"good": SHARE, "hidden": {"share": False}})
        parsed = result(run(env, previous=["hidden"]))
        assert parsed.withheld == ("hidden",)
        assert [c.dataset for c in parsed.catalogs] == ["good"]
        # unlisted and never-indexed mounts (corrupt-b, boom-e, ...) are not named in the result at all
        assert not parsed.unavailable

    def test_removed_binding_is_reported_unavailable(self, env):
        write_policy(env, {"good": SHARE, "gone": SHARE})
        parsed = result(run(env, previous=["gone"]))
        assert "gone" in parsed.unavailable

    def test_missing_policy_withholds_previous_names_only(self, env):
        reply = run(env, ["good", "good-c"], previous=["good"])
        parsed = result(reply)
        assert parsed.status == ResultStatus.FAILED and parsed.reason_code == ReasonCode.POLICY_UNAVAILABLE
        assert parsed.withheld == ("good",) and not parsed.catalogs and not parsed.unavailable

    def test_invalid_catalog_is_unavailable(self, env):
        write_policy(env, {"invalid-f": SHARE})
        assert result(run(env, ["invalid-f"])).unavailable == ("invalid-f",)

    def test_unexpected_exception_is_unavailable_with_no_detail(self, env):
        write_policy(env, {"boom-e": SHARE})
        reply = run(env, ["boom-e"])
        assert result(reply).unavailable == ("boom-e",)
        assert "secret" not in json.dumps(reply[CATALOG_READ_KEY])

    def test_budget_skip_withholds_a_previously_indexed_name(self, env):
        write_policy(env, {"budget-d": SHARE})
        parsed = result(run(env, ["budget-d"], previous=["budget-d"]))
        assert parsed.withheld == ("budget-d",) and parsed.reason_code == ReasonCode.BUDGET_EXCEEDED

    def test_budget_skip_of_a_new_name_lists_nothing(self, env):
        write_policy(env, {"budget-d": SHARE})
        parsed = result(run(env, ["budget-d"]))
        assert not parsed.withheld and not parsed.unavailable and parsed.status == ResultStatus.FAILED

    def test_large_catalog_is_returned_without_a_byte_cap(self, env, monkeypatch):
        write_policy(env, {"good": SHARE})
        catalog = make_large_catalog(dataset="good")
        monkeypatch.setattr(FakeConnector, "read_catalog", lambda self, *args, **kwargs: catalog)

        reply = run(env, ["good"])

        assert len(json.dumps(reply[CATALOG_READ_KEY], separators=(",", ":"))) > 1_000_000
        parsed = result(reply)
        assert parsed.status == ResultStatus.OK and parsed.reason_code is None
        assert parsed.catalogs == (catalog,) and not parsed.withheld and not parsed.unavailable

    def test_abort_skips_reads(self, env):
        write_policy(env, {"good": SHARE})
        signal = Signal()
        signal.trigger("stop")
        parsed = result(run(env, ["good"], previous=["good"], abort=signal))
        assert not parsed.catalogs and parsed.withheld == ("good",) and parsed.reason_code == ReasonCode.CANCELLED

    def test_a_requested_name_that_is_not_mounted_is_unavailable_not_a_failure(self, env):
        write_policy(env, {"good": SHARE, "nope": SHARE})
        parsed = result(run(env, ["good", "nope"]))
        assert parsed.status == ResultStatus.OK
        assert [c.dataset for c in parsed.catalogs] == ["good"] and parsed.unavailable == ("nope",)

    def test_lineage_never_names_an_unshared_dataset(self, env):
        write_policy(env, {"good": SHARE})  # the connector's catalog has lineage to "raw", which has no rule
        assert result(run(env, ["good"])).catalogs[0].lineage == ()
        write_policy(env, {"good": SHARE, "raw": SHARE})
        assert [x.dataset for x in result(run(env, ["good"])).catalogs[0].lineage] == ["raw"]

    def test_connector_that_cannot_load_fails_without_detail(self, env):
        write_policy(env, {"good": SHARE})
        reply = run(
            env, ["good"], connector="tests.unit_test.app_common.fdn.catalog.site.executor_test.BrokenConnector"
        )
        parsed = result(reply)
        assert parsed.status == ResultStatus.FAILED and parsed.reason_code == ReasonCode.INTERNAL
        assert "secret" not in json.dumps(reply[CATALOG_READ_KEY])

    def test_a_connector_that_cannot_load_withholds_every_indexed_name(self, env):
        # the permitted view cannot be rebuilt, so a narrowed policy cannot be applied to what the server holds
        (env / "data" / "study1" / "kept").mkdir()
        write_policy(env, {"good": {"share": False}, "kept": SHARE})
        reply = run(env, ["good", "kept", "new"], previous=["good", "kept"], connector=BROKEN)
        parsed = result(reply)
        assert parsed.status == ResultStatus.FAILED and parsed.withheld == ("good", "kept")

    def test_a_dataset_that_cannot_be_listed_withholds_every_indexed_name(self, env, monkeypatch):
        write_policy(env, {"good": SHARE})

        def broken(self, study, runtime_file=None):
            raise PermissionError("denied")

        monkeypatch.setattr(MountResolver, "resolve", broken)
        parsed = result(run(env, None, previous=["good", "other"]))
        assert parsed.status == ResultStatus.FAILED and parsed.withheld == ("good", "other")

    def test_a_connector_may_only_return_the_dataset_it_was_asked_for(self, env):
        write_policy(env, {"good": SHARE, "swapped-g": SHARE})
        parsed = result(run(env, ["swapped-g"]))
        assert not parsed.catalogs and parsed.unavailable == ("swapped-g",)

    def test_the_record_budget_covers_all_datasets_of_the_read(self, env):
        write_policy(env, {"good": SHARE, "good-c": SHARE})
        records = make_catalog().record_count()
        parsed = result(run(env, ["good", "good-c"], limits={"max_records": records + 1}))
        assert [c.dataset for c in parsed.catalogs] == ["good"] and parsed.reason_code == ReasonCode.BUDGET_EXCEEDED

    @pytest.mark.parametrize("path", ["subprocess.run", "os.system", "builtins.dict"])
    def test_only_catalog_connectors_are_constructed(self, env, path):
        write_policy(env, {"good": SHARE})
        parsed = result(run(env, ["good"], connector=path))
        assert parsed.status == ResultStatus.FAILED and parsed.reason_code == ReasonCode.INTERNAL

    @pytest.mark.parametrize(
        "config",
        [
            {"format_version": 2},
            {"format_version": 1, "unknown": 1},
            {"format_version": 1, "limits": {"max_records": 0}},
            {"format_version": 1, "limits": {"other": 5}},
            {"format_version": 1, "connectors": {"a": {"class": ""}}},
            {"format_version": 1, "default_connector": "undefined"},
            {"format_version": 1, "default_connector": []},
            {"format_version": 1, "default_connector": {}},
            {"format_version": 1, "studies": {"study1": {"good": {"connector": "undefined"}}}},
            {"format_version": 1, "studies": {"study1": {"good": {"connector": []}}}},
            {"format_version": 1, "studies": {"study1": {"good": {"connector": {}}}}},
            {"format_version": 1, "installed_operations": "cohort.count"},
        ],
    )
    def test_an_invalid_site_config_shares_nothing_and_withholds_previous_names(self, env, config):
        write_policy(env, {"good": SHARE, "good-c": SHARE})
        parsed = result(run(env, ["good", "good-c"], previous=["good"], site_config=config))
        assert parsed.status == ResultStatus.FAILED and parsed.reason_code == ReasonCode.POLICY_UNAVAILABLE
        assert parsed.withheld == ("good",) and not parsed.catalogs

    def test_a_missing_site_config_is_handled_like_a_missing_policy(self, env):
        write_policy(env, {"good": SHARE})
        executor = CatalogExecutor()
        executor.policy_root = env / "policy"
        executor.resolver = MountResolver(str(env / "data"))
        ctx = FLContext()
        ctx.set_prop(FLContextKey.JOB_META, {"study": "study1"}, private=False, sticky=False)
        shareable = Shareable()
        shareable[CATALOG_READ_KEY] = build_task_payload(["good"], ["good"])
        parsed = result(executor.execute("catalog.read", shareable, ctx, Signal()), ["good"], ["good"])
        assert parsed.reason_code == ReasonCode.POLICY_UNAVAILABLE and parsed.withheld == ("good",)

    def test_lists_are_disjoint(self, env):
        write_policy(env, {"good": SHARE, "corrupt-b": SHARE})
        parsed = result(run(env))
        names = [c.dataset for c in parsed.catalogs] + list(parsed.withheld) + list(parsed.unavailable)
        assert len(names) == len(set(names))


class TestMountResolver:
    def test_lists_valid_file_and_directory_mounts(self, tmp_path):
        study = tmp_path / "s"
        (study / "ok-1").mkdir(parents=True)
        (study / "Bad Name").mkdir()
        (study / "file").write_text("x")
        assert list(MountResolver(str(tmp_path)).resolve("s")) == ["file", "ok-1"]

    def test_missing_study_has_no_bindings(self, tmp_path):
        assert MountResolver(str(tmp_path)).resolve("nope") == {}

    def test_process_mode_uses_the_study_runtime_datasets_at_their_real_paths(self, tmp_path):
        real = tmp_path / "anywhere" / "omop"
        real.mkdir(parents=True)
        single = tmp_path / "single.parquet"
        single.write_bytes(b"footer")
        runtime = tmp_path / "study_runtime.yaml"
        runtime.write_text(
            f"format_version: 2\nstudies:\n  s:\n    datasets:\n"
            f"      omop: {{source: {real}, mode: ro}}\n      gone: {{source: {tmp_path / 'missing'}, mode: ro}}\n"
            f"      single: {{source: {single}, mode: ro}}\n"
        )
        assert MountResolver(str(tmp_path / "no-mounts")).resolve("s", str(runtime)) == {"omop": real, "single": single}

    def test_container_mounts_win_over_the_study_runtime(self, tmp_path):
        (tmp_path / "s" / "mounted").mkdir(parents=True)
        runtime = tmp_path / "study_runtime.yaml"
        runtime.write_text("format_version: 2\nstudies:\n  s:\n    datasets: {other: {source: /x, mode: ro}}\n")
        assert list(MountResolver(str(tmp_path)).resolve("s", str(runtime))) == ["mounted"]

    def test_an_unreadable_study_runtime_gives_no_bindings(self, tmp_path):
        runtime = tmp_path / "study_runtime.yaml"
        runtime.write_text("format_version: 99\n")
        assert MountResolver(str(tmp_path / "none")).resolve("s", str(runtime)) == {}


class TestSiteLocations:
    def _executor_with_workspace(self, root):
        executor = CatalogExecutor()
        engine = SimpleNamespace(get_workspace=lambda: SimpleNamespace(get_root_dir=lambda: str(root)))
        ctx = FLContext()
        ctx.set_prop(FLContextKey.JOB_META, {"study": "study1"}, private=False, sticky=False)
        ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=False)
        return executor, ctx

    def _run(self, executor, ctx, datasets, previous=()):
        shareable = Shareable()
        shareable[CATALOG_READ_KEY] = build_task_payload(datasets, list(previous))
        return result(executor.execute("catalog.read", shareable, ctx, Signal()), datasets, previous)

    def test_process_mode_reads_local_fdn_and_the_study_runtime(self, tmp_path, monkeypatch):
        monkeypatch.setattr(executor_module, "DEFAULT_POLICY_ROOT", str(tmp_path / "no-mount"))
        monkeypatch.setattr(executor_module, "DEFAULT_MOUNT_ROOT", str(tmp_path / "no-data"))
        real = tmp_path / "site-data" / "good"
        real.mkdir(parents=True)
        local = tmp_path / "ws" / "local"
        (local / "fdn").mkdir(parents=True)
        (local / "study_runtime.yaml").write_text(
            f"format_version: 2\nstudies:\n  study1:\n    datasets:\n      good: {{source: {real}, mode: ro}}\n"
        )
        (local / "fdn" / "catalog_site.json").write_text(
            json.dumps({"format_version": 1, "connectors": FILES, "default_connector": "files"})
        )
        (local / "fdn" / "study1").mkdir()
        (local / "fdn" / "study1" / "catalog_policy.json").write_text(
            json.dumps({"format_version": 1, "datasets": {"good": {"share": True}}})
        )
        executor, ctx = self._executor_with_workspace(tmp_path / "ws")
        executor.resolver = MountResolver(str(tmp_path / "no-data"))
        parsed = self._run(executor, ctx, ["good"])
        assert [c.dataset for c in parsed.catalogs] == ["good"]

    def test_the_policy_mount_wins_over_local_fdn(self, tmp_path, monkeypatch):
        mount = tmp_path / "mount"
        mount.mkdir()
        monkeypatch.setattr(executor_module, "DEFAULT_POLICY_ROOT", str(mount))
        (tmp_path / "ws" / "local" / "fdn").mkdir(parents=True)
        (tmp_path / "ws" / "local" / "fdn" / "catalog_site.json").write_text(
            json.dumps({"format_version": 1, "connectors": FILES, "default_connector": "files"})
        )
        executor, ctx = self._executor_with_workspace(tmp_path / "ws")
        # the mount exists but holds no site file, so nothing is shared even though local/fdn has one
        parsed = self._run(executor, ctx, ["good"], previous=["good"])
        assert parsed.reason_code == ReasonCode.POLICY_UNAVAILABLE and parsed.withheld == ("good",)


class RecordingReader(CatalogReader):
    def __init__(self):
        super().__init__()
        self.calls = []

    def read_catalog(self, dataset, mounts, limits, options=None):
        self.calls.append((dataset, mounts, limits))
        return make_catalog(dataset=dataset, owner="private owner")


def test_reader_component_is_called_only_for_shared_datasets_and_policy_still_applies(env):
    reader = RecordingReader()
    write_policy(env, {"good": {"share": True, "exclude_metadata": ["owner"]}, "hidden": {"share": False}})
    parsed = result(run(env, datasets=["good", "hidden"], reader=reader), datasets=["good", "hidden"])
    assert parsed.status == ResultStatus.OK
    assert [c.dataset for c in parsed.catalogs] == ["good"]
    assert parsed.catalogs[0].owner is None
    assert len(reader.calls) == 1 and reader.calls[0][0] == "good"
    assert reader.calls[0][1]["good"] == env / "data/study1/good"


def test_missing_policy_prevents_reader_invocation(env):
    reader = RecordingReader()
    parsed = result(run(env, datasets=["good"], reader=reader), datasets=["good"])
    assert parsed.status == ResultStatus.FAILED and reader.calls == []


def test_another_study_policy_is_not_used_for_the_active_study(env):
    write_policy(env, {"good": {"share": True}}, study="another-study")
    parsed = result(run(env, datasets=["good"], previous=["good"]), datasets=["good"], previous=["good"])
    assert parsed.status == ResultStatus.FAILED and parsed.reason_code == ReasonCode.POLICY_UNAVAILABLE
    assert parsed.withheld == ("good",) and parsed.catalogs == ()


def test_legacy_and_unscoped_policy_files_are_not_fallbacks(env):
    permissive = json.dumps({"format_version": 1, "datasets": {"good": {"share": True}}})
    (env / "policy/study1.json").write_text(permissive)
    (env / "policy/catalog_policy.json").write_text(permissive)
    parsed = result(run(env, datasets=["good"], previous=["good"]), datasets=["good"], previous=["good"])
    assert parsed.status == ResultStatus.FAILED and parsed.reason_code == ReasonCode.POLICY_UNAVAILABLE
    assert parsed.withheld == ("good",) and parsed.catalogs == ()
