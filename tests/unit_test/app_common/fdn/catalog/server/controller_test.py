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
import threading
from types import SimpleNamespace

import pytest

from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import make_reply
from nvflare.app_common.fdn.catalog.common.payload import (
    CATALOG_JOB_META,
    CATALOG_READ_KEY,
    ManifestEntry,
    PayloadError,
    ResultStatus,
    Selection,
    build_manifest,
    build_result,
    parse_manifest,
    parse_result,
)
from nvflare.app_common.fdn.catalog.server.controller import CatalogController
from nvflare.app_common.fdn.catalog.server.fileio import write_atomically
from tests.unit_test.app_common.fdn.catalog.helpers import make_catalog


def make_controller(tmp_path, clients=("site-a", "site-b"), **kw):
    controller = CatalogController(**kw)
    controller._key, controller._job_id = b"k" * 32, "job-1"
    controller._clients = list(clients)
    controller._manifest_path = str(tmp_path / "catalog_results.json")
    return controller


def client_task(name, result):
    return SimpleNamespace(client=SimpleNamespace(name=name), result=result)


def reply(rc, payload=None):
    shareable = make_reply(rc)
    if payload is not None:
        shareable[CATALOG_READ_KEY] = payload
    return shareable


class TestManifest:
    @pytest.mark.parametrize("bad_value", [b"binary", float("nan"), {1: "a", "b": "c"}, {2: "a", 10: "b"}])
    def test_a_bad_site_cannot_block_other_sites_or_its_own_restrictions(self, tmp_path, bad_value):
        controller = make_controller(tmp_path)
        controller._key, controller._job_id = b"k" * 32, "job-1"
        good = build_result(ResultStatus.OK, [make_catalog()])
        bad = build_result(ResultStatus.OK, withheld=["private"], unavailable=["missing"])
        bad["catalogs"] = [bad_value]
        controller._record(client_task("site-a", reply(ReturnCode.OK, good)), None)
        controller._record(client_task("site-b", reply(ReturnCode.OK, bad)), None)

        controller._write_manifest()

        signed = json.loads((tmp_path / "catalog_results.json").read_text())
        entries = parse_manifest(signed["manifest"])
        assert signed["mac"] is not None
        assert entries["site-a"].result == good
        failed = parse_result(entries["site-b"].result, Selection(None))
        assert failed.status == ResultStatus.FAILED and failed.reason_code == "INTERNAL"
        assert failed.withheld == ("private",) and failed.unavailable == ("missing",)

    def test_the_manifest_is_written_by_one_writer_at_a_time(self, tmp_path, monkeypatch):
        from nvflare.app_common.fdn.catalog.server import controller as controller_module

        controller = make_controller(tmp_path)
        controller._key, controller._job_id = b"k" * 32, "job-1"
        held = []
        monkeypatch.setattr(
            controller_module, "write_atomically", lambda *a: held.append(controller._write_lock.locked())
        )
        controller._write_manifest()
        assert held == [True]

    def test_missing_clients_are_timeouts(self, tmp_path):
        controller = make_controller(tmp_path)
        controller._record(client_task("site-a", reply(ReturnCode.OK, {"status": "OK"})), None)
        controller._write_manifest()
        signed = json.loads((tmp_path / "catalog_results.json").read_text())
        entries = parse_manifest(signed["manifest"])
        assert signed["mac"] is not None
        assert entries["site-a"] == ManifestEntry("OK", {"status": "OK"})
        assert entries["site-b"] == ManifestEntry("TIMEOUT", None)

    def test_the_manifest_is_signed_with_the_servers_key_and_the_job_id(self, tmp_path):
        from nvflare.app_common.fdn.catalog.common.payload import manifest_mac

        controller = make_controller(tmp_path)
        controller._key, controller._job_id = b"k" * 32, "job-1"
        controller._write_manifest()
        signed = json.loads((tmp_path / "catalog_results.json").read_text())
        assert signed["mac"] == manifest_mac(b"k" * 32, "job-1", signed["manifest"])
        assert signed["mac"] != manifest_mac(b"k" * 32, "job-2", signed["manifest"])

    def test_the_client_name_comes_from_the_framework_not_the_payload(self, tmp_path):
        controller = make_controller(tmp_path)
        controller._record(client_task("site-a", reply(ReturnCode.OK, {"client": "site-b"})), None)
        assert set(controller._entries) == {"site-a"}

    def test_error_return_codes_carry_no_payload(self, tmp_path):
        controller = make_controller(tmp_path)
        controller._record(client_task("site-a", reply(ReturnCode.EXECUTION_EXCEPTION, {"x": 1})), None)
        assert controller._entries["site-a"] == ManifestEntry(ReturnCode.EXECUTION_EXCEPTION, None)

    def test_non_shareable_result_is_an_exception_entry(self, tmp_path):
        controller = make_controller(tmp_path)
        controller._record(client_task("site-a", None), None)
        assert controller._entries["site-a"].result is None

    def test_atomic_write_leaves_no_temp_file(self, tmp_path):
        write_atomically(str(tmp_path / "f.json"), {"a": 1})
        assert [p.name for p in tmp_path.iterdir()] == ["f.json"]


class TestAbort:
    def test_an_abort_cancels_the_tasks_once_however_many_sites_are_waiting(self, tmp_path, monkeypatch):
        from nvflare.apis.signal import Signal
        from nvflare.app_common.fdn.catalog.server import controller as controller_module

        controller = make_controller(tmp_path, clients=("site-a", "site-b", "site-c"))
        cancels = []
        monkeypatch.setattr(controller_module, "_POLL_SECONDS", 0.01)
        monkeypatch.setattr(controller, "broadcast", lambda *args, **kwargs: None)
        monkeypatch.setattr(controller, "cancel_all_tasks", lambda **kwargs: cancels.append(kwargs))
        monkeypatch.setattr(controller, "_write_manifest", lambda: None)
        abort_signal = Signal()
        abort_signal.trigger(True)

        controller.control_flow(abort_signal, FLContext())

        assert len(cancels) == 1


class TestCallbacks:
    def test_callbacks_accept_the_keyword_arguments_the_framework_uses(self, tmp_path):
        controller = make_controller(tmp_path)
        event = threading.Event()
        controller._done_callback(event)(task=None, fl_ctx=None)
        assert event.is_set()
        controller._record(client_task=client_task("site-a", reply(ReturnCode.OK, {})), fl_ctx=None)
        assert "site-a" in controller._entries


class TestArguments:
    @pytest.mark.parametrize(
        "datasets", [{}, [], ["omop"], "omop", {"a": []}, {"a": [1]}, {"a": [""]}, {"": ["x"]}, {"a": "x"}]
    )
    def test_bad_datasets(self, datasets):
        with pytest.raises(PayloadError):
            CatalogController(datasets=datasets)

    @pytest.mark.parametrize("timeout", [0, -1, True, 1.5])
    def test_bad_timeout(self, timeout):
        with pytest.raises(PayloadError):
            CatalogController(task_timeout_s=timeout)

    def test_the_arguments_are_attributes_so_the_job_config_can_record_them(self):
        controller = CatalogController(datasets={"site-a": ["a"]}, task_timeout_s=30)
        assert (controller.datasets, controller.task_timeout_s) == ({"site-a": ["a"]}, 30)


class TestContext:
    def load(self, tmp_path, controller, content):
        path = tmp_path / "catalog_context.json"
        path.write_text(content if isinstance(content, str) else json.dumps(content))
        controller._load_context(str(path), FLContext())
        return path

    def test_the_context_is_read_from_the_run_dir_and_removed(self, tmp_path):
        controller = make_controller(tmp_path)
        path = self.load(tmp_path, controller, {"key": "6b" * 32, "previous": {"site-a": ["fhir", "omop"]}})
        assert controller._key == b"k" * 32 and controller._previous("site-a") == ["fhir", "omop"]
        assert controller._previous("site-b") == [] and not path.exists()  # the key is not archived with the job

    def test_a_selection_narrows_the_names(self, tmp_path):
        controller = make_controller(tmp_path, datasets={"site-a": ["omop"], "site-b": ["fhir"]})
        self.load(tmp_path, controller, {"key": "6b" * 32, "previous": {"site-a": ["fhir", "omop"]}})
        assert controller._previous("site-a") == ["omop"]

    @pytest.mark.parametrize(
        "content",
        [
            "{not json",
            {"key": "zz", "previous": {}},
            {"key": "", "previous": {}},
            {"key": "00", "previous": {}},
            {"previous": {}},
            [],
        ],
    )
    def test_without_a_usable_context_nothing_is_signed_and_no_name_is_known(self, tmp_path, content):
        controller = make_controller(tmp_path)
        self.load(tmp_path, controller, content)
        assert controller._key is None and controller._previous("site-a") == []

    def test_a_missing_context_is_the_same(self, tmp_path):
        controller = make_controller(tmp_path)
        controller._load_context(str(tmp_path / "absent.json"), FLContext())
        assert controller._key is None

    def test_each_site_is_asked_for_its_own_names_and_a_site_not_listed_for_all(self, tmp_path):
        controller = make_controller(tmp_path, clients=("site-a", "site-b", "site-c"), datasets={"site-a": ["omop"]})
        assert controller._wanted("site-a") == ["omop"] and controller._wanted("site-b") is None
        controller._record(client_task("site-a", reply(ReturnCode.OK, {"k": 1})), None)
        controller._record(client_task("site-b", reply(ReturnCode.OK, {"k": 1})), None)
        assert controller._entries["site-a"].datasets == ("omop",) and controller._entries["site-b"].datasets is None


class TestManifestFormat:
    @pytest.mark.parametrize(
        "bad",
        [
            {},
            {"results": []},
            {"datasets": None, "results": {}},
            {"results": {"a": {"rc": 1, "result": None, "datasets": None}}},
            {"results": {"a": {"rc": "OK", "result": None}}},
            {"results": {"a": {"rc": "OK", "result": None, "datasets": "x"}}},
            {"results": {"": {"rc": "OK", "result": None, "datasets": None}}},
        ],
    )
    def test_malformed_manifest(self, bad):
        with pytest.raises(PayloadError):
            parse_manifest(bad)

    def test_manifest_round_trip(self):
        entries = {"a": ManifestEntry("OK", {"k": 1}, ("x", "y")), "b": ManifestEntry("TIMEOUT", None)}
        assert parse_manifest(build_manifest(entries)) == entries


@pytest.mark.parametrize("marked", [False, True])
def test_unadmitted_controller_never_collects_or_writes_an_unsigned_manifest(tmp_path, monkeypatch, marked):
    controller = CatalogController()
    engine = SimpleNamespace(
        get_workspace=lambda: SimpleNamespace(get_run_dir=lambda _: str(tmp_path)),
        fire_event=lambda *args: None,
    )
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=False)
    ctx.set_prop(FLContextKey.JOB_META, {CATALOG_JOB_META: marked}, private=True, sticky=False)
    controller._job_id = "job-1"
    broadcasts = []
    monkeypatch.setattr(controller, "broadcast", lambda *a, **k: broadcasts.append(a))
    with pytest.raises(RuntimeError, match="requires"):
        controller.start_controller(ctx)
    controller.stop_controller(ctx)
    with pytest.raises(RuntimeError, match="not been admitted"):
        controller.control_flow(None, ctx)
    assert not broadcasts and not (tmp_path / "catalog_results.json").exists()
