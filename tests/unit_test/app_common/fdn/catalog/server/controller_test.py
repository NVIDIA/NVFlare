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
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.client import Client
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.impl.wf_comm_server import WFCommServer
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.apis.shareable import make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.fdn.catalog.common.payload import (
    CATALOG_JOB_META,
    CATALOG_READ_KEY,
    PayloadError,
    ReasonCode,
    ResultStatus,
    SiteStatus,
    build_result,
    check_catalog_results,
)
from nvflare.app_common.fdn.catalog.server import controller as controller_module
from nvflare.app_common.fdn.catalog.server.controller import CatalogController
from tests.unit_test.app_common.fdn.catalog.helpers import make_catalog


def make_controller(tmp_path, clients=("site-a", "site-b"), **kw):
    controller = CatalogController(**kw)
    controller._job_id, controller._study = "job-1", "oncology"
    controller._clients = list(clients)
    controller._results_path = str(tmp_path / "catalog_results.json")
    return controller


def client_task(name, result):
    return SimpleNamespace(client=SimpleNamespace(name=name), result=result)


def reply(rc, payload=None):
    shareable = make_reply(rc)
    if payload is not None:
        shareable[CATALOG_READ_KEY] = payload
    return shareable


def written(tmp_path):
    return check_catalog_results(json.loads((tmp_path / "catalog_results.json").read_text()))


class TestResults:
    def test_results_file(self, tmp_path):
        controller = make_controller(tmp_path)
        ok = build_result(ResultStatus.OK, [make_catalog(dataset="omop")], unavailable=["registry"], skipped=["big"])
        controller._record(client_task("site-a", reply(ReturnCode.OK, ok)), None)
        controller._write_results()

        doc = written(tmp_path)
        assert doc["study"] == "oncology" and doc["job_id"] == "job-1"
        assert doc["sites"]["site-a"] == {
            "status": "OK",
            "reason_code": None,
            "unavailable": ["registry"],
            "skipped": ["big"],
        }
        assert doc["sites"]["site-b"]["status"] == "NO_RESULT"
        assert [(d["site"], d["dataset"]) for d in doc["datasets"]] == [("site-a", "omop")]
        assert [p.name for p in tmp_path.iterdir()] == ["catalog_results.json"]

    def test_failed_site_keeps_its_reason(self, tmp_path):
        controller = make_controller(tmp_path)
        failed = build_result(ResultStatus.FAILED, reason_code=ReasonCode.CONFIG_UNAVAILABLE)
        controller._record(client_task("site-a", reply(ReturnCode.OK, failed)), None)
        assert controller._entries["site-a"].status == SiteStatus.FAILED
        assert controller._entries["site-a"].reason_code == ReasonCode.CONFIG_UNAVAILABLE

    def test_bad_entry_is_unavailable_and_other_entries_count(self, tmp_path):
        controller = make_controller(tmp_path)
        bad = make_catalog(dataset="bad").to_wire()
        bad["metadata"]["display_name"] = float("nan")
        ok = build_result(ResultStatus.OK, [make_catalog(dataset="good")])
        ok["catalogs"].append(bad)
        controller._record(client_task("site-a", reply(ReturnCode.OK, ok)), None)
        controller._write_results()
        doc = written(tmp_path)
        assert [d["dataset"] for d in doc["datasets"]] == ["good"]
        assert doc["sites"]["site-a"]["unavailable"] == ["bad"]

    def test_names_outside_the_site_selection_are_dropped(self, tmp_path):
        controller = make_controller(tmp_path, datasets={"site-a": ["good"]})
        ok = build_result(ResultStatus.OK, [make_catalog(dataset="good"), make_catalog(dataset="other")])
        controller._record(client_task("site-a", reply(ReturnCode.OK, ok)), None)
        assert [c.dataset for c in controller._entries["site-a"].catalogs] == ["good"]

    @pytest.mark.parametrize("rc", [ReturnCode.EXECUTION_EXCEPTION, ReturnCode.TASK_UNSUPPORTED, None, 5])
    def test_no_ok_reply_is_no_result(self, tmp_path, rc):
        controller = make_controller(tmp_path)
        result = reply(ReturnCode.OK, build_result(ResultStatus.OK, [make_catalog()]))
        result.set_return_code(rc)
        controller._record(client_task("site-a", result), None)
        assert controller._entries["site-a"].status == SiteStatus.NO_RESULT

    def test_unsupported_payload_is_no_result(self, tmp_path):
        controller = make_controller(tmp_path)
        controller._record(client_task("site-a", reply(ReturnCode.OK, {"format_version": 9})), None)
        assert controller._entries["site-a"].status == SiteStatus.NO_RESULT

    def test_the_site_name_comes_from_the_framework_not_the_payload(self, tmp_path):
        controller = make_controller(tmp_path)
        controller._record(client_task("site-a", reply(ReturnCode.OK, {"client": "site-b"})), None)
        assert set(controller._entries) == {"site-a"}

    def test_stop_writes_the_file_only_if_control_flow_did_not(self, tmp_path):
        controller = make_controller(tmp_path)
        controller.stop_controller(None)
        assert written(tmp_path)["sites"]["site-a"]["status"] == "NO_RESULT"
        controller._record(client_task("site-a", reply(ReturnCode.OK, build_result(ResultStatus.OK))), None)
        controller.stop_controller(None)
        assert written(tmp_path)["sites"]["site-a"]["status"] == "NO_RESULT"


class TestControlFlow:
    def test_an_abort_cancels_the_tasks_once(self, tmp_path, monkeypatch):
        controller = make_controller(tmp_path, clients=("site-a", "site-b", "site-c"))
        cancels = []
        monkeypatch.setattr(controller_module, "_POLL_SECONDS", 0.01)
        monkeypatch.setattr(controller, "send", lambda *args, **kwargs: None)
        monkeypatch.setattr(controller, "cancel_all_tasks", lambda **kwargs: cancels.append(kwargs))
        abort_signal = Signal()
        abort_signal.trigger(True)
        controller.control_flow(abort_signal, FLContext())
        assert len(cancels) == 1 and (tmp_path / "catalog_results.json").exists()

    def test_each_site_gets_its_own_selection(self, tmp_path, monkeypatch):
        controller = make_controller(tmp_path, datasets={"site-a": ["omop"]})
        sent = {}

        def send(task, fl_ctx, targets):
            task.schedule_time = time.time()
            task.before_task_sent_cb(client_task=SimpleNamespace(task=task), fl_ctx=fl_ctx)
            sent[targets[0]] = task.data[CATALOG_READ_KEY]
            task.task_done_cb(task=task, fl_ctx=fl_ctx)

        monkeypatch.setattr(controller, "send", send)
        controller.control_flow(Signal(), FLContext())
        assert sent["site-a"]["datasets"] == ["omop"] and "datasets" not in sent["site-b"]
        assert all(0 < p["time_budget_s"] < controller.task_timeout_s for p in sent.values())

    def test_a_resent_task_carries_the_budget_left_at_resend(self, tmp_path, monkeypatch):
        # Drive the real WFCommServer: a resend runs before_task_sent_cb again, and the site must receive the
        # budget that callback computes then, not the one from the first delivery. broadcast() would resend
        # its first-delivery snapshot here.
        site = Client("site-a", "token")
        engine = Mock(spec=ServerEngineSpec)
        fl_ctx = FLContextManager(engine=engine, identity_name="server", job_id="job-1").new_context()
        fl_ctx.set_prop(FLContextKey.JOB_META, {JobMetaKey.MIN_CLIENTS: 1, JobMetaKey.MANDATORY_CLIENTS: []})
        engine.new_context.return_value = fl_ctx
        engine.get_clients.return_value = [site]
        server = WFCommServer()
        server._engine = engine
        server.controller = Mock()
        controller = make_controller(tmp_path, clients=("site-a",), task_timeout_s=60)
        monkeypatch.setattr(controller, "send", server.send)
        monkeypatch.setattr(controller_module, "_POLL_SECONDS", 0.01)
        abort_signal = Signal()
        flow = threading.Thread(target=controller.control_flow, args=(abort_signal, fl_ctx), daemon=True)
        flow.start()
        try:
            deadline = time.time() + 5
            while not server._tasks and time.time() < deadline:
                time.sleep(0.01)
            assert server._tasks, "the controller did not schedule its task"
            scheduled = server._tasks[0].schedule_time
            budgets = []
            for offset in (0, 20):  # the first delivery, then a resend 20 s later with no result in between
                with monkeypatch.context() as clock:
                    clock.setattr(time, "time", lambda offset=offset: scheduled + offset)
                    _, _, data = server.process_task_request(site, fl_ctx)
                budgets.append(data[CATALOG_READ_KEY]["time_budget_s"])
            assert budgets == [48.0, 28.0]
        finally:
            abort_signal.trigger(True)
            flow.join(timeout=5)
        assert not flow.is_alive()

    def test_the_budget_is_what_the_task_has_left_when_sent(self, tmp_path):
        controller = make_controller(tmp_path, task_timeout_s=10)
        task = SimpleNamespace(timeout=10, schedule_time=time.time() - 4, create_time=0, data={CATALOG_READ_KEY: {}})
        controller._set_time_budget(SimpleNamespace(task=task), None)
        assert 3.5 < task.data[CATALOG_READ_KEY]["time_budget_s"] <= 4

    def test_callbacks_accept_the_keyword_arguments_the_framework_uses(self, tmp_path):
        controller = make_controller(tmp_path)
        event = threading.Event()
        controller._done_callback(event)(task=None, fl_ctx=None)
        assert event.is_set()
        controller._record(client_task=client_task("site-a", reply(ReturnCode.OK, {})), fl_ctx=None)
        assert "site-a" in controller._entries


class TestStart:
    def engine(self, tmp_path):
        return SimpleNamespace(
            get_workspace=lambda: SimpleNamespace(get_run_dir=lambda _: str(tmp_path)),
            get_clients=lambda: [SimpleNamespace(name="site-a")],
            fire_event=lambda *args: None,
        )

    def ctx(self, engine, meta):
        ctx = FLContext()
        ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=False)
        ctx.set_prop(FLContextKey.JOB_META, meta, private=True, sticky=False)
        ctx.set_prop(ReservedKey.RUN_NUM, "job-1", private=False, sticky=False)
        return ctx

    def test_a_file_shipped_with_the_job_is_removed(self, tmp_path):
        controller = CatalogController()
        controller._engine = self.engine(tmp_path)
        shipped = tmp_path / "catalog_results.json"
        shipped.write_text("shipped with the job")
        controller.start_controller(self.ctx(controller._engine, {CATALOG_JOB_META: True, "study": "oncology"}))
        assert not shipped.exists() and controller._study == "oncology" and controller._clients == ["site-a"]

    def test_a_job_without_the_marker_is_refused(self, tmp_path):
        controller = CatalogController()
        controller._engine = self.engine(tmp_path)
        with pytest.raises(RuntimeError):
            controller.start_controller(self.ctx(controller._engine, {}))
        controller.stop_controller(None)
        assert not (tmp_path / "catalog_results.json").exists()


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
