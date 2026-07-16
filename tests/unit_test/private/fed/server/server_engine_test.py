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
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.client import Client
from nvflare.apis.fl_constant import AdminCommandNames, RunProcessKey, ServerCommandKey, ServerCommandNames, SiteType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_launcher_spec import JobReturnCode
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.fuel.common.exit_codes import ProcessExitCode
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.private.aux_runner import AuxMsgTarget
from nvflare.private.defs import CellChannel
from nvflare.private.fed.server.server_engine import ServerEngine


class _FakeClientManager:
    def __init__(self):
        self.disabled = set()
        self.disable_errors = {}
        self.enable_errors = {}

    def is_client_disabled(self, client_name):
        return client_name in self.disabled

    def disable_client(self, client_name):
        if client_name in self.disable_errors:
            raise self.disable_errors[client_name]
        self.disabled.add(client_name)
        return []

    def enable_client(self, client_name):
        if client_name in self.enable_errors:
            raise self.enable_errors[client_name]
        was_disabled = client_name in self.disabled
        self.disabled.discard(client_name)
        return was_disabled


class _FakeServer:
    def __init__(self):
        self.client_manager = _FakeClientManager()
        self.admin_server = None
        self.removed_tokens = []

    def remove_client_data(self, token):
        self.removed_tokens.append(token)


def test_disable_clients_reports_already_disabled_state():
    server = _FakeServer()
    engine = ServerEngine.__new__(ServerEngine)
    engine.server = server

    first = engine.disable_clients(["site-1"])
    second = engine.disable_clients(["site-1"])

    assert first["clients"][0]["already_disabled"] is False
    assert second["clients"][0]["already_disabled"] is True


def test_disable_clients_continues_after_per_client_error():
    server = _FakeServer()
    server.client_manager.disable_errors["site-2"] = RuntimeError("persist failed")
    engine = ServerEngine.__new__(ServerEngine)
    engine.server = server

    result = engine.disable_clients(["site-1", "site-2", "site-3"])

    assert [client["client_name"] for client in result["clients"]] == ["site-1", "site-2", "site-3"]
    assert [client["state"] for client in result["clients"]] == ["disabled", "error", "disabled"]
    assert result["clients"][1]["error"] == "persist failed"
    assert server.client_manager.disabled == {"site-1", "site-3"}


def test_disable_clients_single_error_still_raises():
    server = _FakeServer()
    server.client_manager.disable_errors["site-1"] = RuntimeError("persist failed")
    engine = ServerEngine.__new__(ServerEngine)
    engine.server = server

    try:
        engine.disable_clients(["site-1"])
    except RuntimeError as e:
        assert str(e) == "persist failed"
    else:
        raise AssertionError("expected RuntimeError")


def test_enable_clients_continues_after_per_client_error():
    server = _FakeServer()
    server.client_manager.disabled.update({"site-1", "site-2", "site-3"})
    server.client_manager.enable_errors["site-2"] = RuntimeError("persist failed")
    engine = ServerEngine.__new__(ServerEngine)
    engine.server = server

    result = engine.enable_clients(["site-1", "site-2", "site-3"])

    assert [client["client_name"] for client in result["clients"]] == ["site-1", "site-2", "site-3"]
    assert [client["state"] for client in result["clients"]] == ["enabled", "error", "enabled"]
    assert result["clients"][1]["error"] == "persist failed"
    assert server.client_manager.disabled == {"site-2"}


def test_enable_clients_single_error_still_raises():
    server = _FakeServer()
    server.client_manager.disabled.add("site-1")
    server.client_manager.enable_errors["site-1"] = RuntimeError("persist failed")
    engine = ServerEngine.__new__(ServerEngine)
    engine.server = server

    try:
        engine.enable_clients(["site-1"])
    except RuntimeError as e:
        assert str(e) == "persist failed"
    else:
        raise AssertionError("expected RuntimeError")


def _make_wait_engine(run_process_info, exception_run_processes):
    engine = ServerEngine.__new__(ServerEngine)
    engine.lock = threading.Lock()
    engine.run_processes = {"job-1": run_process_info}
    engine.exception_run_processes = exception_run_processes
    engine.engine_info = MagicMock()
    engine.logger = MagicMock()
    return engine


def test_wait_for_complete_preserves_exception_return_code_set_by_fail_run():
    """If fail_run already added the run to exception_run_processes with rc=EXCEPTION,
    a non-zero SJ exit code from the subsequent abort must not overwrite it. Otherwise
    list_jobs would show FINISHED:ABORTED instead of FINISHED:EXECUTION_EXCEPTION.
    """
    run_process_info = {
        RunProcessKey.PARTICIPANTS: {},
        RunProcessKey.PROCESS_RETURN_CODE: ProcessExitCode.EXCEPTION,
    }
    exception_run_processes = {"job-1": run_process_info}
    engine = _make_wait_engine(run_process_info, exception_run_processes)
    process = MagicMock()

    with patch(
        "nvflare.private.fed.server.server_engine.get_return_code",
        return_value=JobReturnCode.ABORTED,
    ):
        engine.wait_for_complete(workspace="/tmp", job_id="job-1", process=process)

    assert run_process_info[RunProcessKey.PROCESS_RETURN_CODE] == ProcessExitCode.EXCEPTION
    assert "job-1" not in engine.run_processes


def test_wait_for_complete_records_nonzero_return_code_for_first_failure():
    """When no external path has marked the run as failed, wait_for_complete should
    still record the SJ's non-zero exit code in exception_run_processes.
    """
    run_process_info = {RunProcessKey.PARTICIPANTS: {}}
    exception_run_processes = {}
    engine = _make_wait_engine(run_process_info, exception_run_processes)
    process = MagicMock()

    with patch(
        "nvflare.private.fed.server.server_engine.get_return_code",
        return_value=ProcessExitCode.EXCEPTION,
    ):
        engine.wait_for_complete(workspace="/tmp", job_id="job-1", process=process)

    assert exception_run_processes["job-1"] is run_process_info
    assert run_process_info[RunProcessKey.PROCESS_RETURN_CODE] == ProcessExitCode.EXCEPTION
    assert "job-1" not in engine.run_processes


def _make_remove_engine(run_processes):
    engine = ServerEngine.__new__(ServerEngine)
    engine.lock = threading.Lock()
    engine.run_processes = run_processes
    engine.logger = MagicMock()
    return engine


def _make_abort_engine(job_handle):
    engine = _make_remove_engine({"job-1": {RunProcessKey.JOB_HANDLE: job_handle}})
    engine.engine_info = MagicMock()
    engine.send_command_to_child_runner_process = MagicMock(return_value="OK")
    return engine


def test_abort_app_on_server_schedules_cleanup_in_background_after_in_band_abort():
    """Cleanup must run in a background thread so the admin reply returns within the
    CLI's default cmd_timeout (5.0s). The thread must be non-daemon so process
    exit waits for bounded cleanup instead of orphaning launcher-managed resources.
    """
    job_handle = MagicMock()
    engine = _make_abort_engine(job_handle)
    engine._remove_run_processes = MagicMock()

    with patch("nvflare.private.fed.server.server_engine.threading.Thread") as thread_cls:
        result = engine.abort_app_on_server("job-1")

    assert result == ""
    thread_cls.assert_called_once_with(
        target=engine._remove_run_processes,
        kwargs={"job_id": "job-1", "job_handle": job_handle, "max_wait": 10.0},
        daemon=False,
    )
    thread_cls.return_value.start.assert_called_once()
    engine._remove_run_processes.assert_not_called()


def test_abort_app_on_server_skips_graceful_wait_when_in_band_abort_fails():
    job_handle = MagicMock()
    engine = _make_abort_engine(job_handle)
    engine.send_command_to_child_runner_process.side_effect = RuntimeError("child unavailable")
    engine._remove_run_processes = MagicMock()

    with patch("nvflare.private.fed.server.server_engine.threading.Thread") as thread_cls:
        result = engine.abort_app_on_server("job-1")

    assert result == ""
    thread_cls.assert_called_once_with(
        target=engine._remove_run_processes,
        kwargs={"job_id": "job-1", "job_handle": job_handle, "max_wait": 0.0},
        daemon=False,
    )
    thread_cls.return_value.start.assert_called_once()


def test_remove_run_processes_terminates_job_handle_when_graceful_wait_expires():
    """If the SJ does not exit on its own (e.g. K8s pod hung during shutdown), the
    abort cleanup must call job_handle.terminate() so launcher resources (the K8s
    server pod) are deleted instead of leaking after an admin abort. Without this
    fallback, FINISHED:ABORTED can be reported while the pod stays Running.
    """
    job_handle = MagicMock()
    run_process_info = {RunProcessKey.JOB_HANDLE: job_handle}
    engine = _make_remove_engine({"job-1": run_process_info})

    engine._remove_run_processes("job-1", max_wait=0.0)

    job_handle.terminate.assert_called_once()
    assert "job-1" not in engine.run_processes


def test_remove_run_processes_terminates_captured_handle_when_run_exits_gracefully():
    """Even if wait_for_complete pops the entry before max_wait expires, abort
    cleanup must still terminate the captured launcher handle. For K8s this is
    what deletes the server job pod after an admin abort.
    """
    job_handle = MagicMock()
    engine = _make_remove_engine({})  # entry already removed by wait_for_complete

    engine._remove_run_processes("job-1", job_handle=job_handle)

    job_handle.terminate.assert_called_once()


def test_remove_run_processes_has_nothing_to_terminate_without_run_or_handle():
    engine = _make_remove_engine({})  # entry already removed by wait_for_complete

    engine._remove_run_processes("job-1")

    assert "job-1" not in engine.run_processes


def test_remove_run_processes_tolerates_terminate_failure():
    """A failure to delete the launcher resource (e.g. K8s API error) must not crash
    abort cleanup; the run_processes entry is still popped so the engine state
    stays consistent.
    """
    job_handle = MagicMock()
    job_handle.terminate.side_effect = RuntimeError("k8s api down")
    run_process_info = {RunProcessKey.JOB_HANDLE: job_handle}
    engine = _make_remove_engine({"job-1": run_process_info})

    engine._remove_run_processes("job-1", max_wait=0.0)

    job_handle.terminate.assert_called_once()
    assert "job-1" not in engine.run_processes


def _basic_engine():
    engine = ServerEngine.__new__(ServerEngine)
    engine.logger = MagicMock()
    engine.lock = threading.Lock()
    engine.client_manager = MagicMock()
    engine.server = MagicMock()
    engine.run_manager = None
    engine.cell = None
    engine.widgets = {}
    engine.asked_to_stop = False
    return engine


def test_client_and_run_manager_accessors_delegate():
    engine = _basic_engine()
    clients = {"token-1": Client("site-1", "token-1")}
    engine.client_manager.get_clients.return_value = clients
    engine.client_manager.get_all_clients_from_inputs.return_value = ([clients["token-1"]], [])
    engine.client_manager.get_client_from_name.return_value = clients["token-1"]

    assert engine.has_relays() is engine.client_manager.has_relays.return_value
    assert engine.get_clients() == [clients["token-1"]]
    assert engine.validate_targets(["site-1"]) == ([clients["token-1"]], [])
    assert engine.get_client_from_name("site-1") is clients["token-1"]
    assert engine.get_run_info() is None

    engine.client_manager = None
    assert not engine.has_relays()


def test_widget_token_and_component_accessors():
    engine = _basic_engine()
    widget = object()
    engine.widgets = {"widget": widget}
    client = Client("site-1", "token-1")
    engine.server.client_manager.clients = {"token-1": client}
    engine.server.runner_config = MagicMock()
    engine.run_manager = MagicMock()

    assert engine.get_widget("widget") is widget
    assert engine.get_client_name_from_token("token-1") == "site-1"
    assert engine.get_client_name_from_token("missing") == ""
    assert engine.get_component("component") is engine.run_manager.get_component.return_value
    engine.add_component("component", widget)
    engine.server.runner_config.add_component.assert_called_once_with("component", widget)
    engine.ask_to_stop()
    assert engine.asked_to_stop


def test_set_run_manager_links_cell_and_registers_widgets():
    engine = _basic_engine()
    engine.cell = object()
    engine.widgets = {"one": object(), "two": object()}
    run_manager = MagicMock()

    engine.set_run_manager(run_manager)

    assert engine.run_manager is run_manager
    assert run_manager.cell is engine.cell
    assert run_manager.add_handler.call_count == 2


def test_initialize_comm_links_run_manager_and_registers_handler():
    engine = _basic_engine()
    engine.run_manager = MagicMock()
    cell = MagicMock()

    engine.initialize_comm(cell)

    assert engine.cell is cell
    assert engine.run_manager.cell is cell
    cell.register_request_cb.assert_called_once_with(
        channel=CellChannel.AUX_COMMUNICATION,
        topic="*",
        cb=engine._handle_aux_message,
    )


def test_aux_target_translation_handles_special_clients_and_invalid_input():
    engine = _basic_engine()
    client = Client("site-1", "token-1")
    client.set_fqcn("site-1.job")
    engine.get_client_from_name = MagicMock(side_effect=lambda name: client if name == "site-1" else None)
    engine.get_clients = MagicMock(return_value=[client])

    assert engine._get_aux_msg_target(SiteType.SERVER).fqcn == "server"
    assert not engine._get_aux_msg_target(SiteType.SERVER_PARENT).job_scoped
    assert engine._get_aux_msg_target("site-1").fqcn == "site-1.job"
    assert engine._get_aux_msg_target("missing") is None
    assert [target.name for target in engine._to_aux_msg_targets([])] == ["site-1"]
    assert engine._to_aux_msg_targets(["site-1"])[0].fqcn == "site-1.job"
    assert not engine._to_aux_msg_targets(["missing"])
    with pytest.raises(TypeError, match="invalid target_names"):
        engine._to_aux_msg_targets("site-1")
    with pytest.raises(TypeError, match="target name must be str"):
        engine._to_aux_msg_targets([1])


def test_send_aux_to_targets_delegates_translated_targets():
    engine = _basic_engine()
    engine.run_manager = MagicMock()
    targets = [AuxMsgTarget.server_target()]
    engine._to_aux_msg_targets = MagicMock(return_value=targets)
    request = Shareable()
    fl_ctx = FLContext()

    result = engine.send_aux_to_targets([SiteType.SERVER], "topic", request, 2.0, fl_ctx, True, False)

    assert result is engine.run_manager.aux_runner.send_aux_request.return_value
    engine.run_manager.aux_runner.send_aux_request.assert_called_once_with(
        targets=targets,
        topic="topic",
        request=request,
        timeout=2.0,
        fl_ctx=fl_ctx,
        optional=True,
        secure=False,
    )


def test_multicast_aux_requests_skips_unknown_targets_and_delegates_known_ones():
    engine = _basic_engine()
    engine.run_manager = MagicMock()
    server_target = AuxMsgTarget.server_target()
    engine._get_aux_msg_target = MagicMock(side_effect=lambda name: server_target if name == "server" else None)
    request = Shareable()

    assert engine.multicast_aux_requests("topic", {}, 1.0, FLContext()) == {}
    result = engine.multicast_aux_requests(
        "topic", {"server": request, "missing": Shareable()}, 1.0, FLContext(), optional=True
    )

    assert result is engine.run_manager.aux_runner.multicast_aux_requests.return_value
    assert engine.run_manager.aux_runner.multicast_aux_requests.call_args.kwargs["target_requests"] == [
        (server_target, request)
    ]


def test_send_child_command_supports_fire_and_forget_and_request_reply():
    engine = _basic_engine()
    engine.server.cell.send_request.return_value = MagicMock(
        payload={"value": 1},
        get_header=MagicMock(return_value=CellReturnCode.OK),
    )

    assert engine.send_command_to_child_runner_process("job-1", "command", {}, timeout=0.0, optional=True) is None
    fire_call = engine.server.cell.fire_and_forget.call_args.kwargs
    assert fire_call["targets"] == "server.job-1"
    assert fire_call["topic"] == "command"
    assert fire_call["optional"] is True

    assert engine.send_command_to_child_runner_process("job-1", "command", {"input": 1}) == {"value": 1}
    engine.server.cell.send_request.return_value.get_header.return_value = CellReturnCode.TIMEOUT
    assert engine.send_command_to_child_runner_process("job-1", "command", {}) is None


def test_retrieve_clients_data_validates_parent_response():
    engine = _basic_engine()
    engine.server.cell.send_request.return_value = MagicMock(
        payload={ServerCommandKey.CLIENTS: ["site-1"]},
        get_header=MagicMock(return_value=CellReturnCode.OK),
    )

    assert engine._retrieve_clients_data("job-1") == ["site-1"]
    call = engine.server.cell.send_request.call_args.kwargs
    assert call["target"] == "server"
    assert call["optional"] is True

    engine.server.cell.send_request.return_value.get_header.return_value = CellReturnCode.TIMEOUT
    assert engine._retrieve_clients_data("job-1") is None


def test_dispatch_returns_not_ready_or_delegates():
    engine = _basic_engine()
    request = Shareable()
    fl_ctx = FLContext()

    assert engine.dispatch("topic", request, fl_ctx).get_return_code() == ReturnCode.SERVER_NOT_READY

    engine.run_manager = MagicMock()
    result = engine.dispatch("topic", request, fl_ctx)
    assert result is engine.run_manager.aux_runner.dispatch.return_value


@pytest.mark.parametrize(
    "method_name, command",
    [("show_stats", ServerCommandNames.SHOW_STATS), ("get_errors", ServerCommandNames.GET_ERRORS)],
)
def test_stats_accessors_return_child_data_or_empty_dict(method_name, command):
    engine = _basic_engine()
    engine.send_command_to_child_runner_process = MagicMock(return_value={"value": 1})

    assert getattr(engine, method_name)("job-1") == {"value": 1}
    assert engine.send_command_to_child_runner_process.call_args.kwargs["command_name"] == command

    engine.send_command_to_child_runner_process.side_effect = RuntimeError("unavailable")
    assert getattr(engine, method_name)("job-1") == {}


def test_reset_errors_and_configure_job_log_handle_child_errors():
    engine = _basic_engine()
    engine.send_command_to_child_runner_process = MagicMock(return_value="configuration error")

    assert engine.reset_errors("job-1") == "reset the server error stats for job: job-1"
    assert engine.configure_job_log("job-1", {"level": "DEBUG"}) == "configuration error"
    assert engine.send_command_to_child_runner_process.call_args.kwargs["command_name"] == (
        AdminCommandNames.CONFIGURE_JOB_LOG
    )

    engine.send_command_to_child_runner_process.side_effect = RuntimeError("unavailable")
    assert "Failed to configure_job_log" in engine.configure_job_log("job-1", {})


def test_streamer_preconditions_and_shutdown():
    engine = _basic_engine()
    with pytest.raises(RuntimeError, match="run_manager has not been created"):
        engine.stream_objects("channel", "topic", MagicMock(), [], MagicMock(), FLContext())
    with pytest.raises(RuntimeError, match="run_manager has not been created"):
        engine.register_stream_processing("channel", "topic", MagicMock())

    engine.run_manager = SimpleNamespace(object_streamer=MagicMock())
    engine.shutdown_streamer()
    engine.run_manager.object_streamer.shutdown.assert_called_once()
