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
from unittest.mock import MagicMock, patch

from nvflare.apis.fl_constant import RunProcessKey
from nvflare.apis.job_launcher_spec import JobReturnCode
from nvflare.fuel.common.exit_codes import ProcessExitCode
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

    with patch("nvflare.private.fed.server.server_engine.time.time", side_effect=[0.0, 99.0]):
        with patch("nvflare.private.fed.server.server_engine.time.sleep"):
            engine._remove_run_processes("job-1")

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

    with patch("nvflare.private.fed.server.server_engine.time.time", side_effect=[0.0, 99.0]):
        with patch("nvflare.private.fed.server.server_engine.time.sleep"):
            engine._remove_run_processes("job-1")

    job_handle.terminate.assert_called_once()
    assert "job-1" not in engine.run_processes
