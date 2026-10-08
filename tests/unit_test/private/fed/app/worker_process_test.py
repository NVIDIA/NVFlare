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

import logging
import os
import subprocess
import sys
import threading
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.apis.signal import Signal
from nvflare.fuel.common.exit_codes import ProcessExitCode
from nvflare.fuel.f3.mpm import MainProcessMonitor
from nvflare.private.fed.app.client import worker_process
from nvflare.private.fed.client.client_app_runner import ClientAppRunner
from nvflare.private.fed.client.client_runner import ClientRunner


@pytest.fixture(autouse=True)
def isolated_cleanup_registration(monkeypatch):
    monkeypatch.setattr(MainProcessMonitor, "_cleanup_cbs", [])


@contextmanager
def _worker_runtime(workspace_root, runner):
    """Use the real worker/app runner with isolated startup and transport."""
    args = SimpleNamespace(
        set=[],
        workspace=str(workspace_root),
        client_name="site-1",
        job_id="job-1",
        token="test",
        token_signature="test",
        ssid="test",
        sp_target="localhost:8002",
        sp_scheme="tcp",
    )
    client = MagicMock()
    config = MagicMock()
    config.base_deployer.create_fed_client.return_value = client
    workspace = MagicMock()
    workspace.get_app_dir.return_value = str(workspace_root / "job-1" / "app_site-1")
    workspace.get_file_path_in_root.side_effect = lambda name: str(workspace_root / name)

    with ExitStack() as stack:
        for name in (
            "download_workspace",
            "refresh_custom_dir_import_path",
            "set_stats_pool_config_for_job",
            "fobs_initialize",
            "security_init_for_job",
            "register_ext_decomposers",
            "configure_logging",
            "upload_results_on_shutdown",
        ):
            stack.enter_context(patch.object(worker_process, name))
        stack.enter_context(patch.object(worker_process, "Workspace", return_value=workspace))
        stack.enter_context(patch.object(worker_process, "FLClientStarterConfiger", return_value=config))
        stack.enter_context(patch.object(worker_process, "create_stats_pool_files_for_job", return_value=None))
        stack.enter_context(patch.object(worker_process, "get_script_logger", return_value=logging.getLogger("worker")))
        stack.enter_context(
            patch.object(worker_process, "monitor_parent_process", lambda _runner, _pid, event: event.wait())
        )
        stack.enter_context(patch.object(ClientAppRunner, "create_client_runner", return_value=runner))
        stack.enter_context(patch.object(ClientAppRunner, "start_command_agent"))
        stack.enter_context(patch.object(ClientAppRunner, "sync_up_parents_process"))
        stack.enter_context(patch.object(ClientAppRunner, "notify_job_status"))
        yield args, client


@pytest.mark.parametrize("execution_fails", [False, True])
def test_worker_preserves_execution_exception_when_cleanup_fails(tmp_path, execution_fails):
    runner = MagicMock()
    execution_error = RuntimeError("runner failed")
    cleanup_error = RuntimeError("publication failed")
    if execution_fails:
        runner.run.side_effect = execution_error

    with (
        _worker_runtime(tmp_path, runner) as (args, client),
        patch.object(worker_process, "shutdown_job_process_runtime", side_effect=cleanup_error),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            worker_process.main(args)
        if execution_fails:
            client.terminate.assert_not_called()
            MainProcessMonitor._do_cleanup(threading.Event())

    assert exc_info.value is (execution_error if execution_fails else cleanup_error)
    client.terminate.assert_called_once()


def test_normal_worker_cleanup_stays_on_the_main_thread(tmp_path):
    cleanup_threads = []
    with (
        _worker_runtime(tmp_path, MagicMock()) as (args, _client),
        patch.object(
            worker_process,
            "shutdown_job_process_runtime",
            side_effect=lambda **_kwargs: cleanup_threads.append(threading.current_thread()),
        ),
    ):
        worker_process.main(args)

    assert cleanup_threads == [threading.current_thread()]


def _exception_worker(workspace_root, failure, cooperative):
    """Subprocess target: exercise real executor shutdown and MPM force exit."""
    from nvflare.fuel.f3.streaming.stream_utils import callback_thread_pool

    logging.basicConfig(level=logging.INFO)
    workspace_root = Path(workspace_root)
    run_dir = workspace_root / "job-1"
    run_dir.mkdir()
    runner = ClientRunner.__new__(ClientRunner)
    runner.engine = MagicMock()
    runner.engine.get_cell.return_value = None
    runner.engine.send_aux_request.return_value = {}
    runner.parent_target = "server"
    runner.run_abort_signal = Signal()
    runner._run_abort_requested = False
    runner.log_info = MagicMock()
    runner.log_debug = MagicMock()
    runner.get_positive_float_var = lambda var_name, default: 0.01
    if failure == "forced":
        runner.run = MagicMock(side_effect=RuntimeError("forced ClientRunner.run failure"))

    started = threading.Event()

    def extension_callback():
        started.set()
        if cooperative == "yes":
            while not runner.run_abort_signal.triggered:
                time.sleep(0.01)
            (run_dir / "callback_completed").write_text("cancelled")
        else:
            threading.Event().wait()

    callback_thread_pool.submit(extension_callback)
    assert started.wait(2)

    # This process-level fallback must follow dependent worker cleanup, never
    # run concurrently with it. It can finish only in the cooperative case.
    MainProcessMonitor.add_cleanup_cb_first(lambda: (run_dir / "fallback_cleanup").write_text("started"))

    with _worker_runtime(workspace_root, runner) as (args, _client):
        rc = MainProcessMonitor.run(
            worker_process.main,
            run_dir=str(run_dir),
            args=args,
            shutdown_grace_time=0,
            cleanup_grace_time=0.1,
        )
    (run_dir / "returned_from_mpm").write_text(str(rc))
    sys.exit(rc)


@pytest.mark.parametrize("cooperative", ["yes", "no"])
@pytest.mark.parametrize(
    "failure, original_error",
    [("sync", "cannot sync with server Runner"), ("forced", "forced ClientRunner.run failure")],
)
def test_escaping_runner_exception_cancels_work_or_uses_mpm_fallback(tmp_path, failure, original_error, cooperative):
    root = Path(__file__).resolve().parents[5]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1])['_exception_worker'](*sys.argv[2:])",
            str(Path(__file__).resolve()),
            str(tmp_path),
            failure,
            cooperative,
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == ProcessExitCode.EXCEPTION, result.stderr
    assert original_error in result.stderr
    run_dir = tmp_path / "job-1"
    if cooperative == "yes":
        assert (run_dir / "callback_completed").read_text() == "cancelled"
        assert (run_dir / "fallback_cleanup").read_text() == "started"
        assert int((run_dir / "returned_from_mpm").read_text()) == ProcessExitCode.EXCEPTION
        assert "Cleanup did not complete within" not in result.stderr
    else:
        assert int((run_dir / FLMetaKey.PROCESS_RC_FILE).read_text()) == ProcessExitCode.EXCEPTION
        assert not (run_dir / "fallback_cleanup").exists()
        assert not (run_dir / "returned_from_mpm").exists()
        assert "Cleanup did not complete within" in result.stderr
    assert "exception_cleanup" not in result.stderr
