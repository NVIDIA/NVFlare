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


"""Real worker archival, executor shutdown, and MPM forced-exit contracts."""

import logging
import os
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.apis.signal import Signal
from nvflare.app_opt.job_launcher import workspace_cell_transfer
from nvflare.fuel.common.exit_codes import ProcessExitCode
from nvflare.fuel.f3.cellnet.defs import ReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply
from nvflare.fuel.f3.mpm import MainProcessMonitor
from nvflare.private.fed.app.client import worker_process
from nvflare.private.fed.client.client_runner import ClientRunner
from tests.timing_utils import WAIT_TIMEOUT
from tests.worker_process_utils import _worker_runtime

pytestmark = pytest.mark.timeout(90)


def _workspace_upload_worker(workspace_root, upload_fails):
    """Run real result archival on the exception-unwinding thread."""
    logging.basicConfig(level=logging.INFO)
    workspace_root = Path(workspace_root)
    run_dir = workspace_root / "job-1"
    run_dir.mkdir()
    (run_dir / "worker.log").write_text("failed job diagnostic")
    runner = MagicMock()
    runner.run.side_effect = RuntimeError("original runner failure")

    def archive_file(_downloader, bundle_path):
        with zipfile.ZipFile(bundle_path) as bundle:
            assert bundle.read("job-1/worker.log") == b"failed job diagnostic"
        return "results-ref"

    def publish_results(**kwargs):
        (run_dir / "upload_started").write_text("started")
        assert threading.current_thread() is threading.main_thread()
        assert str(sys.exc_info()[1]) == "original runner failure"
        assert runner.abort.call_count == 1
        assert kwargs["timeout"] == workspace_cell_transfer.DOWNLOAD_TIMEOUT
        (run_dir / "upload_finished").write_text("finished")
        if upload_fails == "yes":
            raise RuntimeError("parent refused workspace results")
        return make_reply(ReturnCode.OK)

    cell = MagicMock()
    cell.send_request.side_effect = publish_results
    with (
        patch.dict(
            os.environ,
            {
                workspace_cell_transfer.ENV_WORKSPACE_OWNER_FQCN: "site-1",
                workspace_cell_transfer.ENV_WORKSPACE_TRANSFER_TOKEN: "test-transfer-token",
            },
        ),
        patch.object(workspace_cell_transfer, "_get_bootstrap_cell", return_value=cell),
        patch.object(workspace_cell_transfer, "_close_bootstrap_cell"),
        patch.object(workspace_cell_transfer, "ObjectDownloader", return_value=MagicMock()),
        patch.object(workspace_cell_transfer, "add_file", side_effect=archive_file),
        _worker_runtime(workspace_root, runner, mock_upload=False) as (args, client),
    ):
        rc = MainProcessMonitor.run(
            worker_process.main,
            run_dir=str(run_dir),
            args=args,
            shutdown_grace_time=0,
            cleanup_grace_time=WAIT_TIMEOUT,
        )
        client.terminate.assert_called_once()
    (run_dir / "returned_from_mpm").write_text(str(rc))
    sys.exit(rc)


@pytest.mark.parametrize("upload_fails", ["no", "yes"])
def test_failed_worker_archival_runs_on_main_thread_and_preserves_error(tmp_path, upload_fails):
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1])['_workspace_upload_worker'](*sys.argv[2:])",
            str(Path(__file__).resolve()),
            str(tmp_path),
            upload_fails,
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == ProcessExitCode.EXCEPTION, result.stderr
    assert "original runner failure" in result.stderr
    assert (tmp_path / "job-1" / "upload_started").read_text() == "started"
    assert (tmp_path / "job-1" / "upload_finished").is_file(), result.stderr
    assert (tmp_path / "job-1" / "upload_finished").read_text() == "finished", result.stderr
    assert int((tmp_path / "job-1" / "returned_from_mpm").read_text()) == ProcessExitCode.EXCEPTION
    assert "Cleanup did not complete within" not in result.stderr
    if upload_fails == "yes":
        assert "parent refused workspace results" in result.stderr
        assert "while handling another error" in result.stderr
    else:
        assert "upload_results SUCCESS" in result.stderr


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
    assert started.wait(WAIT_TIMEOUT)

    # This process-level fallback must follow dependent worker cleanup, never
    # run concurrently with it. It can finish only in the cooperative case.
    MainProcessMonitor.prepend_cleanup_cb(lambda: (run_dir / "fallback_cleanup").write_text("started"))

    with _worker_runtime(workspace_root, runner) as (args, _client):
        rc = MainProcessMonitor.run(
            worker_process.main,
            run_dir=str(run_dir),
            args=args,
            shutdown_grace_time=0,
            cleanup_grace_time=WAIT_TIMEOUT if cooperative == "yes" else 0.1,
        )
    (run_dir / "returned_from_mpm").write_text(str(rc))
    sys.exit(rc)


@pytest.mark.parametrize("cooperative", ["yes", "no"])
@pytest.mark.parametrize(
    "failure, original_error",
    [("sync", "cannot sync with server Runner"), ("forced", "forced ClientRunner.run failure")],
)
def test_escaping_runner_exception_cancels_work_or_uses_mpm_fallback(tmp_path, failure, original_error, cooperative):
    root = Path(__file__).resolve().parents[3]
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
        timeout=60,
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
