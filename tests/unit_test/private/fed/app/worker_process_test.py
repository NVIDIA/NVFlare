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

import pytest

from nvflare.fuel.f3.mpm import MainProcessMonitor
from nvflare.private.fed.app.client import worker_process
from nvflare.private.fed.client.client_app_runner import ClientAppRunner
from tests.worker_process_utils import _worker_runtime


@pytest.fixture(autouse=True)
def isolated_cleanup_registration(monkeypatch):
    monkeypatch.setattr(MainProcessMonitor, "_cleanup_cbs", [])


@pytest.mark.parametrize("execution_fails", [False, True])
@pytest.mark.parametrize("cleanup_stage", ["shutdown_job_process_runtime", "create_stats_pool_files_for_job"])
def test_worker_preserves_execution_exception_when_cleanup_fails(tmp_path, execution_fails, cleanup_stage):
    runner = MagicMock()
    execution_error = RuntimeError("runner failed")
    cleanup_error = RuntimeError("publication failed")
    if execution_fails:
        runner.run.side_effect = execution_error

    with (
        _worker_runtime(tmp_path, runner) as (args, client),
        patch.object(worker_process, cleanup_stage, side_effect=cleanup_error),
        # Real pool shutdown is exercised in the integration subprocess tests; do not
        # stop the shared pytest process's executors when testing archival errors.
        patch("nvflare.private.fed.app.job_process_cleanup.shutdown_f3_streaming"),
        patch("nvflare.private.fed.app.job_process_cleanup.security_close"),
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


def test_failed_worker_still_archives_after_command_close_failure(tmp_path, caplog):
    calls = []
    runner = MagicMock()
    runner.run.side_effect = RuntimeError("original runner failure")
    runner.abort.side_effect = lambda: calls.append("abort")

    def fail_close():
        calls.append("close")
        raise RuntimeError("command gate failed")

    with (
        _worker_runtime(tmp_path, runner) as (args, client),
        patch.object(ClientAppRunner, "close", side_effect=fail_close),
        patch.object(
            worker_process, "upload_results_on_shutdown", side_effect=lambda *a, **kw: calls.append("archive")
        ),
        patch.object(worker_process, "shutdown_job_process_runtime", side_effect=lambda **kw: calls.append("runtime")),
    ):
        with pytest.raises(RuntimeError, match="original runner failure"):
            worker_process.main(args)
        assert calls == ["close", "abort", "archive"]
        MainProcessMonitor._do_cleanup(threading.Event())

    assert calls == ["close", "abort", "archive", "runtime"]
    assert "command gate failed" in caplog.text
    client.terminate.assert_called_once()
