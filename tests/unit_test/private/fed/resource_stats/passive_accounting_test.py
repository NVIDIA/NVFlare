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

"""The communication foundation does not activate statistics for ordinary jobs."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.private.fed.app import job_process_cleanup
from nvflare.private.fed.app.client import worker_process
from nvflare.private.fed.app.server import runner_process
from nvflare.private.fed.resource_stats import f3_job_counter
from nvflare.private.fed.resource_stats.external_transfer import get_external_transfer_manager
from nvflare.private.fed.server.job_runner import JobRunner


class _ReachedConfiguration(Exception):
    pass


@pytest.mark.parametrize("module", [worker_process, runner_process])
def test_normal_job_startup_does_not_create_a_counter(module, tmp_path, monkeypatch):
    events = []
    original_path = sys.path[:]
    monkeypatch.setenv("PYTHONPATH", "normal-dependency-path")
    monkeypatch.setattr(f3_job_counter, "_counter", None)
    constructor = MagicMock(side_effect=AssertionError("ordinary jobs must not activate statistics"))
    monkeypatch.setattr(f3_job_counter, "F3Counter", constructor)
    monkeypatch.setattr(module, "parse_vars", lambda _: {})
    monkeypatch.setattr(module, "Workspace", MagicMock())
    monkeypatch.setattr(module, "download_workspace", lambda *_: events.append("workspace"))
    monkeypatch.setattr(module, "refresh_custom_dir_import_path", lambda *_: events.append("imports"))

    def configuration(*_):
        events.append("configuration")
        raise _ReachedConfiguration

    monkeypatch.setattr(module, "set_stats_pool_config_for_job", configuration)
    args = SimpleNamespace(set=[], workspace=str(tmp_path), client_name="site-1", job_id="job-1")
    with pytest.raises(_ReachedConfiguration):
        module.main(args)

    assert events == ["workspace", "imports", "configuration"]
    constructor.assert_not_called()
    assert f3_job_counter.get_job_f3_counter() is None
    assert sys.path == original_path
    assert os.environ["PYTHONPATH"] == "normal-dependency-path"


def test_ordinary_parent_and_trainer_do_not_allocate_accounting(tmp_path, monkeypatch):
    monkeypatch.setattr(f3_job_counter, "_counter", None)
    runner = JobRunner(str(tmp_path))
    assert not hasattr(runner, "f3_counters")
    assert get_external_transfer_manager(MagicMock()) is None


@pytest.mark.parametrize("publishing", [False, True])
def test_normal_cleanup_keeps_main_order_without_statistics_drain(publishing, monkeypatch):
    events = []
    monkeypatch.setattr(job_process_cleanup, "shutdown_f3_streaming", lambda: events.append("streams"))
    monkeypatch.setattr(job_process_cleanup, "security_close", lambda: events.append("security"))
    job_process_cleanup.shutdown_job_process_runtime(
        stop_command_admission=lambda: events.append("stop_commands"),
        wait_for_command_callbacks=lambda _: events.append("callbacks") or True,
        stop_cell=lambda: events.append("cell"),
        logger=None,
        before_streaming_shutdown=(lambda: events.append("publication")) if publishing else None,
    )
    assert events == [
        "stop_commands",
        *(["publication"] if publishing else []),
        "streams",
        "cell",
        "callbacks",
        "security",
    ]
