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

from unittest.mock import MagicMock

import pytest

from nvflare.fuel.utils.secret_utils import PotentialSecretWarning
from nvflare.job_config.api import FedJob
from nvflare.recipe.session_mgr import SessionManager, _job_monitor_callback


def test_submit_job_scans_generated_config_before_submission():
    job = FedJob(name="secret-submit-job", min_clients=1)
    job.to_server({"auth_token": "abcd1234efgh"})

    session = MagicMock()
    session.submit_job.return_value = "job-id"
    manager = SessionManager({})
    manager._get_session = MagicMock(return_value=session)

    with pytest.warns(PotentialSecretWarning, match="generated job file"):
        assert manager.submit_job(job) == "job-id"

    session.submit_job.assert_called_once()
    session.close.assert_called_once()


def test_monitor_reports_changes_and_periodic_wait_without_metadata_dump(monkeypatch, capsys):
    now = [0]
    monkeypatch.setattr("nvflare.recipe.session_mgr.time.monotonic", lambda: now[0])
    state = {"count": 0}
    meta = {"status": "RUNNING", "resource_spec": {"gpu": 2}, "deploy_map": {"app": ["@ALL"]}}
    for tick in (0, 1, 14, 15):
        now[0] = tick
        assert _job_monitor_callback(None, "job-id", meta, cb_run_counter=state)
    meta["status"] = "FINISHED:COMPLETED"
    now[0] = 16
    _job_monitor_callback(None, "job-id", meta, cb_run_counter=state)
    output = capsys.readouterr().out
    assert output.count("Job ID:") == 1
    assert output.count("Job status: RUNNING") == 2
    assert "15s monitored" in output
    assert "Job status: FINISHED:COMPLETED" in output
    assert "resource_spec" not in output
    assert "deploy_map" not in output
