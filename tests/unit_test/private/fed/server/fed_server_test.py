# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.fl_constant import RunProcessKey
from nvflare.apis.shareable import Shareable
from nvflare.private.defs import CellMessageHeaderKeys, new_cell_message
from nvflare.private.fed.server.fed_server import FederatedServer
from nvflare.private.fed.server.server_state import ColdState, HotState


def _make_server(get_bool_var_value=None):
    """Create a FederatedServer with ServerEngine patched out.

    Args:
        get_bool_var_value: if not None, ConfigService.get_bool_var is patched to
                            return this value.  If None, the real default is used.
    """
    patches = [patch("nvflare.private.fed.server.fed_server.ServerEngine")]
    if get_bool_var_value is not None:
        patches.append(
            patch(
                "nvflare.private.fed.server.fed_server.ConfigService.get_bool_var",
                return_value=get_bool_var_value,
            )
        )
    ctx_managers = [p.start() for p in patches]
    server = FederatedServer(
        project_name="project_name",
        min_num_clients=1,
        max_num_clients=10,
        cmd_modules=None,
        heart_beat_timeout=600,
        args=MagicMock(),
        secure_train=False,
        snapshot_persistor=MagicMock(),
        overseer_agent=MagicMock(),
    )
    # Ensure cleanup even if the test doesn't use a context manager.
    for p in patches:
        p.stop()
    return server


class TestFederatedServer:
    @pytest.mark.parametrize("server_state, expected", [(HotState(), ["extra_job"]), (ColdState(), [])])
    def test_heart_beat_abort_jobs(self, server_state, expected):
        with patch("nvflare.private.fed.server.fed_server.ServerEngine"):
            server = FederatedServer(
                project_name="project_name",
                min_num_clients=1,
                max_num_clients=100,
                cmd_modules=None,
                heart_beat_timeout=600,
                args=MagicMock(),
                secure_train=False,
                snapshot_persistor=MagicMock(),
                overseer_agent=MagicMock(),
            )

            server.server_state = server_state
            request = new_cell_message(
                {
                    CellMessageHeaderKeys.TOKEN: "token",
                    CellMessageHeaderKeys.SSID: "ssid",
                    CellMessageHeaderKeys.CLIENT_NAME: "client_name",
                    CellMessageHeaderKeys.PROJECT_NAME: "task_name",
                    CellMessageHeaderKeys.JOB_IDS: ["extra_job"],
                },
                Shareable(),
            )

            result = server.client_heartbeat(request)
            assert result.get_header(CellMessageHeaderKeys.ABORT_JOBS, []) == expected

    def test_sync_client_jobs_legacy_reports_missing_immediately(self):
        with (
            patch("nvflare.private.fed.server.fed_server.ServerEngine"),
            patch("nvflare.private.fed.server.fed_server.ConfigService.get_bool_var", return_value=False),
        ):
            server = FederatedServer(
                project_name="project_name",
                min_num_clients=1,
                max_num_clients=10,
                cmd_modules=None,
                heart_beat_timeout=600,
                args=MagicMock(),
                secure_train=False,
                snapshot_persistor=MagicMock(),
                overseer_agent=MagicMock(),
            )

            token = "token-1"
            client = MagicMock()
            client.name = "C1"
            server.engine.run_processes = {"job1": {RunProcessKey.PARTICIPANTS: {token: client}}}
            server.engine.notify_dead_job = MagicMock()

            no_job_request = new_cell_message({CellMessageHeaderKeys.JOB_IDS: []}, Shareable())
            server._sync_client_jobs(no_job_request, token)

            server.engine.notify_dead_job.assert_called_once_with("job1", "C1", "missing job on client")

    def test_sync_client_jobs_reports_missing_only_after_prior_seen_when_enabled(self):
        with (
            patch("nvflare.private.fed.server.fed_server.ServerEngine"),
            patch("nvflare.private.fed.server.fed_server.ConfigService.get_bool_var", return_value=True),
        ):
            server = FederatedServer(
                project_name="project_name",
                min_num_clients=1,
                max_num_clients=10,
                cmd_modules=None,
                heart_beat_timeout=600,
                args=MagicMock(),
                secure_train=False,
                snapshot_persistor=MagicMock(),
                overseer_agent=MagicMock(),
            )

            token = "token-1"
            client = MagicMock()
            client.name = "C1"
            server.engine.run_processes = {"job1": {RunProcessKey.PARTICIPANTS: {token: client}}}
            server.engine.notify_dead_job = MagicMock()

            no_job_request = new_cell_message({CellMessageHeaderKeys.JOB_IDS: []}, Shareable())
            server._sync_client_jobs(no_job_request, token)
            server.engine.notify_dead_job.assert_not_called()

            job_present_request = new_cell_message({CellMessageHeaderKeys.JOB_IDS: ["job1"]}, Shareable())
            server._sync_client_jobs(job_present_request, token)
            server.engine.notify_dead_job.assert_not_called()

            server._sync_client_jobs(no_job_request, token)
            server.engine.notify_dead_job.assert_called_once_with("job1", "C1", "missing job on client")

    def test_sync_client_jobs_default_requires_prior_report(self):
        """Default behaviour (require_previous_report=True) must not fire on the
        first missing-job heartbeat — no config override needed."""
        with (
            patch("nvflare.private.fed.server.fed_server.ServerEngine"),
            patch("nvflare.private.fed.server.fed_server.ConfigService.get_bool_var", return_value=True),
        ):
            server = FederatedServer(
                project_name="project_name",
                min_num_clients=1,
                max_num_clients=10,
                cmd_modules=None,
                heart_beat_timeout=600,
                args=MagicMock(),
                secure_train=False,
                snapshot_persistor=MagicMock(),
                overseer_agent=MagicMock(),
            )

            token = "token-1"
            client = MagicMock()
            client.name = "C1"
            server.engine.run_processes = {"job1": {RunProcessKey.PARTICIPANTS: {token: client}}}
            server.engine.notify_dead_job = MagicMock()

            # First heartbeat: client says it has no job1 — should NOT fire yet
            no_job_request = new_cell_message({CellMessageHeaderKeys.JOB_IDS: []}, Shareable())
            server._sync_client_jobs(no_job_request, token)
            server.engine.notify_dead_job.assert_not_called()

    def test_sync_client_jobs_tracking_in_server_attr_not_job_info(self):
        """Positive observations must be recorded in server._job_reported_clients,
        NOT injected into the job_info dict."""
        with (
            patch("nvflare.private.fed.server.fed_server.ServerEngine"),
            patch("nvflare.private.fed.server.fed_server.ConfigService.get_bool_var", return_value=True),
        ):
            server = FederatedServer(
                project_name="project_name",
                min_num_clients=1,
                max_num_clients=10,
                cmd_modules=None,
                heart_beat_timeout=600,
                args=MagicMock(),
                secure_train=False,
                snapshot_persistor=MagicMock(),
                overseer_agent=MagicMock(),
            )

            token = "token-1"
            client = MagicMock()
            client.name = "C1"
            job_info = {RunProcessKey.PARTICIPANTS: {token: client}}
            server.engine.run_processes = {"job1": job_info}
            server.engine.notify_dead_job = MagicMock()

            # Positive observation heartbeat
            job_present_request = new_cell_message({CellMessageHeaderKeys.JOB_IDS: ["job1"]}, Shareable())
            server._sync_client_jobs(job_present_request, token)

            # Token recorded in server attribute
            assert "job1" in server._job_reported_clients
            assert token in server._job_reported_clients["job1"]

            # NOT injected into job_info dict
            assert "_reported_clients" not in job_info

    def test_sync_client_jobs_cleans_up_stale_job_tracking(self):
        """When a job is removed from run_processes the corresponding tracking
        entry in _job_reported_clients must be purged on the next sync call."""
        with (
            patch("nvflare.private.fed.server.fed_server.ServerEngine"),
            patch("nvflare.private.fed.server.fed_server.ConfigService.get_bool_var", return_value=True),
        ):
            server = FederatedServer(
                project_name="project_name",
                min_num_clients=1,
                max_num_clients=10,
                cmd_modules=None,
                heart_beat_timeout=600,
                args=MagicMock(),
                secure_train=False,
                snapshot_persistor=MagicMock(),
                overseer_agent=MagicMock(),
            )

            token = "token-1"
            client = MagicMock()
            client.name = "C1"
            server.engine.run_processes = {"job1": {RunProcessKey.PARTICIPANTS: {token: client}}}
            server.engine.notify_dead_job = MagicMock()

            # Positive observation — entry created in _job_reported_clients
            job_present = new_cell_message({CellMessageHeaderKeys.JOB_IDS: ["job1"]}, Shareable())
            server._sync_client_jobs(job_present, token)
            assert "job1" in server._job_reported_clients

            # Job finishes — removed from run_processes
            server.engine.run_processes = {}

            # Next sync call for any client should purge the stale entry
            other_request = new_cell_message({CellMessageHeaderKeys.JOB_IDS: []}, Shareable())
            server._sync_client_jobs(other_request, token)
            assert "job1" not in server._job_reported_clients
