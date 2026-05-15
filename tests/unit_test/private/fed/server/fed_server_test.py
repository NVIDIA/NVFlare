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

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.fl_constant import RunProcessKey
from nvflare.apis.job_def import JobMetaKey, RunStatus
from nvflare.apis.job_launcher_spec import JobReturnCode
from nvflare.apis.shareable import Shareable
from nvflare.fuel.common.exit_codes import ProcessExitCode
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as F3ReturnCode
from nvflare.private.defs import CellMessageHeaderKeys, ClientRegMsgKey, JobFailureMsgKey, new_cell_message
from nvflare.private.fed.authenticator import MISSING_CLIENT_FQCN
from nvflare.private.fed.server.fed_server import FederatedServer
from nvflare.private.fed.server.server_state import DEFAULT_SERVICE_SESSION_ID, HotState


class TestFederatedServer:
    def test_resolve_client_fqcn_for_auth_fails_closed_for_registered_client_with_missing_fqcn(self):
        server = object.__new__(FederatedServer)
        client = MagicMock()
        client.name = "site-a"
        client.get_fqcn.return_value = None
        server.client_manager = MagicMock()
        server.client_manager.clients = {"token-a": client}

        assert server._resolve_client_fqcn_for_auth("site-a", "token-a") == MISSING_CLIENT_FQCN

    def test_hot_state_defaults_to_non_empty_session_id(self):
        assert HotState().ssid == DEFAULT_SERVICE_SESSION_ID

    @pytest.mark.parametrize("server_state, expected", [(HotState(), ["extra_job"])])
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

    def test_set_job_aborted_marks_runner_without_publishing_status(self):
        server = object.__new__(FederatedServer)
        server.logger = MagicMock()
        server.engine = MagicMock()

        job_manager = MagicMock()
        server.engine.get_component.return_value = job_manager
        job_manager.get_job.return_value = MagicMock(meta={JobMetaKey.STATUS: RunStatus.RUNNING})

        fl_ctx = MagicMock()
        server.engine.new_context.return_value = nullcontext(fl_ctx)
        server.engine.job_runner.mark_run_aborted.return_value = ""

        server._set_job_aborted("job-1")

        server.engine.job_runner.mark_run_aborted.assert_called_once_with("job-1", fl_ctx)
        job_manager.set_status.assert_not_called()

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

    def test_disabled_client_heartbeat_is_rejected(self, tmp_path):
        with patch("nvflare.private.fed.server.fed_server.ServerEngine"):
            args = MagicMock()
            args.workspace = str(tmp_path)
            server = FederatedServer(
                project_name="project_name",
                min_num_clients=1,
                max_num_clients=10,
                cmd_modules=None,
                heart_beat_timeout=600,
                args=args,
                secure_train=False,
                snapshot_persistor=MagicMock(),
            )
            server.server_state = HotState()
            server.client_manager.disable_client("client_name")

            request = new_cell_message(
                {
                    CellMessageHeaderKeys.TOKEN: "token",
                    CellMessageHeaderKeys.SSID: "ssid",
                    CellMessageHeaderKeys.CLIENT_NAME: "client_name",
                    CellMessageHeaderKeys.PROJECT_NAME: "project_name",
                    CellMessageHeaderKeys.JOB_IDS: [],
                },
                Shareable(),
            )

            result = server.client_heartbeat(request)

            assert result.get_header(MessageHeaderKey.RETURN_CODE) == F3ReturnCode.UNAUTHENTICATED
            assert "disabled" in result.get_header(MessageHeaderKey.ERROR)
            assert "token" not in server.client_manager.clients

    @pytest.mark.parametrize("failure_code", [JobReturnCode.ABORTED, ProcessExitCode.UNSAFE_COMPONENT])
    def test_process_job_failure_stops_run_for_reported_abort_client_failures(self, failure_code):
        with patch("nvflare.private.fed.server.fed_server.ServerEngine"):
            server = FederatedServer(
                project_name="project_name",
                min_num_clients=1,
                max_num_clients=10,
                cmd_modules=None,
                heart_beat_timeout=600,
                args=MagicMock(),
                secure_train=False,
                snapshot_persistor=MagicMock(),
            )

            server.client_manager.is_from_authorized_client = MagicMock(return_value=True)
            fl_ctx = MagicMock()
            server.engine.new_context.return_value = nullcontext(fl_ctx)
            server.engine.job_runner.stop_run = MagicMock()
            server.engine.job_runner.fail_run = MagicMock()

            request = new_cell_message(
                {
                    CellMessageHeaderKeys.TOKEN: "token-1",
                    MessageHeaderKey.ORIGIN: "site-1",
                },
                {
                    JobFailureMsgKey.JOB_ID: "job-1",
                    JobFailureMsgKey.CODE: failure_code,
                    JobFailureMsgKey.REASON: "fatal client failure",
                },
            )

            server.process_job_failure(request)

            server.engine.job_runner.stop_run.assert_called_once_with("job-1", fl_ctx)
            server.engine.job_runner.fail_run.assert_not_called()

    @pytest.mark.parametrize(
        "failure_code",
        [ProcessExitCode.CONFIG_ERROR, ProcessExitCode.EXCEPTION],
    )
    def test_process_job_failure_fails_run_for_reported_exception_client_failures(self, failure_code):
        with patch("nvflare.private.fed.server.fed_server.ServerEngine"):
            server = FederatedServer(
                project_name="project_name",
                min_num_clients=1,
                max_num_clients=10,
                cmd_modules=None,
                heart_beat_timeout=600,
                args=MagicMock(),
                secure_train=False,
                snapshot_persistor=MagicMock(),
            )

            server.client_manager.is_from_authorized_client = MagicMock(return_value=True)
            fl_ctx = MagicMock()
            server.engine.new_context.return_value = nullcontext(fl_ctx)
            server.engine.job_runner.stop_run = MagicMock()
            server.engine.job_runner.fail_run = MagicMock()

            request = new_cell_message(
                {
                    CellMessageHeaderKeys.TOKEN: "token-1",
                    MessageHeaderKey.ORIGIN: "site-1",
                },
                {
                    JobFailureMsgKey.JOB_ID: "job-1",
                    JobFailureMsgKey.CODE: failure_code,
                    JobFailureMsgKey.REASON: "fatal client failure",
                },
            )

            server.process_job_failure(request)

            server.engine.job_runner.fail_run.assert_called_once_with("job-1", ProcessExitCode.EXCEPTION, fl_ctx)
            server.engine.job_runner.stop_run.assert_not_called()

    def test_process_job_failure_ignores_generic_launcher_execution_error(self):
        with patch("nvflare.private.fed.server.fed_server.ServerEngine"):
            server = FederatedServer(
                project_name="project_name",
                min_num_clients=1,
                max_num_clients=10,
                cmd_modules=None,
                heart_beat_timeout=600,
                args=MagicMock(),
                secure_train=False,
                snapshot_persistor=MagicMock(),
            )

            server.client_manager.is_from_authorized_client = MagicMock(return_value=True)
            server.engine.job_runner.stop_run = MagicMock()
            server.engine.job_runner.fail_run = MagicMock()

            request = new_cell_message(
                {
                    CellMessageHeaderKeys.TOKEN: "token-1",
                    MessageHeaderKey.ORIGIN: "site-1",
                },
                {
                    JobFailureMsgKey.JOB_ID: "job-1",
                    JobFailureMsgKey.CODE: JobReturnCode.EXECUTION_ERROR,
                    JobFailureMsgKey.REASON: "generic launcher failure",
                },
            )

            server.process_job_failure(request)

            server.engine.job_runner.fail_run.assert_not_called()
            server.engine.job_runner.stop_run.assert_not_called()


class TestGetValidatedSiteConfig:
    """_get_validated_site_config doesn't depend on instance state beyond
    self.logger and the class-level size cap, so we drive it with a MagicMock
    self instead of constructing a full FederatedServer."""

    def _call(self, shareable):
        mock_self = MagicMock()
        mock_self._SITE_CONFIG_MAX_SERIALIZED_BYTES = FederatedServer._SITE_CONFIG_MAX_SERIALIZED_BYTES
        return FederatedServer._get_validated_site_config(mock_self, shareable, "site-1")

    def test_returns_none_when_missing(self):
        assert self._call(Shareable()) is None

    def test_returns_none_when_not_a_dict(self):
        s = Shareable()
        s[ClientRegMsgKey.SITE_CONFIG] = ["bad"]
        assert self._call(s) is None

    def test_returns_none_when_not_json_serializable(self):
        s = Shareable()
        s[ClientRegMsgKey.SITE_CONFIG] = {"x": {1, 2, 3}}  # set is not JSON-serializable
        assert self._call(s) is None

    def test_returns_none_when_oversized(self):
        s = Shareable()
        s[ClientRegMsgKey.SITE_CONFIG] = {"blob": "a" * (FederatedServer._SITE_CONFIG_MAX_SERIALIZED_BYTES + 1)}
        assert self._call(s) is None

    def test_returns_dict_when_valid(self):
        site_config = {"format_version": 1, "labels": {"region": "us-east"}}
        s = Shareable()
        s[ClientRegMsgKey.SITE_CONFIG] = site_config
        assert self._call(s) == site_config
