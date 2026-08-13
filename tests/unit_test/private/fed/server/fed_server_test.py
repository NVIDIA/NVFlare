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

from nvflare.apis.fl_constant import ConnPropKey, RunProcessKey
from nvflare.apis.job_def import JobMetaKey, RunStatus
from nvflare.apis.job_launcher_spec import JobReturnCode
from nvflare.apis.shareable import Shareable
from nvflare.fuel.common.exit_codes import ProcessExitCode
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, MessagePropKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as F3ReturnCode
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.private.defs import CellChannel, CellMessageHeaderKeys, ClientRegMsgKey, JobFailureMsgKey, new_cell_message
from nvflare.private.fed.authenticator import MISSING_CLIENT_FQCN
from nvflare.private.fed.server.fed_server import FederatedServer
from nvflare.private.fed.server.server_command_agent import ServerCommandAgent
from nvflare.private.fed.server.server_engine import ServerEngine
from nvflare.private.fed.server.server_state import DEFAULT_SERVICE_SESSION_ID, HotState, ServerState


def assert_client_outcome_unresolved(job_runner):
    job_runner.resolve_client_outcome.assert_not_called()


class TestFederatedServer:
    @staticmethod
    def _server_credential_message(origin, peer):
        message = new_cell_message(
            {
                MessageHeaderKey.ORIGIN: origin,
                MessageHeaderKey.DESTINATION: "server.job-1",
                CellMessageHeaderKeys.CLIENT_NAME: "server",
                CellMessageHeaderKeys.TOKEN: "server",
            },
            Shareable(),
        )
        message.set_prop(MessagePropKey.ENDPOINT, Endpoint(peer))
        return message

    @staticmethod
    def _server_for_auth_validation():
        server = object.__new__(FederatedServer)
        server.my_own_auth_client_name = "server"
        server.my_own_token = "server"
        server.logger = MagicMock()
        server.cell = MagicMock()
        server.cell.get_fqcn.return_value = "server"
        server._get_id_asserter = MagicMock(return_value=MagicMock(cert=MagicMock()))
        return server

    @pytest.mark.parametrize(
        "origin,peer",
        [("server", "site-1"), ("server.job-1", "site-1.job-1"), ("server", "server.job-1")],
    )
    def test_server_credentials_cannot_be_replayed_from_client_family(self, origin, peer):
        server = self._server_for_auth_validation()
        message = self._server_credential_message(origin, peer)

        with patch("nvflare.private.fed.server.fed_server.validate_auth_headers", return_value=None):
            reply = server._validate_auth_headers(message)

        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == F3ReturnCode.UNAUTHENTICATED

    @pytest.mark.parametrize("origin,peer", [("server", "server"), ("server.job-1", "server.job-1")])
    def test_server_credentials_are_accepted_from_server_family(self, origin, peer):
        server = self._server_for_auth_validation()
        message = self._server_credential_message(origin, peer)

        with patch("nvflare.private.fed.server.fed_server.validate_auth_headers", return_value=None):
            reply = server._validate_auth_headers(message)

        assert reply is None

    @staticmethod
    def _create_job_cell_with_command_agent(server_state):
        server = object.__new__(FederatedServer)
        engine = ServerEngine.__new__(ServerEngine)
        engine.server = server
        engine.run_manager = None
        server.engine = engine
        server.server_state = server_state

        with (
            patch("nvflare.private.fed.server.fed_server.Cell") as cell_cls,
            patch("nvflare.private.fed.server.fed_server.NetAgent"),
            patch("nvflare.private.fed.server.fed_server.mpm.add_cleanup_cb"),
        ):
            cell = MagicMock()
            cell_cls.return_value = cell
            server.create_job_cell("job-1", "tcp://root", "tcp://parent", False, None)

        engine.set_cell(cell)
        aux_calls = [
            call
            for call in cell.register_request_cb.call_args_list
            if call.kwargs["channel"] == CellChannel.AUX_COMMUNICATION and call.kwargs["topic"] == "*"
        ]
        assert len(aux_calls) == 1
        return server, engine, cell, aux_calls[0].kwargs["cb"]

    def test_resolve_client_fqcn_for_auth_fails_closed_for_registered_client_with_missing_fqcn(self):
        server = object.__new__(FederatedServer)
        client = MagicMock()
        client.name = "site-a"
        client.get_fqcn.return_value = None
        server.client_manager = MagicMock()
        server.client_manager.clients = {"token-a": client}

        assert server._resolve_client_fqcn_for_auth("site-a", "token-a") == MISSING_CLIENT_FQCN

    def test_create_job_cell_allows_missing_server_config_for_non_secure_cell(self):
        server = object.__new__(FederatedServer)
        server.engine = MagicMock()

        with (
            patch("nvflare.private.fed.server.fed_server.Cell") as cell_cls,
            patch("nvflare.private.fed.server.fed_server.NetAgent") as net_agent_cls,
            patch("nvflare.private.fed.server.fed_server.ServerCommandAgent") as command_agent_cls,
            patch("nvflare.private.fed.server.fed_server.mpm.add_cleanup_cb"),
        ):
            cell = MagicMock()
            cell_cls.return_value = cell
            net_agent_cls.return_value = MagicMock()
            command_agent_cls.return_value = MagicMock()

            result = server.create_job_cell("job-1", "tcp://root", "tcp://parent", False, None)

        assert result is cell
        assert cell_cls.call_args.kwargs["credentials"] == {}
        assert cell_cls.call_args.kwargs["auth_identity"] is None
        assert cell_cls.call_args.kwargs["auth_identity_map"] is None

    def test_create_job_cell_uses_auth_identity_from_server_config(self):
        server = object.__new__(FederatedServer)
        server.engine = MagicMock()
        auth_identity_map = {"server": "server-cn"}

        with (
            patch("nvflare.private.fed.server.fed_server.Cell") as cell_cls,
            patch("nvflare.private.fed.server.fed_server.NetAgent") as net_agent_cls,
            patch("nvflare.private.fed.server.fed_server.ServerCommandAgent") as command_agent_cls,
            patch("nvflare.private.fed.server.fed_server.mpm.add_cleanup_cb"),
        ):
            cell_cls.return_value = MagicMock()
            net_agent_cls.return_value = MagicMock()
            command_agent_cls.return_value = MagicMock()

            server.create_job_cell(
                "job-1",
                "tcp://root",
                "tcp://parent",
                False,
                {
                    ConnPropKey.AUTH_IDENTITY: "server-cn",
                    ConnPropKey.AUTH_IDENTITY_MAP: auth_identity_map,
                },
            )

        assert cell_cls.call_args.kwargs["auth_identity"] == "server-cn"
        assert cell_cls.call_args.kwargs["auth_identity_map"] == auth_identity_map

    def test_set_cell_preserves_server_command_agent_aux_callback(self):
        server, engine, cell, aux_callback = self._create_job_cell_with_command_agent(HotState(ssid="ssid"))

        assert engine.cell is cell
        assert aux_callback.__self__ is server.command_agent
        assert aux_callback.__func__ is ServerCommandAgent.aux_communicate

    @pytest.mark.parametrize(
        "server_state,request_ssid",
        [
            (HotState(ssid="expected-ssid"), "wrong-ssid"),
            (ServerState(ssid="expected-ssid"), "expected-ssid"),
        ],
    )
    def test_server_job_aux_callback_rejects_invalid_ssid_or_state(self, server_state, request_ssid):
        _, engine, _, aux_callback = self._create_job_cell_with_command_agent(server_state)
        fl_ctx = MagicMock()
        fl_ctx.get_engine.return_value = engine
        engine.new_context = MagicMock(return_value=nullcontext(fl_ctx))
        engine.dispatch = MagicMock()
        request = new_cell_message(
            {
                MessageHeaderKey.TOPIC: "test_topic",
                CellMessageHeaderKeys.SSID: request_ssid,
            },
            Shareable(),
        )

        result = aux_callback(request)

        assert result.get_header(MessageHeaderKey.RETURN_CODE) == F3ReturnCode.AUTHENTICATION_ERROR
        engine.dispatch.assert_not_called()

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

    @pytest.mark.parametrize(
        "run_processes, exception_run_processes, expected",
        [
            ({"job1": {}}, {}, []),
            ({}, {}, []),
            ({}, {"job1": {}}, ["job1"]),
        ],
        ids=["server-running", "awaiting-client-outcome", "server-failed"],
    )
    def test_sync_client_jobs_keeps_outcome_barrier_only_without_terminal_server_failure(
        self, run_processes, exception_run_processes, expected
    ):
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
            server.engine.run_processes = run_processes
            server.engine.exception_run_processes = exception_run_processes
            server.engine.job_runner.get_client_outcome_jobs.return_value = {"job1"}
            request = new_cell_message({CellMessageHeaderKeys.JOB_IDS: ["job1"]}, Shareable())

            assert server._sync_client_jobs(request, "token-1") == expected
            if exception_run_processes:
                client = MagicMock()
                client.name = "C1"
                server.client_manager.clients = {"token-1": client}
                server.engine.job_runner.is_client_outcome_pending.return_value = True
                request = new_cell_message({CellMessageHeaderKeys.JOB_IDS: []}, Shareable())

                assert server._sync_client_jobs(request, "token-1") == []
                server.engine.job_runner.fail_run.assert_not_called()
                server.engine.job_runner.resolve_client_outcome.assert_called_once_with("job1", "C1")

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

            # A client that no longer has a barrier-only job cannot report again.
            server.client_manager.clients = {token: client}
            server.engine.job_runner.get_client_outcome_jobs.return_value = {"job1"}
            server.engine.job_runner.is_client_outcome_pending.return_value = True
            fl_ctx = MagicMock()
            server.engine.new_context.return_value = nullcontext(fl_ctx)
            server._sync_client_jobs(other_request, token)
            server.engine.job_runner.fail_run.assert_called_once_with(
                "job1", ProcessExitCode.INFRASTRUCTURE_ERROR, fl_ctx
            )
            server.engine.job_runner.resolve_client_outcome.assert_called_once_with("job1", "C1")

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

    def test_process_job_failure_stops_run_for_reported_unsafe_client_failure(self):
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
            server.client_manager.clients = {"token-1": MagicMock(name="site-1")}
            server.client_manager.clients["token-1"].name = "site-1"
            server.engine.job_runner.is_client_outcome_pending.return_value = True
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
                    JobFailureMsgKey.CODE: ProcessExitCode.UNSAFE_COMPONENT,
                    JobFailureMsgKey.REASON: "fatal client failure",
                },
            )

            server.process_job_failure(request)

            server.engine.job_runner.stop_run.assert_called_once_with("job-1", fl_ctx)
            server.engine.job_runner.fail_run.assert_not_called()
            server.engine.job_runner.resolve_client_outcome.assert_called_once_with("job-1", "site-1")

    @pytest.mark.parametrize(
        "failure_code, expected_code",
        [
            (ProcessExitCode.CONFIG_ERROR, ProcessExitCode.EXCEPTION),
            (ProcessExitCode.EXCEPTION, ProcessExitCode.EXCEPTION),
            (ProcessExitCode.INFRASTRUCTURE_ERROR, ProcessExitCode.INFRASTRUCTURE_ERROR),
            (JobReturnCode.ABORTED, JobReturnCode.ABORTED),
        ],
    )
    def test_process_job_failure_fails_run_for_reported_client_failures(self, failure_code, expected_code):
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
            server.client_manager.clients = {"token-1": MagicMock(name="site-1")}
            server.client_manager.clients["token-1"].name = "site-1"
            server.engine.job_runner.is_client_outcome_pending.return_value = True
            fl_ctx = MagicMock()
            server.engine.new_context.return_value = nullcontext(fl_ctx)
            server.engine.job_runner.stop_run = MagicMock()
            server.engine.job_runner.fail_run = MagicMock()
            server.engine.job_runner.fail_run.side_effect = lambda *_: assert_client_outcome_unresolved(
                server.engine.job_runner
            )

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

            server.engine.job_runner.fail_run.assert_called_once_with("job-1", expected_code, fl_ctx)
            server.engine.job_runner.stop_run.assert_not_called()
            server.engine.job_runner.resolve_client_outcome.assert_called_once_with("job-1", "site-1")

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
            server.client_manager.clients = {"token-1": MagicMock(name="site-1")}
            server.client_manager.clients["token-1"].name = "site-1"
            server.engine.job_runner.is_client_outcome_pending.return_value = True
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
            server.engine.job_runner.is_client_outcome_pending.return_value = False
            result = server.process_job_failure(request)
            assert result.get_header(MessageHeaderKey.RETURN_CODE) == F3ReturnCode.OK
            server.engine.job_runner.resolve_client_outcome.assert_called_once_with("job-1", "site-1")

    def test_notify_dead_client_fails_barrier_only_job(self):
        server = object.__new__(FederatedServer)
        server.logger = MagicMock()
        server.engine = MagicMock()
        server.engine.run_processes = {}
        server.engine.job_runner.get_client_outcome_jobs.return_value = {"job-1"}
        server.engine.job_runner.is_client_outcome_pending.return_value = True
        fl_ctx = MagicMock()
        server.engine.new_context.return_value = nullcontext(fl_ctx)
        client = MagicMock()
        client.name = "site-1"

        server.notify_dead_client(client)

        server.engine.job_runner.fail_run.assert_called_once_with("job-1", ProcessExitCode.INFRASTRUCTURE_ERROR, fl_ctx)
        server.engine.job_runner.resolve_client_outcome.assert_called_once_with("job-1", "site-1")


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
