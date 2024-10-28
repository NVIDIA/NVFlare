# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import time
from datetime import datetime
from typing import List

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, Task
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.ccwf.common import (
    Constant,
    StatusReport,
    make_task_name,
    status_report_from_dict,
    topic_for_end_workflow,
)
from nvflare.fuel.utils.validation_utils import (
    DefaultValuePolicy,
    check_number_range,
    check_object_type,
    check_positive_int,
    check_positive_number,
    check_str,
    normalize_config_arg,
    validate_candidate,
    validate_candidates,
)
from nvflare.security.logging import secure_format_traceback


class ClientStatus:
    def __init__(self):
        self.ready_time = None
        self.last_report_time = time.time()
        self.last_progress_time = time.time()
        self.num_reports = 0
        self.status = StatusReport()


class ServerSideController(Controller):
    def __init__(
        self,
        num_rounds: int = 1,
        start_round: int = 0,
        task_name_prefix: str = "wf",
        configure_task_timeout=Constant.CONFIG_TASK_TIMEOUT,
        end_workflow_timeout=Constant.END_WORKFLOW_TIMEOUT,
        start_task_timeout=Constant.START_TASK_TIMEOUT,
        task_check_period: float = Constant.TASK_CHECK_INTERVAL,
        job_status_check_interval: float = Constant.JOB_STATUS_CHECK_INTERVAL,
        starting_client: str = "",
        starting_client_policy: str = DefaultValuePolicy.ANY,
        participating_clients=None,
        result_clients: List[str] = [],
        result_clients_policy: str = DefaultValuePolicy.ALL,
        max_status_report_interval: float = Constant.PER_CLIENT_STATUS_REPORT_TIMEOUT,
        progress_timeout: float = Constant.WORKFLOW_PROGRESS_TIMEOUT,
        private_p2p: bool = True,
    ):
        """
        Constructor

        Args:
            num_rounds - the number of rounds to be performed. This is a workflow config parameter. Defaults to 1.
            start_round - the starting round number. This is a workflow config parameter.
            task_name_prefix - the prefix for task names of this workflow.
                The workflow requires multiple tasks (e.g. config and start) between the server controller and the client.
                The full names of these tasks are <prefix>_config and <prefix>_start.
                Subclasses may send additional tasks. Naming these tasks with a common prefix can make it easier to
                configure task executors for FL clients.
            participating_clients - the names of the clients that will participate in the job. None means all clients.
            result_clients - names of the clients that will receive final learning results.
            result_clients_policy - how to determine result_clients if their names are not explicitly specified.
                Possible values are:
                    ALL - all participating clients
                    ANY - any one of the participating clients
                    EMPTY - no result_clients
                    DISALLOW - does not allow implicit - result_clients must be explicitly specified
            configure_task_timeout - time to wait for clients’ responses to the config task before timeout.
            starting_client - name of the starting client.
            starting_client_policy - how to determine the starting client if the name is not explicitly specified.
                Possible values are:
                    ANY - any one of the participating clients (the first client)
                    RANDOM - a random client
                    EMPTY - no starting client
                    DISALLOW - does not allow implicit - starting_client must be explicitly specified
            start_task_timeout - how long to wait for the starting client to finish the “start” task.
                If timed out, the job will be aborted.
                If the starting_client is not specified, then no start task will be sent.
                max_status_report_interval - the maximum amount of time allowed for a client to miss a status report.
                In other words, if a client fails to report its status for this much time, the client will be considered in
                trouble and the job will be aborted.
            progress_timeout- the maximum amount of time allowed for the workflow to not make any progress.
                In other words, at least one participating client must have made progress during this time.
                Otherwise, the workflow will be considered to be in trouble and the job will be aborted.
            end_workflow_timeout - timeout for ending workflow message.
            private_p2p - whether to make peer-to-peer communications private.
                When set to True, P2P communications will be encrypted.
                Private P2P communication is an additional level of protection on basic communication security
                (such as SSL). Each pair of peers have their own encryption keys to ensure that only they themselves
                can understand their messages, even if the messages may be relayed through other sites (e.g. server).
                Different pairs of peers have different keys.
                Currently, private P2P is enabled only when the system is in secure mode. This is because key exchange
                between peers requires both sides to have PKI certificates and keys, which requires the project
                to be provisioned in secure mode.
        """
        Controller.__init__(self, task_check_period)
        participating_clients = normalize_config_arg(participating_clients)
        if participating_clients is None:
            raise ValueError("participating_clients must not be empty")

        self.task_name_prefix = task_name_prefix
        self.configure_task_name = make_task_name(task_name_prefix, Constant.BASENAME_CONFIG)
        self.configure_task_timeout = configure_task_timeout
        self.start_task_name = make_task_name(task_name_prefix, Constant.BASENAME_START)
        self.start_task_timeout = start_task_timeout
        self.end_workflow_timeout = end_workflow_timeout
        self.num_rounds = num_rounds
        self.start_round = start_round
        self.max_status_report_interval = max_status_report_interval
        self.progress_timeout = progress_timeout
        self.job_status_check_interval = job_status_check_interval
        self.starting_client = starting_client
        self.starting_client_policy = starting_client_policy
        self.participating_clients = participating_clients
        self.result_clients = result_clients
        self.result_clients_policy = result_clients_policy

        # make private_p2p bool
        check_object_type("private_p2p", private_p2p, bool)
        self.private_p2p = private_p2p

        self.client_statuses = {}  # client name => ClientStatus
        self.cw_started = False
        self.asked_to_stop = False
        self.workflow_id = None

        check_positive_int("num_rounds", num_rounds)
        check_number_range("configure_task_timeout", configure_task_timeout, min_value=1)
        check_number_range("end_workflow_timeout", end_workflow_timeout, min_value=1)
        check_positive_number("job_status_check_interval", job_status_check_interval)
        check_number_range("max_status_report_interval", max_status_report_interval, min_value=10.0)
        check_number_range("progress_timeout", progress_timeout, min_value=5.0)
        check_str("starting_client_policy", starting_client_policy)

        if participating_clients and len(participating_clients) < 2:
            raise ValueError(f"Not enough participating_clients: must > 1, but got {participating_clients}")

    def start_controller(self, fl_ctx: FLContext):
        wf_id = fl_ctx.get_prop(FLContextKey.WORKFLOW)
        self.log_debug(fl_ctx, f"starting controller for workflow {wf_id}")
        if not wf_id:
            raise RuntimeError("workflow ID is missing from FL context")
        self.workflow_id = wf_id

        all_clients = self._engine.get_clients()
        if len(all_clients) < 2:
            raise RuntimeError(f"this workflow requires at least 2 clients, but only got {all_clients}")

        all_client_names = [t.name for t in all_clients]
        self.participating_clients = validate_candidates(
            var_name="participating_clients",
            candidates=self.participating_clients,
            base=all_client_names,
            default_policy=DefaultValuePolicy.ALL,
            allow_none=False,
        )

        self.log_info(fl_ctx, f"Using participating clients: {self.participating_clients}")
        self.starting_client = validate_candidate(
            var_name="starting_client",
            candidate=self.starting_client,
            base=self.participating_clients,
            default_policy=self.starting_client_policy,
            allow_none=True,
        )
        self.log_info(fl_ctx, f"Starting client: {self.starting_client}")

        self.result_clients = validate_candidates(
            var_name="result_clients",
            candidates=self.result_clients,
            base=self.participating_clients,
            default_policy=self.result_clients_policy,
            allow_none=True,
        )

        for c in self.participating_clients:
            self.client_statuses[c] = ClientStatus()

    def prepare_config(self) -> dict:
        return {}

    def sub_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        pass

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        # wait for every client to become ready
        self.log_info(fl_ctx, f"Waiting for clients to be ready: {self.participating_clients}")

        # GET STARTED
        self.log_info(fl_ctx, f"Configuring clients {self.participating_clients} for workflow {self.workflow_id}")

        learn_config = {
            Constant.PRIVATE_P2P: self.private_p2p,
            Constant.TASK_NAME_PREFIX: self.task_name_prefix,
            Constant.CLIENTS: self.participating_clients,
            Constant.START_CLIENT: self.starting_client,
            Constant.RESULT_CLIENTS: self.result_clients,
            AppConstants.NUM_ROUNDS: self.num_rounds,
            Constant.START_ROUND: self.start_round,
            FLContextKey.WORKFLOW: self.workflow_id,
        }

        extra_config = self.prepare_config()
        if extra_config:
            learn_config.update(extra_config)

        self.log_info(fl_ctx, f"Workflow Config: {learn_config}")

        # configure all clients
        shareable = Shareable()
        shareable[Constant.CONFIG] = learn_config

        task = Task(
            name=self.configure_task_name,
            data=shareable,
            timeout=self.configure_task_timeout,
            result_received_cb=self._process_configure_reply,
        )

        self.log_info(fl_ctx, f"sending task {self.configure_task_name} to clients {self.participating_clients}")
        start_time = time.time()
        self.broadcast_and_wait(
            task=task,
            targets=self.participating_clients,
            min_responses=len(self.participating_clients),
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )

        time_taken = time.time() - start_time
        self.log_info(fl_ctx, f"client configuration took {time_taken} seconds")

        failed_clients = []
        for c, cs in self.client_statuses.items():
            assert isinstance(cs, ClientStatus)
            if not cs.ready_time:
                failed_clients.append(c)

        if failed_clients:
            self.system_panic(
                f"failed to configure clients {failed_clients}",
                fl_ctx,
            )
            return

        self.log_info(fl_ctx, f"successfully configured clients {self.participating_clients}")

        # starting the starting_client
        if self.starting_client:
            shareable = Shareable()
            task = Task(
                name=self.start_task_name,
                data=shareable,
                timeout=self.start_task_timeout,
                result_received_cb=self._process_start_reply,
            )

            self.log_info(fl_ctx, f"sending task {self.start_task_name} to client {self.starting_client}")

            self.send_and_wait(
                task=task,
                targets=[self.starting_client],
                fl_ctx=fl_ctx,
                abort_signal=abort_signal,
            )

            if not self.cw_started:
                self.system_panic(
                    f"failed to start workflow {self.workflow_id} on client {self.starting_client}",
                    fl_ctx,
                )
                return

            self.log_info(fl_ctx, f"started workflow {self.workflow_id} on client {self.starting_client}")

        # a subclass could provide additional control flow
        self.sub_flow(abort_signal, fl_ctx)

        self.log_info(fl_ctx, f"Waiting for clients to finish workflow {self.workflow_id}  ...")
        while not abort_signal.triggered and not self.asked_to_stop:
            time.sleep(self.job_status_check_interval)
            done = self._check_job_status(fl_ctx)
            if done:
                break

        self.log_info(fl_ctx, f"Workflow {self.workflow_id} finished on all clients")

        # ask all clients to end the workflow
        self.log_info(fl_ctx, f"asking all clients to end workflow {self.workflow_id}")
        engine = fl_ctx.get_engine()
        end_wf_request = Shareable()
        resp = engine.send_aux_request(
            targets=self.participating_clients,
            topic=topic_for_end_workflow(self.workflow_id),
            request=end_wf_request,
            timeout=self.end_workflow_timeout,
            fl_ctx=fl_ctx,
            secure=False,
        )

        assert isinstance(resp, dict)
        num_errors = 0
        for c in self.participating_clients:
            reply = resp.get(c)
            if not reply:
                self.log_error(fl_ctx, f"not reply from client {c} for ending workflow {self.workflow_id}")
                num_errors += 1
                continue

            assert isinstance(reply, Shareable)
            rc = reply.get_return_code(ReturnCode.OK)
            if rc != ReturnCode.OK:
                self.log_error(fl_ctx, f"client {c} failed to end workflow {self.workflow_id}: {rc}")
                num_errors += 1

        if num_errors > 0:
            self.system_panic(f"failed to end workflow {self.workflow_id} on all clients", fl_ctx)

        self.log_info(fl_ctx, f"Workflow {self.workflow_id} done!")

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.BEFORE_PROCESS_TASK_REQUEST:
            self._update_client_status(fl_ctx)

    def process_config_reply(self, client_name: str, reply: Shareable, fl_ctx: FLContext) -> bool:
        return True

    def _process_configure_reply(self, client_task: ClientTask, fl_ctx: FLContext):
        result = client_task.result
        client_name = client_task.client.name

        rc = result.get_return_code()
        if rc == ReturnCode.OK:
            self.log_info(fl_ctx, f"successfully configured client {client_name}")

            try:
                ok = self.process_config_reply(client_name, result, fl_ctx)
                if not ok:
                    return
            except:
                self.log_error(
                    fl_ctx, f"exception processing config reply from client {client_name}: {secure_format_traceback()}"
                )
                return
            cs = self.client_statuses.get(client_name)
            if cs:
                assert isinstance(cs, ClientStatus)
                cs.ready_time = time.time()
        else:
            error = result.get(Constant.ERROR, "?")
            self.log_error(fl_ctx, f"client {client_task.client.name} failed to configure: {rc}: {error}")

    def client_started(self, client_task: ClientTask, fl_ctx: FLContext):
        return True

    def _process_start_reply(self, client_task: ClientTask, fl_ctx: FLContext):
        result = client_task.result
        client_name = client_task.client.name

        rc = result.get_return_code()
        if rc == ReturnCode.OK:
            try:
                ok = self.client_started(client_task, fl_ctx)
                if not ok:
                    return
            except:
                self.log_info(fl_ctx, f"exception in client_started: {secure_format_traceback()}")
                return

            self.cw_started = True
        else:
            error = result.get(Constant.ERROR, "?")
            self.log_error(
                fl_ctx, f"client {client_task.client.name} couldn't start workflow {self.workflow_id}: {rc}: {error}"
            )

    def is_sub_flow_done(self, fl_ctx: FLContext) -> bool:
        return False

    def _check_job_status(self, fl_ctx: FLContext):
        # see whether the server side thinks it's done
        if self.is_sub_flow_done(fl_ctx):
            return True

        now = time.time()
        overall_last_progress_time = 0.0
        for client_name, cs in self.client_statuses.items():
            assert isinstance(cs, ClientStatus)
            assert isinstance(cs.status, StatusReport)

            if cs.status.all_done:
                self.log_info(fl_ctx, f"Got ALL_DONE from client {client_name}")
                return True

            if now - cs.last_report_time > self.max_status_report_interval:
                self.system_panic(
                    f"client {client_name} didn't report status for {self.max_status_report_interval} seconds",
                    fl_ctx,
                )
                return True

            if overall_last_progress_time < cs.last_progress_time:
                overall_last_progress_time = cs.last_progress_time

        if time.time() - overall_last_progress_time > self.progress_timeout:
            self.system_panic(
                f"the workflow {self.workflow_id} has no progress for {self.progress_timeout} seconds",
                fl_ctx,
            )
            return True

        return False

    def _update_client_status(self, fl_ctx: FLContext):
        peer_ctx = fl_ctx.get_peer_context()
        assert isinstance(peer_ctx, FLContext)
        client_name = peer_ctx.get_identity_name()

        # see whether status is available
        reports = peer_ctx.get_prop(Constant.STATUS_REPORTS)
        if not reports:
            self.log_debug(fl_ctx, f"no status report from client {client_name}")
            return

        my_report = reports.get(self.workflow_id)
        if not my_report:
            return

        if client_name not in self.client_statuses:
            self.log_error(fl_ctx, f"received result from unknown client {client_name}!")
            return

        report = status_report_from_dict(my_report)
        cs = self.client_statuses[client_name]
        assert isinstance(cs, ClientStatus)
        now = time.time()
        cs.last_report_time = now
        cs.num_reports += 1

        if report.error:
            self.asked_to_stop = True
            self.system_panic(f"received failure report from client {client_name}: {report.error}", fl_ctx)
            return

        if cs.status != report:
            # updated
            cs.status = report
            cs.last_progress_time = now
            timestamp = datetime.fromtimestamp(report.timestamp) if report.timestamp else False
            self.log_info(
                fl_ctx,
                f"updated status of client {client_name} on round {report.last_round}: "
                f"timestamp={timestamp}, action={report.action}, all_done={report.all_done}",
            )
        else:
            self.log_debug(
                fl_ctx, f"ignored status report from client {client_name} at round {report.last_round}: no change"
            )

    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        self.log_warning(fl_ctx, f"ignored unknown task {task_name} from client {client.name}")

    def stop_controller(self, fl_ctx: FLContext):
        pass
