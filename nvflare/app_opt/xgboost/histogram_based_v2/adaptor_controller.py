# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import time
from abc import ABC, abstractmethod
from typing import Optional

from nvflare.apis.client import Client
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import ClientTask, Controller, Task
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_opt.xgboost.histogram_based_v2.adaptor import XGBServerAdaptor
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.fuel.utils.validation_utils import check_number_range, check_object_type, check_positive_number, check_str
from nvflare.security.logging import secure_format_exception


class ClientStatus:
    """
    Objects of this class keep processing status of each FL client during job execution.
    """

    def __init__(self):
        # Set when the client's config reply is received and the reply return code is OK.
        # If the client failed to reply or the return code is not OK, this value is not set.
        self.configured_time = None

        # Set when the client's start reply is received and the reply return code is OK.
        # If the client failed to reply or the return code is not OK, this value is not set.
        self.started_time = None

        # operation of the last XGB request from this client
        self.last_op = None

        # time of the last XGB op request from this client
        self.last_op_time = time.time()

        # whether the XGB process is done on this client
        self.xgb_done = False


class _XGBRequestHandler(ABC):
    @staticmethod
    @abstractmethod
    def process_request(adaptor: XGBServerAdaptor, request: Shareable, fl_ctx: FLContext) -> Shareable:
        """Process an XGB request.

        Args:
            adaptor: An XGBServerAdaptor
            request: the request containing op params
            fl_ctx: FL context

        Returns:
            A Shareable containing operation result
        """
        pass


class _AllGatherRequestHandler(_XGBRequestHandler):
    """This is the op handler for AllGather using XGBServerAdaptor."""

    @staticmethod
    def process_request(adaptor: XGBServerAdaptor, request: Shareable, fl_ctx: FLContext) -> Shareable:
        rank = request.get(Constant.PARAM_KEY_RANK)
        seq = request.get(Constant.PARAM_KEY_SEQ)
        send_buf = request.get(Constant.PARAM_KEY_SEND_BUF)
        rcv_buf = adaptor.all_gather(rank, seq, send_buf, fl_ctx)
        reply = Shareable()
        reply[Constant.PARAM_KEY_RCV_BUF] = rcv_buf
        return reply


class _AllReduceRequestHandler(_XGBRequestHandler):
    """This is the op handler for AllReduce using XGBServerAdaptor."""

    @staticmethod
    def process_request(adaptor: XGBServerAdaptor, request: Shareable, fl_ctx: FLContext) -> Shareable:
        rank = request.get(Constant.PARAM_KEY_RANK)
        seq = request.get(Constant.PARAM_KEY_SEQ)
        send_buf = request.get(Constant.PARAM_KEY_SEND_BUF)
        data_type = request.get(Constant.PARAM_KEY_DATA_TYPE)
        reduce_op = request.get(Constant.PARAM_KEY_REDUCE_OP)
        rcv_buf = adaptor.all_reduce(rank, seq, data_type, reduce_op, send_buf, fl_ctx)
        reply = Shareable()
        reply[Constant.PARAM_KEY_RCV_BUF] = rcv_buf
        return reply


class _BroadcastRequestHandler(_XGBRequestHandler):
    """This is the op handler for Broadcast using XGBServerAdaptor."""

    @staticmethod
    def process_request(adaptor: XGBServerAdaptor, request: Shareable, fl_ctx: FLContext) -> Shareable:
        rank = request.get(Constant.PARAM_KEY_RANK)
        seq = request.get(Constant.PARAM_KEY_SEQ)
        send_buf = request.get(Constant.PARAM_KEY_SEND_BUF)
        root = request.get(Constant.PARAM_KEY_ROOT)
        rcv_buf = adaptor.broadcast(rank, seq, root, send_buf, fl_ctx)
        reply = Shareable()
        reply[Constant.PARAM_KEY_RCV_BUF] = rcv_buf
        return reply


XGBOpTable = {
    Constant.OP_ALL_GATHER: _AllGatherRequestHandler,
    Constant.OP_ALL_REDUCE: _AllReduceRequestHandler,
    Constant.OP_BROADCAST: _BroadcastRequestHandler,
}


class XGBController(Controller):
    def __init__(
        self,
        adaptor_component_id: str,
        num_rounds: int,
        configure_task_name=Constant.CONFIG_TASK_NAME,
        configure_task_timeout=Constant.CONFIG_TASK_TIMEOUT,
        start_task_name=Constant.START_TASK_NAME,
        start_task_timeout=Constant.START_TASK_TIMEOUT,
        job_status_check_interval: float = Constant.JOB_STATUS_CHECK_INTERVAL,
        max_client_op_interval: float = Constant.MAX_CLIENT_OP_INTERVAL,
        progress_timeout: float = Constant.WORKFLOW_PROGRESS_TIMEOUT,
        client_ranks=None,
    ):
        """Controller for XGB.

        Args:
            adaptor_component_id - the component ID of server target adaptor
            num_rounds - number of rounds
            configure_task_name - name of the config task
            configure_task_timeout - time to wait for clients' responses to the config task before timeout.
            start_task_name - name of the start task
            start_task_timeout - time to wait for clients' responses to the start task before timeout.
            job_status_check_interval - how often to check client statuses of the job
            max_client_op_interval - max amount of time allowed between XGB ops from a client
            progress_timeout- the maximum amount of time allowed for the workflow to not make any progress.
                In other words, at least one participating client must have made progress during this time.
                Otherwise, the workflow will be considered to be in trouble and the job will be aborted.
            client_ranks: client rank assignments.
                If specified, must be a dict of client_name => rank.
                If not specified, client ranks will be randomly assigned.
                No matter how assigned, ranks must be consecutive integers, starting from 0.
        """
        Controller.__init__(self)
        self.adaptor_component_id = adaptor_component_id
        self.num_rounds = num_rounds
        self.configure_task_name = configure_task_name
        self.start_task_name = start_task_name
        self.start_task_timeout = start_task_timeout
        self.configure_task_timeout = configure_task_timeout
        self.max_client_op_interval = max_client_op_interval
        self.progress_timeout = progress_timeout
        self.job_status_check_interval = job_status_check_interval
        self.client_ranks = client_ranks  # client rank assignments

        self.adaptor = None
        self.participating_clients = None
        self.status_lock = threading.Lock()
        self.client_statuses = {}  # client name => ClientStatus
        self.abort_signal = None

        check_str("adaptor_component_id", adaptor_component_id)
        check_number_range("configure_task_timeout", configure_task_timeout, min_value=1)
        check_number_range("start_task_timeout", start_task_timeout, min_value=1)
        check_positive_number("job_status_check_interval", job_status_check_interval)
        check_positive_number("num_rounds", num_rounds)
        check_number_range("max_client_op_interval", max_client_op_interval, min_value=10.0)
        check_number_range("progress_timeout", progress_timeout, min_value=5.0)
        if client_ranks:
            check_object_type("client_ranks", client_ranks, dict)

    def get_adaptor(self, fl_ctx: FLContext) -> XGBServerAdaptor:
        engine = fl_ctx.get_engine()
        return engine.get_component(self.adaptor_component_id)

    def start_controller(self, fl_ctx: FLContext):
        all_clients = self._engine.get_clients()
        self.participating_clients = [t.name for t in all_clients]

        for c in self.participating_clients:
            self.client_statuses[c] = ClientStatus()

        adaptor = self.get_adaptor(fl_ctx)
        if not adaptor:
            self.system_panic(f"cannot get component for {self.adaptor_component_id}", fl_ctx)
            return None

        if not isinstance(adaptor, XGBServerAdaptor):
            self.system_panic(
                f"invalid component '{self.adaptor_component_id}': expect {XGBServerAdaptor.__name__} but got {type(adaptor)}",
                fl_ctx,
            )
            return None

        adaptor.initialize(fl_ctx)
        self.adaptor = adaptor

        engine = fl_ctx.get_engine()
        engine.register_aux_message_handler(
            topic=Constant.TOPIC_XGB_REQUEST,
            message_handle_func=self._process_xgb_request,
        )
        engine.register_aux_message_handler(
            topic=Constant.TOPIC_CLIENT_DONE,
            message_handle_func=self._process_client_done,
        )

    def _trigger_stop(self, fl_ctx: FLContext, error=None):
        # first trigger the abort_signal to tell all components (mainly the controller's control_flow and adaptor)
        # that check this signal to abort.
        if self.abort_signal:
            self.abort_signal.trigger(value=True)

        # if there is error, call system_panic to terminate the job with proper status.
        # if no error, the job will end normally.
        if error:
            self.system_panic(reason=error, fl_ctx=fl_ctx)

    def _is_stopped(self):
        # check whether the abort signal is triggered
        return self.abort_signal and self.abort_signal.triggered

    def _update_client_status(self, fl_ctx: FLContext, op=None, client_done=False):
        """Update the status of the requesting client.

        Args:
            fl_ctx: FL context
            op: the XGB operation requested
            client_done: whether the client is done

        Returns: None

        """
        with self.status_lock:
            peer_ctx = fl_ctx.get_peer_context()
            if not peer_ctx:
                self.log_error(fl_ctx, "missing peer_ctx from fl_ctx")
                return
            if not isinstance(peer_ctx, FLContext):
                self.log_error(fl_ctx, f"expect peer_ctx to be FLContext but got {type(peer_ctx)}")
                return
            client_name = peer_ctx.get_identity_name()
            if not client_name:
                self.log_error(fl_ctx, "missing identity from peer_ctx")
                return
            status = self.client_statuses.get(client_name)
            if not status:
                self.log_error(fl_ctx, f"no status record for client {client_name}")
            assert isinstance(status, ClientStatus)
            if op:
                status.last_op = op
            if client_done:
                status.xgb_done = client_done
            status.last_op_time = time.time()

    def _process_client_done(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        """Process the ClientDone report for a client

        Args:
            topic: topic of the message
            request: request to be processed
            fl_ctx: the FL context

        Returns: reply to the client

        """
        exit_code = request.get(Constant.MSG_KEY_EXIT_CODE)

        # TBD: should we check the exit_code and determine job status?
        # Problem is that even if the exit_code is not 0, we can't say the job failed.
        if exit_code == 0:
            self.log_info(fl_ctx, f"XGB client is done with exit code {exit_code}")
        else:
            self.log_warning(fl_ctx, f"XGB client is done with exit code {exit_code}")

        self._update_client_status(fl_ctx, client_done=True)
        return make_reply(ReturnCode.OK)

    def _process_xgb_request(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        op = request.get_header(Constant.MSG_KEY_XGB_OP)
        if self._is_stopped():
            self.log_error(fl_ctx, f"dropped XGB request '{op}' since server is already stopped")
            return make_reply(ReturnCode.SERVICE_UNAVAILABLE)

        # since XGB protocol is very strict, we'll stop the control flow when any error occurs
        bad_req_error = "bad XGB request"
        process_error = "XGB request process error"
        if not op:
            self.log_error(fl_ctx, "missing op from XGB request")
            self._trigger_stop(fl_ctx, bad_req_error)
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        # find and call the op handlers
        handler: Optional[_XGBRequestHandler] = XGBOpTable.get(op)
        if handler is None:
            self.log_error(fl_ctx, f"invalid op '{op}' from XGB request")
            self._trigger_stop(fl_ctx, bad_req_error)
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        self._update_client_status(fl_ctx, op=op)

        try:
            reply = handler.process_request(self.adaptor, request, fl_ctx)
        except Exception as ex:
            self.log_exception(fl_ctx, f"exception processing {op}: {secure_format_exception(ex)}")
            self._trigger_stop(fl_ctx, process_error)
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        self.log_info(fl_ctx, f"received reply for '{op}'")
        reply.set_header(Constant.MSG_KEY_XGB_OP, op)
        return reply

    def _configure_clients(self, abort_signal: Signal, fl_ctx: FLContext):
        self.log_info(fl_ctx, f"Configuring clients {self.participating_clients}")

        shareable = Shareable()

        # Assumption: all clients are used
        # compute client ranks
        if not self.client_ranks:
            # dynamically assign ranks, starting from 0
            clients = self.participating_clients

            # Sort by client name so rank is consistent
            clients.sort()
            self.client_ranks = {clients[i]: i for i in range(0, len(clients))}
        else:
            # validate ranks - ranks must be unique consecutive integers, starting from 0.
            num_clients = len(self.participating_clients)
            assigned_ranks = {}  # rank => client
            if len(self.client_ranks) != num_clients:
                # either missing client or duplicate client
                self.system_panic(
                    f"expecting rank assignments for {self.participating_clients} but got {self.client_ranks}", fl_ctx
                )
                return False

            # all clients must have ranks
            for c in self.participating_clients:
                if c not in self.client_ranks:
                    self.system_panic(f"missing rank assignment for client '{c}'", fl_ctx)
                    return False

            # check each client's rank
            for c, r in self.client_ranks.items():
                if not isinstance(r, int):
                    self.system_panic(f"bad rank assignment {r} for client '{c}': expect int but got {type(r)}", fl_ctx)
                    return False

                if r < 0 or r >= num_clients:
                    self.system_panic(f"bad rank assignment {r} for client '{c}': must be 0 to {num_clients-1}", fl_ctx)
                    return False

                assigned_client = assigned_ranks.get(r)
                if assigned_client:
                    self.system_panic(f"rank {r} is assigned to both client '{c}' and '{assigned_client}'", fl_ctx)
                    return False

                assigned_ranks[r] = c

        shareable[Constant.CONF_KEY_CLIENT_RANKS] = self.client_ranks
        shareable[Constant.CONF_KEY_NUM_ROUNDS] = self.num_rounds

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
            if not cs.configured_time:
                failed_clients.append(c)

        # if any client failed to configure, terminate the job
        if failed_clients:
            self.system_panic(f"failed to configure clients {failed_clients}", fl_ctx)
            return False

        self.log_info(fl_ctx, f"successfully configured clients {self.participating_clients}")
        return True

    def _start_clients(self, abort_signal: Signal, fl_ctx: FLContext):
        self.log_info(fl_ctx, f"Starting clients {self.participating_clients}")

        task = Task(
            name=self.start_task_name,
            data=Shareable(),
            timeout=self.start_task_timeout,
            result_received_cb=self._process_start_reply,
        )

        self.log_info(fl_ctx, f"sending task {self.start_task_name} to clients {self.participating_clients}")
        start_time = time.time()
        self.broadcast_and_wait(
            task=task,
            targets=self.participating_clients,
            min_responses=len(self.participating_clients),
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )

        time_taken = time.time() - start_time
        self.log_info(fl_ctx, f"client starting took {time_taken} seconds")

        failed_clients = []
        for c, cs in self.client_statuses.items():
            assert isinstance(cs, ClientStatus)
            if not cs.started_time:
                failed_clients.append(c)

        # if any client failed to start, terminate the job
        if failed_clients:
            self.system_panic(f"failed to start clients {failed_clients}", fl_ctx)
            return False

        self.log_info(fl_ctx, f"successfully started clients {self.participating_clients}")
        return True

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        """Control flow of the XGB Controller.

        To ensure smooth XGB execution:

            - ensure that all clients are online and ready to go before starting server
            - ensure that server is started and ready to take requests before asking clients to start operation
            - monitor the health of the clients
            - if anything goes wrong, terminate the job

        Args:
            abort_signal: abort signal that is used to notify components to abort
            fl_ctx: FL context

        Returns:
            None

        """
        self.abort_signal = abort_signal

        # the adaptor uses the same abort signal!
        self.adaptor.set_abort_signal(abort_signal)

        # wait for every client to become online and properly configured
        self.log_info(fl_ctx, f"Waiting for clients to be ready: {self.participating_clients}")

        # configure all clients
        if not self._configure_clients(abort_signal, fl_ctx):
            self.system_panic("failed to configure all clients", fl_ctx)
            return

        # start the server adaptor
        try:
            self.adaptor.configure(
                {
                    Constant.CONF_KEY_WORLD_SIZE: len(self.participating_clients),
                },
                fl_ctx,
            )
            self.adaptor.start(fl_ctx)
        except Exception as ex:
            error = f"failed to start adaptor: {secure_format_exception(ex)}"
            self.log_error(fl_ctx, error)
            self.system_panic(error, fl_ctx)
            return

        self.adaptor.monitor_target(fl_ctx, self._xgb_server_stopped)

        # start all clients
        if not self._start_clients(abort_signal, fl_ctx):
            self.system_panic("failed to start all clients", fl_ctx)
            return

        # monitor client health
        # we periodically check job status until all clients are done or the system is stopped
        self.log_info(fl_ctx, "Waiting for clients to finish ...")
        while not self._is_stopped():
            done = self._check_job_status(fl_ctx)
            if done:
                break
            time.sleep(self.job_status_check_interval)

    def _xgb_server_stopped(self, rc, fl_ctx: FLContext):
        # This CB is called when XGB server target is stopped
        error = None
        if rc != 0:
            self.log_error(fl_ctx, f"XGB Server stopped abnormally with code {rc}")
            error = "XGB server abnormal stop"

        # the XGB server could stop at any moment, we trigger the abort_signal in case it is checked by any
        # other components
        self._trigger_stop(fl_ctx, error)

    def _process_configure_reply(self, client_task: ClientTask, fl_ctx: FLContext):
        result = client_task.result
        client_name = client_task.client.name

        rc = result.get_return_code()
        if rc == ReturnCode.OK:
            self.log_info(fl_ctx, f"successfully configured client {client_name}")
            cs = self.client_statuses.get(client_name)
            if cs:
                assert isinstance(cs, ClientStatus)
                cs.configured_time = time.time()
        else:
            self.log_error(fl_ctx, f"client {client_task.client.name} failed to configure: {rc}")

    def _process_start_reply(self, client_task: ClientTask, fl_ctx: FLContext):
        result = client_task.result
        client_name = client_task.client.name

        rc = result.get_return_code()
        if rc == ReturnCode.OK:
            self.log_info(fl_ctx, f"successfully started client {client_name}")
            cs = self.client_statuses.get(client_name)
            if cs:
                assert isinstance(cs, ClientStatus)
                cs.started_time = time.time()
        else:
            self.log_error(fl_ctx, f"client {client_name} failed to start")

    def _check_job_status(self, fl_ctx: FLContext) -> bool:
        """Check job status and determine whether the job is done.

        Args:
            fl_ctx: FL context

        Returns: whether the job is considered done.

        """
        now = time.time()

        # overall_last_progress_time is the latest time that any client made progress.
        overall_last_progress_time = 0.0
        clients_done = 0
        for client_name, cs in self.client_statuses.items():
            assert isinstance(cs, ClientStatus)

            if cs.xgb_done:
                self.log_info(fl_ctx, f"client {client_name} is Done")
                clients_done += 1
            elif now - cs.last_op_time > self.max_client_op_interval:
                self.system_panic(
                    f"client {client_name} didn't have any activity for {self.max_client_op_interval} seconds",
                    fl_ctx,
                )
                return True

            if overall_last_progress_time < cs.last_op_time:
                overall_last_progress_time = cs.last_op_time

        if clients_done == len(self.client_statuses):
            # all clients are done - the job is considered done
            return True
        elif time.time() - overall_last_progress_time > self.progress_timeout:
            # there has been no progress from any client for too long.
            # this could be because the clients got stuck.
            # consider the job done and abort the job.
            self.system_panic(f"the job has no progress for {self.progress_timeout} seconds", fl_ctx)
            return True
        return False

    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        self.log_warning(fl_ctx, f"ignored unknown task {task_name} from client {client.name}")

    def stop_controller(self, fl_ctx: FLContext):
        if self.adaptor:
            self.log_info(fl_ctx, "Stopping server adaptor")
            self.adaptor.stop(fl_ctx)
