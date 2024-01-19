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
import threading
import time

from nvflare.apis.client import Client
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import ClientTask, Controller, Task
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.security.logging import secure_format_exception
from nvflare.apis.signal import Signal
from nvflare.fuel.utils.validation_utils import (
    check_number_range,
    check_positive_number,
    check_str,
    check_object_type
)
from nvflare.app_common.xgb.bridge import XGBServerBridge
from .defs import Constant


class ClientStatus:
    def __init__(self):
        self.configured_time = None
        self.started_time = None
        self.last_op = None
        self.last_report_time = time.time()
        self.xgb_done = False


class XGBController(Controller):
    def __init__(
        self,
        bridge_component_id: str,
        num_rounds: int,
        configure_task_name=Constant.CONFIG_TASK_NAME,
        configure_task_timeout=Constant.CONFIG_TASK_TIMEOUT,
        start_task_name=Constant.START_TASK_NAME,
        start_task_timeout=Constant.START_TASK_TIMEOUT,
        job_status_check_interval: float = Constant.JOB_STATUS_CHECK_INTERVAL,
        max_status_report_interval: float = Constant.PER_CLIENT_STATUS_REPORT_TIMEOUT,
        progress_timeout: float = Constant.WORKFLOW_PROGRESS_TIMEOUT,
        max_run_time=None,
        client_ranks=None,
    ):
        """
        Constructor

        Args:
            configure_task_timeout - time to wait for clients’ responses to the config task before timeout.
            progress_timeout- the maximum amount of time allowed for the workflow to not make any progress.
                In other words, at least one participating client must have made progress during this time.
                Otherwise, the workflow will be considered to be in trouble and the job will be aborted.
            end_workflow_timeout - timeout for ending workflow message.
        """
        Controller.__init__(self)
        self.bridge_component_id = bridge_component_id
        self.num_rounds = num_rounds
        self.configure_task_name = configure_task_name
        self.start_task_name = start_task_name
        self.start_task_timeout = start_task_timeout
        self.configure_task_timeout = configure_task_timeout
        self.max_status_report_interval = max_status_report_interval
        self.progress_timeout = progress_timeout
        self.job_status_check_interval = job_status_check_interval
        self.max_run_time = max_run_time
        self.client_ranks = client_ranks  # client rank assignments

        self.bridge = None
        self.participating_clients = None
        self.status_lock = threading.Lock()
        self.client_statuses = {}  # client name => ClientStatus
        self.abort_signal = None

        check_str('bridge_component_id', bridge_component_id)
        check_number_range("configure_task_timeout", configure_task_timeout, min_value=1)
        check_positive_number("job_status_check_interval", job_status_check_interval)
        check_positive_number("num_rounds", num_rounds)
        check_number_range("max_status_report_interval", max_status_report_interval, min_value=10.0)
        check_number_range("progress_timeout", progress_timeout, min_value=5.0)
        if client_ranks:
            check_object_type('client_ranks', client_ranks, dict)

        self.op_table = {
            Constant.OP_ALL_GATHER: self._process_all_gather,
            Constant.OP_ALL_GATHER_V: self._process_all_gather_v,
            Constant.OP_ALL_REDUCE: self._process_all_reduce,
            Constant.OP_BROADCAST: self._process_broadcast,
        }

    def start_controller(self, fl_ctx: FLContext):
        all_clients = self._engine.get_clients()
        self.participating_clients = [t.name for t in all_clients]

        for c in self.participating_clients:
            self.client_statuses[c] = ClientStatus()

        engine = fl_ctx.get_engine()
        bridge = engine.get_component(self.bridge_component_id)
        if not bridge:
            self.system_panic(f"cannot get component for {self.bridge_component_id}", fl_ctx)
            return

        if not isinstance(bridge, XGBServerBridge):
            self.system_panic(
                f"invalid component for {self.bridge_component_id}: expect XGBServerBridge but got {type(bridge)}",
                fl_ctx)
            return

        self.bridge = bridge

        engine.register_aux_message_handler(
            topic=Constant.TOPIC_XGB_REQUEST,
            message_handle_func=self._process_xgb_request,
        )
        engine.register_aux_message_handler(
            topic=Constant.TOPIC_CLIENT_DONE,
            message_handle_func=self._process_client_done,
        )

    def _trigger_stop(self, fl_ctx:FLContext, error=None):
        if self.abort_signal:
            self.abort_signal.trigger(value=True)
        if error:
            self.system_panic(reason=error, fl_ctx=fl_ctx)

    def _is_stopped(self):
        return self.abort_signal and self.abort_signal.triggered

    def _update_client_status(self, fl_ctx: FLContext, op=None, client_done=False):
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
            status.last_report_time = time.time()

    def _process_client_done(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        exit_code = request.get(Constant.CONF_KEY_EXIT_CODE)
        self.log_info(fl_ctx, f"XGB client is done with exit code {exit_code}")
        self._update_client_status(fl_ctx, client_done=True)
        return make_reply(ReturnCode.OK)

    def _process_all_gather(self, request: Shareable, fl_ctx: FLContext) -> Shareable:
        rank = request.get(Constant.KEY_XGB_RANK)
        seq = request.get(Constant.KEY_XGB_SEQ)
        send_buf = request.get(Constant.KEY_XGB_SEND_BUF)
        rcv_buf = self.bridge.all_gather(rank, seq, send_buf, fl_ctx)
        reply = Shareable()
        reply[Constant.KEY_XGB_RCV_BUF] = rcv_buf
        return reply

    def _process_all_gather_v(self, request: Shareable, fl_ctx: FLContext) -> Shareable:
        rank = request.get(Constant.KEY_XGB_RANK)
        seq = request.get(Constant.KEY_XGB_SEQ)
        send_buf = request.get(Constant.KEY_XGB_SEND_BUF)
        rcv_buf = self.bridge.all_gather_v(rank, seq, send_buf, fl_ctx)
        reply = Shareable()
        reply[Constant.KEY_XGB_RCV_BUF] = rcv_buf
        return reply

    def _process_all_reduce(self, request: Shareable, fl_ctx: FLContext) -> Shareable:
        rank = request.get(Constant.KEY_XGB_RANK)
        seq = request.get(Constant.KEY_XGB_SEQ)
        send_buf = request.get(Constant.KEY_XGB_SEND_BUF)
        data_type = request.get(Constant.KEY_XGB_DATA_TYPE)
        reduce_op = request.get(Constant.KEY_XGB_REDUCE_OP)
        assert isinstance(self.bridge, XGBServerBridge)
        rcv_buf = self.bridge.all_reduce(rank, seq, data_type, reduce_op, send_buf, fl_ctx)
        reply = Shareable()
        reply[Constant.KEY_XGB_RCV_BUF] = rcv_buf
        return reply

    def _process_broadcast(self, request: Shareable, fl_ctx: FLContext) -> Shareable:
        rank = request.get(Constant.KEY_XGB_RANK)
        seq = request.get(Constant.KEY_XGB_SEQ)
        send_buf = request.get(Constant.KEY_XGB_SEND_BUF)
        root = request.get(Constant.KEY_XGB_ROOT)
        assert isinstance(self.bridge, XGBServerBridge)
        rcv_buf = self.bridge.broadcast(rank, seq, root, send_buf, fl_ctx)
        reply = Shareable()
        reply[Constant.KEY_XGB_RCV_BUF] = rcv_buf
        return reply

    def _process_xgb_request(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        if self._is_stopped():
            self.log_error(fl_ctx, f"dropped XGB request since server is already stopped")
            return make_reply(ReturnCode.SERVICE_UNAVAILABLE)

        # since XGB protocol is very strict, we'll stop the control flow when any error occurs
        op = request.get_header(Constant.KEY_XGB_OP)
        self.log_info(fl_ctx, f"received XGB request '{op}'")
        bad_req_error = "bad XGB request"
        process_error = "XGB request process error"
        if not op:
            self.log_error(fl_ctx, "missing op from XGB request")
            self._trigger_stop(fl_ctx, bad_req_error)
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        process_f = self.op_table.get(op)
        if process_f is None:
            self.log_error(fl_ctx, f"invalid op '{op}' from XGB request")
            self._trigger_stop(fl_ctx, bad_req_error)
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        self._update_client_status(fl_ctx, op=op)
        assert callable(process_f)
        try:
            reply = process_f(request, fl_ctx)
        except Exception as ex:
            self.log_exception(fl_ctx, f"exception processing {op}: {secure_format_exception(ex)}")
            self._trigger_stop(fl_ctx, process_error)
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        self.log_info(fl_ctx, f"received reply for '{op}'")
        reply.set_header(Constant.KEY_XGB_OP, op)
        return reply

    def _configure_clients(self, abort_signal: Signal, fl_ctx: FLContext):
        self.log_info(fl_ctx, f"Configuring clients {self.participating_clients}")

        shareable = Shareable()
        shareable[Constant.CONF_KEY_MAX_RUN_TIME] = self.max_run_time

        # compute client ranks
        if not self.client_ranks:
            # dynamically assign ranks
            self.client_ranks = {}
            for i, c in enumerate(self.participating_clients):
                self.client_ranks[c] = i
        else:
            # validate ranks
            if len(self.client_ranks) != len(self.participating_clients):
                self.system_panic(
                    f"expecting rank assignments for {self.participating_clients} but got {self.client_ranks}",
                    fl_ctx)
                return False

            for c in self.participating_clients:
                if c not in self.client_ranks:
                    self.system_panic(f"missing rank assignment for client '{c}'", fl_ctx)
                    return False

            for c, r in self.client_ranks.items():
                if not isinstance(r, int):
                    self.system_panic(
                        f"bad rank assignment {r} for client '{c}': expect int but got {type(r)}",
                        fl_ctx)
                    return False

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

        if failed_clients:
            self.system_panic(f"failed to start clients {failed_clients}", fl_ctx)
            return False

        self.log_info(fl_ctx, f"successfully started clients {self.participating_clients}")
        return True

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        self.abort_signal = abort_signal
        self.bridge.set_abort_signal(abort_signal)

        # wait for every client to become ready
        self.log_info(fl_ctx, f"Waiting for clients to be ready: {self.participating_clients}")

        # configure all clients
        if not self._configure_clients(abort_signal, fl_ctx):
            self.system_panic("failed to configure all clients", fl_ctx)
            return

        # We start the server bridge here
        try:
            self.bridge.configure({Constant.CONF_KEY_WORLD_SIZE: len(self.participating_clients)}, fl_ctx)
            self.bridge.start(fl_ctx)
        except Exception as ex:
            error = f"failed to start bridge: {secure_format_exception(ex)}"
            self.log_error(fl_ctx, error)
            self.system_panic(error, fl_ctx)
            return

        assert isinstance(self.bridge, XGBServerBridge)
        self.bridge.monitor_target(fl_ctx, self._xgb_server_stopped)

        # start all clients
        if not self._start_clients(abort_signal, fl_ctx):
            self.system_panic("failed to start all clients", fl_ctx)
            return

        self.log_info(fl_ctx, f"Waiting for clients to finish ...")
        while not self._is_stopped():
            done = self._check_job_status(fl_ctx)
            if done:
                break
            time.sleep(self.job_status_check_interval)

    def _xgb_server_stopped(self, rc, fl_ctx: FLContext):
        # XGB server process stopped
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

    def _check_job_status(self, fl_ctx: FLContext):
        now = time.time()
        overall_last_progress_time = 0.0
        clients_done = 0
        for client_name, cs in self.client_statuses.items():
            assert isinstance(cs, ClientStatus)

            if cs.xgb_done:
                self.log_info(fl_ctx, f"client {client_name} is Done")
                clients_done += 1
            elif now - cs.last_report_time > self.max_status_report_interval:
                self.system_panic(
                    f"client {client_name} didn't have any activity for {self.max_status_report_interval} seconds",
                    fl_ctx,
                )
                return True

            if overall_last_progress_time < cs.last_report_time:
                overall_last_progress_time = cs.last_report_time

        if clients_done == len(self.client_statuses):
            # all clients are done
            return True
        elif time.time() - overall_last_progress_time > self.progress_timeout:
            self.system_panic(f"the job has no progress for {self.progress_timeout} seconds", fl_ctx)
            return True
        return False

    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        self.log_warning(fl_ctx, f"ignored unknown task {task_name} from client {client.name}")

    def stop_controller(self, fl_ctx: FLContext):
        if self.bridge:
            self.log_info(fl_ctx, "Stopping server bridge")
            self.bridge.stop(fl_ctx)
