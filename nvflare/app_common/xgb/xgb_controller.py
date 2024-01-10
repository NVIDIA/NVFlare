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
from nvflare.fuel.f3.drivers.net_utils import get_open_tcp_port
from nvflare.fuel.utils.validation_utils import (
    check_number_range,
    check_positive_number,
    check_str,
    check_object_type
)
from .defs import Constant
from .xgb.client import XGBClient
from .process_manager import ProcessManager


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
        run_xgb_server_cmd: str,
        xgb_server_addr=None,
        configure_task_name=Constant.CONFIG_TASK_NAME,
        configure_task_timeout=Constant.CONFIG_TASK_TIMEOUT,
        start_task_name=Constant.START_TASK_NAME,
        start_task_timeout=Constant.START_TASK_TIMEOUT,
        job_status_check_interval: float = Constant.JOB_STATUS_CHECK_INTERVAL,
        max_status_report_interval: float = Constant.PER_CLIENT_STATUS_REPORT_TIMEOUT,
        progress_timeout: float = Constant.WORKFLOW_PROGRESS_TIMEOUT,
        max_run_time=None,
        xgb_server_ready_timeout=Constant.XGB_SERVER_READY_TIMEOUT,
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
        self.run_xgb_server_cmd = run_xgb_server_cmd
        self.xgb_server_addr = xgb_server_addr
        self.configure_task_name = configure_task_name
        self.start_task_name = start_task_name
        self.start_task_timeout = start_task_timeout
        self.configure_task_timeout = configure_task_timeout
        self.max_status_report_interval = max_status_report_interval
        self.progress_timeout = progress_timeout
        self.job_status_check_interval = job_status_check_interval
        self.max_run_time = max_run_time
        self.xgb_server_ready_timeout = xgb_server_ready_timeout
        self.client_ranks = client_ranks  # client rank assignments

        self.participating_clients = None
        self.status_lock = threading.Lock()
        self.client_statuses = {}  # client name => ClientStatus
        self.abort_signal = None
        self.internal_xgb_client = None
        self.xgb_server_monitor = None

        check_str('run_xgb_server_cmd', run_xgb_server_cmd)
        check_number_range("configure_task_timeout", configure_task_timeout, min_value=1)
        check_positive_number("job_status_check_interval", job_status_check_interval)
        check_number_range("max_status_report_interval", max_status_report_interval, min_value=10.0)
        check_number_range("progress_timeout", progress_timeout, min_value=5.0)
        if client_ranks:
            check_object_type('client_ranks', client_ranks, dict)

    def start_controller(self, fl_ctx: FLContext):
        all_clients = self._engine.get_clients()
        self.participating_clients = [t.name for t in all_clients]

        for c in self.participating_clients:
            self.client_statuses[c] = ClientStatus()

        engine = fl_ctx.get_engine()
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

    def _xgb_server_stopped(self, rc, fl_ctx: FLContext):
        # XGB server process stopped
        error = None
        if rc != 0:
            self.log_error(fl_ctx, f"XGB Server stopped abnormally with code {rc}")
            error = "XGB server abnormal stop"

        # the XGB server could stop at any moment, we trigger the abort_signal in case it is checked by any
        # other components
        self._trigger_stop(fl_ctx, error)

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
        exit_code = request.get(Constant.KEY_EXIT_CODE)
        self.log_info(fl_ctx, f"XGB client is done with exit code {exit_code}")
        self._update_client_status(fl_ctx, client_done=True)
        return make_reply(ReturnCode.OK)

    def _process_xgb_request(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        if self._is_stopped():
            self.log_error(fl_ctx, f"dropped XGB request since server is already stopped")
            return make_reply(ReturnCode.SERVICE_UNAVAILABLE)

        # since XGB protocol is very strict, we'll stop the control flow when any error occurs
        op = request.get(Constant.KEY_XGB_OP)
        self.log_info(fl_ctx, f"received XGB request '{op}'")
        bad_req_error = "bad XGB request"
        process_error = "xgb request process error"
        if not op:
            self.log_error(fl_ctx, "missing op from XGB request")
            self._trigger_stop(fl_ctx, bad_req_error)
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        if not op in [Constant.OP_BROADCAST, Constant.OP_ALL_REDUCE, Constant.OP_ALL_GATHER_V, Constant.OP_ALL_GATHER]:
            self.log_error(fl_ctx, f"invalid op '{op}' from XGB request")
            self._trigger_stop(fl_ctx, bad_req_error)
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        self._update_client_status(fl_ctx, op=op)

        serialized_xgb_req = request.get(Constant.KEY_XGB_MSG)
        if not serialized_xgb_req:
            self.log_error(fl_ctx, "missing request data from XGB request")
            self._trigger_stop(fl_ctx, bad_req_error)
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        self.log_info(fl_ctx, f"request size={len(serialized_xgb_req)}")

        try:
            serialized_xgb_reply = self.internal_xgb_client.forward_request(op, serialized_xgb_req)
            self.log_info(fl_ctx, f"forwarded '{op}' to external XGB server")
        except Exception as ex:
            self.log_exception(fl_ctx, f"exception forwarding request to XGB server {secure_format_exception(ex)}")
            self._trigger_stop(fl_ctx, process_error)
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        if not serialized_xgb_reply:
            self._trigger_stop(fl_ctx, process_error)
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        self.log_info(fl_ctx, f"received reply from XGB server for '{op}': {len(serialized_xgb_reply)} bytes")
        result = Shareable()
        result[Constant.KEY_XGB_OP] = op
        result[Constant.KEY_XGB_MSG] = serialized_xgb_reply
        return result

    def _configure_clients(self, abort_signal: Signal, fl_ctx: FLContext):
        self.log_info(fl_ctx, f"Configuring clients {self.participating_clients}")

        shareable = Shareable()
        shareable[Constant.KEY_MAX_RUN_TIME] = self.max_run_time

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

        shareable[Constant.KEY_CLIENT_RANKS] = self.client_ranks

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

        # wait for every client to become ready
        self.log_info(fl_ctx, f"Waiting for clients to be ready: {self.participating_clients}")

        # configure all clients
        if not self._configure_clients(abort_signal, fl_ctx):
            self.system_panic("failed to configure all clients", fl_ctx)
            return

        # We start the XGB server here so that when FL clients are started in the control_flow,
        # we will be ready for their XGB requests
        if not self.xgb_server_addr:
            # we dynamically create server address on localhost
            port = get_open_tcp_port(resources={})
            if not port:
                self.system_panic("failed to get a port for XGB server", fl_ctx)
                return
            self.xgb_server_addr = f"127.0.0.1:{port}"

        self.run_xgb_server_cmd = self.run_xgb_server_cmd.replace("$addr", self.xgb_server_addr)
        self.run_xgb_server_cmd = self.run_xgb_server_cmd.replace(
            "$num_clients", str(len(self.participating_clients)))

        self.xgb_server_monitor = ProcessManager(
            name="XGBServer",
            start_cmd=self.run_xgb_server_cmd,
            stopped_cb=self._xgb_server_stopped,
            fl_ctx=fl_ctx,
        )
        self.xgb_server_monitor.start()

        # start XGB client
        self.internal_xgb_client = XGBClient(self.xgb_server_addr)
        try:
            self.internal_xgb_client.start(ready_timeout=self.xgb_server_ready_timeout)
        except Exception as ex:
            error = f"XGB server not ready in {self.xgb_server_ready_timeout} seconds: {secure_format_exception(ex)}"
            self.log_exception(fl_ctx, error)
            self.system_panic(reason=error, fl_ctx=fl_ctx)
            return

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
        if self.internal_xgb_client:
            self.log_info(fl_ctx, "Stopping internal XGB client")
            self.internal_xgb_client.stop()

        if self.xgb_server_monitor:
            # stop the XGB server
            self.log_info(fl_ctx, "Stopping XGB Server Monitor")
            self.xgb_server_monitor.stop()
