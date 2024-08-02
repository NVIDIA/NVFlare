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

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, Task
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.utils.reliable_message import ReliableMessage
from nvflare.app_common.tie.connector import Connector
from nvflare.fuel.utils.validation_utils import check_number_range, check_positive_number
from nvflare.security.logging import secure_format_exception

from .applet import Applet
from .defs import Constant


class _ClientStatus:
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

        # operation of the last request from this client
        self.last_op = None

        # time of the last op request from this client
        self.last_op_time = time.time()

        # whether the app process is finished on this client
        self.app_done = False


class TieController(Controller, ABC):
    def __init__(
        self,
        configure_task_name=Constant.CONFIG_TASK_NAME,
        configure_task_timeout=Constant.CONFIG_TASK_TIMEOUT,
        start_task_name=Constant.START_TASK_NAME,
        start_task_timeout=Constant.START_TASK_TIMEOUT,
        job_status_check_interval: float = Constant.JOB_STATUS_CHECK_INTERVAL,
        max_client_op_interval: float = Constant.MAX_CLIENT_OP_INTERVAL,
        progress_timeout: float = Constant.WORKFLOW_PROGRESS_TIMEOUT,
    ):
        """
        Constructor

        Args:
            configure_task_name - name of the config task
            configure_task_timeout - time to wait for clients’ responses to the config task before timeout.
            start_task_name - name of the start task
            start_task_timeout - time to wait for clients’ responses to the start task before timeout.
            job_status_check_interval - how often to check client statuses of the job
            max_client_op_interval - max amount of time allowed between app ops from a client
            progress_timeout- the maximum amount of time allowed for the workflow to not make any progress.
                In other words, at least one participating client must have made progress during this time.
                Otherwise, the workflow will be considered to be in trouble and the job will be aborted.
        """
        Controller.__init__(self)
        self.configure_task_name = configure_task_name
        self.start_task_name = start_task_name
        self.start_task_timeout = start_task_timeout
        self.configure_task_timeout = configure_task_timeout
        self.max_client_op_interval = max_client_op_interval
        self.progress_timeout = progress_timeout
        self.job_status_check_interval = job_status_check_interval

        self.connector = None
        self.participating_clients = None
        self.status_lock = threading.Lock()
        self.client_statuses = {}  # client name => ClientStatus
        self.abort_signal = None

        check_number_range("configure_task_timeout", configure_task_timeout, min_value=1)
        check_number_range("start_task_timeout", start_task_timeout, min_value=1)
        check_positive_number("job_status_check_interval", job_status_check_interval)
        check_number_range("max_client_op_interval", max_client_op_interval, min_value=10.0)
        check_number_range("progress_timeout", progress_timeout, min_value=5.0)

    @abstractmethod
    def get_client_config_params(self, fl_ctx: FLContext) -> dict:
        """Called by the TieController to get config parameters to be sent to FL clients.
        Subclass of TieController must implement this method.

        Args:
            fl_ctx: FL context

        Returns: a dict of config params

        """
        pass

    @abstractmethod
    def get_connector_config_params(self, fl_ctx: FLContext) -> dict:
        """Called by the TieController to get config parameters for configuring the connector.
        Subclass of TieController must implement this method.

        Args:
            fl_ctx: FL context

        Returns: a dict of config params

        """
        pass

    @abstractmethod
    def get_connector(self, fl_ctx: FLContext) -> Connector:
        """Called by the TieController to get the Connector to be used with the controller.
        Subclass of TieController must implement this method.

        Args:
            fl_ctx: FL context

        Returns: a Connector object

        """
        pass

    @abstractmethod
    def get_applet(self, fl_ctx: FLContext) -> Applet:
        """Called by the TieController to get the Applet to be used with the controller.
        Subclass of TieController must implement this method.

        Args:
            fl_ctx: FL context

        Returns: an Applet object

        """
        pass

    def start_controller(self, fl_ctx: FLContext):
        """Start the controller.
        It first tries to get the connector and applet to be used.
        It then initializes the applet, set the applet to the connector, and initializes the connector.
        It finally registers message handlers for APP_REQUEST and CLIENT_DONE.
        If error occurs in any step, the job is stopped.

        Note: if a subclass overwrites this method, it must call super().start_controller()!

        Args:
            fl_ctx: the FL context

        Returns: None

        """
        all_clients = self._engine.get_clients()
        self.participating_clients = [t.name for t in all_clients]

        for c in self.participating_clients:
            self.client_statuses[c] = _ClientStatus()

        connector = self.get_connector(fl_ctx)
        if not connector:
            self.system_panic("cannot get connector", fl_ctx)
            return None

        if not isinstance(connector, Connector):
            self.system_panic(
                f"invalid connector: expect Connector but got {type(connector)}",
                fl_ctx,
            )
            return None

        applet = self.get_applet(fl_ctx)
        if not applet:
            self.system_panic("cannot get applet", fl_ctx)
            return

        if not isinstance(applet, Applet):
            self.system_panic(
                f"invalid applet: expect Applet but got {type(applet)}",
                fl_ctx,
            )
            return

        applet.initialize(fl_ctx)
        connector.set_applet(applet)
        connector.initialize(fl_ctx)
        self.connector = connector

        engine = fl_ctx.get_engine()
        engine.register_aux_message_handler(
            topic=Constant.TOPIC_CLIENT_DONE,
            message_handle_func=self._process_client_done,
        )
        ReliableMessage.register_request_handler(
            topic=Constant.TOPIC_APP_REQUEST,
            handler_f=self._handle_app_request,
            fl_ctx=fl_ctx,
        )

    def _trigger_stop(self, fl_ctx: FLContext, error=None):
        # first trigger the abort_signal to tell all components (mainly the controller's control_flow and connector)
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
            op: the app operation requested
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
            assert isinstance(status, _ClientStatus)
            if op:
                status.last_op = op
            if client_done:
                status.app_done = client_done
            status.last_op_time = time.time()

    def _process_client_done(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        """Process the ClientDone report for a client

        Args:
            topic: topic of the message
            request: request to be processed
            fl_ctx: the FL context

        Returns: reply to the client

        """
        self.log_debug(fl_ctx, f"_process_client_done {topic}")
        exit_code = request.get(Constant.MSG_KEY_EXIT_CODE)

        if exit_code == 0:
            self.log_info(fl_ctx, f"app client is done with exit code {exit_code}")
        elif exit_code == Constant.EXIT_CODE_CANT_START:
            self.log_error(fl_ctx, f"app client failed to start (exit code {exit_code})")
            self.system_panic("app client failed to start", fl_ctx)
        else:
            # Should we stop here?
            # Problem is that even if the exit_code is not 0, we can't say the job failed.
            self.log_warning(fl_ctx, f"app client is done with exit code {exit_code}")

        self._update_client_status(fl_ctx, client_done=True)
        return make_reply(ReturnCode.OK)

    def _handle_app_request(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        """Handle app request from applets on other sites
        It calls the connector to process the app request. If the connector fails to process the request, the
        job will be stopped.

        Args:
            topic: message topic
            request: the request data
            fl_ctx: FL context

        Returns: processing result as a Shareable object

        """
        self.log_debug(fl_ctx, f"_handle_app_request {topic}")
        op = request.get_header(Constant.MSG_KEY_OP)
        if self._is_stopped():
            self.log_warning(fl_ctx, f"dropped app request ({op=}) since server is already stopped")
            return make_reply(ReturnCode.SERVICE_UNAVAILABLE)

        # we assume app protocol to be very strict, we'll stop the control flow when any error occurs
        process_error = "app request process error"
        self._update_client_status(fl_ctx, op=op)
        try:
            reply = self.connector.process_app_request(op, request, fl_ctx, self.abort_signal)
        except Exception as ex:
            self.log_exception(fl_ctx, f"exception processing app request {op=}: {secure_format_exception(ex)}")
            self._trigger_stop(fl_ctx, process_error)
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        self.log_info(fl_ctx, f"received reply for app request '{op=}'")
        reply.set_header(Constant.MSG_KEY_OP, op)
        return reply

    def _configure_clients(self, abort_signal: Signal, fl_ctx: FLContext):
        self.log_info(fl_ctx, f"Configuring clients {self.participating_clients}")

        try:
            config = self.get_client_config_params(fl_ctx)
        except Exception as ex:
            self.system_panic(f"exception get_client_config_params: {secure_format_exception(ex)}", fl_ctx)
            return False

        if config is None:
            self.system_panic("no config data is returned", fl_ctx)
            return False

        shareable = Shareable()
        shareable[Constant.MSG_KEY_CONFIG] = config

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
            assert isinstance(cs, _ClientStatus)
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
            assert isinstance(cs, _ClientStatus)
            if not cs.started_time:
                failed_clients.append(c)

        # if any client failed to start, terminate the job
        if failed_clients:
            self.system_panic(f"failed to start clients {failed_clients}", fl_ctx)
            return False

        self.log_info(fl_ctx, f"successfully started clients {self.participating_clients}")
        return True

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        """
        To ensure smooth app execution:
        - ensure that all clients are online and ready to go before starting server
        - ensure that server is started and ready to take requests before asking clients to start operation
        - monitor the health of the clients
        - if anything goes wrong, terminate the job

        Args:
            abort_signal: abort signal that is used to notify components to abort
            fl_ctx: FL context

        Returns: None

        """
        self.abort_signal = abort_signal

        # the connector uses the same abort signal!
        self.connector.set_abort_signal(abort_signal)

        # wait for every client to become online and properly configured
        self.log_info(fl_ctx, f"Waiting for clients to be ready: {self.participating_clients}")

        # configure all clients
        if not self._configure_clients(abort_signal, fl_ctx):
            self.system_panic("failed to configure all clients", fl_ctx)
            return

        # configure and start the connector
        try:
            config = self.get_connector_config_params(fl_ctx)
            self.connector.configure(config, fl_ctx)
            self.log_info(fl_ctx, "starting connector ...")
            self.connector.start(fl_ctx)
        except Exception as ex:
            error = f"failed to start connector: {secure_format_exception(ex)}"
            self.log_error(fl_ctx, error)
            self.system_panic(error, fl_ctx)
            return

        self.connector.monitor(fl_ctx, self._app_stopped)

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

    def _app_stopped(self, rc, fl_ctx: FLContext):
        # This CB is called when app server is stopped
        error = None
        if rc != 0:
            self.log_error(fl_ctx, f"App Server stopped abnormally with code {rc}")
            error = "App server abnormal stop"

        # the app server could stop at any moment, we trigger the abort_signal in case it is checked by any
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
                assert isinstance(cs, _ClientStatus)
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
                assert isinstance(cs, _ClientStatus)
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
            assert isinstance(cs, _ClientStatus)

            if cs.app_done:
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
        """This is called by base controller to stop.
        If a subclass overwrites this method, it must call super().stop_controller(fl_ctx).

        Args:
            fl_ctx:

        Returns:

        """
        if self.connector:
            self.log_info(fl_ctx, "Stopping server connector ...")
            self.connector.stop(fl_ctx)
            self.log_info(fl_ctx, "Server connector stopped")
