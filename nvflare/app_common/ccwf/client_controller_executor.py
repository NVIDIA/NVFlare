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
import copy
import threading
import time
from typing import List

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.task_controller import Task
from nvflare.apis.impl.wf_comm_client import WFCommClient
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learnable import Learnable
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.ccwf.common import Constant, StatusReport, make_task_name, topic_for_end_workflow
from nvflare.fuel.utils.validation_utils import check_number_range
from nvflare.security.logging import secure_format_exception


class ClientControllerExecutor(Executor):
    def __init__(
        self,
        controller_id_list: List,
        task_name_prefix: str = "",
        persistor_id=AppConstants.DEFAULT_PERSISTOR_ID,
        final_result_ack_timeout=Constant.FINAL_RESULT_ACK_TIMEOUT,
        max_task_timeout: int = Constant.MAX_TASK_TIMEOUT,
    ):
        """
        ClientControllerExecutor for running controllers on client-side using WFCommClient.

        Args:
            controller_id_list: List of controller ids, used in order.
            task_name_prefix: prefix of task names. All CCWF task names are prefixed with this.
            persistor_id: ID of the persistor component
            final_result_ack_timeout: timeout for sending final result to participating clients
            max_task_timeout: Maximum task timeout for Controllers using WFCommClient when `task.timeout` is set to 0. Defaults to 3600.
        """
        check_number_range("final_result_ack_timeout", final_result_ack_timeout, min_value=1.0)

        Executor.__init__(self)
        self.controller_id_list = controller_id_list
        self.task_name_prefix = task_name_prefix
        self.persistor_id = persistor_id
        self.final_result_ack_timeout = final_result_ack_timeout
        self.max_task_timeout = max_task_timeout

        self.start_task_name = make_task_name(task_name_prefix, Constant.BASENAME_START)
        self.configure_task_name = make_task_name(task_name_prefix, Constant.BASENAME_CONFIG)
        self.report_final_result_task_name = make_task_name(task_name_prefix, Constant.BASENAME_REPORT_FINAL_RESULT)

        self.persistor = None

        self.current_status = StatusReport()
        self.last_status_report_time = time.time()  # time of last status report to server
        self.config = None
        self.workflow_id = None
        self.finalize_lock = threading.Lock()

        self.asked_to_stop = False
        self.status_lock = threading.Lock()
        self.engine = None
        self.me = None
        self.is_starting_client = False
        self.workflow_done = False
        self.fatal_system_error = False

    def get_config_prop(self, name: str, default=None):
        """
        Get a specified config property.
        Args:
            name: name of the property
            default: default value to return if the property is not defined.
        Returns:
        """
        if not self.config:
            return default
        return self.config.get(name, default)

    def start_run(self, fl_ctx: FLContext):
        self.engine = fl_ctx.get_engine()
        if not self.engine:
            self.system_panic("no engine", fl_ctx)
            return

        runner = fl_ctx.get_prop(FLContextKey.RUNNER)
        if not runner:
            self.system_panic("no client runner", fl_ctx)
            return

        self.me = fl_ctx.get_identity_name()

        self.persistor = self.engine.get_component(self.persistor_id)
        if not isinstance(self.persistor, LearnablePersistor):
            self.log_warning(
                fl_ctx, f"Persistor {self.persistor_id} must be a Persistor instance but got {type(self.persistor)}"
            )
            self.persistor = None

        self.initialize(fl_ctx)

    def initialize_controller(self, controller_id, fl_ctx):
        controller = self.engine.get_component(controller_id)

        comm = WFCommClient(max_task_timeout=self.max_task_timeout)
        controller.set_communicator(comm)
        controller.config = self.config
        controller.initialize(fl_ctx)

        return controller

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.start_run(fl_ctx)

        elif event_type == EventType.BEFORE_PULL_TASK:
            # add my status to fl_ctx
            if not self.workflow_id:
                return

            reports = fl_ctx.get_prop(Constant.STATUS_REPORTS)
            if reports:
                reports.pop(self.workflow_id, None)

            if self.workflow_done:
                return
            report = self._get_status_report()
            if not report:
                self.log_debug(fl_ctx, "nothing to report this time")
                return
            self._add_status_report(report, fl_ctx)
            self.last_status_report_time = report.timestamp

        elif event_type in [EventType.ABORT_TASK, EventType.END_RUN]:
            if not self.asked_to_stop and not self.workflow_done:
                self.asked_to_stop = True
                self.finalize(fl_ctx)

        elif event_type == EventType.FATAL_SYSTEM_ERROR:
            if self.is_starting_client and not self.fatal_system_error:
                self.fatal_system_error = True
                self.fire_fed_event(EventType.FATAL_SYSTEM_ERROR, Shareable(), fl_ctx)

    def _add_status_report(self, report: StatusReport, fl_ctx: FLContext):
        reports = fl_ctx.get_prop(Constant.STATUS_REPORTS)
        if not reports:
            reports = {}
            # set the prop as public, so it will be sent to the peer in peer_context
            fl_ctx.set_prop(Constant.STATUS_REPORTS, reports, sticky=False, private=False)
        reports[self.workflow_id] = report.to_dict()

    def initialize(self, fl_ctx: FLContext):
        """Called to initialize the executor.
        Args:
            fl_ctx: The FL Context
        Returns: None
        """
        fl_ctx.set_prop(Constant.EXECUTOR, self, private=True, sticky=False)
        self.fire_event(Constant.EXECUTOR_INITIALIZED, fl_ctx)

    def finalize(self, fl_ctx: FLContext):
        """Called to finalize the executor.
        Args:
            fl_ctx: the FL Context
        Returns: None
        """
        with self.finalize_lock:
            if self.workflow_done:
                return

            fl_ctx.set_prop(Constant.EXECUTOR, self, private=True, sticky=False)
            fl_ctx.set_prop(FLContextKey.WORKFLOW, self.workflow_id, private=True, sticky=False)
            self.fire_event(Constant.EXECUTOR_FINALIZED, fl_ctx)
            self.workflow_done = True

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if self.workflow_done:
            self.log_error(fl_ctx, f"ClientControllerExecutor is finalized, not executing task {task_name}.")
            return make_reply(ReturnCode.ERROR)

        if task_name == self.configure_task_name:
            self.config = shareable[Constant.CONFIG]
            my_wf_id = self.get_config_prop(FLContextKey.WORKFLOW)
            if not my_wf_id:
                self.log_error(fl_ctx, "missing workflow id in configuration!")
                return make_reply(ReturnCode.BAD_REQUEST_DATA)
            self.log_info(fl_ctx, f"got my workflow id {my_wf_id}")
            self.workflow_id = my_wf_id

            self.engine.register_aux_message_handler(
                topic=topic_for_end_workflow(my_wf_id),
                message_handle_func=self._process_end_workflow,
            )

            return make_reply(ReturnCode.OK)

        elif task_name == self.start_task_name:
            self.is_starting_client = True

            for controller_id in self.controller_id_list:

                if self.asked_to_stop:
                    self.log_info(fl_ctx, "Asked to stop, exiting")
                    return make_reply(ReturnCode.OK)

                self.controller = self.initialize_controller(controller_id, fl_ctx)
                self.log_info(fl_ctx, f"Starting control flow {self.controller.name}")

                try:
                    res = self.controller.control_flow(abort_signal, fl_ctx)
                except Exception as e:
                    error_msg = f"{controller_id} control_flow exception: {secure_format_exception(e)}"
                    self.log_error(fl_ctx, error_msg)
                    self.system_panic(error_msg, fl_ctx)

                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                if hasattr(self.controller, "persistor"):
                    self.broadcast_final_result(self.controller.persistor.load(fl_ctx), fl_ctx)

                self.controller.stop_controller(fl_ctx)

                self.log_info(fl_ctx, f"Finished control flow {self.controller.name}")

                self.update_status(action=f"finished_{controller_id}", error=None, all_done=True)

            self.update_status(action="finished_start_task", error=None, all_done=True)

            return make_reply(ReturnCode.OK)

        elif task_name == self.report_final_result_task_name:
            return self._process_final_result(shareable, fl_ctx)

        else:
            self.log_error(fl_ctx, f"Could not handle task: {task_name}")
            return make_reply(ReturnCode.TASK_UNKNOWN)

    def _get_status_report(self):
        with self.status_lock:
            status = self.current_status
            must_report = False
            if status.error:
                must_report = True
            elif status.timestamp:
                must_report = True

            if not must_report:
                return None

            # do status report
            report = copy.copy(status)
            return report

    def update_status(self, last_round=None, action=None, error=None, all_done=False):
        with self.status_lock:
            status = self.current_status
            status.timestamp = time.time()
            if all_done:
                # once marked all_done, always all_done!
                status.all_done = True
            if error:
                status.error = error
            if action:
                status.action = action
            if status.last_round is None:
                status.last_round = last_round
            elif last_round is not None and last_round > status.last_round:
                status.last_round = last_round

            status_dict = status.to_dict()
            self.logger.info(f"updated my last status: {status_dict}")

    def _process_final_result(self, request: Shareable, fl_ctx: FLContext) -> Shareable:
        peer_ctx = fl_ctx.get_peer_context()
        if peer_ctx:
            client_name = peer_ctx.get_identity_name()
        else:
            self.log_error(fl_ctx, "Request from unknown client")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)
        result = request.get(Constant.RESULT)

        if not result:
            self.log_error(fl_ctx, f"Bad request from client {client_name}: no result")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        if not isinstance(result, Learnable):
            self.log_error(fl_ctx, f"Bad result from client {client_name}: expect Learnable but got {type(result)}")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        self.log_info(fl_ctx, f"Got final result from client {client_name}")

        fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, result, private=True, sticky=True)

        if self.persistor:
            self.persistor.save(result, fl_ctx)
        else:
            self.log_error(fl_ctx, "persistor not configured, model will not be saved")

        return make_reply(ReturnCode.OK)

    def _process_end_workflow(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        self.log_info(fl_ctx, f"ending workflow {self.get_config_prop(FLContextKey.WORKFLOW)}")
        self.asked_to_stop = True
        # self._abort_current_task(fl_ctx)
        self.finalize(fl_ctx)
        return make_reply(ReturnCode.OK)

    def is_task_secure(self, fl_ctx: FLContext) -> bool:
        """
        Determine whether the task should be secure. A secure task requires encrypted communication between the peers.
        The task is secure only when the training is in secure mode AND private_p2p is set to True.
        """
        private_p2p = self.get_config_prop(Constant.PRIVATE_P2P)
        secure_train = fl_ctx.get_prop(FLContextKey.SECURE_MODE, False)
        return private_p2p and secure_train

    def broadcast_final_result(self, result: Learnable, fl_ctx: FLContext):
        targets = self.get_config_prop(Constant.RESULT_CLIENTS)

        if not isinstance(targets, list):
            self.log_warning(fl_ctx, f"expected targets of result clients to be type list, but got {type(targets)}")
            return None

        if self.me in targets:
            targets.remove(self.me)

        if len(targets) == 0:
            # no targets to receive the result!
            self.log_info(fl_ctx, "no targets to receive final result")
            return None

        shareable = Shareable()
        shareable[Constant.RESULT] = result

        self.log_info(fl_ctx, f"broadcasting final result to clients {targets}")

        self.update_status(action="broadcast_final_result")

        task = Task(
            name=self.report_final_result_task_name,
            data=shareable,
            timeout=int(self.final_result_ack_timeout),
            secure=self.is_task_secure(fl_ctx),
        )

        resp = self.controller.broadcast_and_wait(
            task=task,
            targets=targets,
            min_responses=len(targets),
            fl_ctx=fl_ctx,
        )

        if not isinstance(resp, dict):
            self.log_error(fl_ctx, f"bad response for final result from clients, expected dict but got {type(resp)}")
            return

        num_errors = 0
        for t in targets:
            reply = resp.get(t)
            if not isinstance(reply, Shareable):
                self.log_error(
                    fl_ctx,
                    f"bad response for final result from client {t}: " f"reply must be Shareable but got {type(reply)}",
                )
                num_errors += 1
                continue

            rc = reply.get_return_code(ReturnCode.OK)
            if rc != ReturnCode.OK:
                self.log_error(fl_ctx, f"bad response for final result from client {t}: {rc}")
                num_errors += 1

        if num_errors == 0:
            self.log_info(fl_ctx, f"successfully broadcast final result to {targets}")
        return num_errors
