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
import copy
import threading
import time
from abc import abstractmethod

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.task_controller import Task, TaskController
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learnable import Learnable
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.ccwf.common import Constant, ResultType, StatusReport, make_task_name, topic_for_end_workflow
from nvflare.fuel.utils.validation_utils import check_non_empty_str, check_number_range, check_positive_number
from nvflare.security.logging import secure_format_traceback


class _LearnTask:
    def __init__(self, task_name: str, task_data: Shareable, fl_ctx: FLContext):
        self.task_name = task_name
        self.task_data = task_data
        self.fl_ctx = fl_ctx
        self.abort_signal = Signal()


class ClientSideController(Executor, TaskController):
    def __init__(
        self,
        task_name_prefix: str,
        learn_task_name=AppConstants.TASK_TRAIN,
        persistor_id=AppConstants.DEFAULT_PERSISTOR_ID,
        shareable_generator_id=AppConstants.DEFAULT_SHAREABLE_GENERATOR_ID,
        learn_task_check_interval=Constant.LEARN_TASK_CHECK_INTERVAL,
        learn_task_ack_timeout=Constant.LEARN_TASK_ACK_TIMEOUT,
        learn_task_abort_timeout=Constant.LEARN_TASK_ABORT_TIMEOUT,
        final_result_ack_timeout=Constant.FINAL_RESULT_ACK_TIMEOUT,
        allow_busy_task: bool = False,
    ):
        """
        Constructor of a ClientSideController object.

        Args:
            task_name_prefix: prefix of task names. All CCWF task names are prefixed with this.
            learn_task_name: name for the Learning Task (LT)
            persistor_id: ID of the persistor component
            shareable_generator_id: ID of the shareable generator component
            learn_task_check_interval: interval for checking incoming Learning Task (LT)
            learn_task_ack_timeout: timeout for sending the LT to other client(s)
            final_result_ack_timeout: timeout for sending final result to participating clients
            learn_task_abort_timeout: time to wait for the LT to become stopped after aborting it
            allow_busy_task: whether a new learn task is allowed when working on current learn task
        """
        check_non_empty_str("task_name_prefix", task_name_prefix)
        check_positive_number("learn_task_check_interval", learn_task_check_interval)
        check_number_range("learn_task_ack_timeout", learn_task_ack_timeout, min_value=1.0)
        check_positive_number("learn_task_abort_timeout", learn_task_abort_timeout)
        check_number_range("final_result_ack_timeout", final_result_ack_timeout, min_value=1.0)

        Executor.__init__(self)
        TaskController.__init__(self)
        self.task_name_prefix = task_name_prefix
        self.start_task_name = make_task_name(task_name_prefix, Constant.BASENAME_START)
        self.configure_task_name = make_task_name(task_name_prefix, Constant.BASENAME_CONFIG)
        self.do_learn_task_name = make_task_name(task_name_prefix, Constant.BASENAME_LEARN)
        self.report_final_result_task_name = make_task_name(task_name_prefix, Constant.BASENAME_REPORT_FINAL_RESULT)
        self.learn_task_name = learn_task_name
        self.learn_task_abort_timeout = learn_task_abort_timeout
        self.learn_task_check_interval = learn_task_check_interval
        self.learn_task_ack_timeout = learn_task_ack_timeout
        self.final_result_ack_timeout = final_result_ack_timeout
        self.allow_busy_task = allow_busy_task
        self.persistor_id = persistor_id
        self.shareable_generator_id = shareable_generator_id

        self.persistor = None
        self.shareable_generator = None

        self.current_status = StatusReport()
        self.last_status_report_time = time.time()  # time of last status report to server
        self.config = None
        self.workflow_id = None
        self.finalize_lock = threading.Lock()

        self.learn_thread = threading.Thread(target=self._do_learn)
        self.learn_thread.daemon = True
        self.learn_task = None
        self.current_task = None
        self.learn_executor = None
        self.learn_task_lock = threading.Lock()
        self.asked_to_stop = False
        self.status_lock = threading.Lock()
        self.engine = None
        self.me = None
        self.is_starting_client = False
        self.last_result = None
        self.last_round = None
        self.best_result = None
        self.best_metric = None
        self.best_round = 0
        self.workflow_done = False

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
        self.start_controller(fl_ctx)
        self.engine = fl_ctx.get_engine()
        if not self.engine:
            self.system_panic("no engine", fl_ctx)
            return

        runner = fl_ctx.get_prop(FLContextKey.RUNNER)
        if not runner:
            self.system_panic("no client runner", fl_ctx)
            return

        self.me = fl_ctx.get_identity_name()
        if self.learn_task_name:
            self.learn_executor = runner.find_executor(self.learn_task_name)
            if not self.learn_executor:
                self.system_panic(f"no executor for task {self.learn_task_name}", fl_ctx)
                return

        self.persistor = self.engine.get_component(self.persistor_id)
        if not isinstance(self.persistor, LearnablePersistor):
            self.system_panic(
                f"Persistor {self.persistor_id} must be a Persistor instance, but got {type(self.persistor)}",
                fl_ctx,
            )
            return

        if self.shareable_generator_id:
            self.shareable_generator = self.engine.get_component(self.shareable_generator_id)
            if not isinstance(self.shareable_generator, ShareableGenerator):
                self.system_panic(
                    f"Shareable generator {self.shareable_generator_id} must be a Shareable Generator instance, "
                    f"but got {type(self.shareable_generator)}",
                    fl_ctx,
                )
                return

        self.initialize(fl_ctx)

        if self.learn_task_name:
            self.log_info(fl_ctx, "Started learn thread")
            self.learn_thread.start()

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
                self._abort_current_task(fl_ctx)
                self.finalize(fl_ctx)

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

    def process_config(self, fl_ctx: FLContext):
        """This is called to allow the subclass to process config props.

        Returns: None

        """
        pass

    def topic_for_my_workflow(self, base_topic: str):
        return f"{base_topic}.{self.workflow_id}"

    def broadcast_final_result(
        self, fl_ctx: FLContext, result_type: str, result: Learnable, metric=None, round_num=None
    ):
        error = None
        targets = self.get_config_prop(Constant.RESULT_CLIENTS)
        if not targets:
            self.log_info(fl_ctx, f"no clients configured to receive final {result_type} result")
        else:
            try:
                num_errors = self._try_broadcast_final_result(fl_ctx, result_type, result, metric, round_num)
                if num_errors > 0:
                    error = ReturnCode.EXECUTION_EXCEPTION
            except:
                self.log_error(fl_ctx, f"exception broadcast final {result_type} result {secure_format_traceback()}")
                error = ReturnCode.EXECUTION_EXCEPTION

        if result_type == ResultType.BEST:
            action = "finished_broadcast_best_result"
            all_done = False
        else:
            action = "finished_broadcast_last_result"
            all_done = True
        self.update_status(action=action, error=error, all_done=all_done)

    def _try_broadcast_final_result(
        self, fl_ctx: FLContext, result_type: str, result: Learnable, metric=None, round_num=None
    ):
        targets = self.get_config_prop(Constant.RESULT_CLIENTS)

        assert isinstance(targets, list)
        if self.me in targets:
            targets.remove(self.me)

        if len(targets) == 0:
            # no targets to receive the result!
            self.log_info(fl_ctx, f"no targets to receive {result_type} result")
            return 0

        shareable = Shareable()
        shareable.set_header(Constant.RESULT_TYPE, result_type)
        if metric is not None:
            shareable.set_header(Constant.METRIC, metric)
        if round_num is not None:
            shareable.set_header(Constant.ROUND, round_num)
        shareable[Constant.RESULT] = result

        self.log_info(
            fl_ctx, f"broadcasting {result_type} result with metric {metric} at round {round_num} to clients {targets}"
        )

        self.update_status(action=f"broadcast_{result_type}_result")

        task = Task(
            name=self.report_final_result_task_name,
            data=shareable,
            timeout=int(self.final_result_ack_timeout),
            secure=self.is_task_secure(fl_ctx),
        )

        resp = self.broadcast_and_wait(
            task=task,
            targets=targets,
            min_responses=len(targets),
            fl_ctx=fl_ctx,
        )

        assert isinstance(resp, dict)
        num_errors = 0
        for t in targets:
            reply = resp.get(t)
            if not isinstance(reply, Shareable):
                self.log_error(
                    fl_ctx,
                    f"bad response for {result_type} result from client {t}: "
                    f"reply must be Shareable but got {type(reply)}",
                )
                num_errors += 1
                continue

            rc = reply.get_return_code(ReturnCode.OK)
            if rc != ReturnCode.OK:
                self.log_error(fl_ctx, f"bad response for {result_type} result from client {t}: {rc}")
                num_errors += 1

        if num_errors == 0:
            self.log_info(fl_ctx, f"successfully broadcast {result_type} result to {targets}")
        return num_errors

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name == self.configure_task_name:
            self.config = shareable[Constant.CONFIG]
            my_wf_id = self.get_config_prop(FLContextKey.WORKFLOW)
            if not my_wf_id:
                self.log_error(fl_ctx, "missing workflow id in configuration!")
                return make_reply(ReturnCode.BAD_REQUEST_DATA)
            self.log_info(fl_ctx, f"got my workflow id {my_wf_id}")
            self.workflow_id = my_wf_id

            reply = self.process_config(fl_ctx)

            self.engine.register_aux_message_handler(
                topic=topic_for_end_workflow(my_wf_id),
                message_handle_func=self._process_end_workflow,
            )

            learnable = self.persistor.load(fl_ctx)
            fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, learnable, private=True, sticky=True)

            if not reply:
                reply = make_reply(ReturnCode.OK)
            return reply

        elif task_name == self.start_task_name:
            self.is_starting_client = True

            learnable = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
            initial_model = self.shareable_generator.learnable_to_shareable(learnable, fl_ctx)
            return self.start_workflow(initial_model, fl_ctx, abort_signal)

        elif task_name == self.do_learn_task_name:
            return self._process_learn_request(shareable, fl_ctx)

        elif task_name == self.report_final_result_task_name:
            return self._process_final_result(shareable, fl_ctx)

        else:
            self.log_error(fl_ctx, f"Could not handle task: {task_name}")
            return make_reply(ReturnCode.TASK_UNKNOWN)

    @abstractmethod
    def start_workflow(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """
        This is called for the subclass to start the workflow.
        This only happens on the starting_client.

        Args:
            shareable: the initial task data (e.g. initial model weights)
            fl_ctx: FL context
            abort_signal: abort signal for task execution

        Returns:

        """
        pass

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

    def _abort_current_task(self, fl_ctx: FLContext):
        current_task = self.learn_task
        if not current_task:
            return

        current_task.abort_signal.trigger(True)
        fl_ctx.set_prop(FLContextKey.TASK_NAME, current_task.task_name)
        self.fire_event(EventType.ABORT_TASK, fl_ctx)

    def set_learn_task(self, task_data: Shareable, fl_ctx: FLContext) -> bool:
        with self.learn_task_lock:
            task_data.set_header(AppConstants.NUM_ROUNDS, self.get_config_prop(AppConstants.NUM_ROUNDS))
            task = _LearnTask(self.learn_task_name, task_data, fl_ctx)
            current_task = self.learn_task
            if not current_task:
                self.learn_task = task
                return True

            if not self.allow_busy_task:
                return False

            # already has a task!
            self.log_warning(fl_ctx, "already running a task: aborting it")
            self._abort_current_task(fl_ctx)

            # monitor until the task is done
            start = time.time()
            while self.learn_task:
                if time.time() - start > self.learn_task_abort_timeout:
                    self.log_error(
                        fl_ctx, f"failed to stop the running task after {self.learn_task_abort_timeout} seconds"
                    )
                    return False
                time.sleep(0.1)

            self.learn_task = task
            return True

    def _do_learn(self):
        while not self.asked_to_stop:
            if self.learn_task:
                t = self.learn_task
                assert isinstance(t, _LearnTask)
                self.logger.info(f"Got a Learn task {t.task_name}")
                try:
                    self.do_learn_task(t.task_name, t.task_data, t.fl_ctx, t.abort_signal)
                except:
                    self.logger.log(f"exception from do_learn_task: {secure_format_traceback()}")
                self.learn_task = None
            time.sleep(self.learn_task_check_interval)

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

    @abstractmethod
    def do_learn_task(self, name: str, data: Shareable, fl_ctx: FLContext, abort_signal: Signal):
        """This is called to do a Learn Task.
        Subclass must implement this method.

        Args:
            name: task name
            data: task data
            fl_ctx: FL context of the task
            abort_signal: abort signal for the task execution

        Returns:

        """
        pass

    def _process_final_result(self, request: Shareable, fl_ctx: FLContext) -> Shareable:
        peer_ctx = fl_ctx.get_peer_context()
        assert isinstance(peer_ctx, FLContext)
        client_name = peer_ctx.get_identity_name()
        result = request.get(Constant.RESULT)
        metric = request.get_header(Constant.METRIC)
        round_num = request.get_header(Constant.ROUND)
        result_type = request.get_header(Constant.RESULT_TYPE)

        if result_type not in [ResultType.BEST, ResultType.LAST]:
            self.log_error(fl_ctx, f"Bad request from client {client_name}: invalid result type {result_type}")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        if not result:
            self.log_error(fl_ctx, f"Bad request from client {client_name}: no result")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        if not isinstance(result, Learnable):
            self.log_error(fl_ctx, f"Bad result from client {client_name}: expect Learnable but got {type(result)}")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        self.log_info(fl_ctx, f"Got {result_type} from client {client_name} with metric {metric} at round {round_num}")

        fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, result, private=True, sticky=True)

        if result_type == ResultType.BEST:
            fl_ctx.set_prop(Constant.ROUND, round_num, private=True, sticky=False)
            fl_ctx.set_prop(Constant.CLIENT, client_name, private=True, sticky=False)
            fl_ctx.set_prop(AppConstants.VALIDATION_RESULT, metric, private=True, sticky=False)
            self.fire_event(AppEventType.GLOBAL_BEST_MODEL_AVAILABLE, fl_ctx)
        else:
            # last model
            assert isinstance(self.persistor, LearnablePersistor)
            self.persistor.save(result, fl_ctx)
        return make_reply(ReturnCode.OK)

    def _process_end_workflow(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        self.log_info(fl_ctx, f"ending workflow {self.get_config_prop(FLContextKey.WORKFLOW)}")
        self.asked_to_stop = True
        self._abort_current_task(fl_ctx)
        self.finalize(fl_ctx)
        return make_reply(ReturnCode.OK)

    def _process_learn_request(self, request: Shareable, fl_ctx: FLContext) -> Shareable:
        try:
            return self._try_process_learn_request(request, fl_ctx)
        except Exception as ex:
            self.log_exception(fl_ctx, f"exception: {ex}")
            self.update_status(action="process_learn_request", error=ReturnCode.EXECUTION_EXCEPTION)
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _try_process_learn_request(self, request: Shareable, fl_ctx: FLContext) -> Shareable:
        peer_ctx = fl_ctx.get_peer_context()
        assert isinstance(peer_ctx, FLContext)
        sender = peer_ctx.get_identity_name()

        # process request from prev client
        self.log_info(fl_ctx, f"Got Learn request from {sender}")

        if self.learn_task and not self.allow_busy_task:
            # should never happen!
            self.log_error(fl_ctx, f"got Learn request from {sender} while I'm still busy!")
            self.update_status(action="process_learn_request", error=ReturnCode.EXECUTION_EXCEPTION)
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        self.log_info(fl_ctx, f"accepted learn request from {sender}")
        self.set_learn_task(task_data=request, fl_ctx=fl_ctx)
        return make_reply(ReturnCode.OK)

    def send_learn_task(self, targets: list, request: Shareable, fl_ctx: FLContext) -> bool:
        self.log_info(fl_ctx, f"sending learn task to clients {targets}")
        request.set_header(AppConstants.NUM_ROUNDS, self.get_config_prop(AppConstants.NUM_ROUNDS))

        task = Task(
            name=self.do_learn_task_name,
            data=request,
            timeout=int(self.learn_task_ack_timeout),
            secure=self.is_task_secure(fl_ctx),
        )

        resp = self.broadcast_and_wait(
            task=task,
            targets=targets,
            min_responses=len(targets),
            fl_ctx=fl_ctx,
        )

        assert isinstance(resp, dict)
        for t in targets:
            reply = resp.get(t)
            if not isinstance(reply, Shareable):
                self.log_error(fl_ctx, f"failed to send learn request to client {t}")
                self.log_error(fl_ctx, f"reply must be Shareable but got {type(reply)}")
                self.update_status(action="send_learn_task", error=ReturnCode.EXECUTION_EXCEPTION)
                return False

            rc = reply.get_return_code(ReturnCode.OK)
            if rc != ReturnCode.OK:
                self.log_error(fl_ctx, f"bad response for learn request from client {t}: {rc}")
                self.update_status(action="send_learn_task", error=rc)
                return False
        return True

    def execute_learn_task(self, data: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        current_round = data.get_header(AppConstants.CURRENT_ROUND)

        self.log_info(fl_ctx, f"started training round {current_round}")
        try:
            result = self.learn_executor.execute(self.learn_task_name, data, fl_ctx, abort_signal)
        except:
            self.log_exception(fl_ctx, f"trainer exception: {secure_format_traceback()}")
            result = make_reply(ReturnCode.EXECUTION_EXCEPTION)
        self.log_info(fl_ctx, f"finished training round {current_round}")

        # make sure to include cookies in result
        cookie_jar = data.get_cookie_jar()
        result.set_cookie_jar(cookie_jar)
        result.set_header(AppConstants.CURRENT_ROUND, current_round)
        result.add_cookie(AppConstants.CONTRIBUTION_ROUND, current_round)  # to make model selector happy
        return result

    def record_last_result(
        self,
        fl_ctx: FLContext,
        round_num: int,
        result: Learnable,
    ):
        if not isinstance(result, Learnable):
            self.log_error(fl_ctx, f"result must be Learnable but got {type(result)}")
            return

        self.last_result = result
        self.last_round = round_num
        fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, result, private=True, sticky=True)
        if self.persistor:
            self.log_info(fl_ctx, f"Saving result of round {round_num}")
            self.persistor.save(result, fl_ctx)

    def is_task_secure(self, fl_ctx: FLContext) -> bool:
        """
        Determine whether the task should be secure. A secure task requires encrypted communication between the peers.
        The task is secure only when the training is in secure mode AND private_p2p is set to True.
        """
        private_p2p = self.get_config_prop(Constant.PRIVATE_P2P)
        secure_train = fl_ctx.get_prop(FLContextKey.SECURE_MODE, False)
        return private_p2p and secure_train
