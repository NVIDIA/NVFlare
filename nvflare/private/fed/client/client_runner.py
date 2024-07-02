# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import fnmatch
import threading
import time

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import ConfigVarName, FilterKey, FLContextKey, ReservedKey, ReservedTopic, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import UnsafeJobError
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.utils.fl_context_utils import add_job_audit_event
from nvflare.apis.utils.reliable_message import ReliableMessage
from nvflare.apis.utils.task_utils import apply_filters
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.private.defs import SpecialTaskName, TaskConstant
from nvflare.private.fed.client.client_engine_executor_spec import ClientEngineExecutorSpec, TaskAssignment
from nvflare.private.fed.tbi import TBI
from nvflare.private.json_configer import ConfigError
from nvflare.private.privacy_manager import Scope
from nvflare.security.logging import secure_format_exception
from nvflare.widgets.info_collector import GroupInfoCollector, InfoCollector

_TASK_CHECK_RESULT_OK = 0
_TASK_CHECK_RESULT_TRY_AGAIN = 1
_TASK_CHECK_RESULT_TASK_GONE = 2


class TaskRouter:
    def __init__(self):
        self.task_table = {}
        self.patterns = []

    @staticmethod
    def _is_pattern(p: str):
        return "*" in p

    def add_executor(self, tasks: list, executor: Executor):
        for t in tasks:
            assert isinstance(t, str)
            if t in self.task_table:
                raise ConfigError(f'multiple executors defined for task "{t}"')
            self.task_table[t] = executor
            if self._is_pattern(t):
                self.patterns.append((t, executor))

    def route(self, task_name: str):
        e = self.task_table.get(task_name)
        if e:
            return e

        # check patterns
        for p, e in self.patterns:
            if fnmatch.fnmatch(task_name, p):
                return e
        return None


class ClientRunnerConfig(object):
    def __init__(
        self,
        task_router: TaskRouter,
        task_data_filters: dict,  # task_name => list of filters
        task_result_filters: dict,  # task_name => list of filters
        handlers=None,  # list of event handlers
        components=None,  # dict of extra python objects: id => object
        default_task_fetch_interval: float = 0.5,
    ):
        """To init ClientRunnerConfig.

        Args:
            task_router: TaskRouter object to find executor for a task
            task_data_filters: task_name => list of data filters
            task_result_filters: task_name => list of result filters
            handlers: list of event handlers
            components: dict of extra python objects: id => object
            default_task_fetch_interval: default task fetch interval before getting the correct value from server.
                default is set to 0.5.
        """
        self.task_router = task_router
        self.task_data_filters = task_data_filters
        self.task_result_filters = task_result_filters
        self.handlers = handlers
        self.components = components
        self.default_task_fetch_interval = default_task_fetch_interval

        if not components:
            self.components = {}

        if not handlers:
            self.handlers = []

    def add_component(self, comp_id: str, component: object):
        if not isinstance(comp_id, str):
            raise TypeError(f"component id must be str but got {type(comp_id)}")

        if comp_id in self.components:
            raise ValueError(f"duplicate component id {comp_id}")

        self.components[comp_id] = component
        if isinstance(component, FLComponent):
            self.handlers.append(component)


class ClientRunner(TBI):
    def __init__(
        self,
        config: ClientRunnerConfig,
        job_id,
        engine: ClientEngineExecutorSpec,
    ):
        """Initializes the ClientRunner.

        Args:
            config: ClientRunnerConfig
            job_id: job id
            engine: ClientEngine object
        """

        TBI.__init__(self)
        self.task_router = config.task_router
        self.task_data_filters = config.task_data_filters
        self.task_result_filters = config.task_result_filters
        self.default_task_fetch_interval = config.default_task_fetch_interval

        self.job_id = job_id
        self.engine = engine
        self.run_abort_signal = Signal()
        self.task_lock = threading.Lock()
        self.running_tasks = {}  # task_id => TaskAssignment

        self.task_check_timeout = self.get_positive_float_var(ConfigVarName.TASK_CHECK_TIMEOUT, 5.0)
        self.task_check_interval = self.get_positive_float_var(ConfigVarName.TASK_CHECK_INTERVAL, 5.0)
        self.job_heartbeat_interval = self.get_positive_float_var(ConfigVarName.JOB_HEARTBEAT_INTERVAL, 10.0)
        self.get_task_timeout = self.get_positive_float_var(ConfigVarName.GET_TASK_TIMEOUT, None)
        self.submit_task_result_timeout = self.get_positive_float_var(ConfigVarName.SUBMIT_TASK_RESULT_TIMEOUT, None)
        self._register_aux_message_handlers(engine)

    def find_executor(self, task_name):
        return self.task_router.route(task_name)

    def _register_aux_message_handlers(self, engine):
        engine.register_aux_message_handler(topic=ReservedTopic.END_RUN, message_handle_func=self._handle_end_run)
        engine.register_aux_message_handler(topic=ReservedTopic.DO_TASK, message_handle_func=self._handle_do_task)

    @staticmethod
    def _reply_and_audit(reply: Shareable, ref, msg, fl_ctx: FLContext) -> Shareable:
        audit_event_id = add_job_audit_event(fl_ctx=fl_ctx, ref=ref, msg=msg)
        reply.set_header(ReservedKey.AUDIT_EVENT_ID, audit_event_id)
        return reply

    def _process_task(self, task: TaskAssignment, fl_ctx: FLContext) -> Shareable:
        if fl_ctx.is_job_unsafe():
            return make_reply(ReturnCode.UNSAFE_JOB)

        with self.task_lock:
            self.running_tasks[task.task_id] = task

        abort_signal = Signal(parent=self.run_abort_signal)
        try:
            reply = self._do_process_task(task, fl_ctx, abort_signal)
        except Exception as ex:
            self.log_exception(fl_ctx, secure_format_exception(ex))
            reply = make_reply(ReturnCode.EXECUTION_EXCEPTION)

        with self.task_lock:
            self.running_tasks.pop(task.task_id, None)

        if not isinstance(reply, Shareable):
            self.log_error(fl_ctx, f"task reply must be Shareable, but got {type(reply)}")
            reply = make_reply(ReturnCode.EXECUTION_EXCEPTION)

        cookie_jar = task.data.get_cookie_jar()
        if cookie_jar:
            reply.set_cookie_jar(cookie_jar)
        reply.set_header(ReservedHeaderKey.TASK_NAME, task.name)
        return reply

    def _do_process_task(self, task: TaskAssignment, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if not isinstance(task.data, Shareable):
            self.log_error(fl_ctx, f"got invalid task data in assignment: expect Shareable, but got {type(task.data)}")
            return make_reply(ReturnCode.BAD_TASK_DATA)

        fl_ctx.set_prop(FLContextKey.TASK_DATA, value=task.data, private=True, sticky=False)
        fl_ctx.set_prop(FLContextKey.TASK_NAME, value=task.name, private=True, sticky=False)
        fl_ctx.set_prop(FLContextKey.TASK_ID, value=task.task_id, private=True, sticky=False)

        server_audit_event_id = task.data.get_header(ReservedKey.AUDIT_EVENT_ID, "")
        add_job_audit_event(fl_ctx=fl_ctx, ref=server_audit_event_id, msg="received task from server")

        peer_ctx = fl_ctx.get_peer_context()
        if not peer_ctx:
            self.log_error(fl_ctx, "missing peer context in Server task assignment")
            return self._reply_and_audit(
                reply=make_reply(ReturnCode.MISSING_PEER_CONTEXT),
                ref=server_audit_event_id,
                fl_ctx=fl_ctx,
                msg=f"submit result: {ReturnCode.MISSING_PEER_CONTEXT}",
            )

        if not isinstance(peer_ctx, FLContext):
            self.log_error(
                fl_ctx,
                "bad peer context in Server task assignment: expects FLContext but got {}".format(type(peer_ctx)),
            )
            return self._reply_and_audit(
                reply=make_reply(ReturnCode.BAD_PEER_CONTEXT),
                ref=server_audit_event_id,
                fl_ctx=fl_ctx,
                msg=f"submit result: {ReturnCode.BAD_PEER_CONTEXT}",
            )

        task.data.set_peer_props(peer_ctx.get_all_public_props())
        peer_job_id = peer_ctx.get_job_id()
        if peer_job_id != self.job_id:
            self.log_error(fl_ctx, "bad task assignment: not for the same job_id")
            return self._reply_and_audit(
                reply=make_reply(ReturnCode.RUN_MISMATCH),
                ref=server_audit_event_id,
                fl_ctx=fl_ctx,
                msg=f"submit result: {ReturnCode.RUN_MISMATCH}",
            )

        executor = self.find_executor(task.name)
        if not executor:
            self.log_error(fl_ctx, f"bad task assignment: no executor available for task {task.name}")
            return self._reply_and_audit(
                reply=make_reply(ReturnCode.TASK_UNKNOWN),
                ref=server_audit_event_id,
                fl_ctx=fl_ctx,
                msg=f"submit result: {ReturnCode.TASK_UNKNOWN}",
            )

        executor_name = executor.__class__.__name__

        self.log_debug(fl_ctx, "firing event EventType.BEFORE_TASK_DATA_FILTER")
        self.fire_event(EventType.BEFORE_TASK_DATA_FILTER, fl_ctx)

        task_data = task.data
        try:
            filter_name = Scope.TASK_DATA_FILTERS_NAME
            task_data = apply_filters(filter_name, task_data, fl_ctx, self.task_data_filters, task.name, FilterKey.IN)
        except UnsafeJobError:
            self.log_exception(fl_ctx, "UnsafeJobError from Task Data Filters")
            executor.unsafe = True
            fl_ctx.set_job_is_unsafe()
            self.run_abort_signal.trigger(True)
            return self._reply_and_audit(
                reply=make_reply(ReturnCode.UNSAFE_JOB),
                ref=server_audit_event_id,
                fl_ctx=fl_ctx,
                msg=f"submit result: {ReturnCode.UNSAFE_JOB}",
            )
        except Exception as e:
            self.log_exception(fl_ctx, f"Processing error from Task Data Filters : {secure_format_exception(e)}")
            return self._reply_and_audit(
                reply=make_reply(ReturnCode.TASK_DATA_FILTER_ERROR),
                ref=server_audit_event_id,
                fl_ctx=fl_ctx,
                msg=f"submit result: {ReturnCode.TASK_DATA_FILTER_ERROR}",
            )

        if not isinstance(task_data, Shareable):
            self.log_error(
                fl_ctx, "task data was converted to wrong type: expect Shareable but got {}".format(type(task_data))
            )
            return self._reply_and_audit(
                reply=make_reply(ReturnCode.TASK_DATA_FILTER_ERROR),
                ref=server_audit_event_id,
                fl_ctx=fl_ctx,
                msg=f"submit result: {ReturnCode.TASK_DATA_FILTER_ERROR}",
            )

        task.data = task_data

        self.log_debug(fl_ctx, "firing event EventType.AFTER_TASK_DATA_FILTER")
        fl_ctx.set_prop(FLContextKey.TASK_DATA, value=task.data, private=True, sticky=False)
        self.fire_event(EventType.AFTER_TASK_DATA_FILTER, fl_ctx)

        self.log_debug(fl_ctx, "firing event EventType.BEFORE_TASK_EXECUTION")
        fl_ctx.set_prop(FLContextKey.TASK_DATA, value=task.data, private=True, sticky=False)
        self.fire_event(EventType.BEFORE_TASK_EXECUTION, fl_ctx)
        try:
            self.log_info(fl_ctx, f"invoking task executor {executor_name}")
            add_job_audit_event(fl_ctx=fl_ctx, msg=f"invoked executor {executor_name}")

            try:
                reply = executor.execute(task.name, task.data, fl_ctx, abort_signal)
            finally:
                if abort_signal.triggered:
                    return self._reply_and_audit(
                        reply=make_reply(ReturnCode.TASK_ABORTED),
                        ref=server_audit_event_id,
                        fl_ctx=fl_ctx,
                        msg=f"submit result: {ReturnCode.TASK_ABORTED}",
                    )

            if not isinstance(reply, Shareable):
                self.log_error(
                    fl_ctx, f"bad result generated by executor {executor_name}: must be Shareable but got {type(reply)}"
                )
                return self._reply_and_audit(
                    reply=make_reply(ReturnCode.EXECUTION_RESULT_ERROR),
                    ref=server_audit_event_id,
                    fl_ctx=fl_ctx,
                    msg=f"submit result: {ReturnCode.EXECUTION_RESULT_ERROR}",
                )

        except RuntimeError as e:
            self.log_exception(
                fl_ctx, f"RuntimeError from executor {executor_name}: {secure_format_exception(e)}: Aborting the job!"
            )
            return self._reply_and_audit(
                reply=make_reply(ReturnCode.EXECUTION_RESULT_ERROR),
                ref=server_audit_event_id,
                fl_ctx=fl_ctx,
                msg=f"submit result: {ReturnCode.EXECUTION_RESULT_ERROR}",
            )
        except UnsafeJobError:
            self.log_exception(fl_ctx, f"UnsafeJobError from executor {executor_name}")
            executor.unsafe = True
            fl_ctx.set_job_is_unsafe()
            return self._reply_and_audit(
                reply=make_reply(ReturnCode.UNSAFE_JOB),
                ref=server_audit_event_id,
                fl_ctx=fl_ctx,
                msg=f"submit result: {ReturnCode.UNSAFE_JOB}",
            )
        except Exception as e:
            self.log_exception(fl_ctx, f"Processing error from executor {executor_name}: {secure_format_exception(e)}")
            return self._reply_and_audit(
                reply=make_reply(ReturnCode.EXECUTION_EXCEPTION),
                ref=server_audit_event_id,
                fl_ctx=fl_ctx,
                msg=f"submit result: {ReturnCode.EXECUTION_EXCEPTION}",
            )

        fl_ctx.set_prop(FLContextKey.TASK_RESULT, value=reply, private=True, sticky=False)

        self.log_debug(fl_ctx, "firing event EventType.AFTER_TASK_EXECUTION")
        self.fire_event(EventType.AFTER_TASK_EXECUTION, fl_ctx)

        self.log_debug(fl_ctx, "firing event EventType.BEFORE_TASK_RESULT_FILTER")
        self.fire_event(EventType.BEFORE_TASK_RESULT_FILTER, fl_ctx)

        try:
            filter_name = Scope.TASK_RESULT_FILTERS_NAME
            reply = apply_filters(filter_name, reply, fl_ctx, self.task_result_filters, task.name, FilterKey.OUT)
        except UnsafeJobError:
            self.log_exception(fl_ctx, "UnsafeJobError from Task Result Filters")
            executor.unsafe = True
            fl_ctx.set_job_is_unsafe()
            return self._reply_and_audit(
                reply=make_reply(ReturnCode.UNSAFE_JOB),
                ref=server_audit_event_id,
                fl_ctx=fl_ctx,
                msg=f"submit result: {ReturnCode.UNSAFE_JOB}",
            )
        except Exception as e:
            self.log_exception(fl_ctx, f"Processing error in Task Result Filter : {secure_format_exception(e)}")
            return self._reply_and_audit(
                reply=make_reply(ReturnCode.TASK_RESULT_FILTER_ERROR),
                ref=server_audit_event_id,
                fl_ctx=fl_ctx,
                msg=f"submit result: {ReturnCode.TASK_RESULT_FILTER_ERROR}",
            )

        if not isinstance(reply, Shareable):
            self.log_error(
                fl_ctx, "task result was converted to wrong type: expect Shareable but got {}".format(type(reply))
            )
            return self._reply_and_audit(
                reply=make_reply(ReturnCode.TASK_RESULT_FILTER_ERROR),
                ref=server_audit_event_id,
                fl_ctx=fl_ctx,
                msg=f"submit result: {ReturnCode.TASK_RESULT_FILTER_ERROR}",
            )

        fl_ctx.set_prop(FLContextKey.TASK_RESULT, value=reply, private=True, sticky=False)

        self.log_debug(fl_ctx, "firing event EventType.AFTER_TASK_RESULT_FILTER")
        self.fire_event(EventType.AFTER_TASK_RESULT_FILTER, fl_ctx)
        self.log_info(fl_ctx, "finished processing task")

        if not isinstance(reply, Shareable):
            self.log_error(
                fl_ctx, "task processing error: expects result to be Shareable, but got {}".format(type(reply))
            )
            return self._reply_and_audit(
                reply=make_reply(ReturnCode.EXECUTION_RESULT_ERROR),
                ref=server_audit_event_id,
                fl_ctx=fl_ctx,
                msg=f"submit result: {ReturnCode.EXECUTION_RESULT_ERROR}",
            )

        return self._reply_and_audit(reply=reply, ref=server_audit_event_id, fl_ctx=fl_ctx, msg="submit result OK")

    def _try_run(self):
        heartbeat_thread = threading.Thread(target=self._send_job_heartbeat, args=[], daemon=True)
        heartbeat_thread.start()

        while not self.run_abort_signal.triggered:
            with self.engine.new_context() as fl_ctx:
                task_fetch_interval, _ = self.fetch_and_run_one_task(fl_ctx)
            time.sleep(task_fetch_interval)

    def _send_job_heartbeat(self):
        request = Shareable()
        last_heartbeat_sent_time = 0.0
        while not self.run_abort_signal.triggered:
            if time.time() - last_heartbeat_sent_time > self.job_heartbeat_interval:
                with self.engine.new_context() as fl_ctx:
                    self.engine.send_aux_request(
                        targets=[FQCN.ROOT_SERVER],
                        topic=ReservedTopic.JOB_HEART_BEAT,
                        request=request,
                        timeout=0,
                        fl_ctx=fl_ctx,
                        optional=True,
                    )
                last_heartbeat_sent_time = time.time()

            # sleep very short time so that we can check stop condition (e.g. abort signal)
            time.sleep(0.2)

    def fetch_and_run_one_task(self, fl_ctx) -> (float, bool):
        """Fetches and runs a task.

        Returns:
            A tuple of (task_fetch_interval, task_processed).
        """
        default_task_fetch_interval = self.default_task_fetch_interval
        self.log_debug(fl_ctx, "fetching task from server ...")
        task = self.engine.get_task_assignment(fl_ctx, self.get_task_timeout)

        if not task:
            self.log_debug(fl_ctx, "no task received - will try in {} secs".format(default_task_fetch_interval))
            return default_task_fetch_interval, False
        elif task.name == SpecialTaskName.END_RUN:
            self.log_info(fl_ctx, "server asked to end the run")
            self.run_abort_signal.trigger(True)
            return default_task_fetch_interval, False
        elif task.name == SpecialTaskName.TRY_AGAIN:
            task_data = task.data
            task_fetch_interval = default_task_fetch_interval
            if task_data and isinstance(task_data, Shareable):
                task_fetch_interval = task_data.get_header(TaskConstant.WAIT_TIME, task_fetch_interval)
            self.log_debug(fl_ctx, "server asked to try again - will try in {} secs".format(task_fetch_interval))
            return task_fetch_interval, False

        self.log_info(fl_ctx, f"got task assignment: name={task.name}, id={task.task_id}")
        task_data = task.data
        if not isinstance(task_data, Shareable):
            raise TypeError("task_data must be Shareable, but got {}".format(type(task_data)))
        task_fetch_interval = task_data.get_header(TaskConstant.WAIT_TIME, default_task_fetch_interval)

        task_reply = self._process_task(task, fl_ctx)

        self.log_debug(fl_ctx, "firing event EventType.BEFORE_SEND_TASK_RESULT")
        self.fire_event(EventType.BEFORE_SEND_TASK_RESULT, fl_ctx)

        self._send_task_result(task_reply, task.task_id, fl_ctx)
        self.log_debug(fl_ctx, "firing event EventType.AFTER_SEND_TASK_RESULT")
        self.fire_event(EventType.AFTER_SEND_TASK_RESULT, fl_ctx)

        return task_fetch_interval, True

    def _send_task_result(self, result: Shareable, task_id: str, fl_ctx: FLContext):
        try_count = 1
        while True:
            self.log_info(fl_ctx, f"try #{try_count}: sending task result to server")

            if self.run_abort_signal.triggered:
                self.log_info(fl_ctx, "job aborted: stopped trying to send result")
                return False

            try_count += 1
            rc = self._try_send_result_once(result, task_id, fl_ctx)

            if rc == _TASK_CHECK_RESULT_OK:
                return True
            elif rc == _TASK_CHECK_RESULT_TASK_GONE:
                return False
            else:
                # retry
                time.sleep(self.task_check_interval)

    def _try_send_result_once(self, result: Shareable, task_id: str, fl_ctx: FLContext):
        # wait until server is ready to receive
        while True:
            if self.run_abort_signal.triggered:
                return _TASK_CHECK_RESULT_TASK_GONE

            rc = self._check_task_once(task_id, fl_ctx)
            if rc == _TASK_CHECK_RESULT_OK:
                break
            elif rc == _TASK_CHECK_RESULT_TASK_GONE:
                return rc
            else:
                # try again
                time.sleep(self.task_check_interval)

        # try to send the result
        self.log_info(fl_ctx, "start to send task result to server")
        reply_sent = self.engine.send_task_result(result, fl_ctx, timeout=self.submit_task_result_timeout)
        if reply_sent:
            self.log_info(fl_ctx, "task result sent to server")
            return _TASK_CHECK_RESULT_OK
        else:
            self.log_error(fl_ctx, "failed to send task result to server - will try again")
            return _TASK_CHECK_RESULT_TRY_AGAIN

    def _check_task_once(self, task_id: str, fl_ctx: FLContext) -> int:
        """This method checks whether the server is still waiting for the specified task.
        The real reason for this method is to fight against unstable network connections.
        We try to make sure that when we send task result to the server, the connection is available.
        If the task check succeeds, then the network connection is likely to be available.
        Otherwise, we keep retrying until task check succeeds or the server tells us that the task is gone (timed out).
        Args:
            task_id:
            fl_ctx:
        Returns:
        """
        self.log_info(fl_ctx, "checking task ...")
        task_check_req = Shareable()
        task_check_req.set_header(ReservedKey.TASK_ID, task_id)
        resp = self.engine.send_aux_request(
            targets=[FQCN.ROOT_SERVER],
            topic=ReservedTopic.TASK_CHECK,
            request=task_check_req,
            timeout=self.task_check_timeout,
            fl_ctx=fl_ctx,
            optional=True,
        )
        if resp and isinstance(resp, dict):
            reply = resp.get(FQCN.ROOT_SERVER)
            if not isinstance(reply, Shareable):
                self.log_error(fl_ctx, f"bad task_check reply from server: expect Shareable but got {type(reply)}")
                return _TASK_CHECK_RESULT_TRY_AGAIN

            rc = reply.get_return_code()
            if rc == ReturnCode.OK:
                return _TASK_CHECK_RESULT_OK
            elif rc == ReturnCode.COMMUNICATION_ERROR:
                self.log_error(fl_ctx, f"failed task_check: {rc}")
                return _TASK_CHECK_RESULT_TRY_AGAIN
            elif rc == ReturnCode.SERVER_NOT_READY:
                self.log_error(fl_ctx, f"server rejected task_check: {rc}")
                return _TASK_CHECK_RESULT_TRY_AGAIN
            elif rc == ReturnCode.TASK_UNKNOWN:
                self.log_debug(fl_ctx, f"task no longer exists on server: {rc}")
                return _TASK_CHECK_RESULT_TASK_GONE
            else:
                # this should never happen
                self.log_error(fl_ctx, f"programming error: received {rc} from server")
                return _TASK_CHECK_RESULT_OK  # try to push the result regardless
        else:
            self.log_error(fl_ctx, f"bad task_check reply from server: invalid resp {type(resp)}")
            return _TASK_CHECK_RESULT_TRY_AGAIN

    def run(self, app_root, args):
        self.init_run(app_root, args)

        try:
            self._try_run()
        except Exception as e:
            with self.engine.new_context() as fl_ctx:
                self.log_exception(fl_ctx, f"processing error in RUN execution: {secure_format_exception(e)}")
        finally:
            self.end_run_events_sequence()
            ReliableMessage.shutdown()
            with self.task_lock:
                self.running_tasks = {}

    def init_run(self, app_root, args):
        sync_timeout = self.get_positive_float_var(
            var_name=ConfigVarName.RUNNER_SYNC_TIMEOUT,
            default=2.0,
        )
        max_sync_timeout = self.get_positive_float_var(
            var_name=ConfigVarName.MAX_RUNNER_SYNC_TIMEOUT,
            default=60.0,
        )
        target = "server"
        synced = False
        sync_start = time.time()
        with self.engine.new_context() as fl_ctx:
            while True:
                # sync with server runner before starting
                time.sleep(0.5)
                resp = self.engine.send_aux_request(
                    targets=[target],
                    topic=ReservedTopic.SYNC_RUNNER,
                    request=Shareable(),
                    timeout=sync_timeout,
                    fl_ctx=fl_ctx,
                    optional=True,
                    secure=False,
                )
                if time.time() - sync_start > max_sync_timeout:
                    break

                if not resp:
                    continue

                reply = resp.get(target)
                if not reply:
                    continue

                assert isinstance(reply, Shareable)
                rc = reply.get_return_code()
                if rc == ReturnCode.OK:
                    synced = True
                    break

            if not synced:
                raise RuntimeError(f"cannot sync with Server Runner after {max_sync_timeout} seconds")

            self.log_info(fl_ctx, f"synced to Server Runner in {time.time()-sync_start} seconds")
            ReliableMessage.enable(fl_ctx)
            self.fire_event(EventType.ABOUT_TO_START_RUN, fl_ctx)
            fl_ctx.set_prop(FLContextKey.APP_ROOT, app_root, sticky=True)
            fl_ctx.set_prop(FLContextKey.ARGS, args, sticky=True)
            fl_ctx.set_prop(ReservedKey.RUN_ABORT_SIGNAL, self.run_abort_signal, private=True, sticky=True)
            self.log_debug(fl_ctx, "firing event EventType.START_RUN")
            self.fire_event(EventType.START_RUN, fl_ctx)
            self.log_info(fl_ctx, "client runner started")

    def end_run_events_sequence(self):
        with self.engine.new_context() as fl_ctx:
            self.log_info(fl_ctx, "started end-run events sequence")

            with self.task_lock:
                num_running_tasks = len(self.running_tasks)
            if num_running_tasks > 0:
                self.fire_event(EventType.ABORT_TASK, fl_ctx)
                self.log_info(fl_ctx, "fired ABORT_TASK event to abort all running tasks")

            self.fire_event(EventType.ABOUT_TO_END_RUN, fl_ctx)
            self.log_info(fl_ctx, "ABOUT_TO_END_RUN fired")

            self.check_end_run_readiness(fl_ctx)

            # now ready to end run
            self.fire_event(EventType.END_RUN, fl_ctx)
            self.log_info(fl_ctx, "END_RUN fired")

    def abort(self, msg: str = ""):
        """To Abort the current run.

        Returns: N/A

        """
        # This is called when:
        # 1. abort_job command is issued by the user
        # 2. when the job is ended by the server when error conditions occur
        # 3. when the job is ended normally at the end of the workflow
        if not msg:
            msg = "Client is stopping ..."
        with self.engine.new_context() as fl_ctx:
            self.log_info(fl_ctx, msg)
        self.run_abort_signal.trigger(True)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == InfoCollector.EVENT_TYPE_GET_STATS:
            collector = fl_ctx.get_prop(InfoCollector.CTX_KEY_STATS_COLLECTOR)
            if collector:
                if not isinstance(collector, GroupInfoCollector):
                    raise TypeError("collector must be GroupInfoCollector, but got {}".format(type(collector)))
                with self.task_lock:
                    current_tasks = []
                    for _, task in self.running_tasks.items():
                        current_tasks.append(task.name)

                collector.set_info(
                    group_name="ClientRunner",
                    info={"job_id": self.job_id, "current_tasks": current_tasks},
                )
        elif event_type == EventType.FATAL_SYSTEM_ERROR:
            # This happens when a task calls system_panic().
            reason = fl_ctx.get_prop(key=FLContextKey.EVENT_DATA, default="")
            self.log_error(fl_ctx, "Stopped ClientRunner due to FATAL_SYSTEM_ERROR: {}".format(reason))
            self.run_abort_signal.trigger(True)

    def _handle_end_run(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        # This happens when the controller on server asks the client to end the job.
        # Usually at the end of the workflow.
        self.log_info(fl_ctx, "received request from Server to end current RUN")
        self.run_abort_signal.trigger(True)
        return make_reply(ReturnCode.OK)

    def _handle_do_task(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        self.log_info(fl_ctx, "received aux request to do task")
        task_name = request.get_header(ReservedHeaderKey.TASK_NAME)
        task_id = request.get_header(ReservedHeaderKey.TASK_ID)
        task = TaskAssignment(name=task_name, task_id=task_id, data=request)
        reply = self._process_task(task, fl_ctx)
        return reply
