# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.client_engine_spec import ClientEngineSpec, TaskAssignment
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReservedTopic, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.private.defs import SpecialTaskName, TaskConstant
from nvflare.widgets.info_collector import GroupInfoCollector, InfoCollector


class ClientRunnerConfig(object):
    def __init__(
        self,
        task_table: dict,  # task_name => Executor
        task_data_filters: dict,  # task_name => list of filters
        task_result_filters: dict,  # task_name => list of filters
        handlers=None,  # list of event handlers
        components=None,  # dict of extra python objects: id => object
    ):
        """To init ClientRunnerConfig.

        Args:
            task_table: task_name: Executor dict
            task_data_filters: task_name => list of data filters
            task_result_filters: task_name => list of result filters
            handlers: list of event handlers
            components: dict of extra python objects: id => object
        """
        self.task_table = task_table
        self.task_data_filters = task_data_filters
        self.task_result_filters = task_result_filters
        self.handlers = handlers
        self.components = components


class ClientRunner(FLComponent):
    def __init__(
        self,
        config: ClientRunnerConfig,
        job_id,
        engine: ClientEngineSpec,
        task_fetch_interval: int = 5,  # fetch task every 5 secs
    ):
        """To init the ClientRunner.

        Args:
            config: ClientRunnerConfig
            job_id: job id
            engine: ClientEngine object
            task_fetch_interval:  fetch task interval
        """
        FLComponent.__init__(self)
        self.task_table = config.task_table
        self.task_data_filters = config.task_data_filters
        self.task_result_filters = config.task_result_filters

        self.job_id = job_id
        self.engine = engine
        self.task_fetch_interval = task_fetch_interval
        self.run_abort_signal = Signal()
        self.task_abort_signal = None
        self.current_executor = None
        self.current_task = None
        self.asked_to_stop = False
        self.task_lock = threading.Lock()
        self.end_run_fired = False
        self.end_run_lock = threading.Lock()

        engine.register_aux_message_handler(topic=ReservedTopic.END_RUN, message_handle_func=self._handle_end_run)
        engine.register_aux_message_handler(topic=ReservedTopic.ABORT_ASK, message_handle_func=self._handle_abort_task)

    def _process_task(self, task: TaskAssignment, fl_ctx: FLContext) -> Shareable:
        engine = fl_ctx.get_engine()
        if not isinstance(engine, ClientEngineSpec):
            raise TypeError("engine must be ClientEngineSpec, but got {}".format(type(engine)))

        if not isinstance(task.data, Shareable):
            self.log_error(
                fl_ctx, "got invalid task data in assignment: expect Shareable, but got {}".format(type(task.data))
            )
            return make_reply(ReturnCode.BAD_TASK_DATA)

        fl_ctx.set_prop(FLContextKey.TASK_DATA, value=task.data, private=True, sticky=False)
        fl_ctx.set_prop(FLContextKey.TASK_NAME, value=task.name, private=True, sticky=False)
        fl_ctx.set_prop(FLContextKey.TASK_ID, value=task.task_id, private=True, sticky=False)

        peer_ctx = fl_ctx.get_peer_context()
        if not peer_ctx:
            self.log_error(fl_ctx, "missing peer context in Server task assignment")
            return make_reply(ReturnCode.MISSING_PEER_CONTEXT)

        if not isinstance(peer_ctx, FLContext):
            self.log_error(
                fl_ctx,
                "bad peer context in Server task assignment: expects FLContext but got {}".format(type(peer_ctx)),
            )
            return make_reply(ReturnCode.BAD_PEER_CONTEXT)

        task.data.set_peer_props(peer_ctx.get_all_public_props())
        peer_job_id = peer_ctx.get_job_id()
        if peer_job_id != self.job_id:
            self.log_error(fl_ctx, "bad task assignment: not for the same job_id")
            return make_reply(ReturnCode.RUN_MISMATCH)

        executor = self.task_table.get(task.name)
        if not executor:
            self.log_error(fl_ctx, "bad task assignment: no executor available for task {}".format(task.name))
            return make_reply(ReturnCode.TASK_UNKNOWN)

        self.log_debug(fl_ctx, "firing event EventType.BEFORE_TASK_DATA_FILTER")
        self.fire_event(EventType.BEFORE_TASK_DATA_FILTER, fl_ctx)
        filter_list = self.task_data_filters.get(task.name)
        if filter_list:
            task_data = task.data
            for f in filter_list:
                try:
                    task_data = f.process(task_data, fl_ctx)
                except BaseException:
                    self.log_exception(fl_ctx, "processing error in Task Data Filter {}".format(type(f)))
                    return make_reply(ReturnCode.TASK_DATA_FILTER_ERROR)

            if not isinstance(task_data, Shareable):
                self.log_error(
                    fl_ctx, "task data was converted to wrong type: expect Shareable but got {}".format(type(task_data))
                )
                return make_reply(ReturnCode.TASK_DATA_FILTER_ERROR)

            task.data = task_data

        self.log_debug(fl_ctx, "firing event EventType.AFTER_TASK_DATA_FILTER")
        fl_ctx.set_prop(FLContextKey.TASK_DATA, value=task.data, private=True, sticky=False)
        self.fire_event(EventType.AFTER_TASK_DATA_FILTER, fl_ctx)

        self.log_debug(fl_ctx, "firing event EventType.BEFORE_TASK_EXECUTION")
        fl_ctx.set_prop(FLContextKey.TASK_DATA, value=task.data, private=True, sticky=False)
        self.fire_event(EventType.BEFORE_TASK_EXECUTION, fl_ctx)
        try:
            self.log_info(fl_ctx, "invoking task executor {}".format(type(executor)))

            with self.task_lock:
                self.task_abort_signal = Signal()
                self.current_executor = executor
                self.current_task = task

            try:
                reply = executor.execute(task.name, task.data, fl_ctx, self.task_abort_signal)
            finally:
                with self.task_lock:
                    if self.task_abort_signal is None:
                        task_aborted = True
                    else:
                        task_aborted = False

                    self.task_abort_signal = None
                    self.current_task = None
                    self.current_executor = None
                    if task_aborted:
                        return make_reply(ReturnCode.TASK_ABORTED)

            if not isinstance(reply, Shareable):
                self.log_error(
                    fl_ctx,
                    "bad result generated by executor {}: must be Shareable but got {}".format(
                        type(executor), type(reply)
                    ),
                )
                return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)
        except RuntimeError as e:
            self.log_exception(fl_ctx, f"Critical RuntimeError happened with Exception {e}: Aborting the RUN!")
            self.asked_to_stop = True
            return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)
        except BaseException:
            self.log_exception(fl_ctx, "processing error in task executor {}".format(type(executor)))
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        fl_ctx.set_prop(FLContextKey.TASK_RESULT, value=reply, private=True, sticky=False)

        self.log_debug(fl_ctx, "firing event EventType.AFTER_TASK_EXECUTION")
        self.fire_event(EventType.AFTER_TASK_EXECUTION, fl_ctx)

        self.log_debug(fl_ctx, "firing event EventType.BEFORE_TASK_RESULT_FILTER")
        self.fire_event(EventType.BEFORE_TASK_RESULT_FILTER, fl_ctx)
        filter_list = self.task_result_filters.get(task.name)
        if filter_list:
            for f in filter_list:
                try:
                    reply = f.process(reply, fl_ctx)
                except BaseException:
                    self.log_exception(fl_ctx, "processing error in Task Result Filter {}".format(type(f)))
                    return make_reply(ReturnCode.TASK_RESULT_FILTER_ERROR)

            if not isinstance(reply, Shareable):
                self.log_error(
                    fl_ctx, "task result was converted to wrong type: expect Shareable but got {}".format(type(reply))
                )

                return make_reply(ReturnCode.TASK_RESULT_FILTER_ERROR)

        fl_ctx.set_prop(FLContextKey.TASK_RESULT, value=reply, private=True, sticky=False)

        self.log_debug(fl_ctx, "firing event EventType.AFTER_TASK_RESULT_FILTER")
        self.fire_event(EventType.AFTER_TASK_RESULT_FILTER, fl_ctx)
        self.log_info(fl_ctx, "finished processing task")

        if not isinstance(reply, Shareable):
            self.log_error(
                fl_ctx, "task processing error: expects result to be Shareable, but got {}".format(type(reply))
            )
            return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)

        return reply

    def _try_run(self):
        task_fetch_interval = self.task_fetch_interval
        while not self.asked_to_stop:
            with self.engine.new_context() as fl_ctx:
                if self.run_abort_signal.triggered:
                    self.log_info(fl_ctx, "run abort signal received")
                    break

                time.sleep(task_fetch_interval)

                if self.run_abort_signal.triggered:
                    self.log_info(fl_ctx, "run abort signal received")
                    break

                # reset to default fetch interval
                task_fetch_interval = self.task_fetch_interval
                self.log_debug(fl_ctx, "fetching task from server ...")
                task = self.engine.get_task_assignment(fl_ctx)
                if not task:
                    self.log_debug(fl_ctx, "no task received - will try in {} secs".format(task_fetch_interval))
                    continue

                if task.name == SpecialTaskName.END_RUN:
                    self.log_info(fl_ctx, "server asked to end the run")
                    break

                if task.name == SpecialTaskName.TRY_AGAIN:
                    task_data = task.data
                    if task_data and isinstance(task_data, Shareable):
                        task_fetch_interval = task_data.get(TaskConstant.WAIT_TIME, self.task_fetch_interval)
                    self.log_debug(
                        fl_ctx, "server asked to try again - will try in {} secs".format(task_fetch_interval)
                    )
                    continue

                self.log_info(fl_ctx, "got task assignment: name={}, id={}".format(task.name, task.task_id))

                # create a new task abort signal
                task_reply = self._process_task(task, fl_ctx)

                if not isinstance(task_reply, Shareable):
                    raise TypeError("task_reply must be Shareable, but got {}".format(type(task_reply)))
                self.log_debug(fl_ctx, "firing event EventType.BEFORE_SEND_TASK_RESULT")
                self.fire_event(EventType.BEFORE_SEND_TASK_RESULT, fl_ctx)

                # set the cookie in the reply!
                task_data = task.data
                if not isinstance(task_data, Shareable):
                    raise TypeError("task_data must be Shareable, but got {}".format(type(task_data)))

                cookie_jar = task_data.get_cookie_jar()
                if cookie_jar:
                    task_reply.set_cookie_jar(cookie_jar)

                reply_sent = self.engine.send_task_result(task_reply, fl_ctx)
                if reply_sent:
                    self.log_info(
                        fl_ctx, "result sent to server for task: name={}, id={}".format(task.name, task.task_id)
                    )
                else:
                    self.log_error(
                        fl_ctx,
                        "failed to send result to server for task: name={}, id={}".format(task.name, task.task_id),
                    )
                self.log_debug(fl_ctx, "firing event EventType.AFTER_SEND_TASK_RESULT")
                self.fire_event(EventType.AFTER_SEND_TASK_RESULT, fl_ctx)

    def run(self, app_root, args):
        with self.engine.new_context() as fl_ctx:
            self.fire_event(EventType.ABOUT_TO_START_RUN, fl_ctx)
            fl_ctx.set_prop(FLContextKey.APP_ROOT, app_root, sticky=True)
            fl_ctx.set_prop(FLContextKey.ARGS, args, sticky=True)
            fl_ctx.set_prop(ReservedKey.RUN_ABORT_SIGNAL, self.run_abort_signal, private=True, sticky=True)
            self.log_debug(fl_ctx, "firing event EventType.START_RUN")
            self.fire_event(EventType.START_RUN, fl_ctx)
            self.log_info(fl_ctx, "client runner started")
        with self.end_run_lock:
            self.end_run_fired = False

        try:
            self._try_run()
        except BaseException as e:
            with self.engine.new_context() as fl_ctx:
                self.log_exception(fl_ctx, "processing error in RUN execution: {}".format(e))
        finally:
            # in case any task is still running, abort it
            self._abort_current_task()

            self.end_run_events_sequence("run method")

    def _abort_current_task(self):
        with self.task_lock:
            task_abort_signal = self.task_abort_signal
            if task_abort_signal:
                # set task_abort_signal to None to prevent triggering again
                self.task_abort_signal = None
                task_name = ""
                task_id = ""
                task = self.current_task
                if task:
                    task_name = task.name
                    task_id = task.task_id

                with self.engine.new_context() as fl_ctx:
                    fl_ctx.set_prop(FLContextKey.TASK_NAME, value=task_name, private=True, sticky=False)
                    fl_ctx.set_prop(FLContextKey.TASK_ID, value=task_id, private=True, sticky=False)

                    task_abort_signal.trigger(True)
                    self.log_info(fl_ctx, "triggered task_abort_signal to stop task '{}'".format(task_name))

                    self.fire_event(EventType.ABORT_TASK, fl_ctx)
                    self.log_info(fl_ctx, "fired ABORT_TASK event to abort current task {}".format(task_name))

    def abort_task(self, task_names=None):
        has_task_to_abort = False
        with self.engine.new_context() as fl_ctx:
            with self.task_lock:
                if self.current_task:
                    name = self.current_task.name
                    if not task_names or name in task_names:
                        has_task_to_abort = True
                    else:
                        self.log_info(
                            fl_ctx, "Ignored abort_task request since current task '{}' is not target".format(name)
                        )
                else:
                    self.log_info(fl_ctx, "Ignored abort_task request since there is no current task")

        if has_task_to_abort:
            self._abort_current_task()

    def end_run_events_sequence(self, requester):
        with self.engine.new_context() as fl_ctx:
            self.log_info(fl_ctx, f"{requester} requests end run events sequence")
            with self.end_run_lock:
                if not self.end_run_fired:
                    self.fire_event(EventType.ABOUT_TO_END_RUN, fl_ctx)
                    self.log_info(fl_ctx, "ABOUT_TO_END_RUN fired")
                    self.fire_event(EventType.END_RUN, fl_ctx)
                    self.log_info(fl_ctx, "END_RUN fired")
                    self.end_run_fired = True

    def abort(self):
        """To Abort the current run.

        Returns: N/A

        """
        with self.engine.new_context() as fl_ctx:
            self.log_info(fl_ctx, "ABORT (RUN) command received")
        self._abort_current_task()
        self.run_abort_signal.trigger("ABORT (RUN) triggered")
        self.asked_to_stop = True
        self.end_run_events_sequence("ABORT (RUN)")

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == InfoCollector.EVENT_TYPE_GET_STATS:
            collector = fl_ctx.get_prop(InfoCollector.CTX_KEY_STATS_COLLECTOR)
            if collector:
                if not isinstance(collector, GroupInfoCollector):
                    raise TypeError("collector must be GroupInfoCollector, but got {}".format(type(collector)))
                if self.current_task:
                    current_task_name = self.current_task.name
                else:
                    current_task_name = "None"
                collector.set_info(
                    group_name="ClientRunner",
                    info={"job_id": self.job_id, "current_task_name": current_task_name, "status": "started"},
                )
        elif event_type == EventType.FATAL_TASK_ERROR:
            reason = fl_ctx.get_prop(key=FLContextKey.EVENT_DATA, default="")
            self.log_error(fl_ctx, "Aborting current task due to FATAL_TASK_ERROR received: {}".format(reason))
            self._abort_current_task()
        elif event_type == EventType.FATAL_SYSTEM_ERROR:
            reason = fl_ctx.get_prop(key=FLContextKey.EVENT_DATA, default="")
            self.log_error(fl_ctx, "Aborting current RUN due to FATAL_SYSTEM_ERROR received: {}".format(reason))
            self.abort()

    def _handle_end_run(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        self.log_info(fl_ctx, "received aux request from Server to end current RUN")
        self.abort()
        return make_reply(ReturnCode.OK)

    def _handle_abort_task(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        self.log_info(fl_ctx, "received aux request from Server to abort current task")
        task_names = request.get("task_names", None)
        self.abort_task(task_names)
        return make_reply(ReturnCode.OK)
