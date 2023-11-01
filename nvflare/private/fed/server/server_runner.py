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

import threading
import time

from nvflare.apis.client import Client
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FilterKey, FLContextKey, ReservedKey, ReservedTopic, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.utils.fl_context_utils import add_job_audit_event
from nvflare.apis.utils.task_utils import apply_filters
from nvflare.private.defs import SpecialTaskName, TaskConstant
from nvflare.private.privacy_manager import Scope
from nvflare.security.logging import secure_format_exception
from nvflare.widgets.info_collector import GroupInfoCollector, InfoCollector


class ServerRunnerConfig(object):
    def __init__(
        self,
        heartbeat_timeout: int,
        task_request_interval: float,
        workflows: [],
        task_data_filters: dict,
        task_result_filters: dict,
        handlers=None,
        components=None,
    ):
        """Configuration for ServerRunner.

        Args:
            heartbeat_timeout (int): Client heartbeat timeout in seconds
            task_request_interval (float): Task request interval in seconds
            workflows (list): A list of workflow
            task_data_filters (dict):  A dict of  {task_name: list of filters apply to data (pre-process)}
            task_result_filters (dict): A dict of {task_name: list of filters apply to result (post-process)}
            handlers (list, optional):  A list of event handlers
            components (dict, optional):  A dict of extra python objects {id: object}
        """
        self.heartbeat_timeout = heartbeat_timeout
        self.task_request_interval = task_request_interval
        self.workflows = workflows
        self.task_data_filters = task_data_filters
        self.task_result_filters = task_result_filters
        self.handlers = handlers
        self.components = components

    def add_component(self, comp_id: str, component: object):
        if not isinstance(comp_id, str):
            raise TypeError(f"component id must be str but got {type(comp_id)}")

        if comp_id in self.components:
            raise ValueError(f"duplicate component id {comp_id}")

        self.components[comp_id] = component
        if isinstance(component, FLComponent):
            self.handlers.append(component)


class ServerRunner(FLComponent):

    ABORT_RETURN_CODES = [
        ReturnCode.RUN_MISMATCH,
        ReturnCode.TASK_UNKNOWN,
        ReturnCode.UNSAFE_JOB,
    ]

    def __init__(self, config: ServerRunnerConfig, job_id: str, engine: ServerEngineSpec):
        """Server runner class.

        Args:
            config (ServerRunnerConfig): configuration of server runner
            job_id (str): The number to distinguish each experiment
            engine (ServerEngineSpec): server engine
        """
        FLComponent.__init__(self)
        self.job_id = job_id
        self.config = config
        self.engine = engine
        self.abort_signal = Signal()
        self.wf_lock = threading.Lock()
        self.current_wf = None
        self.current_wf_index = 0
        self.status = "init"
        self.turn_to_cold = False
        self._register_aux_message_handler(engine)

    def _register_aux_message_handler(self, engine):
        engine.register_aux_message_handler(
            topic=ReservedTopic.SYNC_RUNNER, message_handle_func=self._handle_sync_runner
        )

        engine.register_aux_message_handler(
            topic=ReservedTopic.JOB_HEART_BEAT, message_handle_func=self._handle_job_heartbeat
        )

        engine.register_aux_message_handler(topic=ReservedTopic.TASK_CHECK, message_handle_func=self._handle_task_check)

    def _handle_sync_runner(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        # simply ack
        return make_reply(ReturnCode.OK)

    def _execute_run(self):
        while self.current_wf_index < len(self.config.workflows):
            wf = self.config.workflows[self.current_wf_index]
            try:
                with self.engine.new_context() as fl_ctx:
                    self.log_info(fl_ctx, "starting workflow {} ({}) ...".format(wf.id, type(wf.responder)))

                    fl_ctx.set_prop(FLContextKey.WORKFLOW, wf.id, sticky=True)
                    wf.responder.initialize_run(fl_ctx)

                    self.log_info(fl_ctx, "Workflow {} ({}) started".format(wf.id, type(wf.responder)))
                    self.log_debug(fl_ctx, "firing event EventType.START_WORKFLOW")
                    self.fire_event(EventType.START_WORKFLOW, fl_ctx)

                    # use the wf_lock to ensure state integrity between workflow change and message processing
                    with self.wf_lock:
                        # we only set self.current_wf to open for business after successful initialize_run!
                        self.current_wf = wf

                with self.engine.new_context() as fl_ctx:
                    wf.responder.control_flow(self.abort_signal, fl_ctx)
            except Exception as e:
                with self.engine.new_context() as fl_ctx:
                    self.log_exception(fl_ctx, "Exception in workflow {}: {}".format(wf.id, secure_format_exception(e)))
                self.system_panic("Exception in workflow {}: {}".format(wf.id, secure_format_exception(e)), fl_ctx)
            finally:
                with self.engine.new_context() as fl_ctx:
                    # do not execute finalize_run() until the wf_lock is acquired
                    with self.wf_lock:
                        # unset current_wf to prevent message processing
                        # then we can release the lock - no need to delay message processing
                        # during finalization!
                        # Note: WF finalization may take time since it needs to wait for
                        # the job monitor to join.
                        self.current_wf = None

                    self.log_info(fl_ctx, f"Workflow: {wf.id} finalizing ...")
                    try:
                        wf.responder.finalize_run(fl_ctx)
                    except Exception as e:
                        self.log_exception(
                            fl_ctx, "Error finalizing workflow {}: {}".format(wf.id, secure_format_exception(e))
                        )

                    self.log_debug(fl_ctx, "firing event EventType.END_WORKFLOW")
                    self.fire_event(EventType.END_WORKFLOW, fl_ctx)

                # Stopped the server runner from the current responder, not continue the following responders.
                if self.abort_signal.triggered:
                    break
            self.current_wf_index += 1

    def run(self):
        with self.engine.new_context() as fl_ctx:
            self.log_info(fl_ctx, "Server runner starting ...")
            self.log_debug(fl_ctx, "firing event EventType.START_RUN")
            fl_ctx.set_prop(ReservedKey.RUN_ABORT_SIGNAL, self.abort_signal, private=True, sticky=True)
            self.fire_event(EventType.START_RUN, fl_ctx)
            self.engine.persist_components(fl_ctx, completed=False)

        self.status = "started"
        try:
            self._execute_run()
        except Exception as e:
            with self.engine.new_context() as fl_ctx:
                self.log_exception(fl_ctx, f"Error executing RUN: {secure_format_exception(e)}")
        finally:
            # use wf_lock to ensure state of current_wf!
            self.status = "done"
            with self.wf_lock:
                with self.engine.new_context() as fl_ctx:
                    self.fire_event(EventType.ABOUT_TO_END_RUN, fl_ctx)
                    self.log_info(fl_ctx, "ABOUT_TO_END_RUN fired")

                    if not self.turn_to_cold:
                        # ask all clients to end run!
                        self.engine.send_aux_request(
                            targets=None,
                            topic=ReservedTopic.END_RUN,
                            request=Shareable(),
                            timeout=0.0,
                            fl_ctx=fl_ctx,
                            optional=True,
                            secure=False,
                        )

                        self.engine.persist_components(fl_ctx, completed=True)
                    self.fire_event(EventType.END_RUN, fl_ctx)
                    self.log_info(fl_ctx, "END_RUN fired")

            self.log_info(fl_ctx, "Server runner finished.")

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == InfoCollector.EVENT_TYPE_GET_STATS:
            collector = fl_ctx.get_prop(InfoCollector.CTX_KEY_STATS_COLLECTOR)
            if collector:
                if not isinstance(collector, GroupInfoCollector):
                    raise TypeError("collector must be GroupInfoCollect but got {}".format(type(collector)))

                with self.wf_lock:
                    if self.current_wf:
                        collector.set_info(
                            group_name="ServerRunner",
                            info={"job_id": self.job_id, "status": self.status, "workflow": self.current_wf.id},
                        )
        elif event_type == EventType.FATAL_SYSTEM_ERROR:
            fl_ctx.set_prop(key=FLContextKey.FATAL_SYSTEM_ERROR, value=True, private=True, sticky=True)
            reason = fl_ctx.get_prop(key=FLContextKey.EVENT_DATA, default="")
            self.log_error(fl_ctx, "Aborting current RUN due to FATAL_SYSTEM_ERROR received: {}".format(reason))
            self.abort(fl_ctx)

    def _task_try_again(self) -> (str, str, Shareable):
        task_data = Shareable()
        task_data.set_header(TaskConstant.WAIT_TIME, self.config.task_request_interval)
        return SpecialTaskName.TRY_AGAIN, "", task_data

    def process_task_request(self, client: Client, fl_ctx: FLContext) -> (str, str, Shareable):
        """Process task request from a client.

        NOTE: the Engine will create a new fl_ctx and call this method:

            with engine.new_context() as fl_ctx:
                name, id, data = runner.process_task_request(client, fl_ctx)
                ...

        Args:
            client (Client): client object
            fl_ctx (FLContext): FL context

        Returns:
            A tuple of (task name, task id, and task data)
        """
        engine = fl_ctx.get_engine()
        if not isinstance(engine, ServerEngineSpec):
            raise TypeError("engine must be ServerEngineSpec but got {}".format(type(engine)))

        self.log_debug(fl_ctx, "process task request from client")

        if self.status == "init":
            self.log_debug(fl_ctx, "server runner still initializing - asked client to try again later")
            return self._task_try_again()

        if self.status == "done":
            self.log_info(fl_ctx, "server runner is finalizing - asked client to end the run")
            return SpecialTaskName.END_RUN, "", None

        peer_ctx = fl_ctx.get_peer_context()
        if not isinstance(peer_ctx, FLContext):
            self.log_debug(fl_ctx, "invalid task request: no peer context - asked client to try again later")
            return self._task_try_again()

        peer_job_id = peer_ctx.get_job_id()
        if not peer_job_id or peer_job_id != self.job_id:
            # the client is in a different RUN
            self.log_info(fl_ctx, "invalid task request: not the same job_id - asked client to end the run")
            return SpecialTaskName.END_RUN, "", None

        try:
            task_name, task_id, task_data = self._try_to_get_task(
                # client, fl_ctx, self.config.task_request_timeout, self.config.task_retry_interval
                client,
                fl_ctx,
            )
            if not task_name or task_name == SpecialTaskName.TRY_AGAIN:
                return self._task_try_again()

            # filter task data
            self.log_debug(fl_ctx, "firing event EventType.BEFORE_TASK_DATA_FILTER")
            self.fire_event(EventType.BEFORE_TASK_DATA_FILTER, fl_ctx)

            try:
                filter_name = Scope.TASK_DATA_FILTERS_NAME
                task_data = apply_filters(
                    filter_name, task_data, fl_ctx, self.config.task_data_filters, task_name, FilterKey.OUT
                )
            except Exception as e:
                self.log_exception(
                    fl_ctx,
                    "processing error in task data filter {}; "
                    "asked client to try again later".format(secure_format_exception(e)),
                )
                with self.wf_lock:
                    if self.current_wf:
                        self.current_wf.responder.handle_exception(task_id, fl_ctx)
                return self._task_try_again()

            self.log_debug(fl_ctx, "firing event EventType.AFTER_TASK_DATA_FILTER")
            self.fire_event(EventType.AFTER_TASK_DATA_FILTER, fl_ctx)
            self.log_info(fl_ctx, f"sent task assignment to client. client_name:{client.name} task_id:{task_id}")

            audit_event_id = add_job_audit_event(fl_ctx=fl_ctx, msg=f'sent task to client "{client.name}"')
            task_data.set_header(ReservedHeaderKey.AUDIT_EVENT_ID, audit_event_id)
            task_data.set_header(TaskConstant.WAIT_TIME, self.config.task_request_interval)
            return task_name, task_id, task_data
        except Exception as e:
            self.log_exception(
                fl_ctx,
                f"Error processing client task request: {secure_format_exception(e)}; asked client to try again later",
            )
            return self._task_try_again()

    def _try_to_get_task(self, client, fl_ctx, timeout=None, retry_interval=0.005):
        start = time.time()
        while True:
            with self.wf_lock:
                if self.current_wf is None:
                    self.log_debug(fl_ctx, "no current workflow - asked client to try again later")
                    return "", "", None

                task_name, task_id, task_data = self.current_wf.responder.process_task_request(client, fl_ctx)

                if task_name and task_name != SpecialTaskName.TRY_AGAIN:
                    if task_data:
                        if not isinstance(task_data, Shareable):
                            self.log_error(
                                fl_ctx,
                                "bad task data generated by workflow {}: must be Shareable but got {}".format(
                                    self.current_wf.id, type(task_data)
                                ),
                            )
                            return "", "", None
                    else:
                        task_data = Shareable()

                    task_data.set_header(ReservedHeaderKey.TASK_ID, task_id)
                    task_data.set_header(ReservedHeaderKey.TASK_NAME, task_name)
                    task_data.add_cookie(ReservedHeaderKey.WORKFLOW, self.current_wf.id)

                    fl_ctx.set_prop(FLContextKey.TASK_NAME, value=task_name, private=True, sticky=False)
                    fl_ctx.set_prop(FLContextKey.TASK_ID, value=task_id, private=True, sticky=False)
                    fl_ctx.set_prop(FLContextKey.TASK_DATA, value=task_data, private=True, sticky=False)

                    self.log_info(fl_ctx, f"assigned task to client {client.name}: name={task_name}, id={task_id}")

                    return task_name, task_id, task_data

            if timeout is None or time.time() - start > timeout:
                break

            time.sleep(retry_interval)

        # ask client to retry
        return "", "", None

    def handle_dead_job(self, client_name: str, fl_ctx: FLContext):
        with self.wf_lock:
            try:
                if self.current_wf is None:
                    return

                self.current_wf.responder.handle_dead_job(client_name=client_name, fl_ctx=fl_ctx)
            except Exception as e:
                self.log_exception(
                    fl_ctx, f"Error processing dead job by workflow {self.current_wf.id}: {secure_format_exception(e)}"
                )

    def process_submission(self, client: Client, task_name: str, task_id: str, result: Shareable, fl_ctx: FLContext):
        """Process task result submitted from a client.

        NOTE: the Engine will create a new fl_ctx and call this method:

            with engine.new_context() as fl_ctx:
                name, id, data = runner.process_submission(client, fl_ctx)

        Args:
            client: Client object
            task_name: task name
            task_id: task id
            result: task result
            fl_ctx: FLContext
        """
        self.log_info(fl_ctx, f"got result from client {client.name} for task: name={task_name}, id={task_id}")

        if not isinstance(result, Shareable):
            self.log_error(fl_ctx, "invalid result submission: must be Shareable but got {}".format(type(result)))
            return

        # set the reply prop so log msg context could include RC from it
        fl_ctx.set_prop(FLContextKey.REPLY, result, private=True, sticky=False)
        fl_ctx.set_prop(FLContextKey.TASK_NAME, value=task_name, private=True, sticky=False)
        fl_ctx.set_prop(FLContextKey.TASK_RESULT, value=result, private=True, sticky=False)
        fl_ctx.set_prop(FLContextKey.TASK_ID, value=task_id, private=True, sticky=False)

        client_audit_event_id = result.get_header(ReservedHeaderKey.AUDIT_EVENT_ID, "")
        add_job_audit_event(
            fl_ctx=fl_ctx, ref=client_audit_event_id, msg=f"received result from client '{client.name}'"
        )

        if self.status != "started":
            self.log_info(fl_ctx, "ignored result submission since server runner's status is {}".format(self.status))
            return

        peer_ctx = fl_ctx.get_peer_context()
        if not isinstance(peer_ctx, FLContext):
            self.log_info(fl_ctx, "invalid result submission: no peer context - dropped")
            return

        peer_job_id = peer_ctx.get_job_id()
        if not peer_job_id or peer_job_id != self.job_id:
            # the client is on a different RUN
            self.log_info(fl_ctx, "invalid result submission: not the same job id - dropped")
            return

        rc = result.get_return_code(default=ReturnCode.OK)
        if rc in self.ABORT_RETURN_CODES:
            self.log_error(fl_ctx, f"aborting ServerRunner due to fatal return code {rc} from client {client.name}")
            self.system_panic(
                reason=f"Aborted job {self.job_id} due to fatal return code {rc} from client {client.name}",
                fl_ctx=fl_ctx,
            )
            return

        result.set_header(ReservedHeaderKey.TASK_NAME, task_name)
        result.set_header(ReservedHeaderKey.TASK_ID, task_id)
        result.set_peer_props(peer_ctx.get_all_public_props())

        with self.wf_lock:
            try:
                if self.current_wf is None:
                    self.log_info(fl_ctx, "no current workflow - dropped submission.")
                    return

                wf_id = result.get_cookie(ReservedHeaderKey.WORKFLOW, None)
                if wf_id is not None and wf_id != self.current_wf.id:
                    self.log_info(
                        fl_ctx,
                        "Got result for workflow {}, but we are running {} - dropped submission.".format(
                            wf_id, self.current_wf.id
                        ),
                    )
                    return

                # filter task result
                self.log_debug(fl_ctx, "firing event EventType.BEFORE_TASK_RESULT_FILTER")
                self.fire_event(EventType.BEFORE_TASK_RESULT_FILTER, fl_ctx)

                try:
                    filter_name = Scope.TASK_RESULT_FILTERS_NAME
                    result = apply_filters(
                        filter_name, result, fl_ctx, self.config.task_result_filters, task_name, FilterKey.IN
                    )
                except Exception as e:
                    self.log_exception(
                        fl_ctx,
                        "processing error in task result filter {}; ".format(secure_format_exception(e)),
                    )
                    result = make_reply(ReturnCode.TASK_RESULT_FILTER_ERROR)

                self.log_debug(fl_ctx, "firing event EventType.AFTER_TASK_RESULT_FILTER")
                self.fire_event(EventType.AFTER_TASK_RESULT_FILTER, fl_ctx)

                self.log_debug(fl_ctx, "firing event EventType.BEFORE_PROCESS_SUBMISSION")
                self.fire_event(EventType.BEFORE_PROCESS_SUBMISSION, fl_ctx)

                self.current_wf.responder.process_submission(
                    client=client, task_name=task_name, task_id=task_id, result=result, fl_ctx=fl_ctx
                )
                self.log_info(fl_ctx, "finished processing client result by {}".format(self.current_wf.id))

                self.log_debug(fl_ctx, "firing event EventType.AFTER_PROCESS_SUBMISSION")
                self.fire_event(EventType.AFTER_PROCESS_SUBMISSION, fl_ctx)
            except Exception as e:
                self.log_exception(
                    fl_ctx,
                    "Error processing client result by {}: {}".format(self.current_wf.id, secure_format_exception(e)),
                )

    def _handle_job_heartbeat(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        self.log_debug(fl_ctx, "received client job_heartbeat")
        return make_reply(ReturnCode.OK)

    def _handle_task_check(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        task_id = request.get_header(ReservedHeaderKey.TASK_ID)
        if not task_id:
            self.log_error(fl_ctx, f"missing {ReservedHeaderKey.TASK_ID} in task_check request")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        self.log_info(fl_ctx, f"received task_check on task {task_id}")

        with self.wf_lock:
            if self.current_wf is None or self.current_wf.responder is None:
                self.log_info(fl_ctx, "no current workflow - dropped task_check.")
                return make_reply(ReturnCode.TASK_UNKNOWN)

            task = self.current_wf.responder.process_task_check(task_id=task_id, fl_ctx=fl_ctx)
            if task:
                self.log_info(fl_ctx, f"task {task_id} is still good")
                return make_reply(ReturnCode.OK)
            else:
                self.log_info(fl_ctx, f"task {task_id} is not found")
                return make_reply(ReturnCode.TASK_UNKNOWN)

    def abort(self, fl_ctx: FLContext, turn_to_cold: bool = False):
        self.status = "done"
        self.abort_signal.trigger(value=True)
        self.turn_to_cold = turn_to_cold
        self.log_info(fl_ctx, "asked to abort - triggered abort_signal to stop the RUN")

    def get_persist_state(self, fl_ctx: FLContext) -> dict:
        return {"job_id": str(self.job_id), "current_wf_index": self.current_wf_index}

    def restore(self, state_data: dict, fl_ctx: FLContext):
        self.job_id = state_data.get("job_id")
        self.current_wf_index = int(state_data.get("current_wf_index", 0))
