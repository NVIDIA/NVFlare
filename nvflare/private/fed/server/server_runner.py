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

from nvflare.apis.client import Client
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReservedTopic
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import WorkflowError
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.apis.signal import Signal
from nvflare.private.defs import SpecialTaskName, TaskConstant
from nvflare.widgets.info_collector import GroupInfoCollector, InfoCollector


class ServerRunnerConfig(object):
    def __init__(
        self,
        heartbeat_timeout: int,
        task_request_interval: int,
        workflows: [],
        task_data_filters: dict,
        task_result_filters: dict,
        handlers=None,
        components=None,
    ):
        """Configuration for ServerRunner.

        Args:
            heartbeat_timeout (int): Client heartbeat timeout in seconds
            task_request_interval (int): Task request interval in seconds
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


class ServerRunner(FLComponent):
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

    def _execute_run(self):
        while self.current_wf_index < len(self.config.workflows):
            wf = self.config.workflows[self.current_wf_index]
            try:
                with self.engine.new_context() as fl_ctx:
                    self.log_info(fl_ctx, "starting workflow {} ({}) ...".format(wf.id, type(wf.responder)))

                    wf.responder.initialize_run(fl_ctx)

                    self.log_info(fl_ctx, "Workflow {} ({}) started".format(wf.id, type(wf.responder)))
                    fl_ctx.set_prop(FLContextKey.WORKFLOW, wf.id, sticky=True)
                    self.log_debug(fl_ctx, "firing event EventType.START_WORKFLOW")
                    self.fire_event(EventType.START_WORKFLOW, fl_ctx)

                    # use the wf_lock to ensure state integrity between workflow change and message processing
                    with self.wf_lock:
                        # we only set self.current_wf to open for business after successful initialize_run!
                        self.current_wf = wf

                with self.engine.new_context() as fl_ctx:
                    wf.responder.control_flow(self.abort_signal, fl_ctx)
            except WorkflowError as e:
                with self.engine.new_context() as fl_ctx:
                    self.log_exception(fl_ctx, f"Fatal error occurred in workflow {wf.id}: {e}. Aborting the RUN")
                self.abort_signal.trigger(True)
            except BaseException as e:
                with self.engine.new_context() as fl_ctx:
                    self.log_exception(fl_ctx, "Exception in workflow {}: {}".format(wf.id, e))
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
                    except BaseException as e:
                        self.log_exception(fl_ctx, "Error finalizing workflow {}: {}".format(wf.id, e))

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
        except BaseException as ex:
            with self.engine.new_context() as fl_ctx:
                self.log_exception(fl_ctx, f"Error executing RUN: {ex}")
        finally:
            # use wf_lock to ensure state of current_wf!
            self.status = "done"
            with self.wf_lock:
                with self.engine.new_context() as fl_ctx:
                    self.fire_event(EventType.ABOUT_TO_END_RUN, fl_ctx)
                    self.log_info(fl_ctx, "ABOUT_TO_END_RUN fired")
                    self.fire_event(EventType.END_RUN, fl_ctx)
                    self.log_info(fl_ctx, "END_RUN fired")
                    self.engine.persist_components(fl_ctx, completed=True)

            # ask all clients to end run!
            self.engine.send_aux_request(
                targets=None, topic=ReservedTopic.END_RUN, request=Shareable(), timeout=0.0, fl_ctx=fl_ctx
            )

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
            reason = fl_ctx.get_prop(key=FLContextKey.EVENT_DATA, default="")
            self.log_error(fl_ctx, "Aborting current RUN due to FATAL_SYSTEM_ERROR received: {}".format(reason))
            self.abort(fl_ctx)

    def _task_try_again(self) -> (str, str, Shareable):
        task = Shareable()
        task[TaskConstant.WAIT_TIME] = self.config.task_request_interval
        return SpecialTaskName.TRY_AGAIN, "", task

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
            with self.wf_lock:
                if self.current_wf is None:
                    self.log_info(fl_ctx, "no current workflow - asked client to try again later")
                    return self._task_try_again()

                task_name, task_id, task_data = self.current_wf.responder.process_task_request(client, fl_ctx)

                if not task_name or task_name == SpecialTaskName.TRY_AGAIN:
                    self.log_debug(fl_ctx, "no task currently for client - asked client to try again later")
                    return self._task_try_again()

                if task_data:
                    if not isinstance(task_data, Shareable):
                        self.log_error(
                            fl_ctx,
                            "bad task data generated by workflow {}: must be Shareable but got {}".format(
                                self.current_wf.id, type(task_data)
                            ),
                        )
                        return self._task_try_again()
                else:
                    task_data = Shareable()

                task_data.set_header(ReservedHeaderKey.TASK_ID, task_id)
                task_data.set_header(ReservedHeaderKey.TASK_NAME, task_name)
                task_data.add_cookie(ReservedHeaderKey.WORKFLOW, self.current_wf.id)

            self.log_info(fl_ctx, "assigned task to client: name={}, id={}".format(task_name, task_id))

            # filter task data
            fl_ctx.set_prop(FLContextKey.TASK_NAME, value=task_name, private=True, sticky=False)
            fl_ctx.set_prop(FLContextKey.TASK_DATA, value=task_data, private=True, sticky=False)
            fl_ctx.set_prop(FLContextKey.TASK_ID, value=task_id, private=True, sticky=False)

            self.log_debug(fl_ctx, "firing event EventType.BEFORE_TASK_DATA_FILTER")
            self.fire_event(EventType.BEFORE_TASK_DATA_FILTER, fl_ctx)
            filter_list = self.config.task_data_filters.get(task_name)
            if filter_list:
                for f in filter_list:
                    try:
                        task_data = f.process(task_data, fl_ctx)
                    except BaseException as ex:
                        self.log_exception(
                            fl_ctx,
                            "processing error in task data filter {}: {}; "
                            "asked client to try again later".format(type(f), ex),
                        )

                        with self.wf_lock:
                            if self.current_wf:
                                self.current_wf.responder.handle_exception(task_id, fl_ctx)
                        return self._task_try_again()

            self.log_debug(fl_ctx, "firing event EventType.AFTER_TASK_DATA_FILTER")
            self.fire_event(EventType.AFTER_TASK_DATA_FILTER, fl_ctx)
            self.log_info(fl_ctx, "sent task assignment to client")
            return task_name, task_id, task_data
        except BaseException as e:
            self.log_exception(fl_ctx, f"Error processing client task request: {e}; asked client to try again later")
            return self._task_try_again()

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
        self.log_info(fl_ctx, "got result from client for task: name={}, id={}".format(task_name, task_id))

        if not isinstance(result, Shareable):
            self.log_error(fl_ctx, "invalid result submission: must be Shareable but got {}".format(type(result)))
            return

        # set the reply prop so log msg context could include RC from it
        fl_ctx.set_prop(FLContextKey.REPLY, result, private=True, sticky=False)

        fl_ctx.set_prop(FLContextKey.TASK_NAME, value=task_name, private=True, sticky=False)
        fl_ctx.set_prop(FLContextKey.TASK_RESULT, value=result, private=True, sticky=False)
        fl_ctx.set_prop(FLContextKey.TASK_ID, value=task_id, private=True, sticky=False)

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

        result.set_header(ReservedHeaderKey.TASK_NAME, task_name)
        result.set_header(ReservedHeaderKey.TASK_ID, task_id)
        result.set_peer_props(peer_ctx.get_all_public_props())

        # filter task result
        self.log_debug(fl_ctx, "firing event EventType.BEFORE_TASK_RESULT_FILTER")
        self.fire_event(EventType.BEFORE_TASK_RESULT_FILTER, fl_ctx)
        filter_list = self.config.task_result_filters.get(task_name)
        if filter_list:
            for f in filter_list:
                try:
                    result = f.process(result, fl_ctx)
                except BaseException as e:
                    self.log_exception(fl_ctx, "Error processing in task result filter {}: {}".format(type(f), e))
                    return

        self.log_debug(fl_ctx, "firing event EventType.AFTER_TASK_RESULT_FILTER")
        self.fire_event(EventType.AFTER_TASK_RESULT_FILTER, fl_ctx)

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

                self.log_debug(fl_ctx, "firing event EventType.BEFORE_PROCESS_SUBMISSION")
                self.fire_event(EventType.BEFORE_PROCESS_SUBMISSION, fl_ctx)

                self.current_wf.responder.process_submission(
                    client=client, task_name=task_name, task_id=task_id, result=result, fl_ctx=fl_ctx
                )
                self.log_info(fl_ctx, "finished processing client result by {}".format(self.current_wf.id))

                self.log_debug(fl_ctx, "firing event EventType.AFTER_PROCESS_SUBMISSION")
                self.fire_event(EventType.AFTER_PROCESS_SUBMISSION, fl_ctx)
            except BaseException as e:
                self.log_exception(fl_ctx, "Error processing client result by {}: {}".format(self.current_wf.id, e))

    def abort(self, fl_ctx: FLContext):
        self.status = "done"
        self.abort_signal.trigger(value=True)
        self.log_info(fl_ctx, "asked to abort - triggered abort_signal to stop the RUN")

    def get_persist_state(self, fl_ctx: FLContext) -> dict:
        return {"job_id": str(self.job_id), "current_wf_index": self.current_wf_index}

    def restore(self, state_data: dict, fl_ctx: FLContext):
        self.job_id = state_data.get("job_id")
        self.current_wf_index = int(state_data.get("current_wf_index", 0))
