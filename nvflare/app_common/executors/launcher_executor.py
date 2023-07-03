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

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.launcher import Launcher
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.fuel.utils.pipe.pipe import Message, Pipe
from nvflare.fuel.utils.pipe.pipe_handler import PipeHandler, Topic
from nvflare.fuel.utils.validation_utils import check_object_type
from nvflare.security.logging import secure_format_exception


class LauncherExecutor(Executor):
    def __init__(
        self,
        pipe_id: str,
        pipe_name: str = "pipe",
        launcher_id: Optional[str] = None,
        launch_timeout: Optional[float] = None,
        task_wait_time: Optional[float] = None,
        task_read_wait_time: Optional[float] = 30.0,
        result_poll_interval: float = 0.1,
        read_interval: float = 0.1,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 30.0,
        workers: int = 1,
    ) -> None:
        """Initializes the LauncherExecutor.

        Args:
            pipe_id (str): Identifier used to get the Pipe from NVFlare components.
            pipe_name (str): Name of the pipe. Defaults to "pipe".
            launcher_id (Optional[str]): Identifier used to get the Launcher from NVFlare components.
            launch_timeout (Optional[float]): Timeout for the "launch" method to end. None means forever.
            task_wait_time (Optional[float]): Time to wait for tasks to complete before exiting the executor.
            task_read_wait_time (Optional[float]): Time to wait for task results from the pipe. Defaults to 30.0.
            result_poll_interval (float): Interval for polling task results from the pipe. Defaults to 0.1.
            read_interval (float): Interval for reading from the pipe. Defaults to 0.1.
            heartbeat_interval (float): Interval for sending heartbeat to the peer. Defaults to 5.0.
            heartbeat_timeout (float): Timeout for waiting for a heartbeat from the peer. Defaults to 30.0.
            workers (int): Number of worker threads needed.
        """
        super().__init__()
        self._launcher_id = launcher_id
        self.launch_timeout = launch_timeout
        self.launcher: Optional[Launcher] = None
        self._thread_pool_executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=self.__class__.__name__)

        self.pipe_handler: Optional[PipeHandler] = None
        self._pipe_id = pipe_id
        self._pipe_name = pipe_name
        self._topic = "data"
        self._read_interval = read_interval
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout
        self._task_wait_time = task_wait_time
        self._result_poll_interval = result_poll_interval
        self._task_read_wait_time = task_read_wait_time

    def initialize(self, fl_ctx: FLContext) -> None:
        engine = fl_ctx.get_engine()
        # init launcher
        launcher: Launcher = engine.get_component(self._launcher_id)
        if launcher is not None:
            check_object_type(self._launcher_id, launcher, Launcher)
            launcher.initialize(fl_ctx)
            self.launcher = launcher

        # gets pipe
        pipe: Pipe = engine.get_component(self._pipe_id)
        check_object_type(self._pipe_id, pipe, Pipe)

        # init pipe
        pipe.open(self._pipe_name)
        self.pipe_handler = PipeHandler(
            pipe,
            read_interval=self._read_interval,
            heartbeat_interval=self._heartbeat_interval,
            heartbeat_timeout=self._heartbeat_timeout,
        )
        self.pipe_handler.start()

    def handle_event(self, event_type: str, fl_ctx: FLContext) -> None:
        if event_type == EventType.START_RUN:
            self.initialize(fl_ctx)
        elif event_type == EventType.END_RUN:
            if self.launcher:
                self.launcher.finalize(fl_ctx)
            self.log_info(fl_ctx, "END_RUN received - telling external to stop")
            if self.pipe_handler is not None:
                self.pipe_handler.notify_end("END_RUN received")
                self.pipe_handler.stop(close_pipe=True)

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        future = self._launch_in_new_thread(task_name, shareable, fl_ctx, abort_signal)

        try:
            launch_success = future.result(timeout=self.launch_timeout)
        except TimeoutError:
            self.log_error(fl_ctx, f"launch task: {task_name} takes too long")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        if not launch_success:
            self.log_error(fl_ctx, f"launch task: {task_name} failed")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        result = self._exchange(task_name, shareable, fl_ctx, abort_signal)
        if self.launcher:
            self._stop_launcher(task_name, fl_ctx)

        return result

    def _launch(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> bool:
        if self.launcher:
            return self.launcher.launch_task(task_name, shareable, fl_ctx, abort_signal)
        return True

    def _launch_in_new_thread(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal):
        future = self._thread_pool_executor.submit(self._launch, task_name, shareable, fl_ctx, abort_signal)
        return future

    def _stop_launcher(self, task_name: str, fl_ctx: FLContext) -> None:
        try:
            if self.launcher:
                self.launcher.stop_task(task_name=task_name, fl_ctx=fl_ctx)
        except Exception as e:
            self.log_exception(fl_ctx, f"launcher stop exception: {secure_format_exception(e)}")

    def _exchange(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if self.pipe_handler is None:
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        model = FLModelUtils.from_shareable(shareable)
        req = Message.new_request(topic=self._topic, data=model)
        has_been_read = self.pipe_handler.send_to_peer(req, timeout=self._task_read_wait_time)
        if self._task_read_wait_time and not has_been_read:
            self.log_error(
                fl_ctx, f"failed to read task '{task_name}' in {self._task_read_wait_time} secs - aborting task!"
            )
            return make_reply(ReturnCode.SERVICE_UNAVAILABLE)

        # wait for result
        start = time.time()
        while True:
            if abort_signal.triggered:
                self.pipe_handler.notify_abort(task_name)
                return make_reply(ReturnCode.TASK_ABORTED)

            reply: Optional[Message] = self.pipe_handler.get_next()
            if reply is None:
                if self._task_wait_time and time.time() - start > self._task_wait_time:
                    self.log_error(fl_ctx, f"task '{task_name}' timeout after {self._task_wait_time} secs")
                    self.pipe_handler.notify_abort(task_name)
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            elif reply.topic == Topic.ABORT:
                self.log_error(fl_ctx, f"the other end ask to abort task '{task_name}'")
                return make_reply(ReturnCode.TASK_ABORTED)
            elif reply.topic in [Topic.END, Topic.PEER_GONE]:
                self.log_error(fl_ctx, f"received {reply.topic} while waiting for result for {task_name}")
                return make_reply(ReturnCode.SERVICE_UNAVAILABLE)
            elif reply.msg_type != Message.REPLY:
                self.log_warning(
                    fl_ctx, f"ignored msg '{reply.topic}.{reply.req_id}' when waiting for '{req.topic}.{req.msg_id}'"
                )
            elif req.topic != reply.topic:
                # ignore wrong task name
                self.log_warning(fl_ctx, f"ignored '{reply.topic}' when waiting for '{req.topic}'")
            elif req.msg_id != reply.req_id:
                self.log_warning(fl_ctx, f"ignored '{reply.req_id}' when waiting for '{req.msg_id}'")
            else:
                self.log_info(fl_ctx, f"got result for task '{task_name}'")
                return FLModelUtils.to_shareable(reply.data)
            time.sleep(self._result_poll_interval)
