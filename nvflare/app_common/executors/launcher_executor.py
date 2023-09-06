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

import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Optional

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.launcher import Launcher, LauncherCompleteStatus
from nvflare.app_common.utils.fl_model_utils import FLModelUtils, ParamsConverter
from nvflare.client.config import ClientConfig, ConfigKey
from nvflare.client.constants import CONFIG_EXCHANGE
from nvflare.fuel.utils.config_factory import ConfigFactory
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
        task_read_wait_time: Optional[float] = None,
        result_poll_interval: float = 0.1,
        read_interval: float = 0.1,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 30.0,
        workers: int = 1,
        training: bool = True,
        global_evaluation: bool = True,
        from_nvflare_converter_id: Optional[str] = None,
        to_nvflare_converter_id: Optional[str] = None,
    ) -> None:
        """Initializes the LauncherExecutor.

        Args:
            pipe_id (str): Identifier used to get the Pipe from NVFlare components.
            pipe_name (str): Name of the pipe. Defaults to "pipe".
            launcher_id (Optional[str]): Identifier used to get the Launcher from NVFlare components.
            launch_timeout (Optional[float]): Timeout for the "launch" method to end. None means never timeout.
            task_wait_time (Optional[float]): Time to wait for tasks to complete before exiting the executor. None means never timeout.
            task_read_wait_time (Optional[float]): Time to wait for task results from the pipe. None means no wait.
            result_poll_interval (float): Interval for polling task results from the pipe. Defaults to 0.1.
            read_interval (float): Interval for reading from the pipe. Defaults to 0.1.
            heartbeat_interval (float): Interval for sending heartbeat to the peer. Defaults to 5.0.
            heartbeat_timeout (float): Timeout for waiting for a heartbeat from the peer. Defaults to 30.0.
            workers (int): Number of worker threads needed.
            training (bool): Whether to run training using global model. Defaults to True.
            global_evaluation (bool): Whether to run evaluation on global model. Defaults to True.
            from_nvflare_converter_id (Optional[str]): Identifier used to get the ParamsConverter from NVFlare components.
                This converter will be called when model is sent from nvflare controller side to executor side.
            to_nvflare_converter_id (Optional[str]): Identifier used to get the ParamsConverter from NVFlare components.
                This converter will be called when model is sent from nvflare executor side to controller side.
        """
        super().__init__()
        self._launcher_id = launcher_id
        self.launch_timeout = launch_timeout
        self.launcher: Optional[Launcher] = None
        self._launcher_finish = Event()
        self._launcher_finish_status = None
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

        # flags to indicate whether the launcher side will send back trained model and/or metrics
        self._training = training
        self._global_evaluation = global_evaluation
        if self._training is False and self._global_evaluation is False:
            raise RuntimeError("training and global_evaluation can't be both False.")
        self._result_fl_model = None
        self._result_metrics = None

        self._from_nvflare_converter_id = from_nvflare_converter_id
        self._from_nvflare_converter: Optional[ParamsConverter] = None
        self._to_nvflare_converter_id = to_nvflare_converter_id
        self._to_nvflare_converter: Optional[ParamsConverter] = None

    def initialize(self, fl_ctx: FLContext) -> None:
        self._init_launcher(fl_ctx)
        self._init_converter(fl_ctx)

        # gets pipe
        engine = fl_ctx.get_engine()
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
        self._update_config_exchange(fl_ctx)

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
            self.log_error(fl_ctx, f"launch task: {task_name} exceeds {self.launch_timeout} seconds")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        if not launch_success:
            self.log_error(fl_ctx, f"launch task: {task_name} failed")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        future = self._wait_in_new_thread(task_name, fl_ctx, self._task_wait_time)
        result = self._exchange(task_name, shareable, fl_ctx, abort_signal)
        try:
            completion_status = future.result(timeout=self._task_wait_time)
            if completion_status != LauncherCompleteStatus.SUCCESS:
                self.log_error(fl_ctx, "launcher execution failed")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        except TimeoutError:
            self.log_error(fl_ctx, f"wait task: {task_name} exceeds {self._task_wait_time} seconds")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        self._clear()

        return result

    def _init_launcher(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        launcher: Launcher = engine.get_component(self._launcher_id)
        if launcher is not None:
            check_object_type(self._launcher_id, launcher, Launcher)
            launcher.initialize(fl_ctx)
            self.launcher = launcher

    def _init_converter(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        from_nvflare_converter: ParamsConverter = engine.get_component(self._from_nvflare_converter_id)
        if from_nvflare_converter is not None:
            check_object_type(self._from_nvflare_converter_id, from_nvflare_converter, ParamsConverter)
            self._from_nvflare_converter = from_nvflare_converter

        to_nvflare_converter: ParamsConverter = engine.get_component(self._to_nvflare_converter_id)
        if to_nvflare_converter is not None:
            check_object_type(self._to_nvflare_converter_id, to_nvflare_converter, ParamsConverter)
            self._to_nvflare_converter = to_nvflare_converter

    def _update_config_exchange(self, fl_ctx: FLContext):
        workspace = fl_ctx.get_engine().get_workspace()
        app_dir = workspace.get_app_dir(fl_ctx.get_job_id())
        config_file = os.path.join(app_dir, workspace.config_folder, CONFIG_EXCHANGE)
        config = ConfigFactory.load_config(config_file)
        if config is None:
            raise RuntimeError(f"Load config file {config} failed.")

        client_config = ClientConfig(config=config.to_dict())
        self._update_config_exchange_dict(client_config.config)
        client_config.to_json(config_file)

    def _update_config_exchange_dict(self, config: dict):
        config[ConfigKey.GLOBAL_EVAL] = self._global_evaluation
        config[ConfigKey.TRAINING] = self._training

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

    def _wait_launcher(self, task_name: str, fl_ctx: FLContext, timeout: Optional[float]) -> LauncherCompleteStatus:
        return_status = LauncherCompleteStatus.FAILED
        try:
            if self.launcher:
                return_status = self.launcher.wait_task(task_name=task_name, fl_ctx=fl_ctx, timeout=timeout)
        except Exception as e:
            self.log_exception(fl_ctx, f"launcher wait exception: {secure_format_exception(e)}")
            self._stop_launcher(task_name=task_name, fl_ctx=fl_ctx)
        self._launcher_finish.set()
        self._launcher_finish_status = return_status
        return return_status

    def _wait_in_new_thread(self, task_name: str, fl_ctx: FLContext, timeout: Optional[float]):
        future = self._thread_pool_executor.submit(self._wait_launcher, task_name, fl_ctx, timeout)
        return future

    def _exchange(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if self.pipe_handler is None:
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        model = FLModelUtils.from_shareable(shareable, self._from_nvflare_converter, fl_ctx)
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
                self.log_error(fl_ctx, f"task '{task_name}' is aborted.")
                self.pipe_handler.notify_abort(task_name)
                self._stop_launcher(task_name, fl_ctx)
                return make_reply(ReturnCode.TASK_ABORTED)

            reply: Optional[Message] = self.pipe_handler.get_next()
            if reply is None:
                if self._task_wait_time and time.time() - start > self._task_wait_time:
                    self.log_error(fl_ctx, f"task '{task_name}' timeout after {self._task_wait_time} secs")
                    self.pipe_handler.notify_abort(task_name)
                    self._stop_launcher(task_name, fl_ctx)
                    self._log_result(fl_ctx)
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            elif reply.topic == Topic.ABORT:
                self.log_error(fl_ctx, f"the other end ask to abort task '{task_name}'")
                self._stop_launcher(task_name, fl_ctx)
                self._log_result(fl_ctx)
                return make_reply(ReturnCode.TASK_ABORTED)
            elif reply.topic in [Topic.END, Topic.PEER_GONE]:
                self.log_error(fl_ctx, f"received {reply.topic} while waiting for result for {task_name}")
                self._stop_launcher(task_name, fl_ctx)
                self._log_result(fl_ctx)
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
                if reply.data.params is not None:
                    self._result_fl_model = reply.data
                if reply.data.metrics is not None:
                    self._result_metrics = reply.data

            if self._check_exchange_exit():
                break

            if self._launcher_finish.is_set():
                self.log_error(
                    fl_ctx,
                    f"Launcher already exited before exchange ended. Exit status is: '{self._launcher_finish_status}'",
                )
                self._log_result(fl_ctx)
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            time.sleep(self._result_poll_interval)
        result_fl_model = self._create_result_fl_model()
        return FLModelUtils.to_shareable(result_fl_model, self._to_nvflare_converter)

    def _log_result(self, fl_ctx):
        if self._training and self._result_fl_model is None:
            self.log_error(fl_ctx, "missing result FLModel with training flag True.")

        if self._global_evaluation and self._result_metrics is None:
            self.log_error(fl_ctx, "missing result metrics with global_evaluation flag True.")

    def _check_exchange_exit(self):
        if self._training and self._result_fl_model is None:
            return False

        if self._global_evaluation and self._result_metrics is None:
            return False

        return True

    def _create_result_fl_model(self):
        if self._result_fl_model is not None:
            if self._result_metrics is not None:
                self._result_fl_model.metrics = self._result_metrics.metrics
            return self._result_fl_model
        elif self._result_metrics is not None:
            return self._result_metrics
        else:
            raise RuntimeError("Missing result fl model and result metrics")

    def _clear(self):
        self._result_fl_model = None
        self._result_metrics = None
        self._launcher_finish_status = None
        self._launcher_finish.clear()
