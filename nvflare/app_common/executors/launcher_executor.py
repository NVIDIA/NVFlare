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
from threading import Event
from typing import Optional

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.exchange_task import ExchangeTask
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.abstract.launcher import Launcher, LauncherCompleteStatus
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.utils.fl_model_utils import FLModelUtils, ParamsConverter
from nvflare.fuel.utils.pipe.pipe import Message, Pipe
from nvflare.fuel.utils.pipe.pipe_handler import PipeHandler, Topic
from nvflare.fuel.utils.validation_utils import check_object_type
from nvflare.security.logging import secure_format_exception


class LauncherExecutor(Executor):
    def __init__(
        self,
        pipe_id: str,
        launcher_id: Optional[str] = None,
        launch_timeout: Optional[float] = None,
        wait_timeout: Optional[float] = None,
        result_timeout: Optional[float] = None,
        last_result_transfer_timeout: float = 5.0,
        peer_read_timeout: Optional[float] = None,
        result_poll_interval: float = 0.5,
        read_interval: float = 0.5,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 30.0,
        workers: int = 1,
        train_with_evaluation: bool = True,
        train_task_name: str = "train",
        evaluate_task_name: str = "evaluate",
        submit_model_task_name: str = "submit_model",
        from_nvflare_converter_id: Optional[str] = None,
        to_nvflare_converter_id: Optional[str] = None,
        launch_once: bool = True,
    ) -> None:
        """Initializes the LauncherExecutor.

        Args:
            pipe_id (str): Identifier for obtaining the Pipe from NVFlare components.
            launcher_id (Optional[str]): Identifier for obtaining the Launcher from NVFlare components.
            launch_timeout (Optional[float]): Timeout for the Launcher's "launch_task" method to complete (None for no timeout).
            wait_timeout (Optional[float]): Timeout for the Launcher's "wait_task" method to complete (None for no timeout).
            result_timeout (Optional[float]): Timeout for retrieving the result (None for no timeout).
            last_result_transfer_timeout (float): Timeout for transmitting the last result from an external process (default: 5.0).
                This value should be greater than the time needed for sending the whole result.
            peer_read_timeout (Optional[float]): Timeout for waiting the task to be read by the peer from the pipe (None for no timeout).
            result_poll_interval (float): Interval for polling task results from the pipe (default: 0.5).
            read_interval (float): Interval for reading from the pipe (default: 0.5).
            heartbeat_interval (float): Interval for sending heartbeat to the peer (default: 5.0).
            heartbeat_timeout (float): Timeout for waiting for a heartbeat from the peer (default: 30.0).
            workers (int): Number of worker threads needed (default: 4).
            train_with_evaluation (bool): Whether to run training with global model evaluation (default: True).
            train_task_name (str): Task name of traini mode (default: train).
            evaluate_task_name (str): Task name of evaluate mode (default: evaluate).
            submit_model_task_name (str): Task name of submit_model mode (default: submit_model).
            from_nvflare_converter_id (Optional[str]): Identifier used to get the ParamsConverter from NVFlare components.
                This converter will be called when model is sent from nvflare controller side to executor side.
            to_nvflare_converter_id (Optional[str]): Identifier used to get the ParamsConverter from NVFlare components.
                This converter will be called when model is sent from nvflare executor side to controller side.
            launch_once (bool): Whether to launch just once for the whole job (default: True). True means only the first task
                will trigger `launcher.launch_task`. Which is efficient when the data setup is taking a lot of time.
        """
        super().__init__()
        self.launcher: Optional[Launcher] = None
        self._launcher_id = launcher_id
        self._launch_timeout = launch_timeout
        self._wait_timeout = wait_timeout
        self._launch_once = launch_once
        self._launched = False
        self._launcher_finish = Event()
        self._launcher_finish_status = None
        self._launcher_finish_time = None
        self._last_result_transfer_timeout = last_result_transfer_timeout
        self._job_end = False

        self._thread_pool_executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=self.__class__.__name__)

        self.pipe_handler: Optional[PipeHandler] = None
        self._pipe: Optional[Pipe] = None
        self._pipe_id = pipe_id
        self._read_interval = read_interval
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout
        self._result_timeout = result_timeout
        self._result_poll_interval = result_poll_interval
        self._peer_read_timeout = peer_read_timeout

        # flags to indicate whether the launcher side will send back trained model and/or metrics
        self._train_with_evaluation = train_with_evaluation
        self._train_task_name = train_task_name
        self._evaluate_task_name = evaluate_task_name
        self._submit_model_task_name = submit_model_task_name
        self._result_fl_model = None
        self._result_metrics = None

        self._from_nvflare_converter_id = from_nvflare_converter_id
        self._from_nvflare_converter: Optional[ParamsConverter] = None
        self._to_nvflare_converter_id = to_nvflare_converter_id
        self._to_nvflare_converter: Optional[ParamsConverter] = None

    def initialize(self, fl_ctx: FLContext) -> None:
        self._init_launcher(fl_ctx)
        self._init_converter(fl_ctx)
        self._init_pipe(fl_ctx)

    def handle_event(self, event_type: str, fl_ctx: FLContext) -> None:
        if event_type == EventType.START_RUN:
            self.initialize(fl_ctx)
        elif event_type == EventType.END_RUN:
            if self.launcher is None:
                raise RuntimeError("Launcher is None.")
            self._job_end = True
            self.launcher.finalize(fl_ctx)
            self.log_info(fl_ctx, "END_RUN event received - telling external to stop")
            if self.pipe_handler is not None:
                self.pipe_handler.notify_end("END_RUN event received")
                self.pipe_handler.stop(close_pipe=True)

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.log_info(fl_ctx, f"execute for task ({task_name})")
        supported_tasks = [self._train_task_name, self._evaluate_task_name, self._submit_model_task_name]
        if task_name not in supported_tasks:
            self.log_error(fl_ctx, f"Task '{task_name}' is not in supported tasks: {supported_tasks}")
            return make_reply(ReturnCode.BAD_TASK_DATA)

        current_round = shareable.get_header(AppConstants.CURRENT_ROUND, None)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS, None)
        if task_name == self._train_task_name:
            if current_round is None:
                self.log_error(fl_ctx, "missing current round")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            if total_rounds is None:
                self.log_error(fl_ctx, "missing total number of rounds")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # if not launched yet
        if not self._launch_once or not self._launched:
            job_name = fl_ctx.get_job_id()
            self.prepare_config_for_launch(job_name, shareable, fl_ctx)
            self._init_pipe_handler(job_name)
            launch_success = self._launch(task_name, shareable, fl_ctx, abort_signal)
            if not launch_success:
                self.log_error(fl_ctx, f"launch task ({task_name}): failed")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            self._launched = True
            self.log_info(fl_ctx, f"External process for task ({task_name}) is launched.")

        if self.pipe_handler is None:
            self.log_error(fl_ctx, "pipe_handler is None")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # pipe handler starts checking for heartbeats only after the 3rd party code has been launched
        self.pipe_handler.start()

        result = self._exchange(task_name, shareable, fl_ctx, abort_signal)
        self._result_fl_model = None
        self._result_metrics = None

        if not self._launch_once or self._job_end:
            req = Message.new_request(topic=Topic.END, data="END")
            has_been_read = self.pipe_handler.send_to_peer(req, timeout=self._peer_read_timeout)
            if self._peer_read_timeout and not has_been_read:
                self.log_warning(
                    fl_ctx,
                    f"3rd party does not get END msg in {self._peer_read_timeout} secs!",
                )
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            launch_finish = self._wait_launch_finish(task_name, shareable, fl_ctx, abort_signal)
            if not launch_finish:
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            self._clear_state()
            self.log_info(fl_ctx, f"Launched external process for task ({task_name}) is finished.")

        return result

    def prepare_config_for_launch(self, task_name: str, shareable: Shareable, fl_ctx: FLContext):
        """Prepares any configuration for the process to be launched."""
        pass

    def _init_pipe(self, fl_ctx: FLContext) -> None:
        engine = fl_ctx.get_engine()
        pipe: Pipe = engine.get_component(self._pipe_id)
        check_object_type(self._pipe_id, pipe, Pipe)
        self._pipe = pipe

    def _init_pipe_handler(self, task_name: str) -> None:
        if self._pipe is None:
            raise RuntimeError("Pipe is None")
        # init pipe handler
        self._pipe.open(task_name)
        self.pipe_handler = PipeHandler(
            self._pipe,
            read_interval=self._read_interval,
            heartbeat_interval=self._heartbeat_interval,
            heartbeat_timeout=self._heartbeat_timeout,
        )

    def _init_launcher(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        launcher: Launcher = engine.get_component(self._launcher_id)
        if launcher is None:
            raise RuntimeError(f"Launcher can not be found using {self._launcher_id}")
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

    def _launch(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> bool:
        future = self._thread_pool_executor.submit(self._launch_task, task_name, shareable, fl_ctx, abort_signal)
        try:
            launch_success = future.result(timeout=self._launch_timeout)
            return launch_success
        except TimeoutError:
            self.log_error(fl_ctx, f"launch task ({task_name}) failed: exceeds {self._launch_timeout} seconds")
            return False
        except Exception as e:
            self.log_error(fl_ctx, f"launch task ({task_name}) failed: {secure_format_exception(e)}")
            return False

    def _launch_task(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> bool:
        if self.launcher is None:
            raise RuntimeError("Launcher is None.")
        return self.launcher.launch_task(task_name, shareable, fl_ctx, abort_signal)

    def _wait_launch_finish(
        self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal
    ) -> bool:
        future = self._thread_pool_executor.submit(self._wait_launcher, task_name, fl_ctx, self._wait_timeout)
        try:
            completion_status = future.result(timeout=self._wait_timeout)
            if completion_status != LauncherCompleteStatus.SUCCESS:
                self.log_error(fl_ctx, f"launcher execution for task ({task_name}) failed")
                return False
        except TimeoutError:
            self.log_error(
                fl_ctx, f"launcher execution for task ({task_name}) timeout: exceeds {self._wait_timeout} seconds"
            )
            return False
        except Exception as e:
            self.log_error(fl_ctx, f"launcher execution for task ({task_name}) failed: {secure_format_exception(e)}")
            return False
        return True

    def _wait_launcher(self, task_name: str, fl_ctx: FLContext, timeout: Optional[float]) -> LauncherCompleteStatus:
        return_status = LauncherCompleteStatus.FAILED
        try:
            if self.launcher is None:
                raise RuntimeError("Launcher is None.")
            return_status = self.launcher.wait_task(task_name=task_name, fl_ctx=fl_ctx, timeout=timeout)
        except Exception as e:
            self.log_exception(fl_ctx, f"launcher wait exception: {secure_format_exception(e)}")
        self._stop_launcher(task_name=task_name, fl_ctx=fl_ctx)
        self._launcher_finish.set()
        self._launcher_finish_status = return_status
        self._launcher_finish_time = time.time()
        return return_status

    def _stop_launcher(self, task_name: str, fl_ctx: FLContext) -> None:
        try:
            if self.launcher is None:
                raise RuntimeError("Launcher is None.")
            self.launcher.stop_task(task_name=task_name, fl_ctx=fl_ctx)
        except Exception as e:
            self.log_exception(fl_ctx, f"launcher stop exception: {secure_format_exception(e)}")

    def _exchange(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if self.pipe_handler is None:
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        model = FLModelUtils.from_shareable(shareable, self._from_nvflare_converter, fl_ctx)
        task_id = shareable.get_header(key=FLContextKey.TASK_ID)
        req = Message.new_request(topic=task_name, data=ExchangeTask(task_name, task_id, meta={}, data=model))
        has_been_read = self.pipe_handler.send_to_peer(req, timeout=self._peer_read_timeout)
        if self._peer_read_timeout and not has_been_read:
            self.log_error(
                fl_ctx,
                f"3rd party does not get req of task '{task_name}' in {self._peer_read_timeout} secs - aborting task!",
            )
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

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
                if self._result_timeout and time.time() - start > self._result_timeout:
                    self.log_error(fl_ctx, f"task '{task_name}' timeout after {self._result_timeout} secs")
                    self.pipe_handler.notify_abort(task_name)
                    self._stop_launcher(task_name, fl_ctx)
                    check_result = self._check_exchange_exit(task_name)
                    self.log_error(fl_ctx, check_result)
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            elif reply.topic == Topic.ABORT:
                self.log_error(fl_ctx, f"the other end ask to abort task '{task_name}'")
                self._stop_launcher(task_name, fl_ctx)
                check_result = self._check_exchange_exit(task_name)
                self.log_error(fl_ctx, check_result)
                return make_reply(ReturnCode.TASK_ABORTED)
            elif reply.topic in [Topic.END, Topic.PEER_GONE]:
                self.log_error(fl_ctx, f"received reply: '{reply}' while waiting for the result of {task_name}")
                self._stop_launcher(task_name, fl_ctx)
                check_result = self._check_exchange_exit(task_name)
                self.log_error(fl_ctx, check_result)
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            elif reply.msg_type != Message.REPLY:
                self.log_warning(
                    fl_ctx, f"ignored reply: '{reply}' (wrong message type) while waiting for the result of {task_name}"
                )
            elif req.topic != reply.topic:
                # ignore wrong topic
                self.log_warning(
                    fl_ctx,
                    f"ignored reply: '{reply}' (reply topic does not match req: '{req}') while waiting for the result of {task_name}",
                )
            elif req.msg_id != reply.req_id:
                self.log_warning(
                    fl_ctx,
                    f"ignored reply: '{reply}' (reply req_id does not match req msg_id: '{req}') while waiting for the result of {task_name}",
                )
            else:
                self.log_info(fl_ctx, f"got result '{reply}' for task '{task_name}'")
                if not isinstance(reply.data, ExchangeTask):
                    self.log_error(fl_ctx, "reply data is not of type ExchangeTask.")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
                if not isinstance(reply.data.data, FLModel):
                    self.log_error(fl_ctx, "reply.data.data is not of type FLModel.")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
                if reply.data.data.params is not None:
                    self._result_fl_model = reply.data.data
                if reply.data.data.metrics is not None:
                    self._result_metrics = reply.data.data

            if self._check_exchange_exit(task_name=task_name) == "":
                break

            if self._launcher_finish.is_set() and self._launcher_finish_time:
                # LauncherExecutor need to wait additional time after the Launcher finishes
                # because it will take some time to communicate the result
                # (from external process sends and LauncherExecutor receives this last result)
                # If we don't wait after the Launcher finishes, then there is possibility
                # that the result is still in transmission, but we mark it as failed.
                if time.time() - self._launcher_finish_time > self._last_result_transfer_timeout:
                    self.log_error(
                        fl_ctx,
                        "Launcher already exited and LauncherExecutor does not receive the last result within "
                        f"{self._last_result_transfer_timeout} seconds. Exit status is: '{self._launcher_finish_status}'",
                    )
                    check_result = self._check_exchange_exit(task_name)
                    self.log_error(fl_ctx, check_result)
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            time.sleep(self._result_poll_interval)
        result_fl_model = self._create_result_fl_model()
        return FLModelUtils.to_shareable(result_fl_model, self._to_nvflare_converter)

    def _check_exchange_exit(self, task_name: str) -> str:
        """Checks if exchange should be exited."""
        if task_name == self._train_task_name:
            if self._result_fl_model is None:
                return f"missing result FLModel for train_task: {self._train_task_name}."

            if self._train_with_evaluation:
                if self._result_fl_model.metrics is None and self._result_metrics is None:
                    return f"missing result metrics for train_task: {self._train_task_name}."
        elif task_name == self._evaluate_task_name:
            if self._result_metrics is None:
                return f"missing result metrics for evaluate_task: {self._evaluate_task_name}."
        elif task_name == self._submit_model_task_name:
            if self._result_fl_model is None:
                return f"missing result FLModel for submit_model_task: {self._submit_model_task_name}."
        return ""

    def _create_result_fl_model(self):
        if self._result_fl_model is not None:
            if self._result_metrics is not None:
                self._result_fl_model.metrics = self._result_metrics.metrics
            return self._result_fl_model
        elif self._result_metrics is not None:
            return self._result_metrics
        else:
            raise RuntimeError("Missing result fl model and result metrics")

    def _clear_state(self):
        self._launcher_finish_status = None
        self._launcher_finish_time = None
        self._launcher_finish.clear()
        if self.pipe_handler is not None:
            self.pipe_handler.stop(close_pipe=True)
            self.pipe_handler = None
        self._launched = False
