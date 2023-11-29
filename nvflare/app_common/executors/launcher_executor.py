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

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Optional

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.launcher import Launcher, LauncherRunStatus
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.data_exchange.params_converter import ParamsConverter
from nvflare.app_common.data_exchange.piper import Piper
from nvflare.app_common.executors.task_exchanger import TaskExchanger
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.fuel.utils.validation_utils import check_object_type
from nvflare.security.logging import secure_format_exception


class LauncherExecutor(TaskExchanger):
    def __init__(
        self,
        pipe_id: str,
        launcher_id: Optional[str] = None,
        launch_timeout: Optional[float] = None,
        wait_timeout: Optional[float] = None,
        task_wait_timeout: Optional[float] = None,
        last_result_transfer_timeout: float = 5.0,
        peer_read_timeout: Optional[float] = None,
        monitor_interval: float = 0.01,
        read_interval: float = 0.001,
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
            task_wait_timeout (Optional[float]): Timeout for retrieving the task result (None for no timeout).
            last_result_transfer_timeout (float): Timeout for transmitting the last result from an external process (default: 5.0).
                This value should be greater than the time needed for sending the whole result.
            peer_read_timeout (Optional[float]): Timeout for waiting the task to be read by the peer from the pipe (None for no timeout).
            monitor_interval (float): Interval for monitoring the launcher (default: 0.01).
            read_interval (float): Interval for reading from the pipe (default: 0.5).
            heartbeat_interval (float): Interval for sending heartbeat to the peer (default: 5.0).
            heartbeat_timeout (float): Timeout for waiting for a heartbeat from the peer (default: 30.0).
            workers (int): Number of worker threads needed (default: 4).
            train_with_evaluation (bool): Whether to run training with global model evaluation (default: True).
            train_task_name (str): Task name of train mode (default: train).
            evaluate_task_name (str): Task name of evaluate mode (default: evaluate).
            submit_model_task_name (str): Task name of submit_model mode (default: submit_model).
            from_nvflare_converter_id (Optional[str]): Identifier used to get the ParamsConverter from NVFlare components.
                This ParamsConverter will be called when model is sent from nvflare controller side to executor side.
            to_nvflare_converter_id (Optional[str]): Identifier used to get the ParamsConverter from NVFlare components.
                This ParamsConverter will be called when model is sent from nvflare executor side to controller side.
            launch_once (bool): Whether to launch just once for the whole job (default: True). True means only the first task
                will trigger `launcher.launch_task`. Which is efficient when the data setup is taking a lot of time.
        """
        TaskExchanger.__init__(
            self,
            pipe_id=pipe_id,
            read_interval=read_interval,
            heartbeat_interval=heartbeat_interval,
            heartbeat_timeout=heartbeat_timeout,
            peer_read_timeout=peer_read_timeout,
            task_wait_time=task_wait_timeout,
        )
        self.launcher: Optional[Launcher] = None
        self._launcher_id = launcher_id
        self._launch_timeout = launch_timeout
        self._wait_timeout = wait_timeout
        self._launch_once = launch_once
        self._launched = Event()
        self._launcher_finish = Event()
        self._launcher_finish_status = None
        self._launcher_finish_time = None
        self._last_result_transfer_timeout = last_result_transfer_timeout
        self._received_result = False
        self._job_end = False

        self._thread_pool_executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=self.__class__.__name__)

        self._monitor_interval = monitor_interval

        # flags to indicate whether the launcher side will send back trained model and/or metrics
        self._train_with_evaluation = train_with_evaluation
        self._train_task_name = train_task_name
        self._evaluate_task_name = evaluate_task_name
        self._submit_model_task_name = submit_model_task_name

        self._from_nvflare_converter_id = from_nvflare_converter_id
        self._from_nvflare_converter: Optional[ParamsConverter] = None
        self._to_nvflare_converter_id = to_nvflare_converter_id
        self._to_nvflare_converter: Optional[ParamsConverter] = None

        self._monitor_launcher_thread = None
        self._abort_signal = None

    def initialize(self, fl_ctx: FLContext) -> None:
        self._init_launcher(fl_ctx)
        self._init_converter(fl_ctx)
        self._monitor_launcher_thread = threading.Thread(target=self._monitor_launcher, args=(fl_ctx,), daemon=True)
        self._monitor_launcher_thread.start()

    def handle_event(self, event_type: str, fl_ctx: FLContext) -> None:
        if event_type == EventType.START_RUN:
            super().handle_event(event_type, fl_ctx)
            self.initialize(fl_ctx)
        elif event_type == EventType.END_RUN:
            if self.launcher is None:
                raise RuntimeError("Launcher is None.")
            self._job_end = True
            self.launcher.finalize(fl_ctx)
            self.log_info(fl_ctx, f"{EventType.END_RUN} event received - telling external to stop")
            super().handle_event(event_type, fl_ctx)

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.log_info(fl_ctx, f"execute for task ({task_name})")

        if not self._launch_external_process(task_name, shareable, fl_ctx, abort_signal):
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        if self._from_nvflare_converter is not None:
            shareable = self._from_nvflare_converter.process(shareable, fl_ctx)

        result = super().execute(task_name, shareable, fl_ctx, abort_signal)

        if self._to_nvflare_converter is not None:
            result = self._to_nvflare_converter.process(result, fl_ctx)

        if not self._end_external_process(task_name, shareable, fl_ctx, abort_signal):
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        return result

    def stop(self, task_name: str, fl_ctx: FLContext):
        """Stops the LauncherExecutor."""
        self._stop_launcher(task_name, fl_ctx)

    def check_input_shareable(self, task_name: str, shareable: Shareable, fl_ctx: FLContext) -> bool:
        supported_tasks = [self._train_task_name, self._evaluate_task_name, self._submit_model_task_name]
        if task_name not in supported_tasks:
            self.log_error(fl_ctx, f"Task '{task_name}' is not in supported tasks: {supported_tasks}")
            return False

        current_round = shareable.get_header(AppConstants.CURRENT_ROUND, None)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS, None)
        if task_name == self._train_task_name:
            if current_round is None:
                self.log_error(fl_ctx, "missing current round")
                return False

            if total_rounds is None:
                self.log_error(fl_ctx, "missing total number of rounds")
                return False
        return True

    def check_output_shareable(self, task_name: str, shareable: Shareable, fl_ctx: FLContext) -> bool:
        check_result = self._check_result_shareable(task_name, shareable)
        if check_result != "":
            self.log_error(fl_ctx, check_result)
            return False
        self._received_result = True
        return True

    def prepare_config_for_launch(self, shareable: Shareable, fl_ctx: FLContext):
        """Prepares any configuration for the process to be launched."""
        pass

    def get_external_pipe_class(self, fl_ctx: FLContext):
        return Piper.get_external_pipe_class(self.pipe_id, fl_ctx)

    def get_external_pipe_args(self, fl_ctx: FLContext):
        return Piper.get_external_pipe_args(self.pipe_id, fl_ctx)

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

    def _launch_external_process(
        self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal
    ) -> bool:
        self._abort_signal = abort_signal
        # if not launched yet
        if not self._launch_once or not self._launched.is_set():
            self.prepare_config_for_launch(shareable, fl_ctx)
            launch_success = self._launch(task_name, shareable, fl_ctx, abort_signal)
            if not launch_success:
                self.log_error(fl_ctx, f"launch task ({task_name}): failed")
                return False
            self._launched.set()
            self.log_info(fl_ctx, f"External process for task ({task_name}) is launched.")
        # wait for external process to set up their pipe_handler
        setup_success = self._wait_external_setup(task_name, fl_ctx, abort_signal)
        if not setup_success:
            self.log_error(fl_ctx, "External process set up failed.")
            return False
        return True

    def _end_external_process(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal):
        if not self._launch_once or self._job_end:
            ask_peer_end_success = self.ask_peer_to_end(fl_ctx)
            if not ask_peer_end_success:
                return False
            launch_finish = self._wait_launch_finish(task_name, shareable, fl_ctx, abort_signal)
            if not launch_finish:
                return False
            self._clear_state()
            self.log_info(fl_ctx, f"Launched external process for task ({task_name}) is finished.")
        return True

    def _wait_external_setup(self, task_name: str, fl_ctx: FLContext, abort_signal: Signal):
        start_time = time.time()
        while True:
            if self._launch_timeout and time.time() - start_time >= self._launch_timeout:
                self.log_error(fl_ctx, f"External process is not set up within timeout: {self._launch_timeout}")
                return False

            if abort_signal.triggered:
                return False

            if self.peer_is_up_or_dead():
                return True

            if self.launcher.check_run_status(task_name, fl_ctx) != LauncherRunStatus.RUNNING:
                return False

            time.sleep(0.1)

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
            if completion_status != LauncherRunStatus.COMPLETE_SUCCESS:
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

    def _wait_launcher(self, task_name: str, fl_ctx: FLContext, timeout: Optional[float]) -> str:
        return_status = LauncherRunStatus.COMPLETE_FAILED
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

    def _check_result_shareable(self, task_name: str, result) -> str:
        """Checks if exchange should be exited."""
        result_fl_model = FLModelUtils.from_shareable(result)

        if task_name == self._train_task_name and self._train_with_evaluation:
            if result_fl_model.metrics is None:
                return f"missing result metrics for train_task: {self._train_task_name}."
        elif task_name == self._evaluate_task_name:
            if result_fl_model.metrics is None:
                return f"missing result metrics for evaluate_task: {self._evaluate_task_name}."
        elif task_name == self._submit_model_task_name:
            if result_fl_model is None:
                return f"missing result FLModel for submit_model_task: {self._submit_model_task_name}."
        return ""

    def _monitor_launcher(self, fl_ctx: FLContext):
        """Monitors the launcher.

        Trigger the abort signal if "_launcher_finish" is set and "_launcher_finish_time" has passed,
        so TaskExchanger will stop waiting.

        Note:
            If we don't wait extra time after the Launcher finishes, then there is possibility
            that the result is still in transmission, but we will mark it as failed.
            (for example: when using FilePipe, if result has been written out but not read.)
        """
        while True:
            # job end
            if self._job_end:
                break

            # launcher is launched
            if self._launched.is_set():
                if self._abort_signal.triggered or self._received_result:
                    continue

                if self._launcher_finish.is_set() and self._launcher_finish_time:
                    # LauncherExecutor need to wait additional time after the Launcher finishes
                    # because it will take some time to communicate the result
                    # (from external process sends and LauncherExecutor receives this last result)
                    if time.time() - self._launcher_finish_time > self._last_result_transfer_timeout:
                        self.log_error(
                            fl_ctx,
                            "Launcher already exited and LauncherExecutor does not receive the last result within "
                            f"{self._last_result_transfer_timeout} seconds. Exit status is: '{self._launcher_finish_status}'",
                        )
                        self._abort_signal.trigger("exceeds_launcher_finish_timeout")

            time.sleep(self._monitor_interval)

    def _clear_state(self):
        self._launcher_finish_status = None
        self._launcher_finish_time = None
        self._launcher_finish.clear()
        self._launched.clear()
        self.clear_pipe()
        self._received_result = False
