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
from threading import Event, Lock
from typing import Any, Optional

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.launcher import Launcher, LauncherRunStatus
from nvflare.app_common.abstract.params_converter import ParamsConverter
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.task_exchanger import TaskExchanger
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.fuel.utils.validation_utils import check_object_type
from nvflare.security.logging import secure_format_exception

LAUNCHER_EXCEPTION = "launcher_exception"


class LauncherExecutor(TaskExchanger):
    def __init__(
        self,
        pipe_id: str,
        launcher_id: Optional[str] = None,
        launch_timeout: Optional[float] = None,
        task_wait_timeout: Optional[float] = None,
        last_result_transfer_timeout: float = 300.0,
        external_pre_init_timeout: float = 60.0,
        peer_read_timeout: Optional[float] = 60.0,
        monitor_interval: float = 0.1,
        read_interval: float = 0.5,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 60.0,
        workers: int = 4,
        train_with_evaluation: bool = True,
        train_task_name: str = AppConstants.TASK_TRAIN,
        evaluate_task_name: str = AppConstants.TASK_VALIDATION,
        submit_model_task_name: str = AppConstants.TASK_SUBMIT_MODEL,
        from_nvflare_converter_id: Optional[str] = None,
        to_nvflare_converter_id: Optional[str] = None,
    ) -> None:
        """Initializes the LauncherExecutor.

        Args:
            pipe_id (str): Identifier for obtaining the Pipe from NVFlare components.
            launcher_id (Optional[str]): Identifier for obtaining the Launcher from NVFlare components.
            launch_timeout (Optional[float]): Timeout for the Launcher's "launch_task" method to complete (None for no timeout).
            task_wait_timeout (Optional[float]): Timeout for retrieving the task result (None for no timeout).
            last_result_transfer_timeout (float): Timeout for transmitting the last result from an external process.
                This value should be greater than the time needed for sending the whole result.
            external_pre_init_timeout (float): Time to wait for external process before it calls flare.init().
            peer_read_timeout (float, optional): time to wait for peer to accept sent message.
            monitor_interval (float): Interval for monitoring the launcher.
            read_interval (float): Interval for reading from the pipe.
            heartbeat_interval (float): Interval for sending heartbeat to the peer.
            heartbeat_timeout (float): Timeout for waiting for a heartbeat from the peer.
            workers (int): Number of worker threads needed.
            train_with_evaluation (bool): Whether to run training with global model evaluation.
            train_task_name (str): Task name of train mode.
            evaluate_task_name (str): Task name of evaluate mode.
            submit_model_task_name (str): Task name of submit_model mode.
            from_nvflare_converter_id (Optional[str]): Identifier used to get the ParamsConverter from NVFlare components.
                This ParamsConverter will be called when model is sent from nvflare controller side to executor side.
            to_nvflare_converter_id (Optional[str]): Identifier used to get the ParamsConverter from NVFlare components.
                This ParamsConverter will be called when model is sent from nvflare executor side to controller side.
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

        self._launcher_finish = False
        self._launcher_finish_time = None
        self._last_result_transfer_timeout = last_result_transfer_timeout
        self._external_pre_init_timeout = external_pre_init_timeout
        self._received_result = Event()
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
        self._current_task = None
        self._lock = Lock()

    def initialize(self, fl_ctx: FLContext) -> None:
        self._init_launcher(fl_ctx)
        self._init_converter(fl_ctx)
        self._monitor_launcher_thread = threading.Thread(target=self._monitor_launcher, args=(fl_ctx,), daemon=True)
        self._monitor_launcher_thread.start()

    def finalize(self, fl_ctx: FLContext) -> None:
        self._execute_launcher_method_in_thread_executor(method_name="finalize", fl_ctx=fl_ctx)

    def handle_event(self, event_type: str, fl_ctx: FLContext) -> None:
        if event_type == EventType.START_RUN:
            super().handle_event(event_type, fl_ctx)
            self.initialize(fl_ctx)
        elif event_type == EventType.END_RUN:
            if self.launcher is None:
                raise RuntimeError("Launcher is None.")
            with self._lock:
                self._job_end = True
            if self._abort_signal is not None:
                self._abort_signal.trigger(f"{EventType.END_RUN} event received - telling external to stop")
            self.finalize(fl_ctx)
            self.log_info(fl_ctx, f"{EventType.END_RUN} event received - telling external to stop")
            super().handle_event(event_type, fl_ctx)
        else:
            super().handle_event(event_type, fl_ctx)

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.log_info(fl_ctx, f"execute for task ({task_name})")

        if not self._initialize_external_execution(task_name, shareable, fl_ctx, abort_signal):
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        if self._from_nvflare_converter is not None:
            shareable = self._from_nvflare_converter.process(task_name, shareable, fl_ctx)

        result = super().execute(task_name, shareable, fl_ctx, abort_signal)

        if result.get_return_code() != ReturnCode.OK:
            abort_signal.trigger("execution exception in TaskExchanger")
            self._execute_launcher_method_in_thread_executor(
                method_name="stop_task", task_name=task_name, fl_ctx=fl_ctx, abort_signal=abort_signal
            )
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        if self._to_nvflare_converter is not None:
            result = self._to_nvflare_converter.process(task_name, result, fl_ctx)

        self._finalize_external_execution(task_name, shareable, fl_ctx, abort_signal)

        return result

    def check_input_shareable(self, task_name: str, shareable: Shareable, fl_ctx: FLContext) -> bool:
        supported_tasks = [self._train_task_name, self._evaluate_task_name, self._submit_model_task_name]
        if task_name not in supported_tasks:
            self.log_error(fl_ctx, f"Task '{task_name}' is not in supported tasks: {supported_tasks}")
            return False

        current_round = shareable.get_header(AppConstants.CURRENT_ROUND, None)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS, None)
        if task_name == self._train_task_name:
            if current_round is None:
                self.log_warning(fl_ctx, f"no current round for task {task_name}")

            if total_rounds is None:
                self.log_warning(fl_ctx, f"no total number of rounds for task {task_name}")

        return True

    def check_output_shareable(self, task_name: str, shareable: Shareable, fl_ctx: FLContext) -> bool:
        check_result = self._check_result_shareable(task_name, shareable)
        if check_result != "":
            self.log_error(fl_ctx, check_result)
            return False
        with self._lock:
            self._received_result.set()
            self._current_task = None
        return True

    def _init_launcher(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        launcher: Launcher = engine.get_component(self._launcher_id)
        if launcher is None:
            raise RuntimeError(f"Launcher can not be found using {self._launcher_id}")
        check_object_type(self._launcher_id, launcher, Launcher)
        self.launcher = launcher
        if (
            self._execute_launcher_method_in_thread_executor(method_name="initialize", fl_ctx=fl_ctx)
            == LAUNCHER_EXCEPTION
        ):
            raise RuntimeError("Launcher initialize failed.")

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

    def _initialize_external_execution(
        self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal
    ) -> bool:
        with self._lock:
            self._abort_signal = abort_signal
            self._current_task = task_name

        launch_task_success = self._execute_launcher_method_in_thread_executor(
            method_name="launch_task",
            task_name=task_name,
            shareable=shareable,
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )
        if not launch_task_success or launch_task_success == LAUNCHER_EXCEPTION:
            abort_signal.trigger("launch task failed")
            return False

        self.log_info(fl_ctx, f"External execution for task ({task_name}) is launched.")
        # wait for external execution to set up their pipe_handler
        setup_success = self._wait_external_setup(task_name, fl_ctx, abort_signal)
        if not setup_success:
            self.log_error(fl_ctx, "External execution set up failed.")
            abort_signal.trigger("External execution set up failed.")
            return False
        return True

    def _execute_launcher_method_in_thread_executor(self, method_name: str, **kwargs) -> Any:
        try:
            if self.launcher is None:
                raise RuntimeError("Launcher is None")

            future = self._thread_pool_executor.submit(getattr(self.launcher, method_name), **kwargs)
            result = future.result(timeout=self._launch_timeout)

            return result
        except TimeoutError:
            self.log_warning(
                kwargs.get("fl_ctx"),
                f"launcher method ({method_name}) execution timeout: exceeds {self._launch_timeout} seconds",
            )
            return LAUNCHER_EXCEPTION
        except Exception as e:
            self.log_warning(
                kwargs.get("fl_ctx"),
                f"launcher method ({method_name}) execution failed: {secure_format_exception(e)}",
            )
            return LAUNCHER_EXCEPTION

    def _wait_external_setup(self, task_name: str, fl_ctx: FLContext, abort_signal: Signal):
        start_time = time.time()
        while True:
            if self._external_pre_init_timeout and time.time() - start_time >= self._external_pre_init_timeout:
                self.log_error(
                    fl_ctx,
                    f"External process has not called flare.init within timeout: {self._external_pre_init_timeout}",
                )
                return False

            if abort_signal.triggered:
                self.log_info(fl_ctx, "External execution has not called flare.init but abort signal is triggered.")
                return False

            if self.peer_is_up_or_dead():
                return True

            run_status = self.launcher.check_run_status(task_name, fl_ctx)
            if run_status != LauncherRunStatus.RUNNING:
                self.log_info(
                    fl_ctx, f"External process has not called flare.init and run status becomes {run_status}."
                )
                return False

            time.sleep(0.1)

    def _finalize_external_execution(
        self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal
    ) -> bool:
        if self._job_end:
            ask_peer_end_success = self.ask_peer_to_end(fl_ctx)
            if not ask_peer_end_success:
                return False

        check_run_status = self._execute_launcher_method_in_thread_executor(
            method_name="check_run_status",
            task_name=task_name,
            fl_ctx=fl_ctx,
        )
        if not self._received_result.is_set() and check_run_status != LauncherRunStatus.COMPLETE_SUCCESS:
            self.log_warning(fl_ctx, f"Try to stop task ({task_name}) when launcher run status is {check_run_status}")

        self.log_info(fl_ctx, f"Calling stop task ({task_name}).")
        stop_task_success = self._execute_launcher_method_in_thread_executor(
            method_name="stop_task", task_name=task_name, fl_ctx=fl_ctx, abort_signal=abort_signal
        )

        if not stop_task_success or stop_task_success == LAUNCHER_EXCEPTION:
            return False

        self.log_info(fl_ctx, f"External execution for task ({task_name}) is finished.")
        return True

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
            time.sleep(self._monitor_interval)

            with self._lock:
                # job end
                if self._job_end:
                    break

                if self._abort_signal is not None and self._abort_signal.triggered:
                    self._job_end = True
                    break

                # result has been received
                if self._received_result.is_set():
                    self._clear_state()
                    continue

                if self.launcher is None:
                    break

                task_name = self._current_task
                run_status = self._execute_launcher_method_in_thread_executor(
                    method_name="check_run_status",
                    task_name=task_name,
                    fl_ctx=fl_ctx,
                )
                if run_status == LAUNCHER_EXCEPTION:
                    msg = "launcher check_run_status failed"
                    self.log_error(fl_ctx, msg)
                    self._abort_signal.trigger(msg)
                    continue

                elif run_status == LauncherRunStatus.NOT_RUNNING:
                    # pause pipe handler because external process is not running
                    self.pause_pipe_handler()
                    continue

                elif run_status == LauncherRunStatus.RUNNING:
                    # resume pipe handler when external process is running
                    self.resume_pipe_handler()
                    continue

                elif (
                    run_status == LauncherRunStatus.COMPLETE_FAILED or run_status == LauncherRunStatus.COMPLETE_SUCCESS
                ):
                    # pause pipe handler because external process is completed
                    self.pause_pipe_handler()
                    if not self._launcher_finish:
                        self._launcher_finish_time = time.time()
                        self._launcher_finish = True
                        self.log_info(
                            fl_ctx,
                            f"launcher completed {task_name} with status {run_status} at time {self._launcher_finish_time}",
                        )

                    if run_status == LauncherRunStatus.COMPLETE_FAILED:
                        msg = f"Launcher failed with at time {self._launcher_finish_time} "
                        self._abort_signal.trigger(msg)
                        break

                if not self._launcher_finish:
                    continue

            # LauncherExecutor need to wait additional time after the Launcher finishes
            # because it will take some time to communicate the result
            # (from external process sends and LauncherExecutor receives this last result)
            if not self._received_result.wait(self._last_result_transfer_timeout):
                msg = (
                    f"Launcher already exited with status {run_status} at time {self._launcher_finish_time} "
                    f"but LauncherExecutor does not receive the last result within {self._last_result_transfer_timeout} seconds."
                )
                self.log_error(
                    fl_ctx,
                    msg,
                )
                self._abort_signal.trigger(msg)

    def _clear_state(self):
        self._launcher_finish_time = None
        self._launcher_finish = False
        self._received_result.clear()
