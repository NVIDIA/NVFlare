# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Client API executor with mode-specific trainer backends.

``in_process`` and ``external_process`` are supported; ``attach`` is reserved. Parameter
conversion and FULL/DIFF state are handled by the trainer-side Client API.
"""

import math
from typing import Optional, Union

from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE
from nvflare.apis.dxo import DXO
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import UnsafeJobError
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.utils.analytix_utils import send_analytic_dxo
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.client_api.backend_spec import ClientAPIBackendContext, ClientAPIBackendSpec
from nvflare.app_common.widgets.convert_to_fed_event import FED_EVENT_PREFIX
from nvflare.client.config import ExchangeFormat, TransferType, normalize_exchange_format, validate_format_pair
from nvflare.security.logging import secure_format_exception, secure_format_traceback


class ExecutionMode:
    """Valid values for ClientAPIExecutor's execution_mode (see design "Overview")."""

    IN_PROCESS = "in_process"
    EXTERNAL_PROCESS = "external_process"
    ATTACH = "attach"


ALL_EXECUTION_MODES = (ExecutionMode.IN_PROCESS, ExecutionMode.EXTERNAL_PROCESS, ExecutionMode.ATTACH)

FED_ANALYTIC_EVENT_TYPE = FED_EVENT_PREFIX + ANALYTIC_EVENT_TYPE

# Shared by constructor defaults and wrong-mode override checks.
_DEFAULT_LAUNCH_ONCE = True
_DEFAULT_LAUNCH_TIMEOUT = 300.0
_DEFAULT_STOP_GRACE_PERIOD = 30.0
_DEFAULT_HEARTBEAT_INTERVAL = 5.0
_DEFAULT_HEARTBEAT_TIMEOUT = 30.0


class ClientAPIExecutor(Executor):
    """Delegates Client API task execution to the configured backend."""

    def __init__(
        self,
        execution_mode: str,
        command: Optional[Union[str, list[str]]] = None,
        task_script_path: Optional[str] = None,
        task_script_args: str = "",
        launch_once: bool = _DEFAULT_LAUNCH_ONCE,
        launch_timeout: Optional[float] = _DEFAULT_LAUNCH_TIMEOUT,
        shutdown_timeout: Optional[float] = None,
        stop_grace_period: float = _DEFAULT_STOP_GRACE_PERIOD,
        heartbeat_interval: float = _DEFAULT_HEARTBEAT_INTERVAL,
        heartbeat_timeout: float = _DEFAULT_HEARTBEAT_TIMEOUT,
        task_wait_timeout: Optional[float] = None,
        result_wait_timeout: Optional[float] = None,
        train_task_name: str = AppConstants.TASK_TRAIN,
        evaluate_task_name: str = AppConstants.TASK_VALIDATION,
        submit_model_task_name: str = AppConstants.TASK_SUBMIT_MODEL,
        train_with_evaluation: bool = False,
        params_exchange_format: ExchangeFormat = ExchangeFormat.RAW,
        server_expected_format: ExchangeFormat = ExchangeFormat.NUMPY,
        params_transfer_type: TransferType = TransferType.FULL,
        memory_gc_rounds: int = 0,
        cuda_empty_cache: bool = False,
        attach_timeout: Optional[float] = None,
        allow_reconnect: bool = False,
    ):
        """Initializes the ClientAPIExecutor.

        Parameter names are part of the public job-config surface: renames are breaking
        changes (guarded by the surface-freeze test).

        Args:
            execution_mode (str): One of "in_process", "external_process", or "attach". Required.
            command (Optional[Union[str, list[str]]]): The trainer launch command, either as a
                command string (e.g. "python custom/train.py") or shell-free argv. Required for
                (and only valid in) "external_process" mode. An empty/whitespace-only string or
                empty argv is treated as unset. Use argv when values must retain exact boundaries
                across platforms.
            task_script_path (Optional[str]): in_process only. Path to the user training script the
                in_process backend runs via TaskScriptRunner. An empty/whitespace-only string is
                treated as unset. (The in_process backend validates presence and ".py" suffix.)
            task_script_args (str): in_process only. Arguments appended to task_script_path.
            launch_once (bool): external_process only. Launch the trainer once per job (default)
                vs once per task.
            launch_timeout (Optional[float]): external_process only. Bound for the launched
                trainer to complete its HELLO/session setup (this replaces the legacy
                external_pre_init_timeout). Defaults to 300 seconds for compatibility with
                ClientAPILauncherExecutor; an explicit None means no timeout.
            shutdown_timeout (Optional[float]): external_process only. How long to wait for the
                trainer to exit naturally after an orderly SHUTDOWN before starting forced
                process-tree termination. None means the backend default.
            stop_grace_period (float): external_process only. Grace period between SIGTERM and
                SIGKILL when terminating the trainer process group (design: "Process-tree
                termination").
            heartbeat_interval (float): out-of-process only (external_process/attach). Interval
                (seconds) for session heartbeats.
            heartbeat_timeout (float): out-of-process only (external_process/attach). Session lease
                timeout (seconds) on missed heartbeats. An in-flight payload transfer keeps the
                lease alive (design: "Heartbeat and Liveness").
            task_wait_timeout (Optional[float]): Bound for the trainer to accept a delivered
                task. None means no timeout.
            result_wait_timeout (Optional[float]): Control-side bound for retrieving the task
                result. Payload transfer completion is governed by the shared transfer layer,
                not by this value. None means no timeout.
            train_task_name (str): Task name treated as "train" by flare.is_train() (rank
                contract). Defaults to AppConstants.TASK_TRAIN.
            evaluate_task_name (str): Task name treated as "evaluate" by flare.is_evaluate().
                Defaults to AppConstants.TASK_VALIDATION.
            submit_model_task_name (str): Task name treated as "submit_model" by
                flare.is_submit_model(). Defaults to AppConstants.TASK_SUBMIT_MODEL.
            train_with_evaluation (bool): Whether the trainer also returns evaluation metrics with
                the trained model.
            params_exchange_format (ExchangeFormat): Framework-native parameter representation
                exposed by ``flare.receive()`` and accepted by ``flare.send()``. The declaration
                is transported to the trainer in ``TASK_EXCHANGE``; the executor does not perform
                conversion. ``RAW`` explicitly disables representation adaptation.
            server_expected_format (ExchangeFormat): Parameter representation expected by the
                server. The declaration is transported to the trainer in ``TASK_EXCHANGE``.
            params_transfer_type (TransferType): Whether training results contain full parameters
                or a difference from the received parameters. This is applied by the trainer-side
                model state and is independent of framework representation conversion.
            memory_gc_rounds (int): Force a GC cycle every N rounds (0 disables).
            cuda_empty_cache (bool): Whether to also empty the CUDA cache during memory cleanup.
            attach_timeout (Optional[float]): attach only. Bound for the externally started
                trainer to attach. None means no timeout.
            allow_reconnect (bool): attach only. Whether a trainer may re-attach to an existing
                session after a disconnect.
        """
        super().__init__()

        if execution_mode not in ALL_EXECUTION_MODES:
            raise ValueError(f"invalid execution_mode {execution_mode!r}: must be one of {list(ALL_EXECUTION_MODES)}")

        # Normalize empty entry points before mode-specific validation.
        command = self._normalize_command(command)
        task_script_path = self._normalize_optional_str(task_script_path)

        is_in_process = execution_mode == ExecutionMode.IN_PROCESS
        is_external = execution_mode == ExecutionMode.EXTERNAL_PROCESS
        is_attach = execution_mode == ExecutionMode.ATTACH

        # --- external_process command / in_process script entry point ---
        if is_external:
            if not command:
                raise ValueError(
                    "execution_mode 'external_process' requires a non-empty command "
                    "(e.g. command='python custom/train.py')"
                )
        elif command is not None:
            raise self._wrong_mode_error("command", command, "'external_process'", execution_mode)

        if not is_in_process:
            if task_script_path is not None:
                raise self._wrong_mode_error("task_script_path", task_script_path, "'in_process'", execution_mode)
            if task_script_args:
                raise self._wrong_mode_error("task_script_args", task_script_args, "'in_process'", execution_mode)

        # Reject external-process settings changed from their defaults in other modes.
        if not is_external:
            if launch_once != _DEFAULT_LAUNCH_ONCE:
                raise self._wrong_mode_error("launch_once", launch_once, "'external_process'", execution_mode)
            if launch_timeout != _DEFAULT_LAUNCH_TIMEOUT:
                raise self._wrong_mode_error("launch_timeout", launch_timeout, "'external_process'", execution_mode)
            if shutdown_timeout is not None:
                raise self._wrong_mode_error("shutdown_timeout", shutdown_timeout, "'external_process'", execution_mode)
            if stop_grace_period != _DEFAULT_STOP_GRACE_PERIOD:
                raise self._wrong_mode_error(
                    "stop_grace_period", stop_grace_period, "'external_process'", execution_mode
                )

        # --- heartbeat knobs are out-of-process only (there is no session heartbeat in_process) ---
        if is_in_process:
            if heartbeat_interval != _DEFAULT_HEARTBEAT_INTERVAL:
                raise self._wrong_mode_error(
                    "heartbeat_interval", heartbeat_interval, "'external_process' or 'attach'", execution_mode
                )
            if heartbeat_timeout != _DEFAULT_HEARTBEAT_TIMEOUT:
                raise self._wrong_mode_error(
                    "heartbeat_timeout", heartbeat_timeout, "'external_process' or 'attach'", execution_mode
                )

        # heartbeat_timeout=0 disables heartbeat timeout checking; otherwise the
        # heartbeat interval must be strictly smaller than the timeout.
        for name, value in (
            ("heartbeat_interval", heartbeat_interval),
            ("heartbeat_timeout", heartbeat_timeout),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, but got {value!r}")
        if heartbeat_interval <= 0:
            raise ValueError(f"heartbeat_interval must > 0, but got {heartbeat_interval}")
        if heartbeat_timeout < 0:
            raise ValueError(f"heartbeat_timeout must >= 0, but got {heartbeat_timeout}")
        if 0 < heartbeat_timeout <= heartbeat_interval:
            raise ValueError(
                f"heartbeat_interval {heartbeat_interval} must be less than heartbeat_timeout {heartbeat_timeout}"
            )

        # --- attach-only knobs ---
        if not is_attach:
            if attach_timeout is not None:
                raise self._wrong_mode_error("attach_timeout", attach_timeout, "'attach'", execution_mode)
            # Treat False and other falsy values as the unset default.
            if allow_reconnect:
                raise self._wrong_mode_error("allow_reconnect", allow_reconnect, "'attach'", execution_mode)

        # Reject invalid bounds before they reach threading/subprocess primitives. In
        # particular, a negative Lock.acquire(timeout=...) can mean "wait forever",
        # which would violate the executor's bounded-shutdown contract.
        for name, value in (
            ("launch_timeout", launch_timeout),
            ("shutdown_timeout", shutdown_timeout),
            ("stop_grace_period", stop_grace_period),
            ("task_wait_timeout", task_wait_timeout),
            ("result_wait_timeout", result_wait_timeout),
            ("attach_timeout", attach_timeout),
        ):
            if value is not None and (
                not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0
            ):
                raise ValueError(f"{name} must be a finite number >= 0 or None, but got {value!r}")
        if not isinstance(memory_gc_rounds, int) or isinstance(memory_gc_rounds, bool) or memory_gc_rounds < 0:
            raise ValueError(f"memory_gc_rounds must be an integer >= 0, but got {memory_gc_rounds!r}")

        self._execution_mode = execution_mode
        self._command = command
        self._task_script_path = task_script_path
        self._task_script_args = task_script_args
        self._launch_once = launch_once
        self._launch_timeout = launch_timeout
        self._shutdown_timeout = shutdown_timeout
        self._stop_grace_period = stop_grace_period
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout
        self._task_wait_timeout = task_wait_timeout
        self._result_wait_timeout = result_wait_timeout
        self._train_task_name = train_task_name
        self._evaluate_task_name = evaluate_task_name
        self._submit_model_task_name = submit_model_task_name
        self._train_with_evaluation = train_with_evaluation
        self._params_exchange_format = normalize_exchange_format(params_exchange_format, "params_exchange_format")
        self._server_expected_format = normalize_exchange_format(server_expected_format, "server_expected_format")
        validate_format_pair(self._params_exchange_format, self._server_expected_format)
        try:
            self._params_transfer_type = TransferType(params_transfer_type)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"invalid params_transfer_type {params_transfer_type!r}: must be one of {list(TransferType)}"
            ) from e
        self._memory_gc_rounds = memory_gc_rounds
        self._cuda_empty_cache = cuda_empty_cache
        self._attach_timeout = attach_timeout
        self._allow_reconnect = bool(allow_reconnect)

        self._backend: Optional[ClientAPIBackendSpec] = None

        # False uses the local event path; Cell backends enable direct federation events as needed.
        self._analytics_fire_fed_event: bool = False

    @property
    def execution_mode(self) -> str:
        """Read-only view of the configured mode, for validation and diagnostics."""
        return self._execution_mode

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            super().handle_event(event_type, fl_ctx)
            try:
                self._backend = self._create_backend()
                self._backend.initialize(self._build_backend_context(), fl_ctx)
            except Exception as e:
                # initialize() owns rollback; finalizing a partial backend is unsafe.
                self._backend = None
                self.log_error(fl_ctx, secure_format_traceback(), fire_event=False)
                self.system_panic(
                    f"ClientAPIExecutor cannot start: backend for execution_mode "
                    f"'{self._execution_mode}' failed to initialize: {secure_format_exception(e)}",
                    fl_ctx,
                )
        elif event_type == EventType.END_RUN:
            backend = self._backend
            self._backend = None
            if backend is not None:
                try:
                    backend.finalize(fl_ctx)
                except Exception:
                    self.log_error(fl_ctx, secure_format_traceback(), fire_event=False)
            super().handle_event(event_type, fl_ctx)
        else:
            super().handle_event(event_type, fl_ctx)

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        backend = self._backend
        if backend is None:
            self.log_error(
                fl_ctx,
                f"no Client API backend available for execution_mode '{self._execution_mode}' - "
                f"backend initialization failed or START_RUN was not handled",
            )
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        try:
            result = backend.execute(task_name, shareable, fl_ctx, abort_signal)
        except UnsafeJobError:
            # ClientRunner has dedicated handling for UnsafeJobError (client_runner.py maps it
            # to ReturnCode.UNSAFE_JOB and marks the job unsafe). Do NOT swallow it into a
            # generic EXECUTION_EXCEPTION reply here - let it propagate so that handling fires.
            # (UnsafeComponentError has no such special handling and is left to the generic
            # branch below, which is its existing behavior.)
            raise
        except Exception:
            self.log_error(fl_ctx, secure_format_traceback())
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        if not isinstance(result, Shareable):
            self.log_error(fl_ctx, f"bad result from backend: expected Shareable but got {type(result)}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        return result

    def set_analytics_fire_fed_event(self, enabled: bool) -> None:
        """Select direct federation events instead of the local ConvertToFedEvent path."""
        self._analytics_fire_fed_event = bool(enabled)

    def fire_log_analytics(self, fl_ctx: FLContext, dxo: DXO) -> None:
        """Emit trainer LOG data through the configured analytics path.

        The local path fires ``analytix_log_stats`` and ConvertToFedEvent prefixes it to
        ``fed.analytix_log_stats``. The direct path must fire that prefixed name itself or
        server-side consumers will miss the event.
        """
        if self._analytics_fire_fed_event:
            send_analytic_dxo(
                self,
                dxo=dxo,
                fl_ctx=fl_ctx,
                event_type=FED_ANALYTIC_EVENT_TYPE,
                fire_fed_event=True,
            )
        else:
            send_analytic_dxo(
                self,
                dxo=dxo,
                fl_ctx=fl_ctx,
                event_type=ANALYTIC_EVENT_TYPE,
                fire_fed_event=False,
            )

    @staticmethod
    def _normalize_optional_str(value: Optional[str]) -> Optional[str]:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @staticmethod
    def _normalize_command(command: Optional[Union[str, list[str]]]) -> Optional[Union[str, list[str]]]:
        if isinstance(command, str) or command is None:
            return ClientAPIExecutor._normalize_optional_str(command)
        if not isinstance(command, list):
            raise ValueError(f"command must be a string, list of strings, or None, but got {type(command).__name__}")
        if not command:
            return None
        if not all(isinstance(arg, str) for arg in command):
            raise ValueError("command argv must contain only strings")
        if not command[0].strip():
            raise ValueError("command argv must start with a non-empty executable")
        return list(command)

    @staticmethod
    def _wrong_mode_error(arg_name: str, value, valid_modes: str, execution_mode: str) -> ValueError:
        return ValueError(
            f"{arg_name} is only valid for execution_mode {valid_modes}, "
            f"but got {arg_name}={value!r} with execution_mode '{execution_mode}'"
        )

    def _build_backend_context(self) -> ClientAPIBackendContext:
        return ClientAPIBackendContext(
            executor=self,
            task_script_path=self._task_script_path,
            task_script_args=self._task_script_args,
            command=self._command,
            launch_once=self._launch_once,
            launch_timeout=self._launch_timeout,
            shutdown_timeout=self._shutdown_timeout,
            stop_grace_period=self._stop_grace_period,
            heartbeat_interval=self._heartbeat_interval,
            heartbeat_timeout=self._heartbeat_timeout,
            task_wait_timeout=self._task_wait_timeout,
            result_wait_timeout=self._result_wait_timeout,
            train_task_name=self._train_task_name,
            evaluate_task_name=self._evaluate_task_name,
            submit_model_task_name=self._submit_model_task_name,
            train_with_evaluation=self._train_with_evaluation,
            params_exchange_format=self._params_exchange_format,
            server_expected_format=self._server_expected_format,
            params_transfer_type=self._params_transfer_type,
            memory_gc_rounds=self._memory_gc_rounds,
            cuda_empty_cache=self._cuda_empty_cache,
        )

    def _create_backend(self) -> ClientAPIBackendSpec:
        if self._execution_mode == ExecutionMode.IN_PROCESS:
            return self._create_in_process_backend()
        if self._execution_mode == ExecutionMode.EXTERNAL_PROCESS:
            return self._create_external_process_backend()
        if self._execution_mode == ExecutionMode.ATTACH:
            raise NotImplementedError("attach execution mode is not yet implemented in this release")

        raise ValueError(f"unexpected execution_mode {self._execution_mode!r}")

    def _create_in_process_backend(self) -> ClientAPIBackendSpec:
        from nvflare.app_common.executors.client_api.in_process_backend import InProcessBackend

        return InProcessBackend()

    def _create_external_process_backend(self) -> ClientAPIBackendSpec:
        from nvflare.app_common.executors.client_api.external_process_backend import ExternalProcessBackend

        return ExternalProcessBackend()
