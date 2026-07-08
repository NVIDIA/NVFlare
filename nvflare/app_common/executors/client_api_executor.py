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

"""The public Client API executor, configured by execution mode.

Design: docs/design/client_api_execution_modes.md ("What We Propose", "Overview",
"Execution Modes", "Configuration Surface"). This module path is normative - job configs
reference ``nvflare.app_common.executors.client_api_executor.ClientAPIExecutor``.

This is the interface-freeze skeleton (plan: EX-2). The constructor surface below is frozen;
the mode backends land in follow-up PRs (in_process, external_process, attach).

Divergence from the design's V1 "Configuration Surface" list
------------------------------------------------------------
The design's V1 arg list is a subset; the frozen surface below adds the following load-bearing
args that the mode backends require and that both legacy executors
(InProcessClientAPIExecutor / ClientAPILauncherExecutor) already expose. They are recorded here
so the design's Configuration Surface can be synced separately:

- ``task_script_path`` / ``task_script_args`` - in_process script entry point (the in_process
  backend runs a user script via TaskScriptRunner; ``command`` names the external_process
  trainer, ``task_script_path`` names the in_process one).
- ``train_task_name`` / ``evaluate_task_name`` / ``submit_model_task_name`` /
  ``train_with_evaluation`` - power the rank-contract APIs flare.is_train()/is_evaluate()/
  is_submit_model().
- ``memory_gc_rounds`` / ``cuda_empty_cache`` - periodic GC / CUDA cache management

Deliberately excluded (per FLARE-2698): ``params_exchange_format`` / ``params_transfer_type`` /
``server_expected_format`` / ``from_nvflare_converter_id`` / ``to_nvflare_converter_id``. Param
conversion moves from executor-owned ParamsConverters to send/receive filters at the client
edge (the intermediate layers pass through), so these are not frozen into this surface. The
transfer type (FULL/DIFF) remains a Client API concern (model_registry) and is decided
separately from the converter removal.
"""

from typing import Callable, Dict, Optional

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
from nvflare.security.logging import secure_format_exception, secure_format_traceback


class ExecutionMode:
    """Valid values for ClientAPIExecutor's execution_mode (see design "Overview")."""

    IN_PROCESS = "in_process"
    EXTERNAL_PROCESS = "external_process"
    ATTACH = "attach"


ALL_EXECUTION_MODES = (ExecutionMode.IN_PROCESS, ExecutionMode.EXTERNAL_PROCESS, ExecutionMode.ATTACH)

# Federation-scoped analytics event, matching MetricRelay's ex-process default
# (job_config/script_runner.py) and flower_job.py: "fed.analytix_log_stats".
FED_ANALYTIC_EVENT_TYPE = FED_EVENT_PREFIX + ANALYTIC_EVENT_TYPE

# Frozen defaults for mode-specific knobs (design "Configuration Surface"). Kept as named
# constants so the constructor default and the wrong-mode "explicitly set a non-default" checks
# below cannot drift apart.
_DEFAULT_LAUNCH_ONCE = True
_DEFAULT_STOP_GRACE_PERIOD = 30.0
_DEFAULT_HEARTBEAT_INTERVAL = 5.0
_DEFAULT_HEARTBEAT_TIMEOUT = 30.0


class ClientAPIExecutor(Executor):
    """One executor for all Client API execution modes.

    The trainer-facing Client API (flare.init/receive/send/log) is unchanged; this executor
    replaces the Pipe/launcher integration stack. It delegates to an internal mode-specific
    backend (ClientAPIBackendSpec) resolved from ``execution_mode`` at START_RUN:

    - ``in_process``: trainer runs inside the CJ process over DataBus.
    - ``external_process``: NVFlare launches and owns the trainer process tree; control over Cell.
    - ``attach``: an externally started/owned trainer attaches over Cell.
    """

    def __init__(
        self,
        execution_mode: str,
        command: Optional[str] = None,
        task_script_path: Optional[str] = None,
        task_script_args: str = "",
        launch_once: bool = _DEFAULT_LAUNCH_ONCE,
        launch_timeout: Optional[float] = None,
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
        memory_gc_rounds: int = 0,
        cuda_empty_cache: bool = False,
        attach_timeout: Optional[float] = None,
        allow_reconnect: bool = False,
    ):
        """Initializes the ClientAPIExecutor.

        This constructor surface is frozen (design: "Configuration Surface"); parameter renames
        break the surface-freeze test and downstream backend PRs. See the module docstring for the
        args added beyond the design's V1 Configuration Surface list.

        Args:
            execution_mode (str): One of "in_process", "external_process", or "attach". Required.
            command (Optional[str]): The trainer launch command, e.g. "python custom/train.py",
                "torchrun ...". Required for (and only valid in) "external_process" mode. An
                empty/whitespace-only string is treated as unset.
            task_script_path (Optional[str]): in_process only. Path to the user training script the
                in_process backend runs via TaskScriptRunner. An empty/whitespace-only string is
                treated as unset. (The in_process backend validates presence and ".py" suffix.)
            task_script_args (str): in_process only. Arguments appended to task_script_path.
            launch_once (bool): external_process only. Launch the trainer once per job (default)
                vs once per task.
            launch_timeout (Optional[float]): external_process only. Bound for the launched
                trainer to complete its HELLO/session setup (this replaces the legacy
                external_pre_init_timeout). None means no timeout.
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

        # Normalize an empty/whitespace command or task_script_path to None up front so "" means
        # "unset" uniformly in every mode. Previously external_process used `if not command` while
        # the other modes used `command is not None`, so command="" was rejected with a misleading
        # "only valid for external_process" message in in_process/attach instead of treated as unset.
        command = self._normalize_optional_str(command)
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

        # --- external_process-only lifecycle knobs (reject only when explicitly set away from the
        # frozen default in a mode that ignores them) ---
        if not is_external:
            if launch_once != _DEFAULT_LAUNCH_ONCE:
                raise self._wrong_mode_error("launch_once", launch_once, "'external_process'", execution_mode)
            if launch_timeout is not None:
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

        # --- attach-only knobs ---
        if not is_attach:
            if attach_timeout is not None:
                raise self._wrong_mode_error("attach_timeout", attach_timeout, "'attach'", execution_mode)
            # Only reject a truthy allow_reconnect: allow_reconnect=False (the frozen default) is
            # indistinguishable from "not set". Using `if allow_reconnect` (not `is not False`)
            # also stops misfires on falsy-but-not-False values (None, 0, numpy.bool_(False)).
            if allow_reconnect:
                raise self._wrong_mode_error("allow_reconnect", allow_reconnect, "'attach'", execution_mode)

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
        self._memory_gc_rounds = memory_gc_rounds
        self._cuda_empty_cache = cuda_empty_cache
        self._attach_timeout = attach_timeout
        self._allow_reconnect = bool(allow_reconnect)

        self._backend: Optional[ClientAPIBackendSpec] = None

        # Analytics-event ownership (design: "Configuration Surface" - "the executor's Cell
        # backend converts [LOG messages] into fed.analytix_log_stats analytics events").
        # False (default): fire the local un-prefixed ANALYTIC_EVENT_TYPE and rely on a
        # ConvertToFedEvent widget (added by BaseFedJob) to re-fire it as
        # "fed.analytix_log_stats" - today's in-process executor behavior.
        # True: fire a federation-scoped event directly (today's MetricRelay behavior for
        # ex-process). Cell backends (EP-4/AT-2) may select this path in initialize().
        self._analytics_fire_fed_event: bool = False

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            super().handle_event(event_type, fl_ctx)
            try:
                self._backend = self._create_backend()
                self._backend.initialize(self._build_backend_context(), fl_ctx)
            except Exception as e:
                # initialize() is contracted to self-unwind its partial setup on failure, so the
                # executor does NOT call finalize() on a half-initialized backend; it just drops
                # the reference and panics so the job fails cleanly.
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
            if self._backend is not None:
                try:
                    self._backend.handle_event(event_type, fl_ctx)
                except Exception:
                    self.log_error(fl_ctx, secure_format_traceback(), fire_event=False)
            super().handle_event(event_type, fl_ctx)

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        backend = self._backend
        if backend is None:
            # START_RUN either never happened or backend initialization failed (and the executor
            # already panicked). Reply with an error instead of waiting on a backend that will
            # never be ready.
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
        """Selects whether trainer LOG data is fired as a federation-scoped analytics event.

        Cell backends call this during initialization when no ConvertToFedEvent widget is
        configured. The default remains the local analytics path used by the in-process backend.

        Args:
            enabled: True to fire federation-scoped events directly; False to fire local events.
        """
        self._analytics_fire_fed_event = bool(enabled)

    def fire_log_analytics(self, fl_ctx: FLContext, dxo: DXO) -> None:
        """Converts trainer LOG data into an analytics event. Executor-owned surface.

        Backends call this for every LOG control message (flare.log from the trainer),
        regardless of execution mode - this replaces MetricRelay (ex-process) and the
        in-process executor's log callback as the single analytics-event ownership point.

        The fire path is mode-selectable via ``set_analytics_fire_fed_event()``:

        - False (default): fires the local, un-prefixed ANALYTIC_EVENT_TYPE
          ("analytix_log_stats") and relies on the ConvertToFedEvent widget (added by
          BaseFedJob) to forward it to the server as "fed.analytix_log_stats".
        - True: fires a federation-scoped event directly to the server, matching what
          MetricRelay does today for ex-process metrics. Cell backends may select this path during
          initialize() by calling ``set_analytics_fire_fed_event(True)`` when no ConvertToFedEvent
          widget is configured.

        The two paths must land on the same server-side event name. ConvertToFedEvent prefixes
        the local event with "fed.", so the fed path must fire the already-prefixed
        FED_ANALYTIC_EVENT_TYPE ("fed.analytix_log_stats"); firing the un-prefixed name
        federation-scoped would miss every consumer listening on "fed.analytix_log_stats"
        (MetricRelay in job_config/script_runner.py, flower_job.py).

        Args:
            fl_ctx: an FLContext to fire the event with.
            dxo: the analytics data (e.g. from create_analytic_dxo) carried by the event.
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
        """Treats an empty/whitespace-only string as unset (None)."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @staticmethod
    def _wrong_mode_error(arg_name: str, value, valid_modes: str, execution_mode: str) -> ValueError:
        """Builds the consistent 'arg only valid for <mode(s)>' rejection error."""
        return ValueError(
            f"{arg_name} is only valid for execution_mode {valid_modes}, "
            f"but got {arg_name}={value!r} with execution_mode '{execution_mode}'"
        )

    def _build_backend_context(self) -> ClientAPIBackendContext:
        """Builds the frozen config snapshot handed to the backend at initialize()."""
        return ClientAPIBackendContext(
            executor=self,
            execution_mode=self._execution_mode,
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
            memory_gc_rounds=self._memory_gc_rounds,
            cuda_empty_cache=self._cuda_empty_cache,
            attach_timeout=self._attach_timeout,
            allow_reconnect=self._allow_reconnect,
        )

    def _backend_registry(self) -> Dict[str, Callable[[], ClientAPIBackendSpec]]:
        """Internal registry mapping each execution_mode to its backend factory.

        Backend PRs replace the corresponding factory below with one that returns a real
        ClientAPIBackendSpec; the registry keys are the frozen mode names.
        """
        return {
            ExecutionMode.IN_PROCESS: self._create_in_process_backend,
            ExecutionMode.EXTERNAL_PROCESS: self._create_external_process_backend,
            ExecutionMode.ATTACH: self._create_attach_backend,
        }

    def _create_backend(self) -> ClientAPIBackendSpec:
        registry = self._backend_registry()
        factory = registry.get(self._execution_mode)
        if factory is None:
            # Unreachable via the public constructor (execution_mode is validated there);
            # guards subclasses that override _backend_registry().
            raise ValueError(
                f"no backend factory registered for execution_mode '{self._execution_mode}': "
                f"registered modes are {list(registry.keys())}"
            )
        return factory()

    def _create_in_process_backend(self) -> ClientAPIBackendSpec:
        # Deferred import: the backend pulls in DataBus/TaskScriptRunner machinery that the
        # other modes never need; the skeleton stays import-light.
        from nvflare.app_common.executors.client_api.in_process_backend import InProcessBackend

        return InProcessBackend()

    def _create_external_process_backend(self) -> ClientAPIBackendSpec:
        # Implemented in a follow-up PR (plan: EP-4).
        raise NotImplementedError("external_process execution mode is not yet implemented in this release")

    def _create_attach_backend(self) -> ClientAPIBackendSpec:
        # Implemented in a follow-up PR (plan: AT-2).
        raise NotImplementedError("attach execution mode is not yet implemented in this release")
