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

"""Tests for the ClientAPIExecutor interface-freeze skeleton (plan: EX-2).

Covers the constructor-validation matrix, the per-mode dispatch failure behavior
(NotImplementedError naming the follow-up PR + system_panic, no hang), the analytics-event
ownership hook, and the surface-freeze contract on the frozen constructor parameter list.
"""

import inspect
from unittest.mock import Mock

import pytest

from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE, AnalyticsDataType
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import EventScope, FLContextKey, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import UnsafeComponentError, UnsafeJobError
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.utils.analytix_utils import create_analytic_dxo
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.client_api.backend_spec import ClientAPIBackendContext, ClientAPIBackendSpec
from nvflare.app_common.executors.client_api_executor import (
    ALL_EXECUTION_MODES,
    FED_ANALYTIC_EVENT_TYPE,
    ClientAPIExecutor,
    ExecutionMode,
)
from nvflare.client.config import ExchangeFormat, TransferType

# The frozen V1 constructor surface (design: "Configuration Surface" in
# docs/design/client_api_execution_modes.md, plus the load-bearing args added beyond that list -
# documented in the ClientAPIExecutor module docstring). Renaming, removing, or reordering any of
# these is a breaking change to the interface freeze consumed by the mode backends.
FROZEN_CONSTRUCTOR_PARAMS = [
    "execution_mode",
    "command",
    "task_script_path",
    "task_script_args",
    "launch_once",
    "launch_timeout",
    "shutdown_timeout",
    "stop_grace_period",
    "heartbeat_interval",
    "heartbeat_timeout",
    "task_wait_timeout",
    "result_wait_timeout",
    "train_task_name",
    "evaluate_task_name",
    "submit_model_task_name",
    "train_with_evaluation",
    "params_exchange_format",
    "server_expected_format",
    "params_transfer_type",
    "memory_gc_rounds",
    "cuda_empty_cache",
    "attach_timeout",
    "allow_reconnect",
]

# Minimal valid constructor kwargs per mode.
MODE_KWARGS = {
    ExecutionMode.IN_PROCESS: {"execution_mode": "in_process"},
    ExecutionMode.EXTERNAL_PROCESS: {"execution_mode": "external_process", "command": "python custom/train.py"},
    ExecutionMode.ATTACH: {"execution_mode": "attach"},
}


def _make_fl_ctx(engine) -> FLContext:
    fl_ctx = FLContext()
    fl_ctx.put(key=ReservedKey.ENGINE, value=engine, private=True, sticky=False)
    return fl_ctx


def _make_recording_engine():
    """Returns (engine, fired) where fired collects (event_type, EVENT_DATA-at-fire-time)."""
    engine = Mock()
    fired = []

    def _record(event_type, fl_ctx):
        fired.append((event_type, fl_ctx.get_prop(FLContextKey.EVENT_DATA)))

    engine.fire_event.side_effect = _record
    return engine, fired


class _StubBackend(ClientAPIBackendSpec):
    """A minimal working backend to verify executor-to-backend plumbing."""

    def __init__(self):
        self.calls = []
        self.result = make_reply(ReturnCode.OK)
        self.context = None

    def initialize(self, context, fl_ctx):
        self.context = context
        self.calls.append("initialize")

    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        self.calls.append(("execute", task_name))
        return self.result

    def finalize(self, fl_ctx):
        self.calls.append("finalize")


class _StubbedInProcessExecutor(ClientAPIExecutor):
    """ClientAPIExecutor with a working in_process backend factory (as EX-3 will provide)."""

    def __init__(self, backend, **kwargs):
        super().__init__(**kwargs)
        self._stub_backend = backend

    def _create_in_process_backend(self):
        return self._stub_backend


class TestConstructorValidation:
    @pytest.mark.parametrize("kwargs", list(MODE_KWARGS.values()), ids=list(MODE_KWARGS.keys()))
    def test_valid_minimal_construction(self, kwargs):
        executor = ClientAPIExecutor(**kwargs)
        assert isinstance(executor, Executor)
        assert executor._execution_mode == kwargs["execution_mode"]

    def test_external_process_default_launch_timeout_is_bounded(self):
        executor = ClientAPIExecutor(execution_mode="external_process", command="python custom/train.py")

        assert executor._launch_timeout == 300.0
        assert executor._build_backend_context().launch_timeout == 300.0

    def test_valid_attach_with_attach_args(self):
        executor = ClientAPIExecutor(execution_mode="attach", attach_timeout=60.0, allow_reconnect=True)
        assert executor._attach_timeout == 60.0
        assert executor._allow_reconnect is True

    def test_valid_full_surface(self):
        executor = ClientAPIExecutor(
            execution_mode="external_process",
            command="torchrun --nproc_per_node=2 custom/train.py",
            launch_once=False,
            launch_timeout=120.0,
            shutdown_timeout=60.0,
            stop_grace_period=10.0,
            heartbeat_interval=2.0,
            heartbeat_timeout=20.0,
            task_wait_timeout=300.0,
            result_wait_timeout=600.0,
            train_task_name="my_train",
            evaluate_task_name="my_eval",
            submit_model_task_name="my_submit",
            train_with_evaluation=True,
            params_exchange_format=ExchangeFormat.PYTORCH,
            server_expected_format=ExchangeFormat.NUMPY,
            params_transfer_type=TransferType.DIFF,
            memory_gc_rounds=5,
            cuda_empty_cache=True,
        )
        assert executor._command == "torchrun --nproc_per_node=2 custom/train.py"
        assert executor._launch_once is False
        assert executor._train_task_name == "my_train"
        assert executor._evaluate_task_name == "my_eval"
        assert executor._submit_model_task_name == "my_submit"
        assert executor._train_with_evaluation is True
        assert executor._params_exchange_format == ExchangeFormat.PYTORCH
        assert executor._server_expected_format == ExchangeFormat.NUMPY
        assert executor._params_transfer_type == TransferType.DIFF
        assert executor._memory_gc_rounds == 5
        assert executor._cuda_empty_cache is True

    def test_task_name_and_memory_defaults(self):
        # Rank-contract task names and memory-management knobs are frozen with the legacy
        # executors' defaults, valid in every mode.
        executor = ClientAPIExecutor(execution_mode="in_process")
        assert executor._train_task_name == AppConstants.TASK_TRAIN
        assert executor._evaluate_task_name == AppConstants.TASK_VALIDATION
        assert executor._submit_model_task_name == AppConstants.TASK_SUBMIT_MODEL
        assert executor._train_with_evaluation is False
        assert executor._params_exchange_format == ExchangeFormat.RAW
        assert executor._server_expected_format == ExchangeFormat.NUMPY
        assert executor._params_transfer_type == TransferType.FULL
        assert executor._memory_gc_rounds == 0
        assert executor._cuda_empty_cache is False

    def test_invalid_or_unsupported_format_declaration_is_rejected(self):
        with pytest.raises(ValueError, match="invalid params_exchange_format"):
            ClientAPIExecutor(execution_mode="in_process", params_exchange_format="unknown")
        with pytest.raises(ValueError, match="unsupported parameter format conversion"):
            ClientAPIExecutor(
                execution_mode="in_process",
                params_exchange_format=ExchangeFormat.PYTORCH,
                server_expected_format=ExchangeFormat.KERAS_LAYER_WEIGHTS,
            )

    @pytest.mark.parametrize(
        "mode,expected",
        [
            (ExecutionMode.IN_PROCESS, False),
            (ExecutionMode.EXTERNAL_PROCESS, True),
            (ExecutionMode.ATTACH, False),
        ],
    )
    def test_only_external_process_supports_task_data_pass_through(self, mode, expected):
        executor = ClientAPIExecutor(**MODE_KWARGS[mode])
        assert executor.supports_task_data_pass_through() is expected

    def test_in_process_accepts_task_script(self):
        # in_process names its script via task_script_path/args; command names the external_process
        # trainer.
        executor = ClientAPIExecutor(
            execution_mode="in_process", task_script_path="custom/train.py", task_script_args="--epochs 3"
        )
        assert executor._task_script_path == "custom/train.py"
        assert executor._task_script_args == "--epochs 3"
        assert executor._command is None

    @pytest.mark.parametrize("bad_mode", ["IN_PROCESS", "In_Process", "subprocess", "launch", "", None, 1])
    def test_unknown_execution_mode_rejected(self, bad_mode):
        with pytest.raises(ValueError, match="invalid execution_mode"):
            ClientAPIExecutor(execution_mode=bad_mode)

    def test_unknown_execution_mode_message_names_valid_modes(self):
        with pytest.raises(ValueError) as exc_info:
            ClientAPIExecutor(execution_mode="bogus")
        for mode in ALL_EXECUTION_MODES:
            assert mode in str(exc_info.value)

    @pytest.mark.parametrize("mode", ["in_process", "attach"])
    def test_command_rejected_for_non_external_process_modes(self, mode):
        with pytest.raises(ValueError, match="command is only valid for execution_mode 'external_process'"):
            ClientAPIExecutor(execution_mode=mode, command="python custom/train.py")

    @pytest.mark.parametrize("command", [None, ""])
    def test_external_process_requires_command(self, command):
        with pytest.raises(ValueError, match="'external_process' requires a non-empty command"):
            ClientAPIExecutor(execution_mode="external_process", command=command)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"execution_mode": "in_process"},
            {"execution_mode": "external_process", "command": "python custom/train.py"},
        ],
        ids=["in_process", "external_process"],
    )
    def test_attach_timeout_rejected_for_non_attach_modes(self, kwargs):
        with pytest.raises(ValueError, match="attach_timeout is only valid for execution_mode 'attach'"):
            ClientAPIExecutor(attach_timeout=30.0, **kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"execution_mode": "in_process"},
            {"execution_mode": "external_process", "command": "python custom/train.py"},
        ],
        ids=["in_process", "external_process"],
    )
    def test_allow_reconnect_rejected_for_non_attach_modes(self, kwargs):
        with pytest.raises(ValueError, match="allow_reconnect is only valid for execution_mode 'attach'"):
            ClientAPIExecutor(allow_reconnect=True, **kwargs)

    @pytest.mark.parametrize("mode", ["in_process", "attach"])
    @pytest.mark.parametrize("command", ["", "   ", "\t"])
    def test_empty_command_treated_as_unset_for_non_external_modes(self, mode, command):
        # An empty/whitespace command is "unset", not a wrong-mode command, so it must not be
        # rejected with the misleading "only valid for external_process" message.
        executor = ClientAPIExecutor(execution_mode=mode, command=command)
        assert executor._command is None

    @pytest.mark.parametrize("command", ["", "   "])
    def test_external_process_rejects_empty_command(self, command):
        # The same normalization must keep external_process requiring a real command.
        with pytest.raises(ValueError, match="'external_process' requires a non-empty command"):
            ClientAPIExecutor(execution_mode="external_process", command=command)

    @pytest.mark.parametrize("mode", ["external_process", "attach"])
    def test_task_script_path_rejected_for_non_in_process_modes(self, mode):
        # task_script_path names the in_process script only.
        kwargs = dict(MODE_KWARGS[mode])
        with pytest.raises(ValueError, match="task_script_path is only valid for execution_mode 'in_process'"):
            ClientAPIExecutor(task_script_path="custom/train.py", **kwargs)

    @pytest.mark.parametrize("mode", ["external_process", "attach"])
    def test_task_script_args_rejected_for_non_in_process_modes(self, mode):
        kwargs = dict(MODE_KWARGS[mode])
        with pytest.raises(ValueError, match="task_script_args is only valid for execution_mode 'in_process'"):
            ClientAPIExecutor(task_script_args="--epochs 3", **kwargs)

    @pytest.mark.parametrize("mode", ["in_process", "attach"])
    def test_launch_once_non_default_rejected_for_non_external_modes(self, mode):
        # external_process-only knobs must be rejected (not silently ignored) when set to a
        # non-default value in a mode that ignores them.
        kwargs = dict(MODE_KWARGS[mode])
        with pytest.raises(ValueError, match="launch_once is only valid for execution_mode 'external_process'"):
            ClientAPIExecutor(launch_once=False, **kwargs)

    @pytest.mark.parametrize("mode", ["in_process", "attach"])
    def test_launch_timeout_rejected_for_non_external_modes(self, mode):
        kwargs = dict(MODE_KWARGS[mode])
        with pytest.raises(ValueError, match="launch_timeout is only valid for execution_mode 'external_process'"):
            ClientAPIExecutor(launch_timeout=120.0, **kwargs)

    @pytest.mark.parametrize("mode", ["in_process", "attach"])
    def test_shutdown_timeout_rejected_for_non_external_modes(self, mode):
        kwargs = dict(MODE_KWARGS[mode])
        with pytest.raises(ValueError, match="shutdown_timeout is only valid for execution_mode 'external_process'"):
            ClientAPIExecutor(shutdown_timeout=60.0, **kwargs)

    @pytest.mark.parametrize("mode", ["in_process", "attach"])
    def test_stop_grace_period_non_default_rejected_for_non_external_modes(self, mode):
        kwargs = dict(MODE_KWARGS[mode])
        with pytest.raises(ValueError, match="stop_grace_period is only valid for execution_mode 'external_process'"):
            ClientAPIExecutor(stop_grace_period=10.0, **kwargs)

    @pytest.mark.parametrize("arg", ["heartbeat_interval", "heartbeat_timeout"])
    def test_heartbeat_non_default_rejected_for_in_process(self, arg):
        # There is no session heartbeat in_process, so a non-default heartbeat knob is dead there
        # and must be rejected (valid for external_process/attach).
        with pytest.raises(ValueError, match=f"{arg} is only valid for execution_mode 'external_process' or 'attach'"):
            ClientAPIExecutor(execution_mode="in_process", **{arg: 99.0})

    @pytest.mark.parametrize("mode", ["external_process", "attach"])
    def test_heartbeat_accepted_for_out_of_process_modes(self, mode):
        kwargs = dict(MODE_KWARGS[mode])
        executor = ClientAPIExecutor(heartbeat_interval=2.0, heartbeat_timeout=20.0, **kwargs)
        assert executor._heartbeat_interval == 2.0
        assert executor._heartbeat_timeout == 20.0

    @pytest.mark.parametrize(
        "heartbeat_interval,heartbeat_timeout,match",
        [
            (0.0, 30.0, "heartbeat_interval must > 0"),
            (5.0, -1.0, "heartbeat_timeout must >= 0"),
            (5.0, 5.0, "must be less than heartbeat_timeout"),
            (float("nan"), 30.0, "finite number"),
            (5.0, float("inf"), "finite number"),
        ],
    )
    def test_invalid_heartbeat_policy_rejected(self, heartbeat_interval, heartbeat_timeout, match):
        with pytest.raises(ValueError, match=match):
            ClientAPIExecutor(
                execution_mode="external_process",
                command="python custom/train.py",
                heartbeat_interval=heartbeat_interval,
                heartbeat_timeout=heartbeat_timeout,
            )

    def test_zero_heartbeat_timeout_disables_lease(self):
        executor = ClientAPIExecutor(
            execution_mode="external_process",
            command="python custom/train.py",
            heartbeat_interval=5.0,
            heartbeat_timeout=0.0,
        )
        assert executor._heartbeat_timeout == 0.0

    @pytest.mark.parametrize(
        "name,value",
        [
            ("launch_timeout", -1.0),
            ("shutdown_timeout", float("nan")),
            ("stop_grace_period", -0.1),
            ("task_wait_timeout", float("inf")),
            ("result_wait_timeout", -1.0),
        ],
    )
    def test_invalid_external_process_bounds_are_rejected(self, name, value):
        with pytest.raises(ValueError, match=rf"{name} must be a finite number >= 0"):
            ClientAPIExecutor(
                execution_mode="external_process",
                command="python custom/train.py",
                **{name: value},
            )

    @pytest.mark.parametrize("value", [-1, 1.5, True])
    def test_invalid_memory_gc_rounds_is_rejected(self, value):
        with pytest.raises(ValueError, match="memory_gc_rounds must be an integer >= 0"):
            ClientAPIExecutor(execution_mode="in_process", memory_gc_rounds=value)

    @pytest.mark.parametrize("mode", ["in_process", "attach"])
    def test_external_process_defaults_accepted_for_other_modes(self, mode):
        # The frozen defaults for external_process-only knobs are indistinguishable from "not set"
        # and must be accepted in every mode.
        kwargs = dict(MODE_KWARGS[mode])
        ClientAPIExecutor(launch_once=True, launch_timeout=300.0, stop_grace_period=30.0, **kwargs)

    @pytest.mark.parametrize("mode", ["in_process", "external_process"])
    def test_allow_reconnect_default_value_accepted_for_non_attach_modes(self, mode):
        # allow_reconnect=False equals the frozen default and is indistinguishable from
        # "not set", so it must not be rejected.
        kwargs = dict(MODE_KWARGS[mode])
        ClientAPIExecutor(allow_reconnect=False, **kwargs)

    @pytest.mark.parametrize("mode", ["in_process", "external_process"])
    @pytest.mark.parametrize("falsy", [None, 0])
    def test_allow_reconnect_falsy_values_accepted_for_non_attach_modes(self, mode, falsy):
        # The wrong-mode check uses `if allow_reconnect` (truthy), not `is not False`, so
        # falsy-but-not-False values (None, 0, numpy.bool_(False)) are treated as "not set"
        # instead of misfiring.
        kwargs = dict(MODE_KWARGS[mode])
        executor = ClientAPIExecutor(allow_reconnect=falsy, **kwargs)
        assert executor._allow_reconnect is False


class TestDispatch:
    """in_process and external_process now resolve real backends; attach remains
    skeleton-only: resolving its backend at START_RUN must fail the job cleanly
    (system_panic naming the mode), and execute() must reply with an error instead of
    hanging. in_process START_RUN without a valid task_script_path — and external_process
    START_RUN without a workspace/cell — fail the same clean way, so the all-modes panic
    tests below still hold."""

    NOT_IMPLEMENTED_MODES = [ExecutionMode.ATTACH]

    def test_in_process_factory_returns_real_backend(self):
        from nvflare.app_common.executors.client_api.in_process_backend import InProcessBackend

        executor = ClientAPIExecutor(**MODE_KWARGS[ExecutionMode.IN_PROCESS])
        assert isinstance(executor._create_backend(), InProcessBackend)

    def test_external_process_factory_returns_real_backend(self):
        from nvflare.app_common.executors.client_api.external_process_backend import ExternalProcessBackend

        executor = ClientAPIExecutor(**MODE_KWARGS[ExecutionMode.EXTERNAL_PROCESS])
        assert isinstance(executor._create_backend(), ExternalProcessBackend)

    @pytest.mark.parametrize("mode", NOT_IMPLEMENTED_MODES)
    def test_backend_factory_raises_not_implemented(self, mode):
        # The user-facing message must not carry an internal plan id (EX-3/EP-4/AT-2); it names the
        # mode and says "not yet implemented".
        executor = ClientAPIExecutor(**MODE_KWARGS[mode])
        with pytest.raises(NotImplementedError, match="not yet implemented") as exc_info:
            executor._create_backend()
        message = str(exc_info.value)
        assert mode in message
        for plan_id in ("EX-3", "EP-4", "AT-2"):
            assert plan_id not in message

    @pytest.mark.parametrize("mode", list(ALL_EXECUTION_MODES))
    def test_start_run_panics_naming_the_mode(self, mode):
        # The panic reason names the mode directly (not via secure_format_exception, so it is robust
        # whether or not NVFLARE_SECURE_LOGGING is set) and carries no plan id.
        # For in_process the failure is now the missing task_script_path (backend initialize
        # raises), not a NotImplementedError factory - the clean-failure contract is the same.
        executor = ClientAPIExecutor(**MODE_KWARGS[mode])
        engine, fired = _make_recording_engine()
        fl_ctx = _make_fl_ctx(engine)

        executor.handle_event(EventType.START_RUN, fl_ctx)

        panics = [(event_type, data) for event_type, data in fired if event_type == EventType.FATAL_SYSTEM_ERROR]
        assert len(panics) == 1, f"expected exactly one system_panic, got events: {fired}"
        reason = panics[0][1]
        assert mode in reason
        for plan_id in ("EX-3", "EP-4", "AT-2"):
            assert plan_id not in reason

    @pytest.mark.parametrize("mode", list(ALL_EXECUTION_MODES))
    def test_execute_without_backend_returns_error_reply_not_hang(self, mode):
        executor = ClientAPIExecutor(**MODE_KWARGS[mode])
        engine, _ = _make_recording_engine()
        fl_ctx = _make_fl_ctx(engine)
        executor.handle_event(EventType.START_RUN, fl_ctx)  # backend init fails -> panic

        reply = executor.execute("train", Shareable(), fl_ctx, Signal())

        assert isinstance(reply, Shareable)
        assert reply.get_return_code() == ReturnCode.EXECUTION_EXCEPTION

    def test_end_run_without_backend_does_not_raise(self):
        executor = ClientAPIExecutor(execution_mode="in_process")
        engine, fired = _make_recording_engine()
        fl_ctx = _make_fl_ctx(engine)
        executor.handle_event(EventType.END_RUN, fl_ctx)
        assert not any(event_type == EventType.FATAL_SYSTEM_ERROR for event_type, _ in fired)


class TestBackendPlumbing:
    """With a working backend registered (as EX-3/EP-4/AT-2 will do), the executor drives the
    ClientAPIBackendSpec lifecycle: initialize at START_RUN, execute per task, and finalize at
    END_RUN."""

    def _make_started_executor(self):
        backend = _StubBackend()
        executor = _StubbedInProcessExecutor(backend, execution_mode="in_process")
        engine, fired = _make_recording_engine()
        fl_ctx = _make_fl_ctx(engine)
        executor.handle_event(EventType.START_RUN, fl_ctx)
        return executor, backend, fl_ctx, fired

    def test_start_run_initializes_backend_without_panic(self):
        executor, backend, fl_ctx, fired = self._make_started_executor()
        assert backend.calls == ["initialize"]
        assert not any(event_type == EventType.FATAL_SYSTEM_ERROR for event_type, _ in fired)

    def test_execute_delegates_to_backend(self):
        executor, backend, fl_ctx, _ = self._make_started_executor()
        reply = executor.execute("train", Shareable(), fl_ctx, Signal())
        assert reply is backend.result
        assert ("execute", "train") in backend.calls

    def test_unrelated_events_are_not_relayed_to_backend(self):
        executor, backend, fl_ctx, _ = self._make_started_executor()
        backend.handle_event = Mock()

        executor.handle_event(EventType.ABOUT_TO_END_RUN, fl_ctx)

        backend.handle_event.assert_not_called()

    def test_end_run_finalizes_and_clears_backend(self):
        executor, backend, fl_ctx, _ = self._make_started_executor()
        executor.handle_event(EventType.END_RUN, fl_ctx)
        assert "finalize" in backend.calls
        reply = executor.execute("train", Shareable(), fl_ctx, Signal())
        assert reply.get_return_code() == ReturnCode.EXECUTION_EXCEPTION

    def test_backend_finalize_exception_is_logged_and_ignored(self):
        executor, backend, fl_ctx, _ = self._make_started_executor()
        backend.finalize = Mock(side_effect=RuntimeError("finalize failed"))

        executor.handle_event(EventType.END_RUN, fl_ctx)

        assert executor._backend is None

    def test_backend_exception_in_execute_becomes_error_reply(self):
        executor, backend, fl_ctx, _ = self._make_started_executor()
        backend.execute = Mock(side_effect=RuntimeError("boom"))
        reply = executor.execute("train", Shareable(), fl_ctx, Signal())
        assert reply.get_return_code() == ReturnCode.EXECUTION_EXCEPTION

    def test_unsafe_job_error_propagates_out_of_execute(self):
        # ClientRunner has dedicated UNSAFE_JOB handling for UnsafeJobError; the executor must
        # let it propagate instead of masking it as EXECUTION_EXCEPTION.
        executor, backend, fl_ctx, _ = self._make_started_executor()
        backend.execute = Mock(side_effect=UnsafeJobError("unsafe"))
        with pytest.raises(UnsafeJobError):
            executor.execute("train", Shareable(), fl_ctx, Signal())

    def test_unsafe_component_error_becomes_execution_exception(self):
        # UnsafeComponentError has no dedicated ClientRunner handling, so it takes the generic
        # error path (its existing behavior) rather than propagating.
        executor, backend, fl_ctx, _ = self._make_started_executor()
        backend.execute = Mock(side_effect=UnsafeComponentError("bad component"))
        reply = executor.execute("train", Shareable(), fl_ctx, Signal())
        assert reply.get_return_code() == ReturnCode.EXECUTION_EXCEPTION

    def test_backend_non_shareable_result_becomes_error_reply(self):
        executor, backend, fl_ctx, _ = self._make_started_executor()
        backend.result = {"not": "a shareable"}
        reply = executor.execute("train", Shareable(), fl_ctx, Signal())
        assert reply.get_return_code() == ReturnCode.EXECUTION_EXCEPTION

    def test_backend_receives_config_context(self):
        # initialize() receives a frozen ClientAPIBackendContext carrying the executor config and a
        # back-reference to the executor (for fire_log_analytics/logging).
        backend = _StubBackend()
        executor = _StubbedInProcessExecutor(
            backend,
            execution_mode="in_process",
            task_script_path="custom/train.py",
            train_task_name="my_train",
            params_exchange_format=ExchangeFormat.PYTORCH,
            server_expected_format=ExchangeFormat.NUMPY,
            memory_gc_rounds=7,
        )
        engine, _ = _make_recording_engine()
        executor.handle_event(EventType.START_RUN, _make_fl_ctx(engine))

        ctx = backend.context
        assert isinstance(ctx, ClientAPIBackendContext)
        assert ctx.executor is executor
        assert ctx.task_script_path == "custom/train.py"
        assert ctx.train_task_name == "my_train"
        assert ctx.params_exchange_format == ExchangeFormat.PYTORCH
        assert ctx.server_expected_format == ExchangeFormat.NUMPY
        assert ctx.memory_gc_rounds == 7

    def test_backend_context_is_frozen(self):
        backend = _StubBackend()
        executor = _StubbedInProcessExecutor(backend, execution_mode="in_process")
        executor.handle_event(EventType.START_RUN, _make_fl_ctx(_make_recording_engine()[0]))
        with pytest.raises(Exception):
            backend.context.task_script_path = "other.py"

    def test_backend_spec_is_abstract(self):
        with pytest.raises(TypeError):
            ClientAPIBackendSpec()


class TestAnalyticsOwnership:
    """fire_log_analytics is the executor-owned LOG-to-analytics conversion surface."""

    def _fire(self, fire_fed_event: bool):
        executor = ClientAPIExecutor(execution_mode="in_process")
        executor.set_analytics_fire_fed_event(fire_fed_event)
        engine = Mock()
        scopes = []
        engine.fire_event.side_effect = lambda event_type, ctx: scopes.append(
            (event_type, ctx.get_prop(FLContextKey.EVENT_SCOPE))
        )
        fl_ctx = _make_fl_ctx(engine)
        dxo = create_analytic_dxo(tag="loss", value=0.1, data_type=AnalyticsDataType.SCALAR)
        executor.fire_log_analytics(fl_ctx, dxo)
        return scopes

    def test_default_path_fires_local_analytics_event(self):
        # Default: local un-prefixed event; ConvertToFedEvent (added by BaseFedJob) is
        # responsible for re-firing it as "fed.analytix_log_stats".
        scopes = self._fire(fire_fed_event=False)
        assert scopes == [(ANALYTIC_EVENT_TYPE, EventScope.LOCAL)]

    def test_fed_path_fires_federation_scoped_event(self):
        # The fed path must fire the already-"fed."-prefixed event name so it lands on the same
        # server-side event as MetricRelay (job_config/script_runner.py) and flower_job.py
        # ("fed.analytix_log_stats"); firing the un-prefixed name federation-scoped would miss every
        # consumer listening on "fed.analytix_log_stats".
        assert FED_ANALYTIC_EVENT_TYPE == "fed.analytix_log_stats"
        scopes = self._fire(fire_fed_event=True)
        assert scopes == [(FED_ANALYTIC_EVENT_TYPE, EventScope.FEDERATION)]


class TestSurfaceFreeze:
    """Interface freeze #2: accidental renames/reorders of the constructor surface fail CI."""

    def test_constructor_parameter_names_match_frozen_list(self):
        params = list(inspect.signature(ClientAPIExecutor.__init__).parameters)
        assert params[0] == "self"
        assert params[1:] == FROZEN_CONSTRUCTOR_PARAMS

    def test_execution_mode_is_required(self):
        sig = inspect.signature(ClientAPIExecutor.__init__)
        assert sig.parameters["execution_mode"].default is inspect.Parameter.empty

    def test_frozen_defaults(self):
        # launch_once/heartbeat_interval/heartbeat_timeout/allow_reconnect defaults are
        # normative from the design's Configuration Surface; the rest are frozen by EX-2.
        sig = inspect.signature(ClientAPIExecutor.__init__)
        expected_defaults = {
            "command": None,
            "task_script_path": None,
            "task_script_args": "",
            "launch_once": True,
            "launch_timeout": 300.0,
            "shutdown_timeout": None,
            "stop_grace_period": 30.0,
            "heartbeat_interval": 5.0,
            "heartbeat_timeout": 30.0,
            "task_wait_timeout": None,
            "result_wait_timeout": None,
            "train_task_name": AppConstants.TASK_TRAIN,
            "evaluate_task_name": AppConstants.TASK_VALIDATION,
            "submit_model_task_name": AppConstants.TASK_SUBMIT_MODEL,
            "train_with_evaluation": False,
            "params_exchange_format": ExchangeFormat.RAW,
            "server_expected_format": ExchangeFormat.NUMPY,
            "params_transfer_type": TransferType.FULL,
            "memory_gc_rounds": 0,
            "cuda_empty_cache": False,
            "attach_timeout": None,
            "allow_reconnect": False,
        }
        actual_defaults = {name: p.default for name, p in sig.parameters.items() if name != "self"}
        actual_defaults.pop("execution_mode")
        assert actual_defaults == expected_defaults

    def test_module_path_is_pinned(self):
        # The design's example job config pins this exact path.
        assert ClientAPIExecutor.__module__ == "nvflare.app_common.executors.client_api_executor"

    def test_execution_mode_values_are_frozen(self):
        assert ALL_EXECUTION_MODES == ("in_process", "external_process", "attach")
