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

"""Tests for the in_process backend of ClientAPIExecutor (plan: EX-3).

Drives the real DataBus round-trip the design defines for in_process: the backend fires the
task on TOPIC_GLOBAL_RESULT, a fake trainer replies on TOPIC_LOCAL_RESULT, and execute()
returns the result. Also covers the backend-contract obligations the legacy executor did not
have: initialize() self-unwinding, finalize() idempotency + DataBus cleanup (the DataBus is a
process singleton, so leaks would cross into later jobs in the same process), bounded result
wait, and LOG routing through the executor-owned fire_log_analytics().
"""

import time
from unittest.mock import MagicMock, Mock

import pytest

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.client_api.backend_spec import ClientAPIBackendContext
from nvflare.app_common.executors.client_api.in_process_backend import InProcessBackend
from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor
from nvflare.client.api_spec import CLIENT_API_KEY
from nvflare.client.config import ConfigKey, ExchangeFormat, TransferType
from nvflare.client.in_process.api import (
    TOPIC_ABORT,
    TOPIC_GLOBAL_RESULT,
    TOPIC_LOCAL_RESULT,
    TOPIC_LOG_DATA,
    TOPIC_STOP,
    InProcessClientAPI,
)
from nvflare.fuel.data_event.data_bus import DataBus

BACKEND_TOPICS = (TOPIC_LOCAL_RESULT, TOPIC_LOG_DATA, TOPIC_ABORT, TOPIC_STOP)


def _backend_callbacks_subscribed(bus, backend) -> bool:
    """True if any of the BACKEND's own callbacks is still subscribed.

    Checked at callback granularity: the InProcessClientAPI the backend creates keeps its own
    ABORT/STOP subscriptions (legacy behavior, unchanged for parity), so topic-level emptiness
    would be the wrong assertion.
    """
    backend_cbs = {backend._local_result_callback, backend._log_result_callback, backend._to_abort_callback}
    for topic in BACKEND_TOPICS:
        for cb, _ in bus.subscribers.get(topic, []):
            if cb in backend_cbs:
                return True
    return False


@pytest.fixture(autouse=True)
def clean_databus():
    """The DataBus is a process singleton: isolate every test from prior subscriptions/data."""
    bus = DataBus()
    bus.subscribers.clear()
    bus.data_store.clear()
    yield bus
    bus.subscribers.clear()
    bus.data_store.clear()


@pytest.fixture
def custom_dir(tmp_path):
    """A workspace custom dir holding a trivial trainer script that exits immediately."""
    (tmp_path / "train.py").write_text("x = 1\n")
    return str(tmp_path)


def _make_engine():
    """An engine whose new_context() works as a context manager (for the LOG callback)."""
    engine = MagicMock()
    engine.new_context.return_value.__enter__.return_value = FLContext()
    return engine


def _make_fl_ctx(engine, custom_dir):
    fl_ctx = FLContext()
    fl_ctx.put(key=ReservedKey.ENGINE, value=engine, private=True, sticky=False)
    fl_ctx.put(key=ReservedKey.RUN_NUM, value="job-1", private=False, sticky=False)
    fl_ctx.put(key=ReservedKey.IDENTITY_NAME, value="site-1", private=False, sticky=False)
    workspace = Mock()
    workspace.get_app_custom_dir.return_value = custom_dir
    fl_ctx.put(key=FLContextKey.WORKSPACE_OBJECT, value=workspace, private=True, sticky=False)
    fl_ctx.put(key=FLContextKey.CURRENT_JOB_ID, value="job-1", private=False, sticky=False)
    return fl_ctx


def _make_context(executor=None, **overrides):
    kwargs = dict(
        executor=executor if executor is not None else MagicMock(),
        execution_mode="in_process",
        task_script_path="train.py",
        task_script_args="",
        # bounded by default so a broken round trip FAILS the test instead of hanging it
        result_wait_timeout=10.0,
    )
    kwargs.update(overrides)
    return ClientAPIBackendContext(**kwargs)


def _initialized_backend(custom_dir, executor=None, **overrides):
    backend = InProcessBackend()
    engine = _make_engine()
    fl_ctx = _make_fl_ctx(engine, custom_dir)
    backend.initialize(_make_context(executor=executor, **overrides), fl_ctx)
    return backend, fl_ctx


def _result_shareable() -> Shareable:
    return DXO(data_kind=DataKind.WEIGHTS, data={"w": [1.0]}).to_shareable()


class TestFactory:
    def test_executor_factory_returns_in_process_backend(self):
        executor = ClientAPIExecutor(execution_mode="in_process", task_script_path="custom/train.py")
        backend = executor._create_in_process_backend()
        assert isinstance(backend, InProcessBackend)

    def test_initialize_rejects_missing_or_non_py_script(self, custom_dir):
        backend = InProcessBackend()
        fl_ctx = _make_fl_ctx(_make_engine(), custom_dir)
        for bad in (None, "", "train.sh"):
            with pytest.raises(ValueError, match="task_script_path"):
                backend.initialize(_make_context(task_script_path=bad), fl_ctx)


class TestInitializeAndFinalize:
    def test_initialize_wires_databus_and_starts_trainer(self, clean_databus, custom_dir):
        backend, _ = _initialized_backend(custom_dir)
        try:
            # the trainer script's flare.init() finds the API instance here
            assert isinstance(clean_databus.get_data(CLIENT_API_KEY), InProcessClientAPI)
            for topic in BACKEND_TOPICS:
                assert topic in clean_databus.subscribers, f"backend must subscribe {topic}"
            # the trivial script exits on its own
            backend._task_fn_thread.join(timeout=5.0)
            assert not backend._task_fn_thread.is_alive()
        finally:
            backend.finalize(FLContext())

    def test_initialize_unwinds_on_failure(self, clean_databus, custom_dir):
        backend = InProcessBackend()
        engine = _make_engine()
        fl_ctx = _make_fl_ctx(engine, custom_dir)
        # break setup after the DataBus subscriptions happened
        workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        workspace.get_app_custom_dir.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            backend.initialize(_make_context(), fl_ctx)

        # contract: self-unwinding - no backend subscriptions and no API instance left behind
        assert clean_databus.get_data(CLIENT_API_KEY) is None
        assert not _backend_callbacks_subscribed(clean_databus, backend)

    def test_finalize_idempotent_and_cleans_databus(self, clean_databus, custom_dir):
        backend, _ = _initialized_backend(custom_dir)
        fl_ctx = FLContext()

        backend.finalize(fl_ctx)
        backend.finalize(fl_ctx)  # contract: idempotent, must not raise

        assert clean_databus.get_data(CLIENT_API_KEY) is None
        assert not _backend_callbacks_subscribed(clean_databus, backend)
        assert not backend._task_fn_thread.is_alive()


class TestExecute:
    def test_execute_round_trip(self, clean_databus, custom_dir):
        backend, fl_ctx = _initialized_backend(custom_dir)
        try:
            result_to_send = _result_shareable()

            received_tasks = []

            def fake_trainer(topic, data, databus):
                # replies as flare.send() does: publish the result on TOPIC_LOCAL_RESULT.
                # No asserts in here: a raise inside a DataBus callback is swallowed by its
                # thread pool and would turn a failure into a timeout.
                received_tasks.append(data)
                clean_databus.publish([TOPIC_LOCAL_RESULT], result_to_send)

            clean_databus.subscribe([TOPIC_GLOBAL_RESULT], fake_trainer)

            task = Shareable()
            task.set_header(AppConstants.CURRENT_ROUND, 3)
            result = backend.execute("train", task, fl_ctx, Signal())

            assert isinstance(result, Shareable)
            assert result.get_return_code() == ReturnCode.OK
            # the round travels back on the result for workflow bookkeeping
            assert result.get_header(AppConstants.CURRENT_ROUND) == 3
            assert len(received_tasks) == 1 and isinstance(received_tasks[0], Shareable)
        finally:
            backend.finalize(FLContext())

    def test_execute_returns_task_aborted_on_triggered_signal(self, clean_databus, custom_dir):
        backend, fl_ctx = _initialized_backend(custom_dir)
        try:
            abort_events = []
            clean_databus.subscribe([TOPIC_ABORT], lambda t, d, b: abort_events.append(d))

            signal = Signal()
            signal.trigger("stop")
            result = backend.execute("train", Shareable(), fl_ctx, signal)

            assert result.get_return_code() == ReturnCode.TASK_ABORTED
            # the trainer was told the task is aborted
            assert abort_events
        finally:
            backend.finalize(FLContext())

    def test_execute_bounded_by_result_wait_timeout(self, clean_databus, custom_dir):
        # contract: never wait unbounded past the configured bound; no trainer reply -> error
        backend, fl_ctx = _initialized_backend(custom_dir, result_wait_timeout=0.05)
        try:
            start = time.time()
            result = backend.execute("train", Shareable(), fl_ctx, Signal())
            elapsed = time.time() - start

            assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
            assert elapsed < 5.0, "timeout path must not hang"
        finally:
            backend.finalize(FLContext())

    def test_multi_round_sequential_execute(self, clean_databus, custom_dir):
        """The same backend serves consecutive rounds (the legacy executor's normal life)."""
        backend, fl_ctx = _initialized_backend(custom_dir)
        try:
            clean_databus.subscribe(
                [TOPIC_GLOBAL_RESULT], lambda t, d, b: clean_databus.publish([TOPIC_LOCAL_RESULT], _result_shareable())
            )
            for round_num in (0, 1, 2):
                task = Shareable()
                task.set_header(AppConstants.CURRENT_ROUND, round_num)
                result = backend.execute("train", task, fl_ctx, Signal())
                assert result.get_return_code() == ReturnCode.OK
                assert result.get_header(AppConstants.CURRENT_ROUND) == round_num
        finally:
            backend.finalize(FLContext())

    def test_execute_after_timeout_fails_fast_with_accurate_rc(self, clean_databus, custom_dir):
        """One result-wait timeout kills the in-process trainer for good (its thread is never
        relaunched); later tasks must fail FAST with EXECUTION_EXCEPTION -- an instant
        TASK_ABORTED would be misleading (task never delivered, abort_signal never triggered)."""
        backend, fl_ctx = _initialized_backend(custom_dir, result_wait_timeout=0.05)
        try:
            first = backend.execute("train", Shareable(), fl_ctx, Signal())
            assert first.get_return_code() == ReturnCode.EXECUTION_EXCEPTION

            start = time.monotonic()
            second = backend.execute("train", Shareable(), fl_ctx, Signal())
            elapsed = time.monotonic() - start
            assert second.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
            assert elapsed < 1.0, "post-abort tasks must fail at entry, not wait the poll loop"
        finally:
            backend.finalize(FLContext())

    def test_trainer_abort_mid_task_returns_task_aborted(self, clean_databus, custom_dir):
        """TOPIC_ABORT arriving mid-wait (what TaskScriptRunner fires when the script raises)
        aborts the current task well before the result-wait bound -- legacy parity."""
        backend, fl_ctx = _initialized_backend(custom_dir, result_wait_timeout=30.0)
        try:
            # deterministic: the abort fires synchronously right after the task is delivered
            clean_databus.subscribe(
                [TOPIC_GLOBAL_RESULT], lambda t, d, b: clean_databus.publish([TOPIC_ABORT], "script failure")
            )
            start = time.monotonic()
            result = backend.execute("train", Shareable(), fl_ctx, Signal())
            elapsed = time.monotonic() - start

            assert result.get_return_code() == ReturnCode.TASK_ABORTED
            assert elapsed < 10.0, "abort must not wait out the result bound"
        finally:
            backend.finalize(FLContext())

    def test_bad_result_type_returns_execution_exception(self, clean_databus, custom_dir):
        backend, fl_ctx = _initialized_backend(custom_dir)
        try:
            clean_databus.subscribe(
                [TOPIC_GLOBAL_RESULT],
                lambda t, d, b: clean_databus.publish([TOPIC_LOCAL_RESULT], "not-a-shareable"),
            )
            result = backend.execute("train", Shareable(), fl_ctx, Signal())
            assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
        finally:
            backend.finalize(FLContext())


class TestMultiBackend:
    def test_finalizing_backend_does_not_clear_successors_client_api(self, clean_databus, custom_dir):
        """Backend A's teardown must not remove the CLIENT_API_KEY a later backend B installed."""
        backend_a, _ = _initialized_backend(custom_dir)
        backend_b, _ = _initialized_backend(custom_dir)
        try:
            api_b = backend_b._client_api
            assert clean_databus.get_data(CLIENT_API_KEY) is api_b

            backend_a.finalize(FLContext())
            assert clean_databus.get_data(CLIENT_API_KEY) is api_b
        finally:
            backend_b.finalize(FLContext())
        assert clean_databus.get_data(CLIENT_API_KEY) is None

    def test_finalize_fires_stop_exactly_once(self, clean_databus, custom_dir):
        """Idempotency means no repeated side effects, not just no raise."""
        backend, _ = _initialized_backend(custom_dir)
        stops = []
        clean_databus.subscribe([TOPIC_STOP], lambda t, d, b: stops.append(d))

        backend.finalize(FLContext())
        backend.finalize(FLContext())

        assert len(stops) == 1

    def test_finalize_detaches_api_own_subscriptions(self, clean_databus, custom_dir):
        """The API instance's own bus subscriptions go away with the backend (close()), so a
        dead job's API can no longer receive -- and pin -- a later job's global model."""
        backend, _ = _initialized_backend(custom_dir)
        api = backend._client_api

        backend.finalize(FLContext())

        for topic, subs in clean_databus.subscribers.items():
            for cb, _kw in subs:
                assert getattr(cb, "__self__", None) is not api, f"dead API still subscribed on {topic}"

    def test_initialize_unwinds_on_late_failure(self, clean_databus, custom_dir, monkeypatch):
        """Failure AFTER the API instance exists: its subscriptions must not leak either."""

        def raising_init(self, *args, **kwargs):
            raise RuntimeError("late boom")

        monkeypatch.setattr(InProcessClientAPI, "init", raising_init)
        backend = InProcessBackend()
        fl_ctx = _make_fl_ctx(_make_engine(), custom_dir)

        with pytest.raises(RuntimeError, match="late boom"):
            backend.initialize(_make_context(), fl_ctx)

        assert clean_databus.get_data(CLIENT_API_KEY) is None
        assert not _backend_callbacks_subscribed(clean_databus, backend)
        for topic, subs in clean_databus.subscribers.items():
            for cb, _kw in subs:
                assert not isinstance(getattr(cb, "__self__", None), InProcessClientAPI), f"leak on {topic}"


class TestLogRouting:
    def test_log_data_routes_through_executor_fire_log_analytics(self, clean_databus, custom_dir):
        executor = MagicMock()
        backend, _ = _initialized_backend(custom_dir, executor=executor)
        try:
            # what flare.log() publishes: a dict with key/value/data_type
            clean_databus.publish(
                [TOPIC_LOG_DATA], {"key": "accuracy", "value": 0.9, "data_type": AnalyticsDataType.SCALAR}
            )

            assert executor.fire_log_analytics.call_count == 1
            dxo = executor.fire_log_analytics.call_args[0][1]
            # the key->tag rename happened and the DXO carries the exact metric
            assert dxo.data == {"track_key": "accuracy", "track_value": 0.9}
        finally:
            backend.finalize(FLContext())


class TestMetaPassThrough:
    def test_meta_uses_raw_full_and_context_task_names(self, custom_dir):
        backend, fl_ctx = _initialized_backend(
            custom_dir, train_task_name="my_train", evaluate_task_name="my_eval", submit_model_task_name="my_submit"
        )
        try:
            meta = backend._prepare_task_meta(fl_ctx, "my_train")
            exchange = meta[ConfigKey.TASK_EXCHANGE]
            # FLARE-2698: pass-through boundary - no converter formats in the frozen surface
            assert exchange[ConfigKey.EXCHANGE_FORMAT] == ExchangeFormat.RAW
            assert exchange[ConfigKey.TRANSFER_TYPE] == TransferType.FULL
            assert exchange[ConfigKey.TRAIN_TASK_NAME] == "my_train"
            assert exchange[ConfigKey.EVAL_TASK_NAME] == "my_eval"
            assert exchange[ConfigKey.SUBMIT_MODEL_TASK_NAME] == "my_submit"
            assert meta[ConfigKey.TASK_NAME] == "my_train"
        finally:
            backend.finalize(FLContext())
