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

import builtins
import sys
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import UnsafeJobError
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.client_api import in_process_backend as ipb_module
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
    """A workspace custom dir holding a trainer thread that stays alive until STOP."""
    (tmp_path / "train.py").write_text(
        "from nvflare.client.api_spec import CLIENT_API_KEY\n"
        "from nvflare.fuel.data_event.data_bus import DataBus\n"
        "api = DataBus().get_data(CLIENT_API_KEY)\n"
        "try:\n"
        "    while api.is_running():\n"
        "        api.clear()\n"
        "except RuntimeError:\n"
        "    pass\n"
    )
    return str(tmp_path)


@pytest.fixture
def exited_custom_dir(tmp_path):
    """A workspace custom dir holding a trainer script that exits normally."""
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
            assert backend._task_fn_thread.is_alive()
        finally:
            backend.finalize(FLContext())
        assert not backend._task_fn_thread.is_alive()

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

    def test_finalize_bounded_when_trainer_stuck_in_user_code(self, tmp_path, caplog, capsys):
        """A trainer wedged in user code never observes TOPIC_STOP: with result_wait_timeout,
        execute() returns while the thread still runs, so finalize() must join with a bound
        and abandon (daemon thread) instead of hanging CJ/simulator teardown forever. Abandoning
        the runner must restore the process globals that TaskScriptRunner owns."""
        (tmp_path / "train.py").write_text("import time\nwhile True: time.sleep(0.5)\n")
        original_print = builtins.print
        original_argv = sys.argv
        original_argv_values = list(sys.argv)
        backend, fl_ctx = _initialized_backend(str(tmp_path), result_wait_timeout=0.05)
        try:
            result = backend.execute("train", Shareable(), fl_ctx, Signal())
            assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
            assert backend._task_fn_thread.is_alive()  # the reachable-hang precondition
            assert backend._task_fn_thread.daemon  # a wedged trainer cannot block process exit
        finally:
            with patch.object(ipb_module, "_TRAINER_STOP_JOIN_TIMEOUT", 0.5):
                start = time.monotonic()
                backend.finalize(FLContext())
                elapsed = time.monotonic() - start
        assert elapsed < 5.0, "finalize must not hang on a stuck trainer"
        assert backend._task_fn_thread.is_alive()
        assert builtins.print is original_print
        assert sys.argv is original_argv
        assert list(sys.argv) == original_argv_values
        capsys.readouterr()
        print("unrelated output")
        assert capsys.readouterr().out == "unrelated output\n"
        assert "did not stop within" in caplog.text

    def test_initialize_configures_memory_management(self, custom_dir):
        backend, _ = _initialized_backend(custom_dir, memory_gc_rounds=3, cuda_empty_cache=True)
        try:
            assert backend._client_api._memory_gc_rounds == 3
            assert backend._client_api._cuda_empty_cache is True
        finally:
            backend.finalize(FLContext())

    def test_finalize_logs_stop_publish_failure(self, exited_custom_dir, caplog):
        backend, _ = _initialized_backend(exited_custom_dir)
        backend._task_fn_thread.join(timeout=1.0)
        backend._event_manager.fire_event = Mock(side_effect=RuntimeError("stop failed"))

        backend.finalize(FLContext())

        assert "stop failed" in caplog.text

    def test_finalize_logs_thread_join_failure(self, exited_custom_dir, caplog):
        backend, _ = _initialized_backend(exited_custom_dir)
        backend._task_fn_thread.join(timeout=1.0)
        thread = Mock()
        thread.is_alive.return_value = True
        thread.join.side_effect = RuntimeError("join failed")
        backend._task_fn_thread = thread

        backend.finalize(FLContext())

        assert "join failed" in caplog.text

    def test_unwind_logs_cleanup_failure(self, caplog):
        backend = InProcessBackend()
        backend._data_bus = Mock()
        backend._subscribed = True
        backend._data_bus.unsubscribe.side_effect = RuntimeError("unsubscribe failed")

        backend._unwind()

        assert "unsubscribe failed" in caplog.text

    def test_unwind_closes_api_even_when_unsubscribe_fails(self, caplog):
        """close() sets the zombie-containment gate (and detaches the API's own subscriptions);
        a failing backend unsubscribe must not skip it -- unwind is best-effort per step."""
        backend = InProcessBackend()
        backend._data_bus = Mock()
        backend._data_bus.unsubscribe.side_effect = RuntimeError("unsubscribe failed")
        backend._data_bus.get_data.return_value = None
        backend._subscribed = True
        api = Mock()
        backend._client_api = api

        backend._unwind()

        api.close.assert_called_once()
        # every unsubscribe was still attempted, not abandoned at the first failure
        assert backend._data_bus.unsubscribe.call_count == 4
        assert "unsubscribe failed" in caplog.text

    def test_finalize_cleans_backend_state_even_when_close_fails(self, clean_databus, custom_dir, caplog):
        """The reverse direction: a failing close() must not skip the CLIENT_API_KEY clear or
        the backend's own unsubscribes."""
        backend, _ = _initialized_backend(custom_dir)
        backend._client_api.close = Mock(side_effect=RuntimeError("close failed"))

        backend.finalize(FLContext())

        assert "close failed" in caplog.text
        assert clean_databus.get_data(CLIENT_API_KEY) is None
        assert not _backend_callbacks_subscribed(clean_databus, backend)


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

    def test_execute_bounded_by_result_wait_timeout(self, clean_databus, custom_dir, monkeypatch):
        # contract: never wait unbounded past the configured bound; no trainer reply -> error
        monkeypatch.setattr("nvflare.app_common.executors.client_api.in_process_backend._RESULT_POLL_INTERVAL", 1.0)
        backend, fl_ctx = _initialized_backend(custom_dir, result_wait_timeout=0.05)
        try:
            start = time.monotonic()
            result = backend.execute("train", Shareable(), fl_ctx, Signal())
            elapsed = time.monotonic() - start

            assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
            assert elapsed < 0.5, "timeout must not be rounded up to the polling interval"
        finally:
            backend.finalize(FLContext())

    def test_execute_fails_fast_when_trainer_thread_exits(self, clean_databus, exited_custom_dir):
        backend, fl_ctx = _initialized_backend(exited_custom_dir, result_wait_timeout=2.0)
        try:
            backend._task_fn_thread.join(timeout=1.0)
            assert not backend._task_fn_thread.is_alive()

            start = time.monotonic()
            result = backend.execute("train", Shareable(), fl_ctx, Signal())
            elapsed = time.monotonic() - start

            assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
            assert elapsed < 0.5, "a dead trainer must be detected before waiting for a result"
            assert "trainer thread exited" in backend._abort_reason
        finally:
            backend.finalize(FLContext())

    def test_execute_detects_trainer_exit_while_waiting(self, clean_databus, custom_dir, monkeypatch):
        backend, fl_ctx = _initialized_backend(custom_dir, result_wait_timeout=2.0)
        monkeypatch.setattr(backend, "_trainer_thread_is_alive", Mock(side_effect=[True, False]))
        try:
            result = backend.execute("train", Shareable(), fl_ctx, Signal())

            assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
            assert backend._abort_reason == "trainer thread exited before producing a result"
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

    def test_none_result_returns_execution_exception_without_waiting_for_timeout(self, clean_databus, custom_dir):
        backend, fl_ctx = _initialized_backend(custom_dir, result_wait_timeout=1.0)
        try:
            clean_databus.subscribe(
                [TOPIC_GLOBAL_RESULT],
                lambda t, d, b: clean_databus.publish([TOPIC_LOCAL_RESULT], None),
            )
            start = time.monotonic()
            result = backend.execute("train", Shareable(), fl_ctx, Signal())
            elapsed = time.monotonic() - start

            assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
            assert elapsed < 0.5, "an invalid None result must fail fast"
        finally:
            backend.finalize(FLContext())

    def test_unsafe_job_error_propagates(self, clean_databus, custom_dir, monkeypatch):
        backend, fl_ctx = _initialized_backend(custom_dir)
        try:
            monkeypatch.setattr(backend, "_prepare_task_meta", Mock(side_effect=UnsafeJobError("unsafe")))
            with pytest.raises(UnsafeJobError, match="unsafe"):
                backend.execute("train", Shareable(), fl_ctx, Signal())
        finally:
            backend.finalize(FLContext())

    def test_execute_exception_returns_execution_exception(self, clean_databus, custom_dir, monkeypatch):
        backend, fl_ctx = _initialized_backend(custom_dir)
        try:
            monkeypatch.setattr(backend, "_prepare_task_meta", Mock(side_effect=RuntimeError("boom")))

            result = backend.execute("train", Shareable(), fl_ctx, Signal())

            assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
            assert backend._abort_reason is not None
        finally:
            backend.finalize(FLContext())

    def test_handle_event_is_noop(self):
        backend = InProcessBackend()
        backend.handle_event("custom_event", FLContext())


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


class TestClosedApiOutgoingGate:
    """A closed API must not publish onto the singleton bus (zombie-trainer containment).

    finalize() can abandon a still-running daemon trainer after the bounded join; that
    thread still holds the API object and may resume later. Its send()/log() must DROP --
    not publish (a successor job's backend is now subscribed to the same topics), and not
    raise (an exception would hit TaskScriptRunner's catch-all, which fires TOPIC_ABORT
    onto the singleton bus and would poison the successor)."""

    def test_finalize_closes_api_and_late_send_log_are_dropped(self, clean_databus, custom_dir):
        backend, _ = _initialized_backend(custom_dir)
        api = backend._client_api
        backend.finalize(FLContext())
        assert api.closed is True

        published = []
        clean_databus.subscribe([TOPIC_LOCAL_RESULT, TOPIC_LOG_DATA], lambda t, d, b: published.append((t, d)))

        # the zombie resumes: neither call may publish or raise
        api.send(None)
        api.log("accuracy", 0.99, "SCALAR")

        assert published == []

    def test_closed_receive_returns_none_fast(self, clean_databus, custom_dir):
        backend, _ = _initialized_backend(custom_dir)
        api = backend._client_api
        backend.finalize(FLContext())

        start = time.monotonic()
        assert api.receive(timeout=10.0) is None
        assert time.monotonic() - start < 1.0, "closed receive must not wait out its timeout"

    def test_open_api_still_publishes(self, clean_databus, custom_dir):
        # the gate must not over-drop: an open API's log still lands on the bus
        backend, _ = _initialized_backend(custom_dir)
        try:
            api = backend._client_api
            api.rank = "0"
            published = []
            clean_databus.subscribe([TOPIC_LOG_DATA], lambda t, d, b: published.append(d))
            api.log("accuracy", 0.99, "SCALAR")
            assert len(published) == 1
        finally:
            backend.finalize(FLContext())


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

    def test_invalid_log_data_is_logged_and_ignored(self, clean_databus, custom_dir, caplog):
        executor = MagicMock()
        backend, _ = _initialized_backend(custom_dir, executor=executor)
        try:
            backend._log_result_callback(TOPIC_LOG_DATA, None, clean_databus)

            executor.fire_log_analytics.assert_not_called()
            assert "invalid result format" in caplog.text
        finally:
            backend.finalize(FLContext())

    def test_log_processing_error_is_logged_and_ignored(self, clean_databus, custom_dir, caplog):
        executor = MagicMock()
        executor.fire_log_analytics.side_effect = RuntimeError("analytics failed")
        backend, _ = _initialized_backend(custom_dir, executor=executor)
        try:
            backend._log_result_callback(
                TOPIC_LOG_DATA,
                {"key": "accuracy", "value": 0.9, "data_type": AnalyticsDataType.SCALAR},
                clean_databus,
            )

            assert "failed to process trainer LOG data" in caplog.text
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
