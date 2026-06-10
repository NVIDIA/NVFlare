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

import time
from unittest.mock import MagicMock

import pytest

import nvflare.fuel.f3.streaming.byte_streamer as byte_streamer_module
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.byte_streamer import TxTask
from nvflare.fuel.f3.streaming.stream_const import STREAM_CHANNEL, STREAM_DATA_TOPIC, StreamHeaderKey
from nvflare.fuel.f3.streaming.stream_types import Stream, StreamError


class DummyStream(Stream):
    def __init__(self, chunks):
        size = sum(len(c) for c in chunks if c)
        super().__init__(size=size, headers={})
        self._chunks = list(chunks)

    def read(self, _size):
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


class TestByteStreamerAckWatchdog:
    def _make_task(
        self,
        monkeypatch,
        *,
        window_size,
        ack_wait,
        ack_progress_timeout,
        ack_progress_check_interval,
        chunks,
        chunk_size=1,
    ):
        monkeypatch.setattr(CommConfigurator, "get_streaming_window_size", lambda self, default: window_size)
        monkeypatch.setattr(CommConfigurator, "get_streaming_ack_wait", lambda self, default: ack_wait)
        monkeypatch.setattr(
            CommConfigurator, "get_streaming_ack_progress_timeout", lambda self, default: ack_progress_timeout
        )
        monkeypatch.setattr(
            CommConfigurator,
            "get_streaming_ack_progress_check_interval",
            lambda self, default: ack_progress_check_interval,
        )

        cell = MagicMock()
        cell.fire_and_forget.return_value = {}
        stream = DummyStream(chunks)
        task = TxTask(
            cell=cell,
            chunk_size=chunk_size,
            channel="ch",
            topic="tp",
            target="peer",
            headers={},
            stream=stream,
            reliable=False,
            secure=False,
            optional=False,
        )
        return task, cell

    def test_ack_progress_check_interval_is_clamped_to_prevent_busy_spin(self, monkeypatch):
        task, _ = self._make_task(
            monkeypatch,
            window_size=0,
            ack_wait=0.5,
            ack_progress_timeout=2.0,
            ack_progress_check_interval=0.0,
            chunks=[b""],
        )

        assert task.ack_progress_check_interval == 0.01

    def test_handle_ack_updates_ack_progress_timestamp(self, monkeypatch):
        task, _ = self._make_task(
            monkeypatch,
            window_size=1024,
            ack_wait=1.0,
            ack_progress_timeout=10.0,
            ack_progress_check_interval=1.0,
            chunks=[b""],
        )

        old_ts = task.last_ack_progress_ts
        msg = Message(
            {
                MessageHeaderKey.ORIGIN: "peer",
                StreamHeaderKey.OFFSET: 128,
            },
            None,
        )

        task.handle_ack(msg)

        assert task.offset_ack == 128
        assert task.last_ack_progress_ts >= old_ts

    def test_watchdog_allows_progress_and_stream_completes(self, monkeypatch):
        task, _ = self._make_task(
            monkeypatch,
            window_size=0,
            ack_wait=0.5,
            ack_progress_timeout=2.0,
            ack_progress_check_interval=0.01,
            chunks=[b"x", b""],
            chunk_size=1,
        )

        # Force flow-control wait path on first iteration
        task.offset = 1
        task.offset_ack = 0
        task.last_ack_progress_ts = time.monotonic()

        task.ack_waiter = MagicMock()

        def wait_side_effect(*_args, **_kwargs):
            task.offset_ack = task.offset
            task.last_ack_progress_ts = time.monotonic()
            return True

        task.ack_waiter.wait.side_effect = wait_side_effect
        task.send_loop()

        assert task.stream_future.done()
        assert task.stream_future.exception() is None

    def test_watchdog_stops_when_no_ack_progress(self, monkeypatch):
        task, _ = self._make_task(
            monkeypatch,
            window_size=0,
            ack_wait=0.5,
            ack_progress_timeout=0.01,
            ack_progress_check_interval=0.01,
            chunks=[b"x"],
            chunk_size=1,
        )

        task.offset = 1
        task.offset_ack = 0
        task.last_ack_progress_ts = time.monotonic() - 10.0
        task.ack_waiter = MagicMock()

        task.send_loop()

        err = task.stream_future.exception()
        assert err is not None
        assert "ack made no progress" in str(err).lower()

    def test_ack_wait_timeout_still_applies_when_progress_timeout_is_large(self, monkeypatch):
        task, _ = self._make_task(
            monkeypatch,
            window_size=0,
            ack_wait=0.02,
            ack_progress_timeout=10.0,
            ack_progress_check_interval=0.005,
            chunks=[b"x"],
            chunk_size=1,
        )

        task.offset = 1
        task.offset_ack = 0
        task.last_ack_progress_ts = time.monotonic()
        task.ack_waiter = MagicMock()
        task.ack_waiter.wait.return_value = False

        task.send_loop()

        err = task.stream_future.exception()
        assert err is not None
        assert "ack timeouts" in str(err).lower()

    def test_fix2_only_progress_timeout_triggers_before_large_ack_wait(self, monkeypatch):
        """Fix #2 only: no-progress timeout should fail fast even when ack_wait is large."""
        task, _ = self._make_task(
            monkeypatch,
            window_size=0,
            ack_wait=1.0,
            ack_progress_timeout=0.02,
            ack_progress_check_interval=0.005,
            chunks=[b"x"],
            chunk_size=1,
        )

        task.offset = 1
        task.offset_ack = 0
        task.last_ack_progress_ts = time.monotonic() - 10.0
        task.ack_waiter = MagicMock()
        task.ack_waiter.wait.return_value = False

        start = time.monotonic()
        task.send_loop()
        elapsed = time.monotonic() - start

        err = task.stream_future.exception()
        assert err is not None
        assert "ack made no progress" in str(err).lower()
        assert elapsed < 0.2


class TestReliableByteStreamer:
    @pytest.fixture
    def retry_scheduler(self, monkeypatch):
        calls = {"registered": [], "unregistered": [], "wakeups": 0}

        def fake_register(task):
            calls["registered"].append(task)

        def fake_unregister(task):
            calls["unregistered"].append(task)

        def fake_wakeup():
            calls["wakeups"] += 1

        monkeypatch.setattr(byte_streamer_module.reliable_retry_scheduler, "register", fake_register)
        monkeypatch.setattr(byte_streamer_module.reliable_retry_scheduler, "unregister", fake_unregister)
        monkeypatch.setattr(byte_streamer_module.reliable_retry_scheduler, "wakeup", fake_wakeup)
        return calls

    def _make_reliable_task(self, monkeypatch, retry_scheduler):
        monkeypatch.setattr(CommConfigurator, "get_streaming_window_size", lambda self, default: 1024)
        monkeypatch.setattr(CommConfigurator, "get_streaming_ack_wait", lambda self, default: 1.0)
        monkeypatch.setattr(CommConfigurator, "get_streaming_ack_progress_timeout", lambda self, default: 10.0)
        monkeypatch.setattr(CommConfigurator, "get_streaming_ack_progress_check_interval", lambda self, default: 1.0)
        monkeypatch.setattr(CommConfigurator, "get_streaming_retry_wait", lambda self, default: 0.01)
        monkeypatch.setattr(CommConfigurator, "get_streaming_retry_timeout", lambda self, default: 1.0)
        monkeypatch.setattr(CommConfigurator, "get_streaming_retry_max_pending_bytes", lambda self, default: default)

        cell = MagicMock()
        cell.fire_and_forget.return_value = {}
        task = TxTask(
            cell=cell,
            chunk_size=4,
            channel="ch",
            topic="tp",
            target="peer",
            headers={},
            stream=DummyStream([b""]),
            reliable=True,
            secure=False,
            optional=False,
        )
        assert retry_scheduler["registered"] == [task]
        return task, cell

    def _patch_common_config(self, monkeypatch):
        monkeypatch.setattr(CommConfigurator, "get_streaming_window_size", lambda self, default: 1024)
        monkeypatch.setattr(CommConfigurator, "get_streaming_ack_wait", lambda self, default: 1.0)
        monkeypatch.setattr(CommConfigurator, "get_streaming_ack_progress_timeout", lambda self, default: 10.0)
        monkeypatch.setattr(CommConfigurator, "get_streaming_ack_progress_check_interval", lambda self, default: 1.0)
        monkeypatch.setattr(CommConfigurator, "get_streaming_retry_wait", lambda self, default: 0.01)
        monkeypatch.setattr(CommConfigurator, "get_streaming_retry_timeout", lambda self, default: 1.0)
        monkeypatch.setattr(CommConfigurator, "get_streaming_retry_max_pending_bytes", lambda self, default: default)

    def _make_task_with_reliable(self, reliable, monkeypatch):
        cell = MagicMock()
        cell.fire_and_forget.return_value = {}
        return TxTask(
            cell=cell,
            chunk_size=4,
            channel="ch",
            topic="tp",
            target="peer",
            headers={},
            stream=DummyStream([b""]),
            reliable=reliable,
            secure=False,
            optional=False,
        )

    def test_reliable_default_comes_from_config(self, monkeypatch, retry_scheduler):
        self._patch_common_config(monkeypatch)
        monkeypatch.setattr(CommConfigurator, "get_streaming_reliable", lambda self, default: False)

        task = self._make_task_with_reliable(None, monkeypatch)

        assert task.reliable is False
        assert task.pending_messages is None
        assert retry_scheduler["registered"] == []

    def test_reliable_default_is_opt_in_when_config_omitted(self, monkeypatch, retry_scheduler):
        self._patch_common_config(monkeypatch)

        def get_streaming_reliable(_self, default):
            assert default is False
            return default

        monkeypatch.setattr(CommConfigurator, "get_streaming_reliable", get_streaming_reliable)

        task = self._make_task_with_reliable(None, monkeypatch)

        assert task.reliable is False
        assert task.pending_messages is None
        assert retry_scheduler["registered"] == []

    def test_explicit_reliable_overrides_config(self, monkeypatch, retry_scheduler):
        self._patch_common_config(monkeypatch)
        monkeypatch.setattr(CommConfigurator, "get_streaming_reliable", lambda self, default: False)

        task = self._make_task_with_reliable(True, monkeypatch)

        assert task.reliable is True
        assert task.pending_messages == {}
        assert retry_scheduler["registered"] == [task]

    def test_retry_pending_byte_limit_comes_from_config(self, monkeypatch, retry_scheduler):
        self._patch_common_config(monkeypatch)
        monkeypatch.setattr(CommConfigurator, "get_streaming_retry_max_pending_bytes", lambda self, default: 7)

        task = self._make_task_with_reliable(True, monkeypatch)

        assert task.retry_max_pending_bytes == 7

    def test_reliable_stream_waits_for_sequence_ack_before_completion(self, monkeypatch, retry_scheduler):
        task, cell = self._make_reliable_task(monkeypatch, retry_scheduler)
        cell.fire_and_forget.return_value = {"peer": "temporary failure"}

        task.buffer[0:1] = b"x"
        task.buffer_size = 1
        task.send_pending_buffer(final=True)

        assert task.pending_messages
        assert task.offset == 1
        assert not task.stopped

        task.stop()

        assert task.stopping is True
        assert not task.stream_future.done()

        ack = Message(
            {
                MessageHeaderKey.ORIGIN: "peer",
                StreamHeaderKey.OFFSET: 1,
                StreamHeaderKey.SEQUENCE: 0,
            },
            None,
        )
        task.handle_ack(ack)

        assert task.stopped is True
        assert task.pending_messages == {}
        assert task.stream_future.result(timeout=0.1) == 1

    def test_reliable_stream_snapshots_retry_payload(self, monkeypatch, retry_scheduler):
        task, _ = self._make_reliable_task(monkeypatch, retry_scheduler)

        task.buffer[0:4] = b"abcd"
        task.buffer_size = 4
        task.send_pending_buffer()

        _start, _last_retry, message = task.pending_messages[0]
        task.buffer[0:4] = b"wxyz"

        assert message.payload == b"abcd"
        assert message.get_header(StreamHeaderKey.RETRY_WAIT) == task.retry_wait
        assert message.get_header(StreamHeaderKey.RETRY_TIMEOUT) == task.retry_timeout

        task.buffer[0:1] = b"e"
        task.buffer_size = 1
        task.send_pending_buffer()

        _start, _last_retry, next_message = task.pending_messages[1]
        assert next_message.get_header(StreamHeaderKey.RETRY_WAIT) is None
        assert next_message.get_header(StreamHeaderKey.RETRY_TIMEOUT) is None

    def test_reliable_stream_rejects_ack_without_sequence(self, monkeypatch, retry_scheduler):
        task, _ = self._make_reliable_task(monkeypatch, retry_scheduler)
        task.pending_messages[0] = (time.monotonic(), time.monotonic(), Message(None, b"x"))

        ack = Message({MessageHeaderKey.ORIGIN: "peer", StreamHeaderKey.OFFSET: 1}, None)
        task.handle_ack(ack)

        err = task.stream_future.exception(timeout=0.1)
        assert err is not None
        assert "doesn't support reliable streaming" in str(err)

    def test_retry_pending_byte_limit_stops_stream(self, monkeypatch, retry_scheduler):
        self._patch_common_config(monkeypatch)
        monkeypatch.setattr(CommConfigurator, "get_streaming_retry_max_pending_bytes", lambda self, default: 1)

        task = self._make_task_with_reliable(True, monkeypatch)
        task.buffer[0:4] = b"abcd"
        task.buffer_size = 4

        assert task.send_pending_buffer() is False

        err = task.stream_future.exception(timeout=0.1)
        assert err is not None
        assert "too many retry messages" in str(err)
        assert task.pending_messages == {}
        assert task.stopped is True

    def test_stop_unregisters_retry_task(self, monkeypatch, retry_scheduler):
        task, _ = self._make_reliable_task(monkeypatch, retry_scheduler)

        task.stop(StreamError("failed"))

        assert retry_scheduler["unregistered"] == [task]

    def test_retry_task_resends_due_pending_message(self, monkeypatch, retry_scheduler):
        task, cell = self._make_reliable_task(monkeypatch, retry_scheduler)
        message = Message(
            {
                StreamHeaderKey.STREAM_ID: task.sid,
                StreamHeaderKey.SEQUENCE: 0,
            },
            b"x",
        )
        task.pending_messages[0] = (0.0, 0.0, message)
        task.retry_wait = 0.01
        task.retry_timeout = 10.0

        monkeypatch.setattr(byte_streamer_module.time, "monotonic", lambda: 1.0)

        assert task.retry_task() == task.retry_wait

        cell.fire_and_forget.assert_called_with(
            STREAM_CHANNEL,
            STREAM_DATA_TOPIC,
            "peer",
            message,
            secure=False,
            optional=False,
        )

    def test_retry_task_failure_fails_stream_future(self, monkeypatch, retry_scheduler):
        task, cell = self._make_reliable_task(monkeypatch, retry_scheduler)
        message = Message(
            {
                StreamHeaderKey.STREAM_ID: task.sid,
                StreamHeaderKey.SEQUENCE: 0,
            },
            b"x",
        )
        task.pending_messages[0] = (0.0, 0.0, message)
        task.retry_wait = 0.01
        task.retry_timeout = 10.0

        monkeypatch.setattr(byte_streamer_module.time, "monotonic", lambda: 1.0)
        cell.fire_and_forget.side_effect = RuntimeError("boom")

        task.retry_task()

        err = task.stream_future.exception(timeout=0.1)
        assert err is not None
        assert "retry thread ended due to error: boom" in str(err)

    def test_retry_timeout_stops_stream_after_releasing_retry_lock(self, monkeypatch, retry_scheduler):
        task, _ = self._make_reliable_task(monkeypatch, retry_scheduler)
        message = Message(
            {
                StreamHeaderKey.STREAM_ID: task.sid,
                StreamHeaderKey.SEQUENCE: 0,
            },
            b"x",
        )
        task.pending_messages[0] = (0.0, 0.0, message)
        task.retry_timeout = 0.5

        monkeypatch.setattr(byte_streamer_module.time, "monotonic", lambda: 1.0)

        original_stop = task.stop

        def checked_stop(*args, **kwargs):
            assert not task.retry_lock._is_owned()
            return original_stop(*args, **kwargs)

        monkeypatch.setattr(task, "stop", checked_stop)

        task.retry_task()

        err = task.stream_future.exception(timeout=0.1)
        assert err is not None
        assert "retry failed" in str(err)
