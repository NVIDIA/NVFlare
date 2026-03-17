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

from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.byte_streamer import TxTask
from nvflare.fuel.f3.streaming.stream_const import StreamHeaderKey
from nvflare.fuel.f3.streaming.stream_types import Stream


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
