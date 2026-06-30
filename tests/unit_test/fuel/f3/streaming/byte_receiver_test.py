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

import logging
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.comm_config import CommConfigurator
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.byte_receiver import RxStream, RxTask
from nvflare.fuel.f3.streaming.stream_const import STREAM_ACK_TOPIC, STREAM_CHANNEL, StreamDataType, StreamHeaderKey
from nvflare.fuel.f3.streaming.stream_types import StreamError


@pytest.fixture(autouse=True)
def clean_rx_task_map():
    with RxTask.map_lock:
        tasks = list(RxTask.rx_task_map.values())
        RxTask.rx_task_map.clear()
    for task in tasks:
        if task.cleanup_timer:
            task.cleanup_timer.cancel()
    yield
    with RxTask.map_lock:
        tasks = list(RxTask.rx_task_map.values())
        RxTask.rx_task_map.clear()
    for task in tasks:
        if task.cleanup_timer:
            task.cleanup_timer.cancel()


def _make_message(origin: str, sid: int, error: str = None) -> Message:
    message = Message()
    headers = {
        StreamHeaderKey.STREAM_ID: sid,
        MessageHeaderKey.ORIGIN: origin,
    }
    if error is not None:
        headers[StreamHeaderKey.ERROR_MSG] = error
    message.add_headers(headers)
    return message


def _make_chunk(
    origin: str,
    sid: int,
    seq: int,
    data_type: int,
    payload: bytes = b"x",
    reliable: bool = True,
    retry_wait: float = None,
    retry_timeout: float = None,
):
    message = Message(None, payload)
    headers = {
        MessageHeaderKey.ORIGIN: origin,
        StreamHeaderKey.STREAM_ID: sid,
        StreamHeaderKey.SEQUENCE: seq,
        StreamHeaderKey.DATA_TYPE: data_type,
        StreamHeaderKey.CHANNEL: "ch",
        StreamHeaderKey.TOPIC: "tp",
        StreamHeaderKey.SIZE: len(payload),
        StreamHeaderKey.RELIABLE: reliable,
    }
    if retry_wait is not None:
        headers[StreamHeaderKey.RETRY_WAIT] = retry_wait
    if retry_timeout is not None:
        headers[StreamHeaderKey.RETRY_TIMEOUT] = retry_timeout
    message.add_headers(headers)
    return message


class _DeadlockDetectingLock:
    """Lock that raises on same-thread re-acquire to model Lock self-deadlock."""

    def __init__(self):
        self._lock = threading.Lock()
        self._owner = None

    def __enter__(self):
        acquired = self.acquire()
        if not acquired:
            raise RuntimeError("failed to acquire lock")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    def acquire(self, blocking=True, timeout=-1):
        tid = threading.get_ident()
        if self._owner == tid:
            raise RuntimeError("self-deadlock: same thread re-acquiring map_lock")
        acquired = self._lock.acquire(blocking, timeout)
        if acquired:
            self._owner = tid
        return acquired

    def release(self):
        self._owner = None
        self._lock.release()

    def locked(self):
        return self._lock.locked()


def _pre_fix_find_or_create_task(message: Message, cell):
    """Original buggy logic: calls stop() while map_lock is still held."""

    sid = message.get_header(StreamHeaderKey.STREAM_ID)
    origin = message.get_header(MessageHeaderKey.ORIGIN)
    error = message.get_header(StreamHeaderKey.ERROR_MSG, None)

    with RxTask.map_lock:
        task = RxTask.rx_task_map.get(sid, None)
        if not task:
            if error:
                return None
            task = RxTask(sid, origin, cell)
            RxTask.rx_task_map[sid] = task
        else:
            if error:
                task.stop(StreamError(f"{task} Received error from {origin}: {error}"), notify=False)
                return None
    return task


def test_pre_fix_find_or_create_task_would_deadlock(monkeypatch):
    monkeypatch.setattr(RxTask, "map_lock", _DeadlockDetectingLock())

    origin = "site-1"
    sid = 99
    fake_cell = SimpleNamespace()

    create_message = _make_message(origin=origin, sid=sid)
    task = _pre_fix_find_or_create_task(create_message, fake_cell)
    assert task is not None

    error_message = _make_message(origin=origin, sid=sid, error="stream failed")
    with pytest.raises(RuntimeError, match="self-deadlock"):
        _pre_fix_find_or_create_task(error_message, fake_cell)


def test_find_or_create_task_stops_outside_map_lock(monkeypatch):
    origin = "site-1"
    sid = 123
    fake_cell = SimpleNamespace()

    create_message = _make_message(origin=origin, sid=sid)
    task = RxTask.find_or_create_task(create_message, fake_cell)
    assert task is not None

    stop_invocation = {"called": False, "lock_held": None}

    def fake_stop(self, error=None, notify=True):
        stop_invocation["called"] = True
        stop_invocation["lock_held"] = RxTask.map_lock.locked()

    monkeypatch.setattr(RxTask, "stop", fake_stop)

    error_message = _make_message(origin=origin, sid=sid, error="stream failed")
    returned_task = RxTask.find_or_create_task(error_message, fake_cell)

    assert returned_task is None
    assert stop_invocation["called"] is True
    assert stop_invocation["lock_held"] is False


def test_stop_ignores_missing_stream_future():
    """stop() before the first chunk (stream_future=None) must not raise AttributeError."""
    origin = "site-1"
    sid = 321
    fire_and_forget_calls = []
    fake_cell = SimpleNamespace(fire_and_forget=lambda *a, **kw: fire_and_forget_calls.append(a))

    task = RxTask(sid=sid, origin=origin, cell=fake_cell)
    with RxTask.map_lock:
        RxTask.rx_task_map[(origin, sid)] = task

    task.stop(StreamError("stream failed"), notify=True)

    with RxTask.map_lock:
        assert (origin, sid) not in RxTask.rx_task_map
    assert len(fire_and_forget_calls) > 0


def test_rxstream_close_ignores_missing_stream_future():
    task = RxTask(sid=456, origin="site-1", cell=SimpleNamespace())
    stream = RxStream(task)

    stream.close()

    assert stream.closed is True


def test_find_or_create_task_records_reliable_header():
    cell = SimpleNamespace()
    message = _make_chunk("site-1", sid=501, seq=0, data_type=StreamDataType.CHUNK, reliable=True)

    task = RxTask.find_or_create_task(message, cell)

    assert task.reliable is True


def test_reliable_duplicate_initial_chunk_sends_sequence_ack():
    cell = SimpleNamespace(fire_and_forget=MagicMock(return_value={}))
    message = _make_chunk("site-1", sid=502, seq=0, data_type=StreamDataType.CHUNK, reliable=True)
    task = RxTask.find_or_create_task(message, cell)

    assert task.process_chunk(message) is True
    assert task.process_chunk(message) is False

    cell.fire_and_forget.assert_called_with(
        STREAM_CHANNEL, STREAM_ACK_TOPIC, "site-1", cell.fire_and_forget.call_args.args[3], optional=False
    )
    ack = cell.fire_and_forget.call_args.args[3]
    assert ack.get_header(StreamHeaderKey.SEQUENCE) == 0
    assert ack.get_header(StreamHeaderKey.OFFSET) == 0


def test_reliable_duplicate_chunk_sends_latest_sequence_ack():
    cell = SimpleNamespace(fire_and_forget=MagicMock(return_value={}))
    chunk_0 = _make_chunk("site-1", sid=506, seq=0, data_type=StreamDataType.CHUNK, payload=b"a", reliable=True)
    chunk_1 = _make_chunk("site-1", sid=506, seq=1, data_type=StreamDataType.CHUNK, payload=b"b", reliable=True)
    task = RxTask.find_or_create_task(chunk_0, cell)

    assert task.process_chunk(chunk_0) is True
    assert task.process_chunk(chunk_1) is False
    assert task.seq == 1
    assert task.process_chunk(chunk_0) is False

    ack = cell.fire_and_forget.call_args.args[3]
    assert ack.get_header(StreamHeaderKey.SEQUENCE) == 1
    assert ack.get_header(StreamHeaderKey.OFFSET) == 0


def test_reliable_duplicate_out_of_sequence_chunk_reacks_latest_contiguous_sequence():
    cell = SimpleNamespace(fire_and_forget=MagicMock(return_value={}))
    chunk_0 = _make_chunk("site-1", sid=517, seq=0, data_type=StreamDataType.CHUNK, payload=b"a", reliable=True)
    chunk_2 = _make_chunk("site-1", sid=517, seq=2, data_type=StreamDataType.CHUNK, payload=b"c", reliable=True)
    task = RxTask.find_or_create_task(chunk_0, cell)

    assert task.process_chunk(chunk_0) is True
    assert task.process_chunk(chunk_2) is False
    assert cell.fire_and_forget.call_count == 0

    assert task.process_chunk(chunk_2) is False

    assert cell.fire_and_forget.call_count == 1
    ack = cell.fire_and_forget.call_args.args[3]
    assert ack.get_header(StreamHeaderKey.SEQUENCE) == 0
    assert ack.get_header(StreamHeaderKey.OFFSET) == 0


def test_reliable_final_chunk_sends_sequence_ack():
    cell = SimpleNamespace(fire_and_forget=MagicMock(return_value={}))
    message = _make_chunk("site-1", sid=503, seq=0, data_type=StreamDataType.FINAL, payload=b"", reliable=True)
    task = RxTask.find_or_create_task(message, cell)

    assert task.process_chunk(message) is True

    ack = cell.fire_and_forget.call_args.args[3]
    assert cell.fire_and_forget.call_args.args[:3] == (STREAM_CHANNEL, STREAM_ACK_TOPIC, "site-1")
    assert cell.fire_and_forget.call_args.kwargs == {"optional": True}
    assert ack.get_header(StreamHeaderKey.SEQUENCE) == 0
    assert ack.get_header(StreamHeaderKey.OFFSET) == 0


def test_reliable_final_ack_uses_received_offset_before_read():
    cell = SimpleNamespace(fire_and_forget=MagicMock(return_value={}))
    message = _make_chunk("site-1", sid=508, seq=0, data_type=StreamDataType.FINAL, payload=b"abc", reliable=True)
    task = RxTask.find_or_create_task(message, cell)

    assert task.process_chunk(message) is True

    ack = cell.fire_and_forget.call_args.args[3]
    assert task.offset == 0
    assert task.received_offset == 3
    assert ack.get_header(StreamHeaderKey.SEQUENCE) == 0
    assert ack.get_header(StreamHeaderKey.OFFSET) == 3


def test_reliable_final_ack_sent_outside_receive_locks():
    cell = SimpleNamespace(fire_and_forget=MagicMock(return_value={}))
    message = _make_chunk("site-1", sid=514, seq=0, data_type=StreamDataType.FINAL, payload=b"abc", reliable=True)
    task = RxTask.find_or_create_task(message, cell)

    def assert_unlocked(*args, **kwargs):
        assert not task.lock.locked()
        assert not task.stop_lock._is_owned()
        return {}

    cell.fire_and_forget.side_effect = assert_unlocked

    assert task.process_chunk(message) is True


def test_reliable_final_partial_reads_do_not_resend_current_ack():
    cell = SimpleNamespace(fire_and_forget=MagicMock(return_value={}))
    message = _make_chunk("site-1", sid=515, seq=0, data_type=StreamDataType.FINAL, payload=b"abcd", reliable=True)
    task = RxTask.find_or_create_task(message, cell)

    assert task.process_chunk(message) is True
    assert cell.fire_and_forget.call_count == 1

    assert task.read(2) == b"ab"
    assert cell.fire_and_forget.call_count == 1

    assert task.read(2) == b"cd"
    assert cell.fire_and_forget.call_count == 1


def test_reliable_success_stop_is_idempotent():
    cell = SimpleNamespace(fire_and_forget=MagicMock(return_value={}))
    message = _make_chunk("site-1", sid=512, seq=0, data_type=StreamDataType.FINAL, payload=b"", reliable=True)
    task = RxTask.find_or_create_task(message, cell)

    assert task.process_chunk(message) is True
    assert cell.fire_and_forget.call_count == 1
    cleanup_timer = task.cleanup_timer

    task.stop()

    assert cell.fire_and_forget.call_count == 1
    assert task.cleanup_timer is cleanup_timer


def test_reliable_late_error_after_completion_keeps_reack_window():
    cell = SimpleNamespace(fire_and_forget=MagicMock(return_value={}))
    message = _make_chunk("site-1", sid=513, seq=0, data_type=StreamDataType.FINAL, payload=b"", reliable=True)
    task = RxTask.find_or_create_task(message, cell)

    assert task.process_chunk(message) is True
    assert cell.fire_and_forget.call_count == 1

    task.stop(StreamError("late failure"), notify=False)

    assert task.completed is True
    assert task.failed is False
    assert cell.fire_and_forget.call_count == 1

    assert task.process_chunk(message) is False

    assert cell.fire_and_forget.call_count == 2
    ack = cell.fire_and_forget.call_args.args[3]
    assert ack.get_header(StreamHeaderKey.DATA_TYPE) == StreamDataType.ACK


def test_reliable_failed_task_rejects_retried_initial_chunk():
    cell = SimpleNamespace(fire_and_forget=MagicMock(return_value={}))
    message = _make_chunk("site-1", sid=516, seq=0, data_type=StreamDataType.CHUNK, payload=b"abc", reliable=True)
    task = RxTask.find_or_create_task(message, cell)

    task.stop(StreamError("failed before callback"), notify=True)

    with RxTask.map_lock:
        assert RxTask.rx_task_map[("site-1", 516)] is task
    assert task.stream_future is None
    assert task.cleanup_timer is not None
    assert cell.fire_and_forget.call_count == 1

    assert task.process_chunk(message) is False

    assert task.stream_future is None
    assert cell.fire_and_forget.call_count == 2
    error_msg = cell.fire_and_forget.call_args.args[3]
    assert error_msg.get_header(StreamHeaderKey.DATA_TYPE) == StreamDataType.ERROR
    assert "failed before callback" in error_msg.get_header(StreamHeaderKey.ERROR_MSG)


def test_stop_error_wakes_blocked_reader():
    task = RxTask(sid=518, origin="site-1", cell=SimpleNamespace(fire_and_forget=MagicMock(return_value={})))
    task.timeout = 60.0
    result = {}

    def read_task():
        try:
            task.read(1)
        except StreamError as ex:
            result["error"] = ex

    thread = threading.Thread(target=read_task)
    thread.start()
    threading.Event().wait(0.05)
    assert thread.is_alive()

    task.stop(StreamError("stream failed"), notify=False)

    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert "stream failed" in str(result["error"])


def test_send_ack_updates_ack_state_only_on_success():
    cell = SimpleNamespace(fire_and_forget=MagicMock(return_value={"site-1": "send failed"}))
    task = RxTask(sid=504, origin="site-1", cell=cell, reliable=True)

    assert task._send_ack(offset=10, seq=2) is False
    assert cell.fire_and_forget.call_args.kwargs == {"optional": False}
    assert task.offset_ack == 0
    assert task.seq_ack == -1

    cell.fire_and_forget.return_value = {}

    assert task._send_ack(offset=10, seq=2) is True
    assert task.offset_ack == 10
    assert task.seq_ack == 2


def test_completed_ack_send_failure_is_debug_only(caplog):
    cell = SimpleNamespace(fire_and_forget=MagicMock(return_value={"site-1": "target_unreachable"}))
    task = RxTask(sid=519, origin="site-1", cell=cell, reliable=True)
    task.completed = True

    with caplog.at_level(logging.DEBUG, logger="nvflare.fuel.f3.streaming.byte_receiver"):
        assert task._send_ack(offset=10, seq=2) is False

    assert cell.fire_and_forget.call_args.kwargs == {"optional": True}
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert "failed to ack seq 2" in caplog.text


def test_reliable_completed_task_reacks_duplicate_final_chunk():
    cell = SimpleNamespace(fire_and_forget=MagicMock(return_value={}))
    message = _make_chunk("site-1", sid=505, seq=0, data_type=StreamDataType.FINAL, payload=b"", reliable=True)
    task = RxTask.find_or_create_task(message, cell)
    task.completed_task_ttl = 60.0

    assert task.process_chunk(message) is True

    with RxTask.map_lock:
        assert RxTask.rx_task_map[("site-1", 505)] is task
    assert task.completed is True
    assert cell.fire_and_forget.call_count == 1

    assert task.process_chunk(message) is False

    assert cell.fire_and_forget.call_count == 2
    duplicate_ack = cell.fire_and_forget.call_args.args[3]
    assert duplicate_ack.get_header(StreamHeaderKey.SEQUENCE) == 0
    assert duplicate_ack.get_header(StreamHeaderKey.OFFSET) == 0


def test_reliable_completed_task_reacks_with_received_offset():
    cell = SimpleNamespace(fire_and_forget=MagicMock(return_value={}))
    message = _make_chunk("site-1", sid=509, seq=0, data_type=StreamDataType.FINAL, payload=b"abc", reliable=True)
    task = RxTask.find_or_create_task(message, cell)

    assert task.process_chunk(message) is True
    assert task.process_chunk(message) is False

    duplicate_ack = cell.fire_and_forget.call_args.args[3]
    assert duplicate_ack.get_header(StreamHeaderKey.SEQUENCE) == 0
    assert duplicate_ack.get_header(StreamHeaderKey.OFFSET) == 3


def test_completed_task_ttl_has_retry_wait_headroom(monkeypatch):
    monkeypatch.setattr(CommConfigurator, "get_streaming_retry_timeout", lambda self, default: 12.0)
    monkeypatch.setattr(CommConfigurator, "get_streaming_retry_wait", lambda self, default: 3.0)

    task = RxTask(sid=507, origin="site-1", cell=SimpleNamespace(), reliable=True)

    assert task.completed_task_ttl == 15.0


def test_completed_task_ttl_uses_sender_retry_window(monkeypatch):
    monkeypatch.setattr(CommConfigurator, "get_streaming_retry_timeout", lambda self, default: 1.0)
    monkeypatch.setattr(CommConfigurator, "get_streaming_retry_wait", lambda self, default: 1.0)
    cell = SimpleNamespace()
    message = _make_chunk(
        "site-1",
        sid=510,
        seq=0,
        data_type=StreamDataType.CHUNK,
        payload=b"x",
        reliable=True,
        retry_wait=4.0,
        retry_timeout=20.0,
    )
    task = RxTask.find_or_create_task(message, cell)

    assert task.completed_task_ttl == 2.0

    task.process_chunk(message)

    assert task.completed_task_ttl == 24.0


def test_completed_task_ttl_keeps_longer_local_retry_window(monkeypatch):
    monkeypatch.setattr(CommConfigurator, "get_streaming_retry_timeout", lambda self, default: 30.0)
    monkeypatch.setattr(CommConfigurator, "get_streaming_retry_wait", lambda self, default: 5.0)
    cell = SimpleNamespace()
    message = _make_chunk(
        "site-1",
        sid=511,
        seq=0,
        data_type=StreamDataType.CHUNK,
        payload=b"x",
        reliable=True,
        retry_wait=4.0,
        retry_timeout=20.0,
    )
    task = RxTask.find_or_create_task(message, cell)

    task.process_chunk(message)

    assert task.completed_task_ttl == 35.0
