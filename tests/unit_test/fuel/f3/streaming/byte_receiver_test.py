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

import threading
from types import SimpleNamespace

import pytest

from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.byte_receiver import RxTask
from nvflare.fuel.f3.streaming.stream_const import StreamHeaderKey
from nvflare.fuel.f3.streaming.stream_types import StreamError


@pytest.fixture(autouse=True)
def clean_rx_task_map():
    with RxTask.map_lock:
        RxTask.rx_task_map.clear()
    yield
    with RxTask.map_lock:
        RxTask.rx_task_map.clear()


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
