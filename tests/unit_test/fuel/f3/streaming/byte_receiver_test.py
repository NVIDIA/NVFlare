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

from types import SimpleNamespace

import pytest

from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.byte_receiver import RxTask
from nvflare.fuel.f3.streaming.stream_const import StreamHeaderKey


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
