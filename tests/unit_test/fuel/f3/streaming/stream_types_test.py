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

import pytest

from nvflare.fuel.f3.streaming.stream_types import StreamCancelled, StreamError, StreamFuture, StreamTaskSpec


class _TaskHandle(StreamTaskSpec):
    def __init__(self):
        self.cancel_called = False

    def cancel(self):
        self.cancel_called = True


def test_stream_future_cancel_marks_future_done_and_invokes_callback():
    task_handle = _TaskHandle()
    future = StreamFuture(stream_id=7, task_handle=task_handle)
    callback_calls = []

    future.add_done_callback(lambda: callback_calls.append("done"))

    assert future.cancel() is True
    assert future.done()
    assert future.cancelled() is True
    assert task_handle.cancel_called is True
    assert callback_calls == ["done"]

    error = future.exception(timeout=0.1)
    assert isinstance(error, StreamCancelled)

    with pytest.raises(StreamCancelled):
        future.result(timeout=0.1)


def test_stream_future_cancel_returns_false_after_completion_with_none_result():
    future = StreamFuture(stream_id=8)
    future.set_result(None)

    assert future.cancel() is False


def test_done_returns_bool():
    future = StreamFuture(stream_id=9)
    assert future.done() is False

    future.set_exception(StreamError("err"))
    assert future.done() is True


def test_add_done_callback_invoked_immediately_when_future_already_done():
    future = StreamFuture(stream_id=10)
    future.set_result(42)

    calls = []
    future.add_done_callback(lambda: calls.append("late"))

    assert calls == ["late"]


def test_set_result_raises_on_double_call():
    future = StreamFuture(stream_id=11)
    future.set_result(1)

    with pytest.raises(StreamError, match="already done"):
        future.set_result(2)
