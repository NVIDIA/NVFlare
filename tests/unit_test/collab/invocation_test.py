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

import queue
import threading
from unittest.mock import MagicMock

from nvflare.apis.signal import Signal
from nvflare.collab.api._invocation import InvocationDispatcher
from nvflare.collab.api.call_opt import CallOption
from nvflare.collab.api.context import Context
from nvflare.collab.api.group_call_context import GroupCallContext, ResultQueue, ResultWaiter


class _GroupOnlyDispatcher(InvocationDispatcher):
    def __init__(self, result):
        super().__init__(Signal())
        self.result = result
        self.group_context = None

    def call_target_in_group(self, gcc, func_name: str, *args, **kwargs):
        self.group_context = gcc
        gcc.set_result(self.result)


def test_call_target_uses_single_member_group():
    dispatcher = _GroupOnlyDispatcher(result="result")
    context = Context(app=object(), caller="server", callee="site-1", abort_signal=dispatcher.abort_signal)

    result = dispatcher.call_target(
        context=context,
        target_name="site-1.trainer",
        call_opt=CallOption(),
        func_name="train",
        value=1,
    )

    assert result == "result"
    assert dispatcher.group_context.target_name == "site-1.trainer"
    assert dispatcher.group_context.context is context
    assert dispatcher.group_context.waiter.sites == ["site-1.trainer"]


def test_call_target_returns_immediately_when_no_result_is_expected():
    dispatcher = _GroupOnlyDispatcher(result="ignored")
    context = Context(app=object(), caller="server", callee="site-1", abort_signal=dispatcher.abort_signal)

    result = dispatcher.call_target(
        context=context,
        target_name="site-1.trainer",
        call_opt=CallOption(expect_result=False),
        func_name="stop",
    )

    assert result is None


def test_group_send_completion_is_idempotent():
    callback = MagicMock()
    gcc = GroupCallContext(
        app=MagicMock(),
        target_name="site-1",
        call_opt=CallOption(),
        func_name="train",
        process_cb=None,
        cb_kwargs={},
        context=MagicMock(),
        waiter=ResultWaiter(["site-1"]),
    )
    gcc.set_send_complete_cb(callback, target="site-1")

    gcc.send_completed()
    gcc.send_completed()

    callback.assert_called_once_with(target="site-1")


def test_result_queue_does_not_drop_last_concurrent_item():
    class RaceQueue(queue.Queue):
        def __init__(self):
            super().__init__()
            self.checked_empty = threading.Event()
            self.release_empty_check = threading.Event()

        def empty(self):
            result = super().empty()
            self.checked_empty.set()
            self.release_empty_check.wait(timeout=1.0)
            return result

        def get_nowait(self):
            try:
                return super().get_nowait()
            except queue.Empty:
                self.checked_empty.set()
                raise

    result_queue = ResultQueue(limit=1)
    race_queue = RaceQueue()
    result_queue.q = race_queue
    item = ("site-1", "result")
    outcome = []

    def consume():
        try:
            outcome.append(("result", next(result_queue)))
        except Exception as ex:
            outcome.append(("error", ex))

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    assert race_queue.checked_empty.wait(timeout=1.0)

    result_queue.append(item)
    race_queue.release_empty_check.set()
    consumer.join(timeout=1.0)

    assert not consumer.is_alive()
    assert outcome == [("result", item)]
