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
from concurrent.futures import CancelledError, ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from nvflare.collab.api.call_opt import CallOption
from nvflare.collab.api.context import get_call_context, set_call_context
from nvflare.collab.api.exceptions import CollabCallError
from nvflare.collab.api.group_call_context import GroupCallContext, ResultWaiter
from nvflare.collab.runtime.cell_dispatcher import CellDispatcher
from nvflare.collab.runtime.defs import CallReplyKey
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import new_cell_message


@pytest.mark.parametrize("error", [None, ValueError("cannot encode")])
def test_group_call_always_completes_parallel_slot(error):
    dispatcher = CellDispatcher(
        manager=MagicMock(),
        engine=MagicMock(),
        caller="server",
        cell=MagicMock(),
        target_fqcn="site-1/job",
        abort_signal=MagicMock(),
        thread_executor=MagicMock(),
    )
    result = object()
    if error:
        dispatcher._call_target = MagicMock(side_effect=error)
    else:
        dispatcher._call_target = MagicMock(return_value=result)
    gcc = MagicMock()

    dispatcher._run_func(gcc, "train", (), {})

    gcc.call_completed.assert_called_once_with()
    if error:
        gcc.set_exception.assert_called_once_with(error)
        gcc.set_result.assert_not_called()
    else:
        gcc.set_result.assert_called_once_with(result)
        gcc.set_exception.assert_not_called()


def test_pretransmission_error_releases_bounded_parallel_slot():
    dispatcher = CellDispatcher(
        manager=MagicMock(),
        engine=MagicMock(),
        caller="server",
        cell=MagicMock(),
        target_fqcn="site-1/job",
        abort_signal=MagicMock(),
        thread_executor=MagicMock(),
    )
    error = ValueError("cannot encode")
    dispatcher._call_target = MagicMock(side_effect=error)
    waiter = ResultWaiter(["site-1"])
    waiter.inc_call_count()
    gcc = GroupCallContext(
        app=MagicMock(),
        target_name="site-1",
        call_opt=CallOption(parallel=1),
        func_name="train",
        process_cb=None,
        cb_kwargs={},
        context=MagicMock(),
        waiter=waiter,
    )
    gcc.set_completion_cb(waiter.dec_call_count)

    dispatcher._run_func(gcc, "train", (), {})

    assert waiter.standing_call_count == 0
    assert list(waiter.results) == []
    call_error = waiter.results.failures["site-1"]
    assert isinstance(call_error, CollabCallError)
    assert call_error.cause is error


def test_cancelled_group_future_completes_site():
    executor = ThreadPoolExecutor(max_workers=1)
    release_worker = threading.Event()
    worker_started = threading.Event()
    completed = threading.Event()

    def occupy_worker():
        worker_started.set()
        release_worker.wait(timeout=5.0)

    try:
        executor.submit(occupy_worker)
        assert worker_started.wait(timeout=1.0)

        dispatcher = CellDispatcher(
            manager=MagicMock(),
            engine=MagicMock(),
            caller="server",
            cell=MagicMock(),
            target_fqcn="site-1/job",
            abort_signal=MagicMock(),
            thread_executor=executor,
        )
        gcc = MagicMock()
        gcc.call_completed.side_effect = completed.set

        dispatcher.call_target_in_group(gcc, "train")
        executor.shutdown(wait=False, cancel_futures=True)

        assert completed.wait(timeout=1.0)
        cancellation = gcc.set_exception.call_args.args[0]
        assert isinstance(cancellation, CancelledError)
        gcc.set_result.assert_not_called()
    finally:
        release_worker.set()
        executor.shutdown(wait=True, cancel_futures=True)


def test_remote_error_preserves_type_and_traceback():
    cell = MagicMock()
    cell.send_request.return_value = new_cell_message(
        {MessageHeaderKey.RETURN_CODE: ReturnCode.PROCESS_EXCEPTION},
        {
            CallReplyKey.ERROR: "ValueError: invalid input",
            CallReplyKey.ERROR_TYPE: "ValueError",
            CallReplyKey.ERROR_TRACEBACK: "remote traceback",
        },
    )
    dispatcher = CellDispatcher(
        manager=MagicMock(),
        engine=MagicMock(),
        caller="server",
        cell=cell,
        target_fqcn="site-1/job",
        abort_signal=MagicMock(),
        thread_executor=MagicMock(),
    )

    try:
        with pytest.raises(CollabCallError) as exc_info:
            dispatcher._call_target(
                context=MagicMock(),
                target_name="site-1.trainer",
                call_opt=CallOption(),
                func_name="train",
            )
    finally:
        set_call_context(None)

    error = exc_info.value
    assert error.site == "site-1"
    assert error.func_name == "train"
    assert error.cause_type == "ValueError"
    assert error.remote_traceback == "remote traceback"


def test_group_worker_restores_previous_context():
    dispatcher = CellDispatcher(
        manager=MagicMock(),
        engine=MagicMock(),
        caller="server",
        cell=MagicMock(),
        target_fqcn="site-1/job",
        abort_signal=MagicMock(),
        thread_executor=MagicMock(),
    )
    previous_ctx = MagicMock()
    call_ctx = MagicMock()

    def call_target(**_kwargs):
        set_call_context(call_ctx)
        return "result"

    dispatcher._call_target = call_target
    gcc = MagicMock()

    set_call_context(previous_ctx)
    try:
        dispatcher._run_func(gcc, "train", (), {})
        assert get_call_context() is previous_ctx
    finally:
        set_call_context(None)
