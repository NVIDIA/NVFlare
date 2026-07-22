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
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from nvflare.collab.api.call_opt import CallOption
from nvflare.collab.runtime.local.direct_dispatcher import DirectDispatcher


@pytest.mark.parametrize("error", [None, ValueError("local call failed")])
def test_group_call_always_completes_send_slot(error):
    target_app = MagicMock()
    dispatcher = DirectDispatcher(
        target_obj_name="",
        target_app=target_app,
        target_obj=MagicMock(),
        abort_signal=MagicMock(),
        thread_executor=MagicMock(),
    )
    result = object()
    if error:
        func = MagicMock(side_effect=error)
    else:
        func = MagicMock(return_value=result)
    dispatcher._preprocess = MagicMock(return_value=(MagicMock(), {}))
    target_app.apply_outgoing_result_filters.side_effect = lambda _target, _func, value, _ctx: value
    gcc = MagicMock()
    gcc.target_name = "site-1"

    dispatcher._run_func_in_group(gcc, "train", func, (), {})

    gcc.send_completed.assert_called_once_with()
    if error:
        gcc.set_exception.assert_called_once_with(error)
        gcc.set_result.assert_not_called()
    else:
        gcc.set_result.assert_called_once_with(result)
        gcc.set_exception.assert_not_called()


def test_group_call_does_not_nest_work_in_same_executor():
    target_app = MagicMock()
    target_func = MagicMock(return_value="result")
    target_app.find_collab_method.return_value = target_func
    target_app.apply_outgoing_result_filters.side_effect = lambda _target, _func, value, _ctx: value
    completed = threading.Event()

    with ThreadPoolExecutor(max_workers=1) as executor:
        dispatcher = DirectDispatcher(
            target_obj_name="",
            target_app=target_app,
            target_obj=MagicMock(),
            abort_signal=MagicMock(triggered=False),
            thread_executor=executor,
        )
        dispatcher._preprocess = MagicMock(return_value=(MagicMock(), {}))
        gcc = MagicMock()
        gcc.target_name = "site-1"
        gcc.context = MagicMock()
        gcc.call_opt = CallOption(timeout=0.05)
        gcc.send_completed.side_effect = completed.set

        dispatcher.call_target_in_group(gcc, "train")

        assert completed.wait(timeout=1.0)
        gcc.set_result.assert_called_once_with("result")
        gcc.set_exception.assert_not_called()
