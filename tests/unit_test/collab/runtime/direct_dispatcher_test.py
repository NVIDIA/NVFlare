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

from unittest.mock import MagicMock

import pytest

from nvflare.collab.runtime.local.direct_dispatcher import DirectDispatcher


@pytest.mark.parametrize("error", [None, ValueError("local call failed")])
def test_group_call_always_completes_send_slot(error):
    dispatcher = DirectDispatcher(
        target_obj_name="",
        target_app=MagicMock(),
        target_obj=MagicMock(),
        abort_signal=MagicMock(),
        thread_executor=MagicMock(),
    )
    result = object()
    if error:
        dispatcher._call_target = MagicMock(side_effect=error)
    else:
        dispatcher._call_target = MagicMock(return_value=result)
    gcc = MagicMock()
    gcc.target_name = "site-1"

    dispatcher._run_func_in_group(gcc, "train", (), {})

    gcc.send_completed.assert_called_once_with()
    if error:
        gcc.set_exception.assert_called_once_with(error)
        gcc.set_result.assert_not_called()
    else:
        gcc.set_result.assert_called_once_with(result)
        gcc.set_exception.assert_not_called()
