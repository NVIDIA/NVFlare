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

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.ccwf.client_ctl import ClientSideController, _LearnTask


class _FailingClientSideController(ClientSideController):
    def __init__(self):
        super().__init__(
            task_name_prefix="test",
            learn_task_check_interval=0.01,
            learn_task_ack_timeout=1.0,
            learn_task_abort_timeout=1.0,
            final_result_ack_timeout=1.0,
        )

    def start_workflow(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        raise NotImplementedError

    def do_learn_task(self, name: str, data: Shareable, fl_ctx: FLContext, abort_signal: Signal):
        self.asked_to_stop = True
        raise RuntimeError("learn failed")


def test_do_learn_logs_exception_from_learn_task():
    ctl = _FailingClientSideController()
    ctl.logger = MagicMock()
    ctl.learn_task = _LearnTask("train", Shareable(), MagicMock())

    ctl._do_learn()

    ctl.logger.error.assert_called_once()
    assert "exception from do_learn_task" in ctl.logger.error.call_args.args[0]
    assert ctl.learn_task is None
