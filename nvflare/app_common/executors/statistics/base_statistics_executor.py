# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC, abstractmethod

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.data_spec import Data
from typing import Dict, Optional, List


class BaseStatsExecutor(Executor, ABC):
    """
       BaseStatsDefaultExecutor intended to capture the following
       1) reduce the boilerplate code for execution
       2) expose only load_data and client_exec functions
       3) this executor package result into Data Exchange and return to sharable, so client don't need to
       4) also handle abort signal.
    """

    def __init__(self):
        super().__init__()
        self.data: Optional[Data] = None

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        client_name = fl_ctx.get_prop(ReservedKey.CLIENT_NAME)
        self.log_info(fl_ctx, f"Executing task '{task_name}' for client: '{client_name}'")
        try:
            if self.data is None:
                self.data = self.load_data(task_name, client_name, fl_ctx)
            if self.data is not None:
                result = self.client_exec(task_name, shareable, fl_ctx, abort_signal)
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)
                if result:
                    dxo = DXO(data_kind=DataKind.ANALYTIC, data=result)
                    return dxo.to_shareable()
                else:
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            else:
                raise ValueError("data is not loaded, please check load_data() function")

        except BaseException as e:
            self.log_exception(fl_ctx, f"Task {task_name} failed. Exception: {e.__str__()}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    @abstractmethod
    def load_data(self, task_name: str, client_name: str, fl_ctx: FLContext) -> Data:
        pass

    @abstractmethod
    def client_exec(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        pass
