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
import traceback
from abc import ABC, abstractmethod
from typing import Optional

from nvflare.apis.dxo import DXO
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.executors.client_executor import ClientExecutor
from nvflare.security.logging import is_secure


class CommonExecutor(Executor, ABC):
    def __init__(self):
        super().__init__()
        self.init_status_ok = True
        self.init_failure = {"abort_job": None, "fail_client": None}
        self.client_name = None
        self.client_executor: Optional[ClientExecutor] = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.initialize(fl_ctx)
        elif event_type == EventType.END_RUN:
            self.finalize(fl_ctx)

    def initialize(self, fl_ctx: FLContext):
        try:
            self.client_name = fl_ctx.get_identity_name()
            self.client_executor = self.get_client_executor(fl_ctx)
        except TypeError as te:
            if not is_secure():
                self.log_exception(fl_ctx, traceback.format_exc())
            self.init_status_ok = False
            self.init_failure = {"abort_job": te}
        except Exception as e:
            if not is_secure():
                self.log_exception(fl_ctx, traceback.format_exc())
            self.init_status_ok = False
            self.init_failure = {"fail_client": e}

    @abstractmethod
    def get_client_executor(self, fl_ctx: FLContext) -> ClientExecutor:
        pass

    @abstractmethod
    def get_data_kind(self) -> str:
        pass

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        init_rc = self._check_init_status(fl_ctx)
        if init_rc:
            return make_reply(init_rc)
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        try:
            result = self.client_executor.client_exec(task_name, shareable, fl_ctx)
            if result:
                dxo = DXO(data_kind=self.get_data_kind(), data=result)
                return dxo.to_shareable()
            else:
                return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)

        except BaseException as e:
            self.log_exception(fl_ctx, f"Task {task_name} failed. Exception: {e.__str__()}")
            return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)

    def _check_init_status(self, fl_ctx: FLContext):

        if not self.init_status_ok:
            for fail_key in self.init_failure:
                reason = self.init_failure[fail_key]
                if fail_key == "abort_job":
                    return ReturnCode.EXECUTION_EXCEPTION
                else:
                    self.system_panic(reason, fl_ctx)
                    # probably never reach here, but just for type consistency
                    return ReturnCode.EXECUTION_RESULT_ERROR
        return None

    def finalize(self, fl_ctx: FLContext):
        try:
            if self.client_executor:
                self.client_executor.finalize()
        except Exception as e:
            self.log_exception(fl_ctx, f"Statistics generator finalize exception: {e}")
