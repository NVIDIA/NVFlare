# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Optional

from nvflare.apis.dxo import DXO
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.task_handler import TaskHandler


class ErrorHandlingExecutor(Executor, ABC):
    """This class adds error handling mechanisms to Executor spec.

    It also makes sharable convertible to DXO.
    It delegates the task execution to TaskHandler.
    """

    def __init__(self):
        super().__init__()
        self.init_status_ok = True
        self.init_failure = {"abort_job": None, "fail_client": None}
        self.client_name = None
        self.task_handler: Optional[TaskHandler] = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.initialize(fl_ctx)
        elif event_type == EventType.END_RUN:
            self.finalize(fl_ctx)

    def initialize(self, fl_ctx: FLContext):
        try:
            self.client_name = fl_ctx.get_identity_name()
            self.task_handler = self.get_task_handler(fl_ctx)

        except TypeError as te:
            self.log_exception(fl_ctx, f"{self.__class__.__name__} initialize failed.")
            self.init_status_ok = False
            self.init_failure = {"abort_job": te}
        except Exception as e:
            self.log_exception(fl_ctx, f"{self.__class__.__name__} initialize failed.")
            self.init_status_ok = False
            self.init_failure = {"fail_client": e}

    @abstractmethod
    def get_task_handler(self, fl_ctx: FLContext) -> TaskHandler:
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
            result = self.task_handler.execute_task(task_name, shareable, fl_ctx, abort_signal)
            if result is not None:
                dxo = DXO(data_kind=self.get_data_kind(), data=result)
                return dxo.to_shareable()

            self.log_error(
                fl_ctx,
                f"task:{task_name} failed on client:{fl_ctx.get_identity_name()} due to result is '{result}'\n",
            )
            return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)

        except Exception:
            self.log_exception(fl_ctx, f"{self.__class__.__name__} executes task {task_name} failed.")
            return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)

    def _check_init_status(self, fl_ctx: FLContext):

        if not self.init_status_ok:
            for fail_key in self.init_failure:
                reason = self.init_failure[fail_key]
                if fail_key == "abort_job":
                    return ReturnCode.EXECUTION_EXCEPTION
                self.system_panic(reason, fl_ctx)
                return ReturnCode.EXECUTION_RESULT_ERROR
        return None

    def finalize(self, fl_ctx: FLContext):
        try:
            if self.task_handler:
                self.task_handler.finalize(fl_ctx)
        except Exception:
            self.log_exception(fl_ctx, f"{self.__class__.__name__} finalize exception.")
