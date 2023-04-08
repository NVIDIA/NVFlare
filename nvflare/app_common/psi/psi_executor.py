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

from nvflare.apis.dxo import DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.task_handler import TaskHandler
from nvflare.app_common.executors.error_handling_executor import ErrorHandlingExecutor
from nvflare.app_common.utils.component_utils import check_component_type


class PSIExecutor(ErrorHandlingExecutor):
    def __init__(self, psi_algo_id: str):
        super().__init__()
        self.psi_algo_id = psi_algo_id

    def get_data_kind(self) -> str:
        return DataKind.PSI

    def get_task_handler(self, fl_ctx: FLContext) -> TaskHandler:
        return self.load_task_handler(self.psi_algo_id, fl_ctx)

    def load_task_handler(self, psi_algo: str, fl_ctx: FLContext) -> TaskHandler:
        engine = fl_ctx.get_engine()
        psi_task_handler = engine.get_component(psi_algo) if psi_algo else None

        self.check_psi_algo(psi_task_handler, fl_ctx)
        psi_task_handler.initialize(fl_ctx)
        return psi_task_handler

    def check_psi_algo(self, psi_task_handler: TaskHandler, fl_ctx):
        if not psi_task_handler:
            self.log_error(fl_ctx, f"PSI algorithm specified by {self.psi_algo_id} is not implemented")
            raise NotImplementedError

        check_component_type(psi_task_handler, TaskHandler)
