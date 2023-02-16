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
from typing import Optional

from nvflare.apis.dxo import DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.executors.client_executor import ClientExecutor
from nvflare.app_common.executors.common_executor import CommonExecutor
from nvflare.app_common.utils.component_utils import check_component_type
from nvflare.app_opt.psi.dh_psi.dh_psi_executor import DhPSIExecutor


class PSIExecutor(CommonExecutor):
    def __init__(self, local_psi_id: str, psi_algo: Optional[str] = None):
        super().__init__()
        self.local_psi_id = local_psi_id
        self.psi_algo = psi_algo

    def get_data_kind(self) -> str:
        return DataKind.PSI

    def get_client_executor(self, fl_ctx: FLContext) -> ClientExecutor:
        return self.load_client_executor(self.psi_algo, fl_ctx)

    def load_client_executor(self, psi_algo: str, fl_ctx: FLContext) -> ClientExecutor:
        engine = fl_ctx.get_engine()
        psi_client_executor = engine.get_component(psi_algo) if psi_algo else None
        if not psi_client_executor:
            # use default
            psi_client_executor = DhPSIExecutor(self.local_psi_id)

        self.check_psi_algo(psi_client_executor, fl_ctx)
        psi_client_executor.initialize(fl_ctx)
        return psi_client_executor

    def check_psi_algo(self, psi_client_exec: ClientExecutor, fl_ctx):
        if not psi_client_exec:
            self.log_error(fl_ctx, f"PSI algorithm specified by {self.psi_algo} is not implemented")
            raise NotImplementedError

        check_component_type(psi_client_exec, ClientExecutor)
