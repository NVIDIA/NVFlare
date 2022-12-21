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
from nvflare.apis.dxo import DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.executors.client_executor import ClientExecutor
from nvflare.app_common.executors.common_executor import CommonExecutor
from nvflare.app_common.executors.psi.dh_psi_executor import DhPSIExecutor


class PSIExecutor(CommonExecutor):
    def __init__(self, local_psi_id: str, psi_algo: str = "open_mined_psi"):
        super().__init__()
        self.local_psi_id = local_psi_id
        self.psi_algo = psi_algo
        self.psi_algo_impls = {"open_mined_psi": DhPSIExecutor(self.local_psi_id)}

    def get_data_kind(self) -> str:
        return DataKind.PSI

    def get_client_executor(self, fl_ctx: FLContext) -> ClientExecutor:
        self.log_info(fl_ctx, "get_client_executor")
        client_executor = self.psi_algo_impls.get(self.psi_algo, None)
        self.check_psi_algo(fl_ctx)
        client_executor.initialize(fl_ctx)
        self.log_info(fl_ctx, "return client_executor")
        return client_executor

    def check_psi_algo(self, fl_ctx):
        if not ClientExecutor:
            self.log_error(fl_ctx, f"PSI algorithm specified by {self.psi_algo} is not implemented")
            raise NotImplementedError
