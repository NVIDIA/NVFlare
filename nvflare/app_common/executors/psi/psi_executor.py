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
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.app_constant import PSIConst
from nvflare.app_common.executors.client_executor import ClientExecutor
from nvflare.app_common.psi.psi_exec_handler_spec import PsiExecutorHandler
from nvflare.app_common.psi.psi_spec import PSI
from nvflare.app_common.utils.component_utils import check_component_type


class PSIExecutor(ClientExecutor):
    """
    PSIExecutor is handles the communication and FLARE server task delegation
    User will interface local component : PSI to provide client items and  get intersection
    The actual is done by PsiHandler, which provide receiving setup, calculate intersection from response.
    If there are different PSI algorithm, one can register a different PsiHandler.
    """

    def __init__(self, local_psi_id: str, psi_handler_id: str):
        super().__init__(local_psi_id, PSI)
        self.psi_handler_id = psi_handler_id

    def initialize(self, fl_ctx: FLContext):
        super().initialize(fl_ctx)

        engine = fl_ctx.get_engine()
        psi_handler: PsiExecutorHandler = engine.get_component(self.local_comp_id)
        psi_handler.initialize(fl_ctx)
        check_component_type(psi_handler, PsiExecutorHandler)

    def client_exec(self, task_name: str, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        client_name = fl_ctx.get_identity_name()
        self.log_info(fl_ctx, f"Executing task '{task_name}' for client: '{client_name}'")
        if PSIConst.PSI_TASK == task_name:
            # todo:
            # todo:
            # todo:
            pass
        else:
            raise RuntimeError(ReturnCode.TASK_UNKNOWN)
