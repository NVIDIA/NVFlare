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
from typing import List, Optional

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.app_constant import PSIConst
from nvflare.app_common.executors.client_executor import ClientExecutor
from nvflare.app_common.psi.dh_psi.dh_psi_client import PsiClient
from nvflare.app_common.psi.dh_psi.dh_psi_server import PsiServer
from nvflare.app_common.psi.psi_spec import PSI


class DhPSIExecutor(ClientExecutor):
    """
    DhPSIExecutor is the executor for Diffie-Hellman-based Algorithm PSI.It handles the communication and FLARE server task delegation
    User will interface local component : PSI to provide client items and  get intersection
    """

    def __init__(self, local_psi_id: str):
        super().__init__(local_psi_id, PSI)
        self.bloom_filter_fpr = None
        self.psi_client = None
        self.psi_server = None
        self.intersects: Optional[List[str]] = None
        self.local_psi_handler: Optional[PSI] = None

    def initialize(self, fl_ctx: FLContext):
        super().initialize(fl_ctx)
        self.local_psi_handler = self.local_comp

    def client_exec(self, task_name: str, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        client_name = fl_ctx.get_identity_name()
        self.log_info(fl_ctx, f"Executing task '{task_name}' for {client_name}")

        if PSIConst.PSI_TASK == task_name:
            psi_stage_task = shareable.get(PSIConst.PSI_TASK_KEY)
            self.log_info(fl_ctx, f"Executing psi_stage_task {psi_stage_task} for {client_name}")

            if psi_stage_task == PSIConst.PSI_TASK_PREPARE:
                self.bloom_filter_fpr = shareable[PSIConst.PSI_BLOOM_FILTER_FPR]
                items = self.get_items()
                print("**********************", client_name, items)
                self.psi_client = PsiClient(items)
                self.psi_server = PsiServer(items, self.bloom_filter_fpr)
                return self.get_items_size()
            else:
                if psi_stage_task == PSIConst.PSI_TASK_SETUP:
                    return self.setup(shareable)
                elif psi_stage_task == PSIConst.PSI_TASK_REQUEST:
                    return self.create_request(shareable)
                elif psi_stage_task == PSIConst.PSI_TASK_RESPONSE:
                    return self.process_request(shareable)
                elif psi_stage_task == PSIConst.PSI_TASK_INTERSECT:
                    return self.calculate_intersection(shareable)
        else:
            raise RuntimeError(ReturnCode.TASK_UNKNOWN)

    def create_request(self, shareable: Shareable):
        setup_msg = shareable.get(PSIConst.PSI_SETUP_MSG)
        self.psi_client.receive_setup(setup_msg)
        request = self.psi_client.get_request()
        result = Shareable()
        result[PSIConst.PSI_REQUEST_MSG] = request
        return result

    def setup(self, shareable: Shareable):
        items = self.get_items()
        print("**********************", items)
        self.psi_client = PsiClient(items)
        self.psi_server = PsiServer(items, self.bloom_filter_fpr)

        if PSIConst.PSI_ITEMS_SIZE in shareable:
            target_item_size = shareable.get(PSIConst.PSI_ITEMS_SIZE)
            setup_msg = self.psi_server.setup(target_item_size)
            result = Shareable()
            result[PSIConst.PSI_SETUP_MSG] = setup_msg
            return result
        elif PSIConst.PSI_ITEMS_SIZE_SET in shareable:
            target_item_size_set = shareable.get(PSIConst.PSI_ITEMS_SIZE_SET)
            result = Shareable()
            setup_sets = {}
            for client_iterm_size in target_item_size_set:
                setup_msg = self.psi_server.setup(client_iterm_size)
                setup_sets[str(client_iterm_size)] = setup_msg

            result[PSIConst.PSI_SETUP_MSG] = setup_sets
            return result

    def get_items_size(self):
        result = Shareable()
        result[PSIConst.PSI_ITEMS_SIZE] = len(self.get_items())
        return result

    def process_request(self, shareable: Shareable):
        request_msg = shareable.get(PSIConst.PSI_REQUEST_MSG)
        response = self.psi_server.process_request(request_msg)
        result = Shareable()
        result[PSIConst.PSI_RESPONSE_MSG] = response
        return result

    def calculate_intersection(self, shareable: Shareable):
        response_msg = shareable.get(PSIConst.PSI_RESPONSE_MSG)
        intersections = self.psi_client.get_intersection(response_msg)
        self.intersects = intersections
        client_name = self.fl_ctx.get_identity_name()
        print(f"******** client {client_name}, intersections = {intersections}")
        self.local_psi_handler.save(intersections)
        result = Shareable()
        result[PSIConst.PSI_ITEMS_SIZE] = len(intersections)
        return result

    def get_items(self):

        if not self.intersects:
            return self.local_psi_handler.load_items()
        else:
            return self.intersects
