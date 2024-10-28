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


import collections
from typing import List, Optional

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.task_handler import TaskHandler
from nvflare.app_common.app_constant import PSIConst
from nvflare.app_common.psi.psi_spec import PSI
from nvflare.app_opt.psi.dh_psi.dh_psi_client import PSIClient
from nvflare.app_opt.psi.dh_psi.dh_psi_server import PSIServer


def check_items_uniqueness(items):
    duplicates = {item: count for item, count in collections.Counter(items).items() if count > 1}
    if duplicates:
        raise ValueError(f"the items must be unique, the following items with duplicates {duplicates}")


class DhPSITaskHandler(TaskHandler):
    """Executor for Diffie-Hellman-based Algorithm PSI.

    It handles the communication and FLARE server task delegation
    User will write an interface local component : PSI to provide client items and  get intersection
    """

    def __init__(self, local_psi_id: str):
        super().__init__(local_psi_id, PSI)
        self.bloom_filter_fpr = None
        self.psi_client = None
        self.psi_server = None
        self.intersects: Optional[List[str]] = None
        self.local_psi_handler: Optional[PSI] = None
        self.client_name = None
        self.items = None

    def initialize(self, fl_ctx: FLContext):
        super().initialize(fl_ctx)
        self.local_psi_handler = self.local_comp

    def execute_task(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        client_name = fl_ctx.get_identity_name()
        self.client_name = client_name
        self.log_info(fl_ctx, f"Executing task '{task_name}' for {client_name}")

        if PSIConst.TASK == task_name:
            psi_stage_task = shareable.get(PSIConst.TASK_KEY)
            self.log_info(fl_ctx, f"Executing psi_stage_task {psi_stage_task} for {client_name}")

            if psi_stage_task == PSIConst.TASK_PREPARE:
                self.bloom_filter_fpr = shareable[PSIConst.BLOOM_FILTER_FPR]
                items = self.get_items()
                self.psi_client = PSIClient(items)
                self.psi_server = PSIServer(items, self.bloom_filter_fpr)
                return self.get_items_size()
            else:
                if psi_stage_task == PSIConst.TASK_SETUP:
                    return self.setup(shareable, client_name)
                elif psi_stage_task == PSIConst.TASK_REQUEST:
                    return self.create_request(shareable)
                elif psi_stage_task == PSIConst.TASK_RESPONSE:
                    return self.process_request(shareable)
                elif psi_stage_task == PSIConst.TASK_INTERSECT:
                    return self.calculate_intersection(shareable)
        else:
            raise RuntimeError(ReturnCode.TASK_UNKNOWN)

    def create_request(self, shareable: Shareable):
        setup_msg = shareable.get(PSIConst.SETUP_MSG)
        self.psi_client.receive_setup(setup_msg)
        request = self.psi_client.get_request(self.get_items())
        result = Shareable()
        result[PSIConst.REQUEST_MSG] = request
        return result

    def setup(self, shareable: Shareable, client_name: str):
        items = self.get_items()
        if len(items) == 0:
            raise RuntimeError(f"site {client_name} doesn't have any items for to perform PSI")

        # note, each interaction with client requires a new client,server keys to be secure.
        self.psi_client = PSIClient(items)
        self.psi_server = PSIServer(items, self.bloom_filter_fpr)

        if PSIConst.ITEMS_SIZE in shareable:
            target_item_size = shareable.get(PSIConst.ITEMS_SIZE)
            setup_msg = self.psi_server.setup(target_item_size)
            result = Shareable()
            result[PSIConst.SETUP_MSG] = setup_msg
            return result
        elif PSIConst.ITEMS_SIZE_SET in shareable:
            target_item_size_set = shareable.get(PSIConst.ITEMS_SIZE_SET)
            result = Shareable()
            setup_sets = {}
            for client_iterm_size in target_item_size_set:
                setup_msg = self.psi_server.setup(client_iterm_size)
                setup_sets[str(client_iterm_size)] = setup_msg

            result[PSIConst.SETUP_MSG] = setup_sets
            return result

    def get_items_size(self):
        result = Shareable()
        result[PSIConst.ITEMS_SIZE] = len(self.get_items())
        return result

    def process_request(self, shareable: Shareable):
        if PSIConst.REQUEST_MSG in shareable:
            request_msg = shareable.get(PSIConst.REQUEST_MSG)
            response = self.psi_server.process_request(request_msg)
            result = Shareable()
            result[PSIConst.RESPONSE_MSG] = response
            return result
        elif PSIConst.REQUEST_MSG_SET in shareable:
            request_msgs = shareable.get(PSIConst.REQUEST_MSG_SET)
            result = Shareable()
            client_responses = {}
            for client_name in request_msgs:
                response = self.psi_server.process_request(request_msgs[client_name])
                client_responses[client_name] = response
            result[PSIConst.RESPONSE_MSG] = client_responses
        else:
            raise ValueError(
                "Required PSI Message PSIConst.PSI_REQUEST_MSG or PSIConst.PSI_REQUEST_MSG_SET is not provided"
            )

        return result

    def calculate_intersection(self, shareable: Shareable):
        response_msg = shareable.get(PSIConst.RESPONSE_MSG)
        intersections = self.psi_client.get_intersection(response_msg)
        self.intersects = intersections
        self.local_psi_handler.save(intersections)
        result = Shareable()
        result[PSIConst.ITEMS_SIZE] = len(intersections)
        return result

    def get_items(self):
        if not self.intersects:
            if self.items is None:
                items = self.local_psi_handler.load_items()
                check_items_uniqueness(items)
                self.items = items
        else:
            self.items = self.intersects

        return self.items
