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
from typing import Dict, List, NamedTuple

from nvflare.apis.dxo import DXO
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import PSIConst
from nvflare.app_common.psi.psi_control_handler_spec import PsiControlHandler
from nvflare.app_common.workflows.broadcast_operator import BroadcastAndWait


class SiteSize(NamedTuple):
    name: str
    size: int


class DHPsiControlHandler(PsiControlHandler):
    def __init__(self):
        super().__init__()
        self.abort_signal = None
        self.fl_ctx = None
        self.task_name = PSIConst.PSI_TASK
        self.wait_time_after_min_received = 0
        self.controller = None
        self.ordered_sites: List[SiteSize] = []
        self.forward_processed = {}
        self.backward_processed = {}

    def initialize(self, fl_ctx: FLContext, **kwargs):
        self.fl_ctx = fl_ctx
        self.controller = kwargs["controller"]

    def pre_workflow(self, abort_signal: Signal) -> bool:
        # ask client send back their item sizes
        # sort client by ascending order and select the 1st site based on size (smallest size)
        # todo:
        # ask client if he already has the intersection,
        # if it has, it means the FORWARD PATH intersection has already done at this point before failure
        # so that we can resume were left off. If all clients have intersection calculated, will continue with next step.
        #
        if abort_signal:
            return False
        self.abort_signal = abort_signal

        inputs = Shareable()
        inputs[PSIConst.PSI_TASK_KEY] = PSIConst.PSI_TASK_PREPARE
        task_props = {PSIConst.PSI_TASK_KEY: PSIConst.PSI_TASK_PREPARE}
        targets = None
        min_responses = len(self.fl_ctx.clients())
        bop = BroadcastAndWait(self.fl_ctx, self.controller)
        results = bop.broadcast_and_wait(self.task_name, task_props, inputs, targets, min_responses, abort_signal)
        self.ordered_sites = self.get_ordered_sites(results)

    def workflow(self, abort_signal: Signal):
        if abort_signal:
            return False

        self.abort_signal = abort_signal
        self.forward_processed.update(self.forward_pass(self.ordered_sites))
        self.backward_processed.update(self.backward_pass(self.ordered_sites))

    def post_workflow(self, abort_signal: Signal):
        pass

    def finalize(self):
        pass

    def get_ordered_sites(self, results: Dict[str, DXO]):
        def compare_fn(e):
            return c.size

        site_sizes = []
        for site_name in results:
            data = results[site_name].data
            if PSIConst.PSI_ITEM_SIZE in data:
                size = data[PSIConst.PSI_ITEM_SIZE]
            else:
                size = 0
            if size > 0:
                c = SiteSize(site_name, size)
                site_sizes.append(c)

            site_sizes.sort(key=compare_fn)

        return site_sizes

    def forward_pass(self, ordered_sites: List[SiteSize], reverse=False) -> dict:
        processed = {}
        if self.abort_signal:
            return processed

        #   FORWARD PASS
        #   for each client pair selected from sorted clients
        #   for two clients:
        #     step 1:  ask  C1 (PsiServer) prepare setup message, FL server receive result
        #     step 2:  send C2 C1's setup message and C2 return with it's request
        #     step 3:  send C1 (PsiServer) C2's request, and C1 process the request and send back the response
        #     step 4:  send C2 (PsiClient) C1's response and C2 compute intersect
        #
        #   for next two clients: C2, C3
        #      Step 1: select C2 as PsiServer, and prepare setup message, FL server will receive setup message
        #      Step 2: repeat the same steps for C2, C3

        total_sites = len(ordered_sites)
        if total_sites <= 1:
            return processed

        start, end, step = self.get_directional_range(total_sites, reverse)
        for i in range(start, end, step):
            s = ordered_sites[i]
            c = ordered_sites[i + 1]
            setup_msg = self.prepare_setup_message(s, c)
            request = self.prepare_request(c, setup_msg)
            response = self.process_request(s, c.name, request[c.name])
            status = self.calculate_intersection(c, response)
            processed.update(status)

        return processed

    def get_directional_range(self, total: int, reverse: bool = False):
        if reverse:
            start = total - 1
            end = -1
            step = -1
        else:
            start = 0
            end = total
            step = 1

        return start, end, step

    def backward_pass(self, ordered_clients: list) -> dict:
        processed = {}
        if self.abort_signal:
            return processed

        #   BACKWARD STEPS
        #   Once we exhausted all clients, we starts the Backward Path, with the last node as the PsiServer
        #   all other nodes as PsiClients
        #      Step 1: last Cn (PsiServer) prepare setup message, FL server receive setup message
        #      step 2: broadcast the Cn's setup message to all other clients and all clients will send back requests
        #      step 3: broadcast other clients' requests to Cn, Cn will process the requests and send response for each
        #      step 4: broadcast Cn's response to all other clients, and all other clients will compute intersects
        #      and save the result locally

        total_clients = len(ordered_clients)
        if total_clients <= 1:
            return processed

        if len(self.forward_processed) == total_clients:
            # Sequential version
            return self.forward_pass(ordered_clients, reverse=True)

            # todo parallel version
            # cn = self.ordered_sites[total_clients - 1]
            # others = [x for x in self.ordered_sites if x.name != cn.name]
            # if self.forward_processed[cn.name] == PSIConst.PSI_STATUS_DONE:
            #     setup_msg = self.prepare_setup_message(cn, )
            #     requests = self.prepare_requests(others, setup_msg)
            #     responses = self.process_request(cn, list(requests.items())[0])
            #     intersections = self.calculate_intersections(others, responses)
        else:
            raise ValueError("started backward pass before finish the forward pass")

    def prepare_setup_message(self, s: SiteSize, c: SiteSize) -> dict:
        inputs = Shareable()
        inputs[PSIConst.PSI_TASK_KEY] = PSIConst.PSI_TASK_SETUP
        inputs[PSIConst.PSI_ITEM_SIZE] = c.size

        task_props = {PSIConst.PSI_DIRECTION_KEY: PSIConst.PSI_FORWARD}
        targets = [s.name]

        min_responses = 1
        bop = BroadcastAndWait(self.fl_ctx, self.controller)
        results = bop.broadcast_and_wait(self.task_name, task_props, inputs, targets, min_responses, self.abort_signal)

        dxo = results[s.name]
        setup_msg = dxo.data[PSIConst.PSI_SETUP_MSG]
        return {c.name: setup_msg}

    def prepare_request(self, c: SiteSize, setup_msg: dict) -> dict:
        inputs = Shareable()
        inputs[PSIConst.PSI_TASK_KEY] = PSIConst.PSI_TASK_REQUEST
        inputs[PSIConst.PSI_SETUP_MSG] = setup_msg[c.name]

        task_props = {PSIConst.PSI_DIRECTION_KEY: PSIConst.PSI_FORWARD}
        targets = [c.name]

        min_responses = 1
        bop = BroadcastAndWait(self.fl_ctx, self.controller)
        results = bop.broadcast_and_wait(self.task_name, task_props, inputs, targets, min_responses, self.abort_signal)

        dxo = results[c.name]
        request_msg = dxo.data[PSIConst.PSI_REQUEST_MSG]
        return {c.name: request_msg}

    def process_request(self, s: SiteSize, c_name: str, response_msg: str):
        inputs = Shareable()
        inputs[PSIConst.PSI_TASK_KEY] = PSIConst.PSI_TASK_RESPONSE
        inputs[PSIConst.PSI_REQUEST_MSG] = response_msg
        task_props = {PSIConst.PSI_DIRECTION_KEY: PSIConst.PSI_FORWARD}
        targets = [s.name]

        min_responses = 1
        bop = BroadcastAndWait(self.fl_ctx, self.controller)
        results = bop.broadcast_and_wait(self.task_name, task_props, inputs, targets, min_responses, self.abort_signal)
        dxo = results[s.name]
        response_msg = dxo.data[PSIConst.PSI_RESPONSE_MSG]
        return {c_name: response_msg}

    def calculate_intersection(self, c: SiteSize, response: dict):
        inputs = Shareable()
        inputs[PSIConst.PSI_TASK_KEY] = PSIConst.PSI_TASK_INTERSECT
        inputs[PSIConst.PSI_RESPONSE_MSG] = response[c.name]
        task_props = {PSIConst.PSI_DIRECTION_KEY: PSIConst.PSI_FORWARD}
        targets = [c.name]
        min_responses = 1
        bop = BroadcastAndWait(self.fl_ctx, self.controller)
        results = bop.broadcast_and_wait(self.task_name, task_props, inputs, targets, min_responses, self.abort_signal)
        dxo = results[c.name]
        status = dxo.data[PSIConst.PSI_STATUS]
        return {c.name: status}
