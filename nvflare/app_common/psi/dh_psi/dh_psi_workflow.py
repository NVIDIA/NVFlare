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

from typing import Dict, List, NamedTuple, Set

from nvflare.apis.dxo import DXO
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import PSIConst
from nvflare.app_common.psi.psi_workflow_spec import PSIWorkflow
from nvflare.app_common.workflows.broadcast_operator import BroadcastAndWait
from nvflare.utils.decorators import measure_time


class SiteSize(NamedTuple):
    name: str
    size: int


class DhPSIWorkFlow(PSIWorkflow):
    def __init__(self, bloom_filter_fpr: float = 1e-11):
        super().__init__()
        self.task_name = PSIConst.TASK
        self.bloom_filter_fpr: float = bloom_filter_fpr
        self.wait_time_after_min_received = 0
        self.abort_signal = None
        self.fl_ctx = None
        self.controller = None
        self.ordered_sites: List[SiteSize] = []
        self.forward_processed: Dict[str, int] = {}
        self.backward_processed: Dict[str, int] = {}

    def initialize(self, fl_ctx: FLContext, **kwargs):
        self.fl_ctx = fl_ctx
        self.controller = kwargs["controller"]

    def pre_process(self, abort_signal: Signal) -> bool:
        # ask client send back their item sizes
        # sort client by ascending order
        self.log_info(self.fl_ctx, f"pre_process on task {self.task_name}")

        if abort_signal.triggered:
            return False
        self.abort_signal = abort_signal
        self.prepare_sites(abort_signal)

    def run(self, abort_signal: Signal):
        if abort_signal.triggered:
            return False

        self.abort_signal = abort_signal
        self.log_info(self.fl_ctx, f"order sites = {self.ordered_sites}")

        intersect_site = self.forward_pass(self.ordered_sites, self.forward_processed)

        self.log_info(
            self.fl_ctx,
            f"forward_processed sites {self.forward_processed}\n,"
            f"intersect_sites={intersect_site}\n"
            f"ordered sites = {self.ordered_sites}\n",
        )

        self.check_processed_sites(intersect_site, self.forward_processed)

        self.backward_processed.update(self.backward_pass(self.ordered_sites, intersect_site))

        self.log_info(
            self.fl_ctx,
            f"backward_processed sites {self.backward_processed}\n,"
            f"intersect_sites={intersect_site}\n"
            f"ordered sites = {self.ordered_sites}\n",
        )

        self.check_final_intersection_sizes(intersect_site)

        self.log_pass_time_taken()

    def check_processed_sites(self, last_site: SiteSize, processed_sites: Dict[str, int]):
        valid = all(value >= last_site.size for value in processed_sites.values())
        if not valid:
            raise RuntimeError(
                f"Intersection calculation failed:\n"
                f"processed sites :{processed_sites},\n"
                f"last_site  ={last_site} \n"
                f"ordered sites = {self.ordered_sites} \n"
            )

    def check_final_intersection_sizes(self, intersect_site: SiteSize):
        all_equal = all(value == intersect_site.size for value in self.backward_processed.values())
        if not all_equal:
            raise RuntimeError(
                f"Intersection calculation failed: the intersection sizes from all sites must be equal.\n"
                f"backward processed sites:{self.backward_processed},\n"
                f"intersect sites ={intersect_site} \n"
                f"ordered sites = {self.ordered_sites} \n"
            )
        else:
            self.log_info(self.fl_ctx, "Intersection calculation succeed")

    def log_pass_time_taken(self):
        self.log_info(self.fl_ctx, f"'forward_pass' took {self.forward_pass.time_taken} ms.")
        self.log_info(self.fl_ctx, f"'backward_pass' took {self.backward_pass.time_taken} ms.")

    def post_process(self, abort_signal: Signal):
        pass

    def finalize(self, fl_ctx: FLContext):
        pass

    @staticmethod
    def get_ordered_sites(results: Dict[str, DXO]):
        def compare_fn(e):
            return e.size

        site_sizes = []
        for site_name in results:
            data = results[site_name].data
            if PSIConst.ITEMS_SIZE in data:
                size = data[PSIConst.ITEMS_SIZE]
            else:
                size = 0

            if size > 0:
                c = SiteSize(site_name, size)
                site_sizes.append(c)
        site_sizes.sort(key=compare_fn)
        return site_sizes

    @measure_time
    def forward_pass(self, ordered_sites: List[SiteSize], processed: Dict[str, int]) -> SiteSize:
        if self.abort_signal.triggered:
            return ordered_sites[0]

        total_sites = len(ordered_sites)
        if total_sites <= 1:
            return ordered_sites[0]

        return self.parallel_forward_pass(ordered_sites, processed)

    def pairwise_setup(self, ordered_sites: List[SiteSize]):
        total_sites = len(ordered_sites)
        n = int(total_sites / 2)
        task_inputs = {}
        for i in range(n):
            s = ordered_sites[i]
            c = ordered_sites[i + n]
            inputs = Shareable()
            inputs[PSIConst.TASK_KEY] = PSIConst.TASK_SETUP
            inputs[PSIConst.ITEMS_SIZE] = c.size
            task_inputs[s.name] = inputs

        bop = BroadcastAndWait(self.fl_ctx, self.controller)
        results = bop.multicasts_and_wait(
            task_name=self.task_name, task_inputs=task_inputs, abort_signal=self.abort_signal
        )
        return {site_name: results[site_name].data[PSIConst.SETUP_MSG] for site_name in results}

    def pairwise_requests(self, ordered_sites: List[SiteSize], setup_msgs: Dict[str, str]):
        total_sites = len(ordered_sites)
        n = int(total_sites / 2)
        task_inputs = {}
        for i in range(n):
            s = ordered_sites[i]
            c = ordered_sites[i + n]
            inputs = Shareable()
            inputs[PSIConst.TASK_KEY] = PSIConst.TASK_REQUEST
            inputs[PSIConst.SETUP_MSG] = setup_msgs[s.name]
            task_inputs[c.name] = inputs

        bop = BroadcastAndWait(self.fl_ctx, self.controller)
        results = bop.multicasts_and_wait(
            task_name=self.task_name, task_inputs=task_inputs, abort_signal=self.abort_signal
        )
        return {site_name: results[site_name].data[PSIConst.REQUEST_MSG] for site_name in results}

    def pairwise_responses(self, ordered_sites: List[SiteSize], request_msgs: Dict[str, str]):
        total_sites = len(ordered_sites)
        n = int(total_sites / 2)
        task_inputs = {}
        for i in range(n):
            s = ordered_sites[i]
            c = ordered_sites[i + n]
            inputs = Shareable()
            inputs[PSIConst.TASK_KEY] = PSIConst.TASK_RESPONSE
            inputs[PSIConst.REQUEST_MSG] = request_msgs[c.name]
            task_inputs[s.name] = inputs

        bop = BroadcastAndWait(self.fl_ctx, self.controller)
        results = bop.multicasts_and_wait(
            task_name=self.task_name, task_inputs=task_inputs, abort_signal=self.abort_signal
        )
        return {site_name: results[site_name].data[PSIConst.RESPONSE_MSG] for site_name in results}

    def pairwise_intersect(self, ordered_sites: List[SiteSize], response_msg: Dict[str, str]):
        total_sites = len(ordered_sites)
        n = int(total_sites / 2)
        task_inputs = {}
        for i in range(n):
            s = ordered_sites[i]
            c = ordered_sites[i + n]
            inputs = Shareable()
            inputs[PSIConst.TASK_KEY] = PSIConst.TASK_INTERSECT
            inputs[PSIConst.RESPONSE_MSG] = response_msg[s.name]
            task_inputs[c.name] = inputs

        bop = BroadcastAndWait(self.fl_ctx, self.controller)
        results = bop.multicasts_and_wait(
            task_name=self.task_name, task_inputs=task_inputs, abort_signal=self.abort_signal
        )
        return {site_name: results[site_name].data[PSIConst.ITEMS_SIZE] for site_name in results}

    def parallel_forward_pass(self, target_sites, processed: dict):
        self.log_info(self.fl_ctx, f"target_sites: {target_sites}")
        total_sites = len(target_sites)
        if total_sites < 2:
            final_site = target_sites[0]
            processed.update({final_site.name: final_site.size})
            return final_site
        else:
            setup_msgs = self.pairwise_setup(target_sites)
            request_msgs = self.pairwise_requests(target_sites, setup_msgs)
            response_msgs = self.pairwise_responses(target_sites, request_msgs)
            it_sites = self.pairwise_intersect(target_sites, response_msgs)
            processed.update(it_sites)
            new_targets = [SiteSize(site.name, it_sites[site.name]) for site in target_sites if site.name in it_sites]
            if total_sites % 2 == 1:
                new_targets.append(target_sites[total_sites - 1])

            return self.parallel_forward_pass(new_targets, processed)

    @measure_time
    def backward_pass(self, ordered_clients: list, intersect_site: SiteSize) -> dict:
        processed = {}
        if self.abort_signal.triggered:
            return processed

        total_clients = len(ordered_clients)
        if total_clients <= 1:
            return processed
        status = self.parallel_backward_pass(ordered_clients, intersect_site)

        time_taken = self.parallel_backward_pass.time_taken
        self.log_info(self.fl_ctx, f"parallel_back_pass took {time_taken} (ms)")
        return status

    @measure_time
    def parallel_backward_pass(self, ordered_clients: list, intersect_site: SiteSize):
        # parallel version
        other_sites = [site for site in ordered_clients if site.name != intersect_site.name]
        other_sites = self.get_updated_site_sizes(other_sites)

        s = intersect_site
        other_site_sizes = set([site.size for site in other_sites])
        setup_msgs: Dict[str, str] = self.prepare_setup_messages(s, other_site_sizes)

        site_setup_msgs = {site.name: setup_msgs[str(site.size)] for site in other_sites}
        request_msgs: Dict[str, str] = self.create_requests(site_setup_msgs)
        response_msgs: Dict[str, str] = self.process_requests(s, request_msgs)
        return self.calculate_intersections(response_msgs)

    def calculate_intersections(self, response_msg) -> Dict[str, int]:
        task_inputs = {}
        for client_name in response_msg:
            inputs = Shareable()
            inputs[PSIConst.TASK_KEY] = PSIConst.TASK_INTERSECT
            inputs[PSIConst.RESPONSE_MSG] = response_msg[client_name]
            task_inputs[client_name] = inputs
        bop = BroadcastAndWait(self.fl_ctx, self.controller)
        results = bop.multicasts_and_wait(
            task_name=self.task_name, task_inputs=task_inputs, abort_signal=self.abort_signal
        )

        intersects = {client_name: results[client_name].data[PSIConst.ITEMS_SIZE] for client_name in results}
        self.log_info(self.fl_ctx, f"received intersections : {intersects} ")
        return intersects

    def process_requests(self, s: SiteSize, request_msgs: Dict[str, str]) -> Dict[str, str]:
        task_inputs = Shareable()
        task_inputs[PSIConst.TASK_KEY] = PSIConst.TASK_RESPONSE
        task_inputs[PSIConst.REQUEST_MSG_SET] = request_msgs
        bop = BroadcastAndWait(self.fl_ctx, self.controller)
        results = bop.broadcast_and_wait(
            task_name=self.task_name, task_input=task_inputs, targets=[s.name], abort_signal=self.abort_signal
        )

        dxo = results[s.name]
        response_msgs = dxo.data[PSIConst.RESPONSE_MSG]
        return response_msgs

    def create_requests(self, site_setup_msgs) -> Dict[str, str]:
        task_inputs = {}
        for client_name in site_setup_msgs:
            inputs = Shareable()
            inputs[PSIConst.TASK_KEY] = PSIConst.TASK_REQUEST
            inputs[PSIConst.SETUP_MSG] = site_setup_msgs[client_name]
            task_inputs[client_name] = inputs

        bop = BroadcastAndWait(self.fl_ctx, self.controller)
        results = bop.multicasts_and_wait(
            task_name=self.task_name, task_inputs=task_inputs, abort_signal=self.abort_signal
        )
        request_msgs = {client_name: results[client_name].data[PSIConst.REQUEST_MSG] for client_name in results}
        return request_msgs

    def get_updated_site_sizes(self, ordered_sites):
        updated_sites = []
        for site in ordered_sites:
            new_size = self.forward_processed.get(site.name, site.size)
            updated_sites.append(SiteSize(site.name, new_size))

        return updated_sites

    def prepare_sites(self, abort_signal):

        inputs = Shareable()
        inputs[PSIConst.TASK_KEY] = PSIConst.TASK_PREPARE
        inputs[PSIConst.BLOOM_FILTER_FPR] = self.bloom_filter_fpr
        targets = None
        engine = self.fl_ctx.get_engine()
        min_responses = len(engine.get_clients())

        bop = BroadcastAndWait(self.fl_ctx, self.controller)
        results = bop.broadcast_and_wait(
            task_name=self.task_name,
            task_input=inputs,
            targets=targets,
            min_responses=min_responses,
            abort_signal=abort_signal,
        )
        self.log_info(self.fl_ctx, f"{PSIConst.TASK_PREPARE} results = {results}")
        if not results:
            abort_signal.trigger("no items to perform PSI")
            raise RuntimeError("There is no item to perform PSI calculation")
        else:
            self.ordered_sites = self.get_ordered_sites(results)

    def prepare_setup_messages(self, s: SiteSize, other_site_sizes: Set[int]) -> Dict[str, str]:
        inputs = Shareable()
        inputs[PSIConst.TASK_KEY] = PSIConst.TASK_SETUP
        inputs[PSIConst.ITEMS_SIZE_SET] = other_site_sizes
        bop = BroadcastAndWait(self.fl_ctx, self.controller)
        results = bop.broadcast_and_wait(
            task_name=self.task_name, task_input=inputs, targets=[s.name], abort_signal=self.abort_signal
        )
        dxo = results[s.name]
        return dxo.data[PSIConst.SETUP_MSG]
