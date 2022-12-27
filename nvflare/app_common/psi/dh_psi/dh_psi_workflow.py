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
from typing import Dict, List, NamedTuple, Set

from nvflare.apis.dxo import DXO
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import PSIConst
from nvflare.app_common.psi.psi_workflow_spec import PSIWorkflow
from nvflare.app_common.workflows.broadcast_operator import BroadcastAndWait
from nvflare.utils.decorators import collect_time, measure_time


class SiteSize(NamedTuple):
    name: str
    size: int


class DhPSIWorkFlow(PSIWorkflow):
    def __init__(self, bloom_filter_fpr: float = 1e-11):
        super().__init__()
        self.task_name = PSIConst.PSI_TASK
        self.bloom_filter_fpr: float = bloom_filter_fpr
        self.wait_time_after_min_received = 0
        self.abort_signal = None
        self.fl_ctx = None
        self.controller = None
        self.ordered_sites: List[SiteSize] = []
        self.forward_processed = {}
        self.backward_processed = {}

    def initialize(self, fl_ctx: FLContext, **kwargs):
        self.fl_ctx = fl_ctx
        self.controller = kwargs["controller"]

    def pre_workflow(self, abort_signal: Signal) -> bool:
        # ask client send back their item sizes
        # sort client by ascending order
        self.log_info(self.fl_ctx, f"pre_workflow on task {self.task_name}")

        if abort_signal.triggered:
            return False
        self.abort_signal = abort_signal
        self.prepare_sites(PSIConst.PSI_FORWARD, abort_signal)

    def workflow(self, abort_signal: Signal):
        if abort_signal.triggered:
            return False

        self.abort_signal = abort_signal

        self.log_info(self.fl_ctx, f"order sites = {self.ordered_sites}")

        self.forward_pass(self.ordered_sites, self.forward_processed)
        self.backward_processed.update(self.backward_pass(self.ordered_sites))
        self.log_pass_time_taken()

    def log_pass_time_taken(self):
        self.log_info(self.fl_ctx, f"'********* forward_pass' took {self.forward_pass.time_taken} ms.")
        self.log_info(self.fl_ctx, f"'********* backward_pass' took {self.backward_pass.time_taken} ms.")

    def post_workflow(self, abort_signal: Signal):
        pass

    def finalize(self):
        pass

    def get_ordered_sites(self, results: Dict[str, DXO]):

        def compare_fn(e):
            return e.size

        site_sizes = []
        for site_name in results:
            data = results[site_name].data
            if PSIConst.PSI_ITEMS_SIZE in data:
                size = data[PSIConst.PSI_ITEMS_SIZE]
            else:
                size = 0

            self.log_info(self.fl_ctx, f"****** site:{site_name}, size = {size}")

            if size > 0:
                c = SiteSize(site_name, size)
                site_sizes.append(c)
        site_sizes.sort(key=compare_fn)
        return site_sizes

    @measure_time
    def forward_pass(self, ordered_sites: List[SiteSize], processed: Dict[str, int]) -> dict:
        if self.abort_signal.triggered:
            return {}

        total_sites = len(ordered_sites)
        if total_sites <= 1:
            return {}

        # reset time measurements
        self.parallel_forward_pass(ordered_sites, processed)

    def pairwise_setup(self, ordered_sites: List[SiteSize]):
        total_sites = len(ordered_sites)
        n = int(total_sites / 2)
        task_inputs = {}
        for i in range(n):
            k = 2 * i
            s = ordered_sites[k]
            c = ordered_sites[k + 1]
            inputs = Shareable()
            inputs[PSIConst.PSI_TASK_KEY] = PSIConst.PSI_TASK_SETUP
            inputs[PSIConst.PSI_ITEMS_SIZE] = c.size
            task_inputs[s.name] = inputs

        bop = BroadcastAndWait(self.fl_ctx, self.controller)
        results = bop.multicasts_and_wait(task_name=self.task_name,
                                          task_inputs=task_inputs,
                                          abort_signal=self.abort_signal)
        return {site_name: results[site_name].data[PSIConst.PSI_SETUP_MSG] for site_name in results}

    def pairwise_requests(self, ordered_sites: List[SiteSize], setup_msgs: Dict[str, str]):
        total_sites = len(ordered_sites)
        n = int(total_sites / 2)
        task_inputs = {}
        for i in range(n):
            k = 2 * i
            s = ordered_sites[k]
            c = ordered_sites[k + 1]
            inputs = Shareable()
            inputs[PSIConst.PSI_TASK_KEY] = PSIConst.PSI_TASK_REQUEST
            inputs[PSIConst.PSI_SETUP_MSG] = setup_msgs[s.name]
            task_inputs[c.name] = inputs
        bop = BroadcastAndWait(self.fl_ctx, self.controller)
        results = bop.multicasts_and_wait(task_name=self.task_name,
                                          task_inputs=task_inputs,
                                          abort_signal=self.abort_signal)
        return {site_name: results[site_name].data[PSIConst.PSI_REQUEST_MSG] for site_name in results}

    def pairwise_responses(self, ordered_sites: List[SiteSize], request_msgs: Dict[str, str]):
        total_sites = len(ordered_sites)
        n = int(total_sites / 2)
        task_inputs = {}
        for i in range(n):
            k = 2 * i
            s = ordered_sites[k]
            c = ordered_sites[k + 1]
            inputs = Shareable()
            inputs[PSIConst.PSI_TASK_KEY] = PSIConst.PSI_TASK_RESPONSE
            inputs[PSIConst.PSI_REQUEST_MSG] = request_msgs[c.name]
            task_inputs[s.name] = inputs

        bop = BroadcastAndWait(self.fl_ctx, self.controller)
        results = bop.multicasts_and_wait(task_name=self.task_name,
                                          task_inputs=task_inputs,
                                          abort_signal=self.abort_signal)
        return {site_name: results[site_name].data[PSIConst.PSI_RESPONSE_MSG] for site_name in results}

    def pairwise_intersect(self, ordered_sites: List[SiteSize], response_msg: Dict[str, str]):
        total_sites = len(ordered_sites)
        n = int(total_sites / 2)
        task_inputs = {}
        for i in range(n):
            k = 2 * i
            s = ordered_sites[k]
            c = ordered_sites[k + 1]
            inputs = Shareable()
            inputs[PSIConst.PSI_TASK_KEY] = PSIConst.PSI_TASK_INTERSECT
            inputs[PSIConst.PSI_RESPONSE_MSG] = response_msg[s.name]
            task_inputs[c.name] = inputs
        bop = BroadcastAndWait(self.fl_ctx, self.controller)
        results = bop.multicasts_and_wait(task_name=self.task_name,
                                          task_inputs=task_inputs,
                                          abort_signal=self.abort_signal)
        return {site_name: results[site_name].data[PSIConst.PSI_ITEMS_SIZE] for site_name in results}

    def parallel_forward_pass(self, target_sites, processed: dict):
        total_sites = len(target_sites)
        if total_sites < 2:
            final_site = target_sites[0]
            processed.update({final_site.name: final_site.size})
        else:
            setup_msgs = self.pairwise_setup(target_sites)
            request_msgs = self.pairwise_requests(target_sites, setup_msgs)
            response_msg = self.pairwise_responses(target_sites, request_msgs)
            it_sites = self.pairwise_intersect(target_sites, response_msg)
            processed.update(it_sites)
            new_targets = [SiteSize(site.name, it_sites[site.name]) for site in target_sites if site.name in it_sites]
            if total_sites % 2 == 1:
                new_targets.append(target_sites[int(total_sites / 2) + 1])

            return self.parallel_forward_pass(new_targets, processed)

    @measure_time
    def backward_pass(self, ordered_clients: list) -> dict:
        processed = {}
        if self.abort_signal.triggered:
            return processed

        total_clients = len(ordered_clients)
        if total_clients <= 1:
            return processed
        self.log_info(self.fl_ctx, f"forward_processed = {self.forward_processed}")
        status = self.parallel_back_pass(ordered_clients)

        time_taken = self.parallel_back_pass.time_taken
        self.log_info(self.fl_ctx, f"parallel_back_pass took {time_taken} (ms)")
        return status

    @measure_time
    def parallel_back_pass(self, ordered_clients: list):
        # parallel version
        updated_sites = self.get_updated_site_sizes(ordered_clients)
        total_clients = len(updated_sites)
        s = updated_sites[total_clients - 1]

        other_site_sizes = set([site.size for site in updated_sites if site.name != s.name])
        setup_msgs: Dict[str, str] = self.prepare_setup_messages(s, other_site_sizes)

        site_setup_msgs = {site.name: setup_msgs[str(site.size)] for site in updated_sites if site.name != s.name}
        request_msgs: Dict[str, str] = self.create_requests(site_setup_msgs)
        response_msg: Dict[str, str] = self.process_requests(s, request_msgs)
        return self.calculate_intersections(response_msg)

    def calculate_intersections(self, response_msg) -> Dict[str, int]:
        task_inputs = {}
        for client_name in response_msg:
            inputs = Shareable()
            inputs[PSIConst.PSI_TASK_KEY] = PSIConst.PSI_TASK_INTERSECT
            inputs[PSIConst.PSI_RESPONSE_MSG] = response_msg[client_name]
            task_inputs[client_name] = inputs
        bop = BroadcastAndWait(self.fl_ctx, self.controller)
        results = bop.multicasts_and_wait(task_name=self.task_name,
                                          task_inputs=task_inputs,
                                          abort_signal=self.abort_signal)

        intersects = {client_name: results[client_name].data[PSIConst.PSI_ITEMS_SIZE] for client_name in results}
        self.log_info(self.fl_ctx, f"received intersections : {intersects} ")
        return intersects

    def process_requests(self, s: SiteSize, request_msgs: Dict[str, str]) -> Dict[str, str]:
        task_inputs = Shareable()
        task_inputs[PSIConst.PSI_TASK_KEY] = PSIConst.PSI_TASK_RESPONSE
        task_inputs[PSIConst.PSI_REQUEST_MSG_SET] = request_msgs
        bop = BroadcastAndWait(self.fl_ctx, self.controller)
        results = bop.broadcast_and_wait(task_name=self.task_name,
                                         task_input=task_inputs,
                                         targets=[s.name],
                                         abort_signal=self.abort_signal)

        dxo = results[s.name]
        response_msgs = dxo.data[PSIConst.PSI_RESPONSE_MSG]
        return response_msgs

    def create_requests(self, site_setup_msgs) -> Dict[str, str]:
        task_inputs = {}
        for client_name in site_setup_msgs:
            inputs = Shareable()
            inputs[PSIConst.PSI_TASK_KEY] = PSIConst.PSI_TASK_REQUEST
            inputs[PSIConst.PSI_SETUP_MSG] = site_setup_msgs[client_name]
            task_inputs[client_name] = inputs

        bop = BroadcastAndWait(self.fl_ctx, self.controller)
        results = bop.multicasts_and_wait(task_name=self.task_name,
                                          task_inputs=task_inputs,
                                          abort_signal=self.abort_signal)
        request_msgs = {client_name: results[client_name].data[PSIConst.PSI_REQUEST_MSG] for client_name in results}
        return request_msgs

    def get_updated_site_sizes(self, ordered_sites):
        updated_sites = []
        for site in ordered_sites:
            new_size = self.forward_processed.get(site.name, site.size)
            updated_sites.append(SiteSize(site.name, new_size))

        return updated_sites

    def prepare_sites(self, direction: str, abort_signal):
        self.log_info(self.fl_ctx, f" start to prepare_sites, stage task : {PSIConst.PSI_TASK_PREPARE}")
        inputs = Shareable()
        inputs[PSIConst.PSI_TASK_KEY] = PSIConst.PSI_TASK_PREPARE
        inputs[PSIConst.PSI_DIRECTION_KEY] = direction
        inputs[PSIConst.PSI_BLOOM_FILTER_FPR] = self.bloom_filter_fpr
        targets = None
        engine = self.fl_ctx.get_engine()
        min_responses = len(engine.get_clients())
        self.log_info(self.fl_ctx, f"{PSIConst.PSI_TASK_PREPARE} BroadcastAndWait() on task {self.task_name}")
        bop = BroadcastAndWait(self.fl_ctx, self.controller)
        results = bop.broadcast_and_wait(task_name=self.task_name,
                                         task_input=inputs,
                                         targets=targets,
                                         min_responses=min_responses,
                                         abort_signal=abort_signal)
        self.log_info(self.fl_ctx, f"{PSIConst.PSI_TASK_PREPARE} results = {results}")
        self.ordered_sites = self.get_ordered_sites(results)

    def prepare_setup_messages(self, s: SiteSize, other_site_sizes: Set[int]) -> Dict[str, str]:
        inputs = Shareable()
        inputs[PSIConst.PSI_TASK_KEY] = PSIConst.PSI_TASK_SETUP
        inputs[PSIConst.PSI_ITEMS_SIZE_SET] = other_site_sizes
        bop = BroadcastAndWait(self.fl_ctx, self.controller)
        results = bop.broadcast_and_wait(task_name=self.task_name,
                                         task_input=inputs,
                                         targets=[s.name],
                                         abort_signal=self.abort_signal)
        dxo = results[s.name]
        return dxo.data[PSIConst.PSI_SETUP_MSG]

