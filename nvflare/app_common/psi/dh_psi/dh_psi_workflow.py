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

        self.forward_processed.update(self.forward_pass(self.ordered_sites))
        self.backward_processed.update(self.backward_pass(self.ordered_sites))
        if len(self.backward_processed) < len(self.ordered_sites) - 1:
            self.log_error(self.fl_ctx, "incomplete for all sites' intersections")
            self.log_error(self.fl_ctx, "completed ones are {self.backward_processed}")
            raise RuntimeError("process failed without completing all sites ")

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
            print("****** site:", site_name, "data = ", data)
            if PSIConst.PSI_ITEMS_SIZE in data:
                size = data[PSIConst.PSI_ITEMS_SIZE]
            else:
                size = 0

            if size > 0:
                c = SiteSize(site_name, size)
                site_sizes.append(c)
        site_sizes.sort(key=compare_fn)
        return site_sizes

    @measure_time
    def forward_pass(self, ordered_sites: List[SiteSize], reverse=False) -> dict:
        processed = {}
        if self.abort_signal.triggered:
            return processed

        total_sites = len(ordered_sites)
        if total_sites <= 1:
            return processed
        #
        self.prepare_setup_message(reset=True)
        self.prepare_request(reset=True)
        self.process_request(reset=True)
        self.calculate_intersection(reset=True)

        start, end, step = self.get_directional_range(total_sites, reverse)
        for i in range(start, end, step):
            s = ordered_sites[i]
            c = ordered_sites[i + step]
            setup_msg = self.prepare_setup_message(s, c)
            request = self.prepare_request(c, setup_msg)
            response = self.process_request(s, c.name, request[c.name])
            client_intersect = self.calculate_intersection(c, response)
            processed.update(client_intersect)

        self.report_time_taken(reverse)
        return processed

    def report_time_taken(self, reverse):
        direction = "backward" if reverse else "forward"
        self.log_info(
            self.fl_ctx,
            f"{direction} pass, prepare_setup_message {self.prepare_setup_message.time_taken} "
            f"(ms) with {self.prepare_setup_message.count} calls",
        )
        self.log_info(
            self.fl_ctx,
            f"{direction} pass, prepare_request {self.prepare_request.time_taken} "
            f"(ms) with {self.prepare_request.count} calls",
        )
        self.log_info(
            self.fl_ctx,
            f"{direction} pass, process_request {self.process_request.time_taken} "
            f"(ms) with {self.process_request.count} calls",
        )
        self.log_info(
            self.fl_ctx,
            f"{direction} pass, calculate_intersection {self.calculate_intersection.time_taken} "
            f"(ms) with {self.calculate_intersection.count} calls",
        )

    def get_directional_range(self, total: int, reverse: bool = False):
        if reverse:
            start = total - 1
            end = -1
            step = -1
        else:
            start = 0
            end = total - 1
            step = 1

        return start, end, step

    @measure_time
    def backward_pass(self, ordered_clients: list) -> dict:
        processed = {}
        if self.abort_signal.triggered:
            return processed

        total_clients = len(ordered_clients)
        if total_clients <= 1:
            return processed

        self.log_info(self.fl_ctx, f"forward_processed = {self.forward_processed}")

        if len(self.forward_processed) == total_clients - 1:
            # status = self.sequential_back_pass(ordered_clients)
            # self.log_info(self.fl_ctx, f"*********** sequential_back_pass took {self.sequential_back_pass.time_taken} (ms)")

            status = self.parallel_back_pass(ordered_clients)
            self.log_info(self.fl_ctx, f"********************* parallel_back_pass took {self.parallel_back_pass.time_taken} (ms)")
            return status
        else:
            raise ValueError("started backward pass before finish the forward pass")

    @measure_time
    def sequential_back_pass(self, ordered_clients: list):
        # Sequential version
        return self.forward_pass(ordered_clients, reverse=True)

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

    @collect_time
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

    @collect_time
    def prepare_setup_message(self, s: SiteSize, c: SiteSize) -> dict:
        inputs = Shareable()
        inputs[PSIConst.PSI_TASK_KEY] = PSIConst.PSI_TASK_SETUP
        inputs[PSIConst.PSI_ITEMS_SIZE] = c.size
        targets = [s.name]

        bop = BroadcastAndWait(self.fl_ctx, self.controller)
        results = bop.broadcast_and_wait(task_name=self.task_name,
                                         task_input=inputs,
                                         targets=targets,
                                         abort_signal=self.abort_signal)

        dxo = results[s.name]
        setup_msg = dxo.data[PSIConst.PSI_SETUP_MSG]
        self.log_info(self.fl_ctx, f"received setup message from {s.name} for {c.name}")
        return {c.name: setup_msg}

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

    @collect_time
    def prepare_request(self, c: SiteSize, setup_msg: dict) -> dict:
        inputs = Shareable()
        inputs[PSIConst.PSI_TASK_KEY] = PSIConst.PSI_TASK_REQUEST
        inputs[PSIConst.PSI_SETUP_MSG] = setup_msg[c.name]
        targets = [c.name]
        bop = BroadcastAndWait(self.fl_ctx, self.controller)
        results = bop.broadcast_and_wait(task_name=self.task_name,
                                         task_input=inputs,
                                         targets=targets,
                                         abort_signal=self.abort_signal)

        dxo = results[c.name]
        request_msg = dxo.data[PSIConst.PSI_REQUEST_MSG]
        self.log_info(self.fl_ctx, f"received request message from {c.name}")
        return {c.name: request_msg}

    @collect_time
    def process_request(self, s: SiteSize, c_name: str, response_msg: str):
        inputs = Shareable()
        inputs[PSIConst.PSI_TASK_KEY] = PSIConst.PSI_TASK_RESPONSE
        inputs[PSIConst.PSI_REQUEST_MSG] = response_msg
        targets = [s.name]

        bop = BroadcastAndWait(self.fl_ctx, self.controller)
        results = bop.broadcast_and_wait(task_name=self.task_name,
                                         task_input=inputs,
                                         targets=targets,
                                         abort_signal=self.abort_signal)
        dxo = results[s.name]
        response_msg = dxo.data[PSIConst.PSI_RESPONSE_MSG]
        self.log_info(self.fl_ctx, f"received response message from {s.name} for {c_name}")
        return {c_name: response_msg}

    @collect_time
    def calculate_intersection(self, c: SiteSize, response: dict):
        inputs = Shareable()
        inputs[PSIConst.PSI_TASK_KEY] = PSIConst.PSI_TASK_INTERSECT
        inputs[PSIConst.PSI_RESPONSE_MSG] = response[c.name]
        targets = [c.name]

        bop = BroadcastAndWait(self.fl_ctx, self.controller)
        results = bop.broadcast_and_wait(task_name=self.task_name,
                                         task_input=inputs,
                                         targets=targets,
                                         abort_signal=self.abort_signal)

        dxo = results[c.name]
        intersect_size = dxo.data[PSIConst.PSI_ITEMS_SIZE]
        self.log_info(self.fl_ctx, f"received intersection size from {c.name}: {intersect_size}")
        return {c.name: intersect_size}
