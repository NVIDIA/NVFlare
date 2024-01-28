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


import logging
import threading
from typing import Callable, Dict, List, Optional, Union

from nvflare.apis.controller_spec import SendOrder
from nvflare.apis.fl_constant import ReturnCode
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.wf_comm.wf_comm_api_spec import (
    CURRENT_ROUND,
    DATA,
    MIN_RESPONSES,
    NUM_ROUNDS,
    RESP_MAX_WAIT_TIME,
    RESULT,
    SITE_NAMES,
    START_ROUND,
    STATUS,
    TARGET_SITES,
    TASK_NAME,
    WFCommAPISpec,
)
from nvflare.fuel.data_event.data_bus import DataBus
from nvflare.fuel.data_event.event_manager import EventManager


class WFCommAPI(WFCommAPISpec):
    def __init__(self):
        self.meta = {SITE_NAMES: []}
        self.logger = logging.getLogger(self.__class__.__name__)

        self.task_results = {}
        self.task_result_lock = threading.Lock()

        data_bus = DataBus()
        data_bus.subscribe(topics=["TASK_RESULT"], callback=self.result_callback)

        self.event_manager = EventManager(data_bus)
        self.ctrl = data_bus.receive_data("communicator")
        self._check_inputs()

    def get_site_names(self):
        return self.meta.get(SITE_NAMES)

    def broadcast_and_wait(
        self,
        task_name: str,
        min_responses: int,
        data: any,
        meta: dict = None,
        targets: Optional[List[str]] = None,
        callback: Callable = None,
    ) -> Union[int, Dict[str, Dict[str, FLModel]]]:

        meta = {} if meta is None else meta
        msg_payload = self._prepare_input_payload(task_name, data, meta, min_responses, targets)
        self.register_callback(callback)
        print("\ncalling broadcast_to_peers_and_wait\n")
        self.ctrl.broadcast_to_peers_and_wait(msg_payload)
        print("\nafter broadcast_to_peers_and_wait\n")

        if callback is None:
            return self._get_results(task_name)

    def register_callback(self, callback):
        if callback:
            self.event_manager.data_bus.subscribe(["POST_PROCESS_RESULT"], callback)

    def send_and_wait(
        self,
        task_name: str,
        min_responses: int,
        data: any,
        meta: dict = None,
        send_order: SendOrder = SendOrder.SEQUENTIAL,
        targets: Optional[List[str]] = None,
        callback: Callable = None,
    ):
        meta = {} if meta is None else meta
        msg_payload = self._prepare_input_payload(task_name, data, meta, min_responses, targets)

        if callback is not None:
            self.register_callback(callback)

        self.ctrl.send_to_peers_and_wait(msg_payload, send_order)

        if callback is not None:
            return self._get_results(task_name)

    def relay_and_wait(
        self,
        task_name: str,
        min_responses: int,
        data: any,
        meta: dict = None,
        targets: Optional[List[str]] = None,
        relay_order: str = "sequential",
        callback: Callable = None,
    ) -> Dict[str, Dict[str, FLModel]]:

        meta = {} if meta is None else meta
        msg_payload = self._prepare_input_payload(task_name, data, meta, min_responses, targets)

        self.register_callback(callback)

        self.ctrl.relay_to_peers_and_wait(msg_payload, SendOrder(relay_order))

        if callback is None:
            return self._get_results(task_name)

        return self._get_results(task_name)

    def broadcast(self, task_name: str, data: any, meta: dict = None, targets: Optional[List[str]] = None):
        msg_payload = self._prepare_input_payload(task_name, data, meta, min_responses=0, targets=targets)
        self.ctrl.broadcast_to_peers(pay_load=msg_payload)

    def send(
        self,
        task_name: str,
        data: any,
        meta: dict = None,
        targets: Optional[List[str]] = None,
        send_order: str = "sequential",
    ):
        msg_payload = self._prepare_input_payload(task_name, data, meta, min_responses=0, targets=targets)
        self.ctrl.send_to_peers(pay_load=msg_payload, send_order=send_order)

    def relay(
        self,
        task_name: str,
        data: any,
        meta: dict = None,
        targets: Optional[List[str]] = None,
        send_order: str = "sequential",
    ):
        msg_payload = self._prepare_input_payload(task_name, data, meta, min_responses=0, targets=targets)
        self.ctrl.relay_to_peers(msg_payload, send_order)

    def _process_one_result(self, site_result) -> Dict[str, FLModel]:
        self._check_result(site_result)
        rc = site_result.get(STATUS)
        if rc == ReturnCode.OK:
            result = site_result.get(RESULT, {})
            site_name, data = next(iter(result.items()))
            task_result = {site_name: data}
        else:
            msg = f"task failed with '{rc}' status"
            raise RuntimeError(msg)

        return task_result

    def _get_results(self, task_name) -> Dict[str, Dict[str, FLModel]]:
        print("_get_results\n")
        batch_result: Dict = {}
        site_results = self.task_results.get(task_name)
        if not site_results:
            raise RuntimeError(f"not result for given task {task_name}")

        for i in range(len(site_results)):
            item = site_results[i]
            one_result = self._process_one_result(item)
            task_result = batch_result.get(task_name, {})
            task_result.update(one_result)
            batch_result[task_name] = task_result

        with self.task_result_lock:
            self.task_results[task_name] = []

        print("return batch_result=", batch_result)
        return batch_result

    def _check_result(self, site_result):

        if site_result is None:
            raise RuntimeError("expecting site_result to be dictionary, but get None")

        if not isinstance(site_result, dict):
            raise RuntimeError(f"expecting site_result to be dictionary, but get '{type(site_result)}', {site_result=}")

        keys = [RESULT, STATUS]
        all_keys_present = all(key in site_result for key in keys)
        if not all_keys_present:
            raise RuntimeError(f"expecting all keys {keys} present in site_result")

    def _check_inputs(self):
        if self.ctrl is None:
            raise RuntimeError("missing Controller")

    def result_callback(self, topic, data, data_bus):
        if topic == "TASK_RESULT":
            task, site_result = next(iter(data.items()))
            # fire event with process data
            one_result = self._process_one_result(site_result)
            self.event_manager.fire_event("POST_PROCESS_RESULT", {task: one_result})
            site_task_results = self.task_results.get(task, [])
            site_task_results.append(site_result)
            self.task_results[task] = site_task_results

    def _prepare_input_payload(self, task_name, data, meta, min_responses, targets):

        if data and isinstance(data, FLModel):
            start_round = data.start_round
            current_round = data.current_round
            num_rounds = data.total_rounds
        else:
            start_round = meta.get(START_ROUND, 0)
            current_round = meta.get(CURRENT_ROUND, 0)
            num_rounds = meta.get(NUM_ROUNDS, 1)

        resp_max_wait_time = meta.get(RESP_MAX_WAIT_TIME, 15)

        msg_payload = {
            TASK_NAME: task_name,
            MIN_RESPONSES: min_responses,
            RESP_MAX_WAIT_TIME: resp_max_wait_time,
            CURRENT_ROUND: current_round,
            NUM_ROUNDS: num_rounds,
            START_ROUND: start_round,
            DATA: data,
            TARGET_SITES: targets,
        }
        return msg_payload
