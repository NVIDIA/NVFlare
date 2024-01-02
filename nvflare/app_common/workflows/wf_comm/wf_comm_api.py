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
import time
from queue import Empty
from typing import Dict, Optional, Tuple

from nvflare.apis.fl_constant import ReturnCode
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.workflows.wf_comm.wf_comm_api_spec import (
    CMD,
    CMD_ABORT,
    CMD_BROADCAST,
    CMD_RELAY,
    CMD_SEND,
    CMD_STOP,
    MIN_RESPONSES,
    PAYLOAD,
    RESP_MAX_WAIT_TIME,
    RESULT,
    SITE_NAMES,
    STATUS,
    WFCommAPISpec,
)
from nvflare.app_common.workflows.wf_comm.wf_queue import WFQueue


class WFCommAPI(WFCommAPISpec):
    def __init__(self):
        self.result_pull_interval = 2
        self.wf_queue: Optional[WFQueue] = None
        self.meta = {SITE_NAMES: []}
        self.logger = logging.getLogger(self.__class__.__name__)

    def set_result_pull_interval(self, pull_interval: float):
        self.result_pull_interval = pull_interval

    def set_queue(self, wf_queue: WFQueue):
        self.wf_queue = wf_queue

    def broadcast_and_wait(self, msg_payload: Dict):
        self.broadcast(msg_payload)
        min_responses = msg_payload.get(MIN_RESPONSES, 0)
        resp_max_wait_time = msg_payload.get(RESP_MAX_WAIT_TIME, 5)
        return self.wait_all(min_responses, resp_max_wait_time)

    def broadcast(self, msg_payload):
        self._check_wf_queue()
        message = {
            CMD: CMD_BROADCAST,
            PAYLOAD: msg_payload,
        }
        self.wf_queue.put_ctrl_msg(message)

    def send(self, msg_payload: Dict):
        self._check_wf_queue()
        message = {
            CMD: CMD_SEND,
            PAYLOAD: msg_payload,
        }
        self.wf_queue.put_ctrl_msg(message)

    def send_and_wait(self, msg_payload: Dict):
        self.send(msg_payload)
        min_responses = msg_payload.get(MIN_RESPONSES, 0)
        return self.wait_all(min_responses)

    def get_site_names(self):
        return self.meta.get(SITE_NAMES)

    def wait_all(self, min_responses: int, resp_max_wait_time: Optional[float] = None) -> Dict[str, Dict[str, FLModel]]:
        acc_size = 0
        start = None
        while True:
            if self.wf_queue.has_result():
                start = time.time() if start is None else start
                items_size = self.wf_queue.result_size()
                acc_size = items_size + acc_size
                time_waited = time.time() - start
                self.logger.info(
                    f"\n\n {items_size=}, {acc_size=}, {min_responses=}, {time_waited=}, {resp_max_wait_time=}"
                )
                if time_waited < resp_max_wait_time and acc_size >= min_responses:
                    return self._get_results()
                else:
                    if time_waited < resp_max_wait_time:
                        self.logger.info(f" wait for more results, sleep {self.result_pull_interval} sec")
                        time.sleep(self.result_pull_interval)
                    else:
                        msg = f"not enough responses {acc_size} compare with min responses requirement {min_responses} within the max allowed time {resp_max_wait_time} seconds"
                        self.logger.info(msg)
                        raise RuntimeError(msg)

            else:
                time.sleep(self.result_pull_interval)

    def relay_and_wait(self, msg_payload: Dict):
        self.relay(msg_payload)
        min_responses = msg_payload.get(MIN_RESPONSES, 1)
        return self.wait_all(min_responses)

    def relay(self, msg_payload: Dict):
        self._check_wf_queue()
        message = {
            CMD: CMD_RELAY,
            PAYLOAD: msg_payload,
        }
        self.wf_queue.put_ctrl_msg(message)

    def wait_one(self, resp_max_wait_time: Optional[float] = None) -> Tuple[str, str, FLModel]:
        try:
            item = self.wf_queue.get_result(resp_max_wait_time)
            if item:
                return self._process_one_result(item)
        except Empty as e:
            raise RuntimeError(f"failed to get result within the given timeout {resp_max_wait_time} sec.")

    def _process_one_result(self, item) -> Tuple[str, str, FLModel]:
        cmd = item.get(CMD, None)

        if cmd is None:
            msg = f"get None command, expecting {CMD} key'"
            self.logger.error(msg)
            raise RuntimeError(msg)

        elif cmd == CMD_STOP or cmd == CMD_ABORT:
            msg = item.get(PAYLOAD)
            self.logger.info(f"receive {cmd} command, {msg}")
            raise RuntimeError(msg)

        elif cmd == RESULT:
            payload = item.get(PAYLOAD)
            task_result = None
            task, site_result = next(iter(payload.items()))
            self._check_result(site_result)
            rc = site_result.get(STATUS)
            if rc == ReturnCode.OK:
                result = site_result.get(RESULT, {})
                site_name, data = next(iter(result.items()))
                task_result = (task, site_name, data)
            else:
                msg = f"task {task} failed with '{rc}' status"
                self.wf_queue.ask_abort(msg)
                raise RuntimeError(msg)

            return task_result
        else:
            raise RuntimeError(f"Unknown command {cmd}")

    def _get_results(self) -> Dict[str, Dict[str, FLModel]]:
        items_size = self.wf_queue.result_size()
        batch_result: Dict = {}
        for i in range(items_size):
            item = self.wf_queue.get_result()
            task, site_name, data = self._process_one_result(item)
            task_result = batch_result.get(task, {})
            task_result.update({site_name: data})
            batch_result[task] = task_result
        return batch_result

    def wait_for_responses(self, items_size, min_responses, resp_max_wait_time):
        start = time.time()
        while items_size < min_responses:
            time_waited = time.time() - start
            if time_waited < resp_max_wait_time:
                time.sleep(1)
                items_size = self.wf_queue.result_size()
            else:
                break
        return items_size

    def _check_result(self, site_result):

        if site_result is None:
            raise RuntimeError("expecting site_result to be dictionary, but get None")

        if not isinstance(site_result, dict):
            raise RuntimeError(f"expecting site_result to be dictionary, but get '{type(site_result)}'")

        keys = [RESULT, STATUS]
        all_keys_present = all(key in site_result for key in keys)
        if not all_keys_present:
            raise RuntimeError(f"expecting all keys {keys} present in site_result")

    def _check_wf_queue(self):
        if self.wf_queue is None:
            raise RuntimeError("missing WFQueue")
