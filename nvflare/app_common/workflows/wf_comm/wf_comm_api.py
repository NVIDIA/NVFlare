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
from typing import Dict, Optional

from nvflare.apis.fl_constant import ReturnCode
from nvflare.app_common.workflows.wf_comm.wf_comm_api_spec import (
    CMD,
    CMD_ABORT,
    CMD_BROADCAST,
    CMD_RELAY,
    CMD_SEND,
    CMD_STOP,
    MIN_RESPONSES,
    PAYLOAD,
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
        return self.wait(min_responses)

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
        return self.wait(min_responses)

    def get_site_names(self):
        return self.meta.get(SITE_NAMES)

    def wait(self, min_responses):
        while True:
            if self.wf_queue.has_result():
                items_size = self.wf_queue.result_size()
                if items_size >= min_responses:
                    return self._get_results()
                else:
                    self.logger.info(f" wait for more results, sleep {self.result_pull_interval} sec")
                    time.sleep(self.result_pull_interval)
            else:
                # self.logger.info(f"no result available, sleep {self.result_pull_interval} sec")
                time.sleep(self.result_pull_interval)

    def relay_and_wait(self, msg_payload: Dict):
        self.relay(msg_payload)
        min_responses = msg_payload.get(MIN_RESPONSES, 1)
        return self.wait(min_responses)

    def relay(self, msg_payload: Dict):
        self._check_wf_queue()
        message = {
            CMD: CMD_RELAY,
            PAYLOAD: msg_payload,
        }
        self.wf_queue.put_ctrl_msg(message)

    def _get_results(self) -> dict:
        items_size = self.wf_queue.result_size()
        batch_result: Dict = {}

        for i in range(items_size):
            item = self.wf_queue.get_result()
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
                one_site_result = item.get(PAYLOAD)
                for task, site_result in one_site_result.items():
                    task_result = batch_result.get(task, {})
                    self._check_result(site_result)
                    rc = site_result.get(STATUS)
                    if rc == ReturnCode.OK:
                        result = site_result.get(RESULT, {})
                        task_result.update(result)
                        batch_result[task] = task_result
                    else:
                        msg = f"task {task} failed with '{rc}' status"
                        self.wf_queue.ask_abort(msg)
                        raise RuntimeError(msg)
            else:
                raise RuntimeError(f"Unknown command {cmd}")

        return batch_result

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
