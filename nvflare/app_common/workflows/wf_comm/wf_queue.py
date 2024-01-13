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


from queue import Queue
from typing import Dict, Optional

from nvflare.app_common.workflows.wf_comm.wf_comm_api_spec import CMD, CMD_ABORT, CMD_STOP, PAYLOAD


class WFQueue:
    def __init__(self, result_queue: Queue):
        self.result_queue = result_queue

    def put_result(self, msg):
        self.result_queue.put(msg)

    def has_result(self) -> bool:
        return not self.result_queue.empty()

    def result_size(self) -> int:
        return self.result_queue.qsize()

    def get_result(self, timeout: Optional[float] = None) -> Dict:
        item = self.result_queue.get(timeout=timeout)
        self.result_queue.task_done()
        return item

    def stop(self, msg: Optional[str] = None):
        msg = msg if msg else {}
        self.put_result({CMD: CMD_STOP, PAYLOAD: msg})

    def ask_abort(self, msg: Optional[str] = None):
        msg = msg if msg else {}
        self.put_result({CMD: CMD_ABORT, PAYLOAD: msg})
