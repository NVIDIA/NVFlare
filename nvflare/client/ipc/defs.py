# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.apis.fl_constant import ReturnCode as RC
from nvflare.fuel.f3.cellnet.fqcn import FQCN

CHANNEL = "flare_agent"

TOPIC_GET_TASK = "get_task"
TOPIC_SUBMIT_RESULT = "submit_result"
TOPIC_HEARTBEAT = "heartbeat"
TOPIC_HELLO = "hello"
TOPIC_BYE = "bye"
TOPIC_ABORT = "abort"

JOB_META_KEY_AGENT_ID = "agent_id"


class MsgHeader:

    TASK_ID = "task_id"
    TASK_NAME = "task_name"
    RC = "rc"


class PayloadKey:
    DATA = "data"
    META = "meta"


class MetaKey:
    CURRENT_ROUND = "current_round"
    TOTAL_ROUND = "total_round"
    DATA_KIND = "data_kind"
    NUM_STEPS_CURRENT_ROUND = "NUM_STEPS_CURRENT_ROUND"
    PROCESSED_ALGORITHM = "PROCESSED_ALGORITHM"
    PROCESSED_KEYS = "PROCESSED_KEYS"
    INITIAL_METRICS = "initial_metrics"
    FILTER_HISTORY = "filter_history"


class Task:

    NEW = 0
    FETCHED = 1
    PROCESSED = 2

    def __init__(self, task_name: str, task_id: str, meta: dict, data):
        self.task_name = task_name
        self.task_id = task_id
        self.meta = meta
        self.data = data
        self.status = Task.NEW
        self.last_send_result_time = None
        self.aborted = False
        self.already_received = False

    def __str__(self):
        return f"'{self.task_name} {self.task_id}'"


class TaskResult:
    def __init__(self, meta: dict, data, return_code=RC.OK):
        if not meta:
            meta = {}

        if not isinstance(meta, dict):
            raise TypeError(f"meta must be dict but got {type(meta)}")

        if not data:
            data = {}

        if not isinstance(return_code, str):
            raise TypeError(f"return_code must be str but got {type(return_code)}")

        self.return_code = return_code
        self.meta = meta
        self.data = data


class AgentClosed(Exception):
    pass


class CallStateError(Exception):
    pass


def agent_site_fqcn(site_name: str, agent_id: str):
    # add the "-" prefix to the agent_id to make a child of the site
    # this prefix will make the agent site's FQCN < the CJ's FQCN
    # this is necessary to enable ad-hoc connections between CJ and agent, where CJ listens
    # with ad-hoc connection, the cell with greater FQCN listens.
    return FQCN.join([site_name, f"-{agent_id}"])
