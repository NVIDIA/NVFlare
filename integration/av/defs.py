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
from nvflare.fuel.f3.cellnet.fqcn import FQCN

CHANNEL = "flare_agent"

TOPIC_GET_TASK = "get_task"
TOPIC_SUBMIT_RESULT = "submit_result"
TOPIC_HEARTBEAT = "heartbeat"
TOPIC_HELLO = "hello"
TOPIC_BYE = "bye"
TOPIC_ABORT = "abort"

JOB_META_KEY_AGENT_ID = "agent_id"


class RC:
    OK = "OK"
    BAD_TASK_DATA = "BAD_TASK_DATA"
    EXECUTION_EXCEPTION = "EXECUTION_EXCEPTION"


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


def agent_site_fqcn(site_name: str, agent_id: str, job_id=None):
    if not job_id:
        return f"{site_name}--{agent_id}"
    else:
        return FQCN.join([site_name, job_id, agent_id])
