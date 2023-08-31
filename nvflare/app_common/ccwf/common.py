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

from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.app_common.abstract.learnable import Learnable


class Constant:

    TN_PREFIX_CYCLIC = "cyclic"
    TN_PREFIX_SWARM = "swarm"
    TN_PREFIX_CROSS_SITE_EVAL = "cse"

    BASENAME_CONFIG = "config"
    BASENAME_START = "start"
    BASENAME_LEARN = "learn"
    BASENAME_EVAL = "eval"
    BASENAME_RCV_LEARN_RESULT = "rcv_learn_result"
    BASENAME_RCV_FINAL_RESULT = "rcv_final_result"
    BASENAME_ASK_FOR_MODEL = "ask_for_model"

    TASK_NAME_PREFIX = "cwf.task_prefix"
    ERROR = "cwf.error"
    ORDER = "cwf.order"
    CLIENTS = "cwf.clients"
    START_CLIENT = "cwf.start_client"
    RESULT_CLIENTS = "cwf.result_clients"
    CLIENT_ORDER = "cwf.client_order"
    LAST_ROUND = "cwf.last_round"
    START_ROUND = "cwf.start_round"
    TIMESTAMP = "cwf.timestamp"
    ACTION = "cwf.action"
    ALL_DONE = "cwf.all_done"
    AGGR_CLIENTS = "cwf.aggr_clients"
    TRAIN_CLIENTS = "cwf.train_clients"
    AGGREGATOR = "cwf.aggr"
    METRIC = "cwf.metric"
    CLIENT = "cwf.client"
    ROUND = "cwf.round"
    CONFIG = "cwf.config"
    STATUS_REPORTS = "cwf.status_reports"
    RESULT = "cwf.result"
    RESULT_TYPE = "cwf.result_type"
    EVAL_LOCAL = "cwf.eval_local"
    EVAL_GLOBAL = "cwf.eval_global"
    EVALUATORS = "cwf.evaluators"
    EVALUATEES = "cwf.evaluatees"
    GLOBAL_CLIENT = "cwf.global_client"
    MODEL_OWNER = "cwf.model_owner"
    MODEL_NAME = "cwf.model_name"
    MODEL_TYPE = "cwf.model_type"
    GLOBAL_NAMES = "cwf.global_names"
    EXECUTOR = "cwf.executor"
    EXECUTOR_INITIALIZED = "cwf.executor_initialized"
    EXECUTOR_FINALIZED = "cwf.executor_finalized"

    TOPIC_SHARE_RESULT = "cwf.share_result"
    TOPIC_END_WORKFLOW = "cwf.end_wf"

    RC_NO_GLOBAL_MODELS = "cwf.no_global_models"
    RC_NO_LOCAL_MODEL = "cwf.no_local_model"
    RC_UNABLE_TO_EVAL = "cwf.unable_to_eval"

    CONFIG_TASK_TIMEOUT = 300
    START_TASK_TIMEOUT = 10
    END_WORKFLOW_TIMEOUT = 2.0
    TASK_CHECK_INTERVAL = 0.5
    JOB_STATUS_CHECK_INTERVAL = 2.0
    PER_CLIENT_STATUS_REPORT_TIMEOUT = 3600.0
    WORKFLOW_PROGRESS_TIMEOUT = 3600.0

    MAX_STATUS_REPORT_INTERVAL = 600.0
    LEARN_TASK_CHECK_INTERVAL = 1.0
    LEARN_TASK_SEND_TIMEOUT = 10
    LEARN_TASK_ABORT_TIMEOUT = 5.0
    FINAL_RESULT_SEND_TIMEOUT = 10
    GET_MODEL_TIMEOUT = 10


class ModelType:

    LOCAL = "local"
    GLOBAL = "global"


class ResultType:

    BEST = "best"
    LAST = "last"


class RROrder:

    FIXED = "fixed"
    RANDOM = "random"


class StatusReport:
    def __init__(
        self,
        timestamp=None,
        action: str = "",
        last_round=None,
        all_done=False,
        error: str = "",
    ):
        self.timestamp = timestamp
        self.action = action
        self.last_round = last_round
        self.all_done = all_done
        self.error = error

    def to_dict(self) -> dict:
        result = {
            Constant.TIMESTAMP: self.timestamp,
            Constant.ACTION: self.action,
            Constant.ALL_DONE: self.all_done,
        }

        if self.last_round is not None:
            result[Constant.LAST_ROUND] = self.last_round

        if self.error:
            result[Constant.ERROR] = self.error
        return result

    def __eq__(self, other):
        if not isinstance(other, StatusReport):
            # don't attempt to compare against unrelated types
            return ValueError(f"cannot compare to object of type {type(other)}")

        return (
            self.last_round == other.last_round
            and self.timestamp == other.timestamp
            and self.action == other.action
            and self.all_done == other.all_done
            and self.error == other.error
        )


def status_report_from_dict(d: dict) -> StatusReport:
    last_round = d.get(Constant.LAST_ROUND)
    timestamp = d.get(Constant.TIMESTAMP)
    all_done = d.get(Constant.ALL_DONE)
    error = d.get(Constant.ERROR)
    action = d.get(Constant.ACTION)

    return StatusReport(
        last_round=last_round,
        timestamp=timestamp,
        action=action,
        all_done=all_done,
        error=error,
    )


def learnable_to_shareable(learnable: Learnable) -> Shareable:
    s = Shareable()
    s.update(learnable)
    return s


def shareable_to_learnable(shareable: Shareable) -> Learnable:
    learnable = Learnable()
    learnable.update(shareable)
    learnable.pop(ReservedHeaderKey.HEADERS, None)
    return learnable


def rotate_to_front(item, items: list):
    num_items = len(items)
    idx = items.index(item)
    if idx != 0:
        new_list = [None] * num_items
        for i in range(num_items):
            new_pos = i - idx
            if new_pos < 0:
                new_pos += num_items
            new_list[new_pos] = items[i]

        for i in range(num_items):
            items[i] = new_list[i]


def topic_for_end_workflow(wf_id):
    return f"{Constant.TOPIC_END_WORKFLOW}.{wf_id}"


def make_task_name(prefix: str, base_name: str) -> str:
    return f"{prefix}_{base_name}"
