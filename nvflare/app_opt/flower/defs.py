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

from nvflare.app_common.tie.defs import Constant as TieConstant
from nvflare.fuel.f3.drivers.net_utils import MAX_FRAME_SIZE


class Constant:

    # task name defaults
    CONFIG_TASK_NAME = TieConstant.CONFIG_TASK_NAME
    START_TASK_NAME = TieConstant.START_TASK_NAME

    # keys of config parameters
    CONF_KEY_NUM_ROUNDS = "num_rounds"

    PARAM_KEY_HEADERS = "flower.headers"
    PARAM_KEY_CONTENT = "flower.content"
    PARAM_KEY_MSG_NAME = "flower.name"

    # default component config values
    CONFIG_TASK_TIMEOUT = TieConstant.CONFIG_TASK_TIMEOUT
    START_TASK_TIMEOUT = TieConstant.START_TASK_TIMEOUT
    FLOWER_SERVER_READY_TIMEOUT = 10.0

    TASK_CHECK_INTERVAL = TieConstant.TASK_CHECK_INTERVAL
    JOB_STATUS_CHECK_INTERVAL = TieConstant.JOB_STATUS_CHECK_INTERVAL
    MAX_CLIENT_OP_INTERVAL = TieConstant.MAX_CLIENT_OP_INTERVAL
    WORKFLOW_PROGRESS_TIMEOUT = TieConstant.WORKFLOW_PROGRESS_TIMEOUT

    APP_CTX_SERVER_ADDR = "flower_server_addr"
    APP_CTX_PORT = "flower_port"
    APP_CTX_CLIENT_NAME = "flower_client_name"
    APP_CTX_NUM_ROUNDS = "flower_num_rounds"
    APP_CTX_FL_CONTEXT = TieConstant.APP_CTX_FL_CONTEXT


GRPC_DEFAULT_OPTIONS = [
    ("grpc.max_send_message_length", MAX_FRAME_SIZE),
    ("grpc.max_receive_message_length", MAX_FRAME_SIZE),
]
