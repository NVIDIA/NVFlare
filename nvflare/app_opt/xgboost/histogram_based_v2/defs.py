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


class Constant:

    # task name defaults
    CONFIG_TASK_NAME = "config"
    START_TASK_NAME = "start"
    TRAIN_TASK_NAME = "train"

    # keys of adaptor config parameters
    CONF_KEY_CLIENT_RANKS = "client_ranks"
    CONF_KEY_RANK = "rank"
    CONF_KEY_WORLD_SIZE = "world_size"
    CONF_KEY_NUM_ROUNDS = "num_rounds"

    # default component config values
    CONFIG_TASK_TIMEOUT = 10
    START_TASK_TIMEOUT = 10
    XGB_SERVER_READY_TIMEOUT = 5.0

    TASK_CHECK_INTERVAL = 0.5
    JOB_STATUS_CHECK_INTERVAL = 2.0
    MAX_CLIENT_OP_INTERVAL = 900.0
    WORKFLOW_PROGRESS_TIMEOUT = 3600.0

    # message topics
    TOPIC_XGB_REQUEST = "xgb.request"
    TOPIC_XGB_REQUEST_CHECK = "xgb.req_check"
    TOPIC_CLIENT_DONE = "xgb.client_done"

    # keys for Shareable between client and server
    MSG_KEY_EXIT_CODE = "exit_code"
    MSG_KEY_XGB_OP = "xgb.op"
    MSG_KEY_XGB_REQ_ID = "xgb.req_id"
    MSG_KEY_XGB_REQ_TRY_NUM = "xgb.req_try_num"
    MSG_KEY_XGB_REQ_RECEIVED = "xgb.req_received"

    # XGB operation names
    OP_ALL_GATHER = "all_gather"
    OP_ALL_GATHER_V = "all_gather_v"
    OP_ALL_REDUCE = "all_reduce"
    OP_BROADCAST = "broadcast"

    # XGB operation codes
    OPCODE_NONE = 0
    OPCODE_ALL_GATHER = 1
    OPCODE_ALL_GATHER_V = 2
    OPCODE_ALL_REDUCE = 3
    OPCODE_BROADCAST = 4
    OPCODE_DONE = 99

    # XGB operation error codes
    ERR_OP_MISMATCH = -1
    ERR_INVALID_RANK = -2
    ERR_NO_CLIENT_FOR_RANK = -3
    ERR_TARGET_ERROR = -4

    # XGB operation parameter keys
    PARAM_KEY_RANK = "xgb.rank"
    PARAM_KEY_SEQ = "xgb.seq"
    PARAM_KEY_SEND_BUF = "xgb.send_buf"
    PARAM_KEY_DATA_TYPE = "xgb.data_type"
    PARAM_KEY_REDUCE_OP = "xgb.reduce_op"
    PARAM_KEY_ROOT = "xgb.root"
    PARAM_KEY_RCV_BUF = "xgb.rcv_buf"

    RUNNER_CTX_SERVER_ADDR = "server_addr"
    RUNNER_CTX_PORT = "port"
    RUNNER_CTX_CLIENT_NAME = "client_name"
    RUNNER_CTX_NUM_ROUNDS = "num_rounds"
    RUNNER_CTX_WORLD_SIZE = "world_size"
    RUNNER_CTX_RANK = "rank"
    RUNNER_CTX_MODEL_DIR = "model_dir"

    EXIT_CODE_CANT_START_XGB = 101
