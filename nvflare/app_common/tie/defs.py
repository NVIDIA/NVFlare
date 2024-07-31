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

    # default component config values
    CONFIG_TASK_TIMEOUT = 10
    START_TASK_TIMEOUT = 10

    TASK_CHECK_INTERVAL = 0.5
    JOB_STATUS_CHECK_INTERVAL = 2.0
    MAX_CLIENT_OP_INTERVAL = 90.0
    WORKFLOW_PROGRESS_TIMEOUT = 3600.0

    # message topics
    TOPIC_APP_REQUEST = "tie.request"
    TOPIC_CLIENT_DONE = "tie.client_done"

    # keys for Shareable between client and server
    MSG_KEY_EXIT_CODE = "tie.exit_code"
    MSG_KEY_OP = "tie.op"
    MSG_KEY_CONFIG = "tie.config"

    EXIT_CODE_CANT_START = 101
    EXIT_CODE_FATAL_ERROR = 102

    APP_CTX_FL_CONTEXT = "tie.fl_context"
