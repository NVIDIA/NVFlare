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


class SSLConstants:
    """hard coded names related to SSL."""

    CERT = "ssl_cert"
    PRIVATE_KEY = "ssl_private_key"
    ROOT_CERT = "ssl_root_cert"


class CellChannel:

    CLIENT_MAIN = "admin"
    AUX_COMMUNICATION = "aux_communication"
    SERVER_MAIN = "task"
    SERVER_COMMAND = "server_command"
    SERVER_PARENT_LISTENER = "server_parent_listener"
    CLIENT_COMMAND = "client_command"
    CLIENT_SUB_WORKER_COMMAND = "client_sub_worker_command"
    MULTI_PROCESS_EXECUTOR = "multi_process_executor"
    SIMULATOR_RUNNER = "simulator_runner"
    RETURN_ONLY = "return_only"


class CellChannelTopic:

    Register = "register"
    Quit = "quit"
    GET_TASK = "get_task"
    SUBMIT_RESULT = "submit_result"
    HEART_BEAT = "heart_beat"
    EXECUTE_RESULT = "execute_result"
    FIRE_EVENT = "fire_event"
    REPORT_JOB_FAILURE = "report_job_failure"

    SIMULATOR_WORKER_INIT = "simulator_worker_init"
