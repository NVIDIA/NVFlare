# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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


class CommunicationMetaData(object):
    COMMAND = "command"
    TASK_NAME = "task_name"
    FL_CTX = "fl_ctx"
    EVENT_TYPE = "event_type"
    HANDLE_CONN = "handle_conn"
    EXE_CONN = "exe_conn"
    COMPONENTS = "MPExecutor_components"
    HANDLERS = "MPExecutor_handlers"
    LOCAL_EXECUTOR = "local_executor"
    RANK_NUMBER = "rank_number"
    SHAREABLE = "shareable"
    RELAYER = "relayer"
    RANK_PROCESS_STARTED = "rank_process_started"
    PARENT_PASSWORD = "parent process secret password"
    CHILD_PASSWORD = "client process secret password"


class CommunicateData(object):
    EXECUTE = "execute"
    HANDLE_EVENT = "handle_event"
    CLOSE = "close"
    SUB_WORKER_PROCESS = "sub_worker_process"
    MULTI_PROCESS_EXECUTOR = "multi_process_executor"


class MultiProcessCommandNames:
    INITIALIZE = "initialize"
    TASK_EXECUTION = "task_execution"
    FIRE_EVENT = "fire_event"
    EXECUTE_RESULT = "execute_result"
    CLOSE = "close"
