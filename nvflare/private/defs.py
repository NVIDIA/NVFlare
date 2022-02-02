# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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


class SpecialTaskName(object):

    TRY_AGAIN = "__try_again__"
    END_RUN = "__end_run__"


class TaskConstant(object):
    WAIT_TIME = "__wait_time__"


class EngineConstant(object):

    FEDERATE_CLIENT = "federate_client"
    FL_TOKEN = "fl_token"
    CLIENT_TOKEN_FILE = "client_token.txt"
    ENGINE_TASK_NAME = "engine_task_name"


class InfoCollectorTopic(object):

    SHOW_STATS = "info.show_stats"
    SHOW_ERRORS = "info.show_errors"
    RESET_ERRORS = "info.reset_errors"


class ComponentCallerTopic(object):

    CALL_COMPONENT = "comp_caller.call"


class TrainingTopic(object):

    START = "train.start"
    ABORT = "train.abort"
    ABORT_TASK = "train.abort_task"
    DELETE_RUN = "train.delete_run"
    DEPLOY = "train.deploy"
    SHUTDOWN = "train.shutdown"
    RESTART = "train.restart"
    CHECK_STATUS = "train.check_status"
    SET_RUN_NUMBER = "train.set_run_number"


class RequestHeader(object):

    RUN_NUM = "run_number"
    APP_NAME = "app_name"
    CONTROL_COMMAND = "control_command"
    CALL_NAME = "call_name"
    COMPONENT_TARGET = "component_target"


class SysCommandTopic(object):

    SYS_INFO = "sys.info"
    SHELL = "sys.shell"


class ControlCommandTopic(object):

    DO_COMMAND = "control.do_command"


class ControlCommandName(object):

    ABORT_TASK = "abort_task"
    END_RUN = "end_run"


class ClientStatusKey(object):

    RUN_NUM = "run_number"
    CURRENT_TASK = "current_task"
    STATUS = "status"
    APP_NAME = "app_name"
