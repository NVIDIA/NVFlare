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

import time
import uuid

from nvflare.apis.fl_constant import CellMessageAuthHeaderKey

# this import is to let existing scripts import from nvflare.private.defs
from nvflare.fuel.f3.cellnet.defs import CellChannel, CellChannelTopic, SSLConstants  # noqa: F401
from nvflare.fuel.f3.cellnet.utils import new_cell_message  # noqa: F401
from nvflare.fuel.hci.server.constants import ConnProps


class SpecialTaskName(object):

    TRY_AGAIN = "__try_again__"
    END_RUN = "__end_run__"


class TaskConstant(object):
    WAIT_TIME = "__wait_time__"


class EngineConstant(object):

    FEDERATE_CLIENT = "federate_client"
    AUTH_TOKEN = "auth_token"
    AUTH_TOKEN_SIGNATURE = "auth_token_signature"
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
    SET_JOB_ID = "train.set_job_id"
    CHECK_RESOURCE = "scheduler.check_resource"
    ALLOCATE_RESOURCE = "scheduler.allocate_resource"
    CANCEL_RESOURCE = "scheduler.cancel_resource"
    START_JOB = "train.start_job"
    GET_SCOPES = "train.get_scopes"
    NOTIFY_JOB_STATUS = "train.notify_job_status"
    CONFIGURE_JOB_LOG = "train.configure_job_log"


class RequestHeader(object):

    JOB_ID = "job_id"
    JOB_STATUS = "job_status"
    TOPIC = "topic"
    JOB_META = "job_meta"
    APP_NAME = "app_name"
    CONTROL_COMMAND = "control_command"
    CALL_NAME = "call_name"
    COMPONENT_TARGET = "component_target"
    ADMIN_COMMAND = "admin_command"
    USER_NAME = ConnProps.USER_NAME
    USER_ORG = ConnProps.USER_ORG
    USER_ROLE = ConnProps.USER_ROLE
    SUBMITTER_NAME = ConnProps.SUBMITTER_NAME
    SUBMITTER_ORG = ConnProps.SUBMITTER_ORG
    SUBMITTER_ROLE = ConnProps.SUBMITTER_ROLE
    REQUIRE_AUTHZ = "require_authz"


class SysCommandTopic(object):

    SYS_INFO = "sys.info"
    SHELL = "sys.shell"
    REPORT_RESOURCES = "resource_manager.report_resources"
    REPORT_ENV = "sys.report_env"
    CONFIGURE_SITE_LOG = "sys.configure_site_log"


class ControlCommandTopic(object):

    DO_COMMAND = "control.do_command"


class ControlCommandName(object):

    ABORT_TASK = "abort_task"
    END_RUN = "end_run"


class ClientStatusKey(object):

    JOB_ID = "job_id"
    CURRENT_TASK = "current_task"
    STATUS = "status"
    APP_NAME = "app_name"
    CLIENT_NAME = "client_name"
    RUNNING_JOBS = "running_jobs"


class ScopeInfoKey(object):

    SCOPE_NAMES = "scope_names"
    DEFAULT_SCOPE = "default_scope"


# TODO:: Remove some of these constants
class AppFolderConstants:
    """hard coded file names inside the app folder."""

    CONFIG_TRAIN = "config_train.json"
    CONFIG_ENV = "environment.json"


ERROR_MSG_PREFIX = "NVFLARE_ERROR"


class CellMessageHeaderKeys:

    CLIENT_NAME = CellMessageAuthHeaderKey.CLIENT_NAME
    CLIENT_TYPE = "client_type"
    TOKEN = CellMessageAuthHeaderKey.TOKEN
    TOKEN_SIGNATURE = CellMessageAuthHeaderKey.TOKEN_SIGNATURE
    CLIENT_IP = "client_ip"
    PROJECT_NAME = "project_name"
    SSID = CellMessageAuthHeaderKey.SSID
    UNAUTHENTICATED = "unauthenticated"
    JOB_ID = "job_id"
    JOB_IDS = "job_ids"
    MESSAGE = "message"
    ABORT_JOBS = "abort_jobs"


class ClientType:
    RELAY = "relay"
    REGULAR = "regular"


AUTH_CLIENT_NAME_FOR_SJ = "server_job"


class JobFailureMsgKey:

    JOB_ID = "job_id"
    CODE = "code"
    REASON = "reason"


class InternalFLContextKey:

    CLIENT_REG_SESSION = "client_reg_session"


class ClientRegSession:
    def __init__(self, client_name: str):
        self.client_name = client_name
        self.nonce = str(uuid.uuid4())
        self.reg_start_time = time.time()
