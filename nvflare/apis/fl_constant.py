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

from enum import Enum


class ReturnCode(object):

    OK = "OK"

    BAD_PEER_CONTEXT = "BAD_PEER_CONTEXT"
    BAD_REQUEST_DATA = "BAD_REQUEST_DATA"
    BAD_TASK_DATA = "BAD_TASK_DATA"
    COMMUNICATION_ERROR = "COMMUNICATION_ERROR"
    ERROR = "ERROR"
    EXECUTION_EXCEPTION = "EXECUTION_EXCEPTION"
    EXECUTION_RESULT_ERROR = "EXECUTION_RESULT_ERROR"
    HANDLER_EXCEPTION = "HANDLER_EXCEPTION"
    MISSING_PEER_CONTEXT = "MISSING_PEER_CONTEXT"
    RUN_MISMATCH = "RUN_MISMATCH"
    TASK_ABORTED = "TASK_ABORTED"
    TASK_DATA_FILTER_ERROR = "TASK_DATA_FILTER_ERROR"
    TASK_RESULT_FILTER_ERROR = "TASK_RESULT_FILTER_ERROR"
    TASK_UNKNOWN = "TASK_UNKNOWN"
    TASK_UNSUPPORTED = "TASK_UNSUPPORTED"
    TOPIC_UNKNOWN = "TOPIC_UNKNOWN"
    MODEL_UNRECOGNIZED = "MODEL_UNRECOGNIZED"
    VALIDATE_TYPE_UNKNOWN = "VALIDATE_TYPE_UNKNOWN"
    EMPTY_RESULT = "EMPTY_RESULT"

    SERVER_NOT_READY = "SERVER_NOT_READY"


class MachineStatus(Enum):
    """Constants for machine status.

    Status Lifecycle
        STOPPED <-> STARTING -> STARTED -> STOPPING -> STOPPED
    """

    STARTING = "starting"
    STARTED = "started"
    STOPPING = "stopping"
    STOPPED = "stopped"


class ReservedKey(object):

    MANAGER = "__manager__"
    ENGINE = "__engine__"
    AUX_RUNNER = "__aux_runner__"
    RUN_NUM = "__run_num__"
    IDENTITY_NAME = "__identity_name__"  # identity of the endpoint (e.g. client name)
    PEER_CTX = "__peer_ctx__"
    RC = "__rc__"
    COOKIE_JAR = "__cookie_jar__"
    WORKSPACE_ROOT = "__workspace_root__"
    APP_ROOT = "__app_root__"
    CLIENT_NAME = "__client_name__"
    TASK_NAME = "__task_name__"
    TASK_DATA = "__task_data__"
    TASK_RESULT = "__task_result__"
    TASK_ID = "__task_id__"
    EVENT_ID = "__event_id__"
    IS_RESEND = "__is_resend__"
    RUNNER = "__runner__"
    WORKFLOW = "__workflow__"
    REPLY = "__reply__"
    EVENT_ORIGIN = "__event_origin__"
    EVENT_ORIGIN_SITE = "__event_origin_site__"
    EVENT_DATA = "__event_data__"
    EVENT_SCOPE = "__event_scope__"
    RUN_ABORT_SIGNAL = "__run_abort_signal__"
    SHAREABLE = "__shareable__"
    SHARED_FL_CONTEXT = "__shared_fl_context__"
    ARGS = "__args__"
    WORKSPACE_OBJECT = "__workspace_object__"
    RANK_NUMBER = "__rank_number__"
    NUM_OF_PROCESSES = "__num_of_processes__"
    FROM_RANK_NUMBER = "__from_rank_number__"
    SECURE_MODE = "__secure_mode__"
    SP_END_POINT = "__sp_end_point__"
    JOB_INFO = "__job_info__"
    CURRENT_JOB_ID = "__current_job_id__"
    JOB_RUN_NUMBER = "__job_run_number__"


class FLContextKey(object):

    TASK_NAME = ReservedKey.TASK_NAME
    TASK_DATA = ReservedKey.TASK_DATA
    TASK_RESULT = ReservedKey.TASK_RESULT
    TASK_ID = ReservedKey.TASK_ID
    EVENT_ID = ReservedKey.EVENT_ID
    EVENT_ORIGIN = ReservedKey.EVENT_ORIGIN
    EVENT_ORIGIN_SITE = ReservedKey.EVENT_ORIGIN_SITE
    EVENT_DATA = ReservedKey.EVENT_DATA
    EVENT_SCOPE = ReservedKey.EVENT_SCOPE
    CLIENT_NAME = ReservedKey.CLIENT_NAME
    WORKSPACE_ROOT = ReservedKey.WORKSPACE_ROOT
    CURRENT_RUN = ReservedKey.RUN_NUM
    APP_ROOT = ReservedKey.APP_ROOT
    PEER_CONTEXT = ReservedKey.PEER_CTX
    IS_CLIENT_TASK_RESEND = ReservedKey.IS_RESEND
    RUNNER = ReservedKey.RUNNER
    WORKFLOW = ReservedKey.WORKFLOW
    SHAREABLE = ReservedKey.SHAREABLE
    RUN_ABORT_SIGNAL = ReservedKey.RUN_ABORT_SIGNAL
    ARGS = ReservedKey.ARGS
    REPLY = ReservedKey.REPLY
    WORKSPACE_OBJECT = ReservedKey.WORKSPACE_OBJECT
    RANK_NUMBER = ReservedKey.RANK_NUMBER
    NUM_OF_PROCESSES = ReservedKey.NUM_OF_PROCESSES
    FROM_RANK_NUMBER = ReservedKey.FROM_RANK_NUMBER
    SECURE_MODE = ReservedKey.SECURE_MODE
    SP_END_POINT = ReservedKey.SP_END_POINT
    JOB_INFO = ReservedKey.JOB_INFO
    CURRENT_JOB_ID = ReservedKey.CURRENT_JOB_ID
    JOB_RUN_NUMBER = ReservedKey.JOB_RUN_NUMBER


class ReservedTopic(object):

    END_RUN = "__end_run__"
    ABORT_ASK = "__abort_task__"
    AUX_COMMAND = "__aux_command__"


class AdminCommandNames(object):

    SUBMIT_JOB = "submit_job"
    LIST_JOBS = "list_jobs"
    DOWNLOAD_JOB = "download_job"
    ABORT_JOB = "abort_job"
    DELETE_JOB = "delete_job"
    CLONE_JOB = "clone_job"
    DELETE_WORKSPACE = "delete_workspace"
    DEPLOY_APP = "deploy_app"
    START_APP = "start_app"
    CHECK_STATUS = "check_status"
    ABORT = "abort"
    ABORT_TASK = "abort_task"
    REMOVE_CLIENT = "remove_client"
    SHUTDOWN = "shutdown"
    RESTART = "restart"
    SET_TIMEOUT = "set_timeout"
    SHOW_STATS = "show_stats"
    SHOW_ERRORS = "show_errors"
    RESET_ERRORS = "reset_errors"
    AUX_COMMAND = "aux_command"


class ServerCommandNames(object):

    GET_RUN_INFO = "get_run_info"
    GET_TASK = "get_task"
    SUBMIT_UPDATE = "submit_update"
    AUX_COMMUNICATE = "aux_communicate"
    HEARTBEAT = "heartbeat"
    GET_CLIENTS = "get_clients"
    AUX_SEND = "aux_send"
    SHOW_STATS = "show_stats"
    GET_ERRORS = "get_errors"


class ServerCommandKey(object):

    COMMAND = "command"
    DATA = "data"
    FL_CONTEXT = "fl_context"
    PEER_FL_CONTEXT = "peer_fl_ctx"
    SHAREABLE = "shareable"
    TASK_NAME = "task_name"
    TASK_ID = "task_id"
    FL_CLIENT = "fl_client"
    TOPIC = "topic"
    AUX_REPLY = "aux_reply"
    JOB_ID = "job_id"
    CLIENTS = "clients"
    COLLECTOR = "collector"


class FedEventHeader(object):

    TIMESTAMP = "_timestamp"
    EVENT_TYPE = "_event_type"
    DIRECTION = "_direction"
    ORIGIN = "_origin"
    TARGETS = "_targets"


class EventScope(object):

    FEDERATION = "federation"
    LOCAL = "local"


class NonSerializableKeys(object):

    KEYS = [ReservedKey.ENGINE, ReservedKey.MANAGER, ReservedKey.RUNNER]


class LogMessageTag(object):

    DEBUG = "log/debug"
    ERROR = "log/error"
    EXCEPTION = "log/exception"
    INFO = "log/info"
    WARNING = "log/warning"
    CRITICAL = "log/critical"
    LOG_RECORD = "log_record"


class SnapshotKey(object):

    FL_CONTEXT = "fl_context"
    SERVER_RUNNER = "_Server_Runner"
    WORKSPACE = "_workspace"
    JOB_INFO = "_job_info"
    JOB_ID = "_job_id"
    JOB_CLIENTS = "_job_clients"


class RunProcessKey(object):

    LISTEN_PORT = "_listen_port"
    CONNECTION = "_comm_conn"
    CHILD_PROCESS = "_child_process"
    STATUS = "_status"
    JOB_ID = "_job_id"
    PARTICIPANTS = "_participants"


class SystemComponents(object):

    JOB_SCHEDULER = "job_scheduler"
    JOB_MANAGER = "job_manager"
    JOB_RUNNER = "job_runner"
    SERVER_RUNNER = "server_runner"
    CLIENT_RUNNER = "client_runner"
    CHECK_RESOURCE_PROCESSOR = "check_resource_processor"
    CANCEL_RESOURCE_PROCESSOR = "cancel_resource_processor"
    RESOURCE_MANAGER = "resource_manager"
    RESOURCE_CONSUMER = "resource_consumer"


class WorkspaceConstants:
    """hard coded file names inside the workspace folder."""

    LOGGING_CONFIG = "log.config"
    AUDIT_LOG = "audit.log"

    # these two files is used by shell scripts to determine restart / shutdown
    RESTART_FILE = "restart.fl"
    SHUTDOWN_FILE = "shutdown.fl"

    WORKSPACE_PREFIX = ""
    APP_PREFIX = "app_"
