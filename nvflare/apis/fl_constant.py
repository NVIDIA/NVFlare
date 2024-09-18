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
    UNSAFE_JOB = "UNSAFE_JOB"
    EARLY_TERMINATION = "EARLY_TERMINATION"
    SERVER_NOT_READY = "SERVER_NOT_READY"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"


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
    AUDIT_EVENT_ID = "__audit_event_id__"
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
    SIMULATE_MODE = "__simulate_mode__"
    SP_END_POINT = "__sp_end_point__"
    JOB_INFO = "__job_info__"
    JOB_META = "__job_meta__"
    CURRENT_JOB_ID = "__current_job_id__"
    JOB_RUN_NUMBER = "__job_run_number__"
    JOB_DEPLOY_DETAIL = "__job_deploy_detail__"
    FATAL_SYSTEM_ERROR = "__fatal_system_error__"
    JOB_IS_UNSAFE = "__job_is_unsafe__"
    CUSTOM_PROPS = "__custom_props__"
    EXCEPTIONS = "__exceptions__"


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
    EXCEPTIONS = ReservedKey.EXCEPTIONS
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
    SIMULATE_MODE = ReservedKey.SIMULATE_MODE
    SP_END_POINT = ReservedKey.SP_END_POINT
    JOB_INFO = ReservedKey.JOB_INFO
    JOB_META = ReservedKey.JOB_META
    CURRENT_JOB_ID = ReservedKey.CURRENT_JOB_ID
    JOB_RUN_NUMBER = ReservedKey.JOB_RUN_NUMBER
    JOB_DEPLOY_DETAIL = ReservedKey.JOB_DEPLOY_DETAIL
    CUSTOM_PROPS = ReservedKey.CUSTOM_PROPS
    JOB_SCOPE_NAME = "__job_scope_name__"
    EFFECTIVE_JOB_SCOPE_NAME = "__effective_job_scope_name__"
    SCOPE_PROPERTIES = "__scope_props__"
    SCOPE_OBJECT = "__scope_object__"
    FATAL_SYSTEM_ERROR = ReservedKey.FATAL_SYSTEM_ERROR
    COMMUNICATION_ERROR = "Flare_communication_error__"
    UNAUTHENTICATED = "Flare_unauthenticated__"
    CLIENT_RESOURCE_SPECS = "__client_resource_specs"
    RESOURCE_CHECK_RESULT = "__resource_check_result"
    JOB_PARTICIPANTS = "__job_participants"
    JOB_BLOCK_REASON = "__job_block_reason"  # why the job should be blocked from scheduling
    SSID = "__ssid__"
    CLIENT_TOKEN = "__client_token"
    AUTHORIZATION_RESULT = "_authorization_result"
    AUTHORIZATION_REASON = "_authorization_reason"
    DISCONNECTED_CLIENT_NAME = "_disconnected_client_name"
    RECONNECTED_CLIENT_NAME = "_reconnected_client_name"

    CLIENT_REGISTER_DATA = "_client_register_data"
    SECURITY_ITEMS = "_security_items"
    COMMAND_NAME = "_command_name"
    SITE_NAME = "__site_name"
    USER_NAME = "__user_name"
    USER_ORG = "__user_org"
    USER_ROLE = "__user_role"
    SUBMITTER_NAME = "_submitterName"
    SUBMITTER_ORG = "_submitterOrg"
    SUBMITTER_ROLE = "_submitterRole"
    COMPONENT_BUILD_ERROR = "__component_build_error__"
    COMPONENT_CONFIG = "__component_config__"
    COMPONENT_NODE = "__component_node__"
    CONFIG_CTX = "__config_ctx__"
    FILTER_DIRECTION = "__filter_dir__"
    ROOT_URL = "__root_url__"  # the URL for accessing the FL Server
    NOT_READY_TO_END_RUN = "not_ready_to_end_run__"  # component sets this to indicate it's not ready to end run yet


class ReservedTopic(object):

    END_RUN = "__end_run__"
    ABORT_ASK = "__abort_task__"
    DO_TASK = "__do_task__"
    AUX_COMMAND = "__aux_command__"
    SYNC_RUNNER = "__sync_runner__"
    JOB_HEART_BEAT = "__job_heartbeat__"
    TASK_CHECK = "__task_check__"


class AdminCommandNames(object):

    SUBMIT_JOB = "submit_job"
    LIST_JOBS = "list_jobs"
    GET_JOB_META = "get_job_meta"
    DOWNLOAD_JOB = "download_job"
    DOWNLOAD_JOB_FILE = "download_job_file"
    ABORT_JOB = "abort_job"
    DELETE_JOB = "delete_job"
    CLONE_JOB = "clone_job"
    DELETE_WORKSPACE = "delete_workspace"
    CHECK_RESOURCES = "check_resources"
    DEPLOY_APP = "deploy_app"
    START_APP = "start_app"
    CHECK_STATUS = "check_status"
    ADMIN_CHECK_STATUS = "admin_check_status"
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
    SYS_INFO = "sys_info"
    REPORT_RESOURCES = "report_resources"
    REPORT_ENV = "report_env"
    SHOW_SCOPES = "show_scopes"
    CALL = "call"
    SHELL_PWD = "pwd"
    SHELL_LS = "ls"
    SHELL_CAT = "cat"
    SHELL_HEAD = "head"
    SHELL_TAIL = "tail"
    SHELL_GREP = "grep"
    APP_COMMAND = "app_command"


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
    RESET_ERRORS = "reset_errors"
    UPDATE_RUN_STATUS = "update_run_status"
    HANDLE_DEAD_JOB = "handle_dead_job"
    SERVER_STATE = "server_state"
    APP_COMMAND = "app_command"


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
    TURN_TO_COLD = "__turn_to_cold__"
    REASON = "reason"


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

    KEYS = [
        ReservedKey.ENGINE,
        ReservedKey.MANAGER,
        ReservedKey.RUNNER,
        FLContextKey.SCOPE_PROPERTIES,
        FLContextKey.SCOPE_OBJECT,
        FLContextKey.WORKSPACE_OBJECT,
        FLContextKey.TASK_DATA,
        FLContextKey.SHAREABLE,
    ]


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
    CONNECTION = "_conn"
    CHILD_PROCESS = "_child_process"
    STATUS = "_status"
    JOB_ID = "_job_id"
    PARTICIPANTS = "_participants"
    PROCESS_FINISHED = "_process_finished"
    PROCESS_EXE_ERROR = "_process_exe_error"
    PROCESS_RETURN_CODE = "_process_return_code"


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
    APP_DEPLOYER = "app_deployer"
    DEFAULT_APP_DEPLOYER = "default_app_deployer"
    JOB_META_VALIDATOR = "job_meta_validator"
    FED_CLIENT = "fed_client"
    RUN_MANAGER = "run_manager"


class JobConstants:
    SERVER_JOB_CONFIG = "config_fed_server.json"
    CLIENT_JOB_CONFIG = "config_fed_client.json"
    META_FILE = "meta.json"
    META = "meta"


class WorkspaceConstants:
    """hard coded file names inside the workspace folder."""

    STARTUP_FOLDER_NAME = "startup"
    SITE_FOLDER_NAME = "local"
    CUSTOM_FOLDER_NAME = "custom"

    LOGGING_CONFIG = "log.config"
    DEFAULT_LOGGING_CONFIG = LOGGING_CONFIG + ".default"
    AUDIT_LOG = "audit.log"
    LOG_FILE_NAME = "log.txt"
    STATS_POOL_SUMMARY_FILE_NAME = "stats_pool_summary.json"
    STATS_POOL_RECORDS_FILE_NAME = "stats_pool_records.csv"

    # these two files is used by shell scripts to determine restart / shutdown
    RESTART_FILE = "restart.fl"
    SHUTDOWN_FILE = "shutdown.fl"

    WORKSPACE_PREFIX = ""
    APP_PREFIX = "app_"

    SERVER_STARTUP_CONFIG = "fed_server.json"
    CLIENT_STARTUP_CONFIG = "fed_client.json"

    SERVER_APP_CONFIG = JobConstants.SERVER_JOB_CONFIG
    CLIENT_APP_CONFIG = JobConstants.CLIENT_JOB_CONFIG

    JOB_META_FILE = "meta.json"

    AUTHORIZATION_CONFIG = "authorization.json"
    DEFAULT_AUTHORIZATION_CONFIG = AUTHORIZATION_CONFIG + ".default"
    RESOURCES_CONFIG = "resources.json"
    DEFAULT_RESOURCES_CONFIG = RESOURCES_CONFIG + ".default"
    PRIVACY_CONFIG = "privacy.json"
    SAMPLE_PRIVACY_CONFIG = PRIVACY_CONFIG + ".sample"
    JOB_RESOURCES_CONFIG = "job_resources.json"

    ADMIN_STARTUP_CONFIG = "fed_admin.json"


class SiteType:
    SERVER = "server"
    CLIENT = "client"
    ALL = "@ALL"


class SystemConfigs:
    STARTUP_CONF = "start_config"
    RESOURCES_CONF = "resources_config"
    APPLICATION_CONF = "application_config"


class SecureTrainConst:
    SSL_ROOT_CERT = "ssl_root_cert"
    SSL_CERT = "ssl_cert"
    PRIVATE_KEY = "ssl_private_key"


class FLMetaKey:
    NUM_STEPS_CURRENT_ROUND = "NUM_STEPS_CURRENT_ROUND"
    PROCESSED_ALGORITHM = "PROCESSED_ALGORITHM"
    PROCESSED_KEYS = "PROCESSED_KEYS"
    INITIAL_METRICS = "initial_metrics"
    FILTER_HISTORY = "filter_history"
    CONFIGS = "configs"
    VALIDATE_TYPE = "validate_type"
    START_ROUND = "start_round"
    CURRENT_ROUND = "current_round"
    TOTAL_ROUNDS = "total_rounds"
    JOB_ID = "job_id"
    SITE_NAME = "site_name"
    PROCESS_RC_FILE = "_process_rc.txt"
    SUBMIT_MODEL_NAME = "submit_model_name"


class FilterKey:
    IN = "in"
    OUT = "out"
    INOUT = "inout"
    DELIMITER = "/"


class ConfigVarName:
    # These variables can be set in job config files (config_fed_server or config_fed_client)
    RUNNER_SYNC_TIMEOUT = "runner_sync_timeout"  # client: runner sync message timeout
    MAX_RUNNER_SYNC_TIMEOUT = "max_runner_sync_timeout"  # client: max timeout of runner sync attempts
    TASK_CHECK_TIMEOUT = "task_check_timeout"  # client: timeout for task_check message (before submitting task)

    # client: how long to wait before sending task_check again (if previous task_check fails)
    TASK_CHECK_INTERVAL = "task_check_interval"

    # client: how often to send job heartbeats
    JOB_HEARTBEAT_INTERVAL = "job_heartbeat_interval"

    # client and server: max time to wait for components to become ready for end-run
    END_RUN_READINESS_TIMEOUT = "end_run_readiness_timeout"

    # client and server: how long to wait before checking components end-run readiness again (if previous check fails)
    END_RUN_READINESS_CHECK_INTERVAL = "end_run_readiness_check_interval"

    # client: timeout for getTask requests
    GET_TASK_TIMEOUT = "get_task_timeout"

    # client: timeout for submitTaskResult requests
    SUBMIT_TASK_RESULT_TIMEOUT = "submit_task_result_timeout"

    # client and server: max number of request workers for reliable message
    RM_MAX_REQUEST_WORKERS = "rm_max_request_workers"

    # client and server: query interval for reliable message
    RM_QUERY_INTERVAL = "rm_query_interval"

    # server: wait this long since client death report before treating the client as dead/disconnected
    DEAD_CLIENT_GRACE_PERIOD = "dead_client_grace_period"

    # server: wait this long since job schedule time before starting to check dead/disconnected clients
    DEAD_CLIENT_CHECK_LEAD_TIME = "dead_client_check_lead_time"

    # customized nvflare decomposers module name
    DECOMPOSER_MODULE = "nvflare_decomposers"

    # client and server: max amount of time to wait for communication cell to be created
    CELL_WAIT_TIMEOUT = "cell_wait_timeout"


class SystemVarName:
    """
    These vars are automatically generated by FLARE and can be referenced in job config (config_fed_client and
    config_fed_server). For example, you can reference SITE_NAME as "{SITE_NAME}" in your config.

    To avoid potential conflict with user-defined var names, these var names are in UPPER CASE.
    """

    SITE_NAME = "SITE_NAME"  # name of client site or server
    WORKSPACE = "WORKSPACE"  # directory of the workspace
    JOB_ID = "JOB_ID"  # Job ID
    ROOT_URL = "ROOT_URL"  # the URL of the Service Provider (server)
    SECURE_MODE = "SECURE_MODE"  # whether the system is running in secure mode
    JOB_CUSTOM_DIR = "JOB_CUSTOM_DIR"  # custom dir of the job
    PYTHONPATH = "PYTHONPATH"


class RunnerTask:

    INIT = "init"
    TASK_EXEC = "task_exec"
    END_RUN = "end_run"
