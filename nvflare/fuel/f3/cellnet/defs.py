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
CELLNET_PREFIX = "cn__"


class ConnectorRequirementKey:

    URL = "url"
    HOST = "host"
    SECURE = "secure"  # bool: secure or not


class MessageHeaderKey:

    MSG_TYPE = CELLNET_PREFIX + "msg_type"
    REQ_ID = CELLNET_PREFIX + "req_id"
    REPLY_EXPECTED = CELLNET_PREFIX + "reply_expected"
    TOPIC = CELLNET_PREFIX + "topic"
    ORIGIN = CELLNET_PREFIX + "origin"
    DESTINATION = CELLNET_PREFIX + "destination"
    FROM_CELL = CELLNET_PREFIX + "from"
    TO_CELL = CELLNET_PREFIX + "to"
    CONN_URL = CELLNET_PREFIX + "conn_url"
    CHANNEL = CELLNET_PREFIX + "channel"
    RETURN_CODE = CELLNET_PREFIX + "return_code"
    ERROR = CELLNET_PREFIX + "error"
    PAYLOAD_ENCODING = CELLNET_PREFIX + "payload_encoding"
    ROUTE = CELLNET_PREFIX + "route"
    ORIGINAL_HEADERS = CELLNET_PREFIX + "original_headers"
    SEND_TIME = CELLNET_PREFIX + "send_time"
    RETURN_REASON = CELLNET_PREFIX + "return_reason"
    SECURE = CELLNET_PREFIX + "secure"
    PAYLOAD_LEN = CELLNET_PREFIX + "payload_len"
    CLEAR_PAYLOAD_LEN = CELLNET_PREFIX + "clear_payload_len"
    ENCRYPTED = CELLNET_PREFIX + "encrypted"
    OPTIONAL = CELLNET_PREFIX + "optional"


class ReturnReason:

    CANT_FORWARD = "cant_forward"
    INTERCEPT = "intercept"


class MessagePropKey:

    ENDPOINT = CELLNET_PREFIX + "endpoint"
    COMMON_NAME = CELLNET_PREFIX + "common_name"


class Encoding:

    BYTES = "bytes"
    FOBS = "fobs"  # FOBS coded
    NONE = "none"


class ReturnCode:

    OK = "ok"
    TIMEOUT = "timeout"
    INVALID_TARGET = "invalid_target"
    TARGET_UNREACHABLE = "target_unreachable"
    COMM_ERROR = "comm_error"
    MSG_TOO_BIG = "msg_too_big"
    FILTER_ERROR = "filter_error"
    INVALID_REQUEST = "invalid_request"
    PROCESS_EXCEPTION = "process_exception"  # receiver error processing request
    AUTHENTICATION_ERROR = "authentication_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    INVALID_SESSION = "invalid_session"
    ABORT_RUN = "abort_run"
    UNAUTHENTICATED = "unauthenticated"


ALL_RETURN_CODES = [
    ReturnCode.OK,
    ReturnCode.TIMEOUT,
    ReturnCode.INVALID_TARGET,
    ReturnCode.TARGET_UNREACHABLE,
    ReturnCode.COMM_ERROR,
    ReturnCode.MSG_TOO_BIG,
    ReturnCode.FILTER_ERROR,
    ReturnCode.INVALID_REQUEST,
    ReturnCode.PROCESS_EXCEPTION,
    ReturnCode.AUTHENTICATION_ERROR,
    ReturnCode.SERVICE_UNAVAILABLE,
    ReturnCode.INVALID_SESSION,
    ReturnCode.ABORT_RUN,
    ReturnCode.UNAUTHENTICATED,
]


class MessageType:

    REQ = "req"
    REPLY = "reply"
    RETURN = "return"  # return to sender due to forward error


class CellPropertyKey:
    FQCN = "fqcn"


class TargetCellUnreachable(Exception):
    pass


class AuthenticationError(Exception):
    pass


class ServiceUnavailable(Exception):
    pass


class InvalidSession(Exception):
    pass


class AbortRun(Exception):
    pass


class InvalidRequest(Exception):
    pass


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
