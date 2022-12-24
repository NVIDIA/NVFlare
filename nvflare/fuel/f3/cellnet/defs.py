#  Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from nvflare.fuel.f3.message import Message


class ConnectorRequirementKey:

    URL = "url"
    HOST = "host"
    SECURE = "secure"           # bool: secure or not


CELLNET_PREFIX = "cn."


class MessageHeaderKey:

    MSG_TYPE = CELLNET_PREFIX + "msg_type"
    REQ_ID = CELLNET_PREFIX + "req_id"
    REPLY_EXPECTED = CELLNET_PREFIX + "reply_expected"
    TOPIC = CELLNET_PREFIX + "topic"
    WAIT_UNTIL = CELLNET_PREFIX + "wait_until"
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


class Encoding:

    BYTES = "bytes"
    FOBS = "fobs"       # FOBS coded
    NONE = "none"


class ReturnCode:

    OK = "ok"
    TIMEOUT = "timeout"
    COMM_ERROR = "comm_error"
    INVALID_REQUEST = "invalid_request"
    PROCESS_EXCEPTION = "process_exception"   # receiver error processing request


class MessageType:

    REQ = "req"
    REPLY = "reply"
    RETURN = "return"   # return to sender due to forward error


class CellPropertyKey:

    FQCN = "fqcn"


class TargetCellUnreachable(Exception):
    pass


class TargetMessage:

    def __init__(
            self,
            target: str,
            channel: str,
            topic: str,
            message: Message,
    ):
        self.target = target
        self.channel = channel
        self.topic = topic
        self.message = message
