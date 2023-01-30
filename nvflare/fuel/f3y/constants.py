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

class ConnectorRequirementKey:

    URL = "url"
    HOST = "host"
    SECURE = "secure"           # bool: secure or not


class MessageHeaderKey:

    MSG_TYPE = "cellnet.msg_type"
    REQ_ID = "cellnet.req_id"
    REPLY_EXPECTED = "cellnet.reply_expected"
    TOPIC = "cellnet.topic"
    WAIT_UNTIL = "cellnet.wait_until"
    ORIGIN = "cellnet.origin"
    DESTINATION = "cellnet.destination"
    FROM_CELL = "cellnet.from"
    TO_CELL = "cellnet.to"
    CONN_URL = "cellnet.conn_url"
    CHANNEL = "cellnet.channel"
    RETURN_CODE = "cellnet.return_code"
    ERROR = "cellnet.error"
    CONTENT_TYPE = "cellnet.content_type"
    ROUTE = "cellnet.route"


class ContentType:

    BYTES = "bytes"
    FOBS = "fobs"       # FOBS coded
    NONE = "none"


class ReturnCode:

    OK = "ok"
    TIMEOUT = "timeout"
    COMM_ERROR = "comm_error"
    PROCESS_EXCEPTION = "process_exception"   # receiver error processing request


class MessageType:

    REQ = "req"
    REPLY = "reply"
    RETURN = "return"   # return to sender due to forward error


class CellPropertyKey:

    FQCN = "fqcn"
