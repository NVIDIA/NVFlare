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

import json
from datetime import datetime
from enum import Enum
from typing import List

from .table import Table

LINE_END = "\x03"  # Indicates the end of the line (end of text)
ALL_END = "\x04"  # Marks the end of a complete transmission (End of Transmission)

MAX_BLOCK_SIZE = 1024


class ProtoKey(object):

    TIME = "time"
    DATA = "data"
    META = "meta"
    TYPE = "type"
    STRING = "string"
    TABLE = "table"
    ROWS = "rows"
    DICT = "dict"
    SUCCESS = "success"
    ERROR = "error"
    SHUTDOWN = "shutdown"
    COMMAND = "command"
    TOKEN = "token"
    DETAILS = "details"
    STATUS = "status"


class MetaKey(object):

    STATUS = "status"
    INFO = "info"
    JOB_ID = "job_id"
    DATA_TYPE = "data_type"
    JOB_META = "job_meta"
    JOB_DATA = "job_data"
    WORKSPACE = "workspace"
    JOB_DOWNLOAD_URL = "job_download_url"
    APP_NAME = "app_name"
    SERVER_STATUS = "server_status"
    SERVER_START_TIME = "server_start_time"
    CLIENT_NAME = "client_name"
    CLIENT_LAST_CONNECT_TIME = "client_last_conn_time"
    CLIENTS = "clients"
    CLIENT_STATUS = "client_status"
    JOBS = "jobs"
    JOB_NAME = "job_name"
    SUBMIT_TIME = "submit_time"
    DURATION = "duration"
    CMD_TIMEOUT = "cmd_timeout"
    CUSTOM_PROPS = "custom_props"
    CMD_PROPS = "cmd_props"
    FILES = "files"
    CMD_NAME = "cmd_name"
    TX_ID = "tx_id"
    FOLDER_NAME = "folder_name"
    LOCATION = "location"


class MetaStatusValue(object):

    OK = "ok"
    SYNTAX_ERROR = "syntax_error"
    NOT_AUTHORIZED = "not_authorized"
    NOT_AUTHENTICATED = "not_authenticated"
    ERROR = "error"
    INTERNAL_ERROR = "internal_error"
    INVALID_TARGET = "invalid_target"
    INVALID_JOB_DEFINITION = "invalid_job_def"
    INVALID_JOB_ID = "invalid_job_id"
    JOB_RUNNING = "job_running"
    JOB_NOT_RUNNING = "job_not_running"
    CLIENTS_RUNNING = "clients_running"
    NO_JOBS = "no_jobs"
    NO_REPLY = "no_reply"
    NO_CLIENTS = "no_clients"


class CredentialType(str, Enum):

    PASSWORD = "password"
    CERT = "cert"
    LOCAL_CERT = "local_cert"


class InternalCommands(object):

    PWD_LOGIN = "_login"
    CERT_LOGIN = "_cert_login"
    LOGOUT = "_logout"
    GET_CMD_LIST = "_commands"
    CHECK_SESSION = "_check_session"
    LIST_SESSIONS = "list_sessions"

    commands = [PWD_LOGIN, CERT_LOGIN, LOGOUT, GET_CMD_LIST, CHECK_SESSION, LIST_SESSIONS]

    @classmethod
    def contains_commmand(cls, command: str):
        return command in InternalCommands.commands


class ConfirmMethod(object):

    AUTH = "auth"
    PASSWORD = "pwd"
    YESNO = "yesno"
    USER_NAME = "username"


class Buffer(object):
    def __init__(self):
        """Buffer to append to for :class:`nvflare.fuel.hci.conn.Connection`."""
        self.meta = {}
        self.data = []
        self.output = {ProtoKey.TIME: f"{format(datetime.now())}", ProtoKey.DATA: self.data, ProtoKey.META: self.meta}

    def append_table(self, headers: List[str], name=None) -> Table:
        meta_rows = []
        if name:
            self.meta.update({name: meta_rows})
        t = Table(headers, meta_rows)
        self.data.append({ProtoKey.TYPE: ProtoKey.TABLE, ProtoKey.ROWS: t.rows})
        return t

    def update_meta(self, meta: dict):
        if meta:
            self.meta.update(meta)

    def append_string(self, data: str, meta: dict = None):
        self.data.append({ProtoKey.TYPE: ProtoKey.STRING, ProtoKey.DATA: data})
        self.update_meta(meta)

    def append_dict(self, data: dict, meta: dict = None):
        self.data.append({ProtoKey.TYPE: ProtoKey.DICT, ProtoKey.DATA: data})
        self.update_meta(meta)

    def append_success(self, data: str, meta: dict = None):
        self.data.append({ProtoKey.TYPE: ProtoKey.SUCCESS, ProtoKey.DATA: data})
        if not meta:
            meta = make_meta(MetaStatusValue.OK, data)
        self.update_meta(meta)

    def append_error(self, data: str, meta: dict = None):
        self.data.append({ProtoKey.TYPE: ProtoKey.ERROR, ProtoKey.DATA: data})
        if not meta:
            meta = make_meta(MetaStatusValue.ERROR, data)
        self.update_meta(meta)

    def append_command(self, cmd: str):
        self.data.append({ProtoKey.TYPE: ProtoKey.COMMAND, ProtoKey.DATA: cmd})

    def append_token(self, token: str):
        self.data.append({ProtoKey.TYPE: ProtoKey.TOKEN, ProtoKey.DATA: token})

    def append_shutdown(self, msg: str):
        self.data.append({ProtoKey.TYPE: ProtoKey.SHUTDOWN, ProtoKey.DATA: msg})

    def encode(self):
        if len(self.data) <= 0:
            return None

        return json.dumps(self.output)

    def reset(self):
        self.data = []
        self.meta = {}
        self.output = {ProtoKey.TIME: f"{format(datetime.now())}", ProtoKey.DATA: self.data, ProtoKey.META: self.meta}


def make_error(data: str):
    buf = Buffer()
    buf.append_error(data)
    return buf.output


def validate_proto(line: str):
    """Validate that the line being received is of the expected format.

    Args:
        line: str containing a JSON document

    Returns: deserialized JSON document
    """
    all_types = [
        ProtoKey.STRING,
        ProtoKey.SUCCESS,
        ProtoKey.ERROR,
        ProtoKey.TABLE,
        ProtoKey.COMMAND,
        ProtoKey.TOKEN,
        ProtoKey.SHUTDOWN,
        ProtoKey.DICT,
    ]
    types_with_data = [
        ProtoKey.STRING,
        ProtoKey.SUCCESS,
        ProtoKey.ERROR,
        ProtoKey.DICT,
        ProtoKey.COMMAND,
        ProtoKey.TOKEN,
        ProtoKey.SHUTDOWN,
    ]
    try:
        json_data = json.loads(line)
        assert isinstance(json_data, dict)
        assert ProtoKey.DATA in json_data
        data = json_data[ProtoKey.DATA]
        assert isinstance(data, list)
        for item in data:
            assert isinstance(item, dict)
            assert ProtoKey.TYPE in item
            it = item[ProtoKey.TYPE]
            assert it in all_types

            if it in types_with_data:
                item_data = item.get(ProtoKey.DATA, None)
                assert item_data is not None
                assert isinstance(item_data, str) or isinstance(item_data, dict)
            elif it == ProtoKey.TABLE:
                assert ProtoKey.ROWS in item
                rows = item[ProtoKey.ROWS]
                assert isinstance(rows, list)
                for row in rows:
                    assert isinstance(row, list)

        return json_data
    except Exception:
        return None


def make_meta(status: str, info: str = "", extra: dict = None) -> dict:
    meta = {MetaKey.STATUS: status, MetaKey.INFO: info}
    if extra:
        meta.update(extra)
    return meta
