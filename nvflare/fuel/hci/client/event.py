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

from abc import ABC, abstractmethod

from nvflare.fuel.common.ctx import SimpleContext


class EventType:

    WAIT_FOR_SERVER_ADDR = "wait_for_server_addr"
    SERVER_ADDR_OBTAINED = "server_addr_obtained"
    SESSION_CLOSED = "session_closed"  # close the current session
    LOGIN_SUCCESS = "login_success"  # logged in to server
    LOGIN_FAILURE = "login_failure"  # cannot log in to server
    TRYING_LOGIN = "trying_login"  # still try to log in
    SP_ADDR_CHANGED = "sp_addr_changed"  # service provider address changed
    SESSION_TIMEOUT = "session_timeout"  # server timed out current session
    BEFORE_LOGIN = "before_login"
    BEFORE_EXECUTE_CMD = "before_execute_cmd"
    BEFORE_DOWNLOAD_FILE = "before_download_file"
    AFTER_DOWNLOAD_FILE = "after_download_file"


class EventPropKey:

    MSG = "msg"
    USER_NAME = "user_name"
    CMD_NAME = "cmd_name"
    CMD_CTX = "cmd_ctx"
    CUSTOM_PROPS = "custom_props"


class EventContext(SimpleContext):
    def get_custom_prop(self, key: str, default):
        props = self.get_prop(EventPropKey.CUSTOM_PROPS)
        if not props:
            return default
        return props.get(key, default)

    def set_custom_prop(self, key: str, value):
        props = self.get_prop(EventPropKey.CUSTOM_PROPS)
        if not props:
            props = {}
            self.set_prop(EventPropKey.CUSTOM_PROPS, props)
        props[key] = value


class EventHandler(ABC):
    @abstractmethod
    def handle_event(self, event_type: str, ctx: SimpleContext):
        pass
