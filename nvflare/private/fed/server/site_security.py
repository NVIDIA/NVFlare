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
import logging
from typing import List

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import InternalCommands
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.hci.server.reg import CommandFilter


logger = logging.getLogger(__name__)


class SiteSecurityFilter(CommandFilter):
    def __init__(self) -> None:
        super().__init__()

    def pre_command(self, conn: Connection, args: List[str]) -> (bool, str):
        engine = conn.app_ctx
        command = args[0]

        self._set_security_data(conn, engine)
        filter_succeed, messages = self.security_check(engine, command)

        if filter_succeed:
            return True, ""
        else:
            return False, messages

    def security_check(self, engine, command):
        filter_succeed = True
        messages = ""
        if command not in InternalCommands.commands:
            with engine.new_context() as fl_ctx:
                fl_ctx.set_prop(FLContextKey.COMMAND_NAME, command, sticky=False)
                engine.fire_event(EventType.SECURITY_CHECK, fl_ctx)

                authentication_result = fl_ctx.get_prop(FLContextKey.AUTHENTICATION_RESULT, True)
                if not authentication_result:
                    reasons = fl_ctx.get_prop(FLContextKey.AUTHENTICATION_REASON, {})
                    messages = self._get_messages(reasons)
                    logger.error(f"Authentication failed. Reason: {messages}")
                    fl_ctx.remove_prop(FLContextKey.AUTHENTICATION_RESULT)
                    fl_ctx.remove_prop(FLContextKey.AUTHENTICATION_REASON)
                    filter_succeed = False

                authorization_result = fl_ctx.get_prop(FLContextKey.AUTHORIZATION_RESULT, True)
                if not authorization_result:
                    reasons = fl_ctx.get_prop(FLContextKey.AUTHORIZATION_REASON, {})
                    messages = self._get_messages(reasons)
                    logger.error(f"Authorization failed. Reason: {messages}")
                    filter_succeed = False
        return filter_succeed, messages

    def _get_messages(self, reasons):
        messages = ""
        for id, reason in reasons.items():
            messages += id + ": " + reason + "; "
        return messages

    def _set_security_data(self, conn: Connection, engine):
        security_items = {}
        with engine.new_context() as fl_ctx:
            security_items[FLContextKey.USER_NAME] = conn.get_prop(ConnProps.USER_NAME, "")
            security_items[FLContextKey.USER_ORG] = conn.get_prop(ConnProps.USER_ORG, "")
            security_items[FLContextKey.USER_ROLE] = conn.get_prop(ConnProps.USER_ROLE, "")
            fl_ctx.set_prop(FLContextKey.SECURITY_ITEMS, security_items, private=True, sticky=False)
