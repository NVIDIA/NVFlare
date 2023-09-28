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
import logging

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import NotAuthorized
from nvflare.fuel.hci.proto import InternalCommands

logger = logging.getLogger(__name__)


class SiteSecurity:
    """
    SiteSecurity is the function is to check if the command is authorized to execute. SiteSecurity use the
    FLContext data to perform the authorization security check. Based on the
    check result, set the AUTHORIZATION_RESULT and AUTHORIZATION_REASON accordingly.

    """

    def __init__(self) -> None:
        super().__init__()

    def authorization_check(self, engine, command, fl_ctx: FLContext):
        authorized = True
        reasons = ""
        if not InternalCommands.contains_commmand(command):
            fl_ctx.set_prop(FLContextKey.COMMAND_NAME, command, sticky=False)
            engine.fire_event(EventType.AUTHORIZE_COMMAND_CHECK, fl_ctx)

            authorization_result = fl_ctx.get_prop(FLContextKey.AUTHORIZATION_RESULT, True)
            if not authorization_result:
                reasons = fl_ctx.get_prop(FLContextKey.AUTHORIZATION_REASON, "")
                logger.error(f"Authorization failed. Reason: {reasons}")
                fl_ctx.remove_prop(FLContextKey.AUTHORIZATION_RESULT)
                fl_ctx.remove_prop(FLContextKey.AUTHORIZATION_REASON)
                authorized = False
            else:
                exceptions = fl_ctx.get_prop(FLContextKey.EXCEPTIONS)
                if exceptions and isinstance(exceptions, dict):
                    for handler_name, ex in exceptions.items():
                        if isinstance(ex, NotAuthorized):
                            logger.error(f"NotAuthorized to execute. Reason: {ex}")
                            fl_ctx.remove_prop(FLContextKey.AUTHORIZATION_RESULT)
                            fl_ctx.remove_prop(FLContextKey.AUTHORIZATION_REASON)
                            authorized = False
        return authorized, reasons
