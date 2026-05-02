# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.apis.fl_context import FLContext
from nvflare.fuel.utils.log_utils import get_module_logger

ACTION = "_action"
MESSAGE = "_message"

NIS = "Not In Service"
ABORT_RUN = "Abort Run"
SERVICE = "In Service"
DEFAULT_SERVICE_SESSION_ID = "NA"


class ServerState:
    NOT_IN_SERVICE = {ACTION: NIS, MESSAGE: "Server not in service"}
    ABORT_CURRENT_RUN = {ACTION: ABORT_RUN, MESSAGE: "Abort current run"}
    IN_SERVICE = {ACTION: SERVICE, MESSAGE: "Server in service"}

    logger = get_module_logger(__module__, __qualname__)

    def __init__(self, host: str = "", port: str = "", ssid: str = DEFAULT_SERVICE_SESSION_ID) -> None:
        self.host = host
        self.service_port = port
        self.ssid = ssid

    def register(self, fl_ctx: FLContext) -> dict:
        return ServerState.NOT_IN_SERVICE

    def heartbeat(self, fl_ctx: FLContext) -> dict:
        return ServerState.NOT_IN_SERVICE

    def get_task(self, fl_ctx: FLContext) -> dict:
        return ServerState.NOT_IN_SERVICE

    def submit_result(self, fl_ctx: FLContext) -> dict:
        return ServerState.NOT_IN_SERVICE

    def aux_communicate(self, fl_ctx: FLContext) -> dict:
        return ServerState.NOT_IN_SERVICE


class HotState(ServerState):
    def register(self, fl_ctx: FLContext) -> dict:
        return ServerState.IN_SERVICE

    def heartbeat(self, fl_ctx: FLContext) -> dict:
        return ServerState.IN_SERVICE

    def get_task(self, fl_ctx: FLContext) -> dict:
        return ServerState.IN_SERVICE

    def submit_result(self, fl_ctx: FLContext) -> dict:
        return ServerState.IN_SERVICE

    def aux_communicate(self, fl_ctx: FLContext) -> dict:
        return ServerState.IN_SERVICE
