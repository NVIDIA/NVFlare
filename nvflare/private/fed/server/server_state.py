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

from __future__ import annotations

import logging

from nvflare.apis.fl_context import FLContext
from nvflare.apis.overseer_spec import SP

ACTION = "_action"
MESSAGE = "_messsage"

NIS = "Not In Service"
ABORT_RUN = "Abort Run"
SERVICE = "In Service"


class ServiceSession:
    def __init__(self, host: str = "", port: str = "", ssid: str = "") -> None:
        self.host = host
        self.service_port = port
        self.ssid = ssid


class ServerState(object):
    NOT_IN_SERVICE = {ACTION: NIS, MESSAGE: "Server not in service"}
    ABORT_CURRENT_RUN = {ACTION: ABORT_RUN, MESSAGE: "Abort current run"}
    IN_SSERVICE = {ACTION: SERVICE, MESSAGE: "Server in service"}

    def __init__(self, host: str = "", port: str = "", ssid: str = "") -> None:
        self.host = host
        self.service_port = port
        self.ssid = ssid

        self.logger = logging.getLogger("FederatedServer")

    def register(self, fl_ctx: FLContext) -> dict:
        pass

    def heartbeat(self, fl_ctx: FLContext) -> dict:
        pass

    def get_task(self, fl_ctx: FLContext) -> dict:
        pass

    def submit_result(self, fl_ctx: FLContext) -> dict:
        pass

    def aux_communicate(self, fl_ctx: FLContext) -> dict:
        pass

    def handle_sd_callback(self, sp: SP, fl_ctx: FLContext) -> ServerState:
        pass


class ColdState(ServerState):
    def __init__(self, host: str = "", port: str = "", ssid: str = "") -> None:
        super().__init__(host, port, ssid)

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

    def handle_sd_callback(self, sp: SP, fl_ctx: FLContext) -> ServerState:
        if sp and sp.primary is True:
            if sp.name == self.host and sp.fl_port == self.service_port:
                self.primary = True
                self.ssid = sp.service_session_id
                self.logger.info(
                    f"Got the primary sp: {sp.name} fl_port: {sp.fl_port} SSID: {sp.service_session_id}. "
                    f"Turning to hot."
                )
                return Cold2HotState(host=self.host, port=self.service_port, ssid=sp.service_session_id)
            else:
                self.primary = False
                return self
        return self


class Cold2HotState(ServerState):
    def __init__(self, host: str = "", port: str = "", ssid: str = "") -> None:
        super().__init__(host, port, ssid)

    def register(self, fl_ctx: FLContext) -> dict:
        return ServerState.IN_SSERVICE

    def heartbeat(self, fl_ctx: FLContext) -> dict:
        return ServerState.IN_SSERVICE

    def get_task(self, fl_ctx: FLContext) -> dict:
        return ServerState.ABORT_CURRENT_RUN

    def submit_result(self, fl_ctx: FLContext) -> dict:
        return ServerState.ABORT_CURRENT_RUN

    def aux_communicate(self, fl_ctx: FLContext) -> dict:
        return ServerState.ABORT_CURRENT_RUN

    def handle_sd_callback(self, sp: SP, fl_ctx: FLContext) -> ServerState:
        return self


class HotState(ServerState):
    def __init__(self, host: str = "", port: str = "", ssid: str = "") -> None:
        super().__init__(host, port, ssid)

    def register(self, fl_ctx: FLContext) -> dict:
        return ServerState.IN_SSERVICE

    def heartbeat(self, fl_ctx: FLContext) -> dict:
        return ServerState.IN_SSERVICE

    def get_task(self, fl_ctx: FLContext) -> dict:
        return ServerState.IN_SSERVICE

    def submit_result(self, fl_ctx: FLContext) -> dict:
        return ServerState.IN_SSERVICE

    def aux_communicate(self, fl_ctx: FLContext) -> dict:
        return ServerState.IN_SSERVICE

    def handle_sd_callback(self, sp: SP, fl_ctx: FLContext) -> ServerState:
        if sp and sp.primary is True:
            if sp.name == self.host and sp.fl_port == self.service_port:
                self.primary = True
                if sp.service_session_id != self.ssid:
                    self.ssid = sp.service_session_id
                    self.logger.info(
                        f"Primary sp changed to: {sp.name} fl_port: {sp.fl_port} SSID: {sp.service_session_id}. "
                        f"Turning to Cold"
                    )
                    return Hot2ColdState(host=self.host, port=self.service_port, ssid=sp.service_session_id)
                else:
                    return self
            else:
                self.primary = False
                self.logger.info(
                    f"Primary sp changed to: {sp.name} fl_port: {sp.fl_port} SSID: {sp.service_session_id}. "
                    f"Turning to Cold"
                )
                return Hot2ColdState(host=self.host, port=self.service_port)
        return self


class Hot2ColdState(ServerState):
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

    def handle_sd_callback(self, sp: SP, fl_ctx: FLContext) -> ServerState:
        return self


class ShutdownState(ServerState):
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

    def handle_sd_callback(self, sp: SP, fl_ctx: FLContext) -> ServerState:
        return self
