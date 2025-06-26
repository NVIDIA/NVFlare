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
from typing import Union

from nvflare.apis.fl_context import FLContext
from nvflare.apis.streaming import StreamContext
from nvflare.app_common.streamers.file_streamer import FileStreamer
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import CellChannel
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import MetaKey, MetaStatusValue, ProtoKey, StreamChannel, make_meta, validate_proto
from nvflare.fuel.utils.log_utils import get_obj_logger
from nvflare.private.fed.server.cred_keeper import CredKeeper
from nvflare.security.logging import secure_log_traceback

from .constants import ConnProps
from .reg import ServerCommandRegister

logger = logging.getLogger(__name__)


class AdminServer:

    def __init__(
        self,
        cell: Cell,
        cmd_reg: ServerCommandRegister,
        engine,
        extra_conn_props=None,
    ):
        """Base class of FedAdminServer to create a server that can receive commands.

        Args:
            cell: the communication cell
            cmd_reg: CommandRegister
            extra_conn_props: a dict of extra conn props, if specified
        """
        if extra_conn_props is not None:
            assert isinstance(extra_conn_props, dict), "extra_conn_props must be dict but got {}".format(
                extra_conn_props
            )

        self.cell = cell
        self.engine = engine
        self.fl_ctx = None
        self.extra_conn_props = extra_conn_props
        self.cmd_reg = cmd_reg
        self.cred_keeper = CredKeeper()
        self.logger = get_obj_logger(self)

        cmd_reg.finalize()

        cell.register_request_cb(
            channel=CellChannel.HCI,
            topic="*",
            cb=self._process_admin_request,
        )

        if engine:
            self.fl_ctx = engine.new_context()
            FileStreamer.register_stream_processing(
                fl_ctx=self.fl_ctx,
                channel=StreamChannel.UPLOAD,
                topic="*",
                stream_done_cb=self._process_upload,
            )

    def get_id_asserter(self):
        return self.cred_keeper.get_id_asserter(self.fl_ctx)

    def get_id_verifier(self):
        return self.cred_keeper.get_id_verifier(self.fl_ctx)

    def _create_conn(self, conn_data: str, cmd_headers=None) -> (bool, str, Connection):
        conn = Connection(
            props={
                ConnProps.ENGINE: self.engine,
                ConnProps.HCI_SERVER: self,
            }
        )

        if self.extra_conn_props:
            conn.set_props(self.extra_conn_props)
        if self.cmd_reg.conn_props:
            conn.set_props(self.cmd_reg.conn_props)

        if cmd_headers:
            conn.set_prop(ConnProps.CMD_HEADERS, cmd_headers)

        try:
            req = conn_data.strip()
            command = None
            req_json = validate_proto(req)
            conn.request = req_json

            if req_json is not None:
                meta = req_json.get(ProtoKey.META, None)
                if meta and isinstance(meta, dict):
                    cmd_timeout = meta.get(MetaKey.CMD_TIMEOUT)
                    if cmd_timeout:
                        conn.set_prop(ConnProps.CMD_TIMEOUT, cmd_timeout)

                    custom_props = meta.get(MetaKey.CUSTOM_PROPS)
                    if custom_props:
                        conn.set_prop(ConnProps.CUSTOM_PROPS, custom_props)

                    cmd_props = meta.get(MetaKey.CMD_PROPS)
                    if cmd_props:
                        conn.set_prop(ConnProps.CMD_PROPS, cmd_props)

                data = req_json[ProtoKey.DATA]
                for item in data:
                    it = item[ProtoKey.TYPE]
                    if it == ProtoKey.COMMAND:
                        command = item[ProtoKey.DATA]
                        break

                if command is None:
                    self.logger.error("protocol violation: no command specified in request")
                    conn.append_error(
                        "protocol violation",
                        meta=make_meta(MetaStatusValue.INTERNAL_ERROR, "protocol violation"),
                    )
                    return False, "", conn
                else:
                    return True, command, conn
            else:
                # not json encoded
                conn.append_error(
                    "protocol violation", meta=make_meta(MetaStatusValue.INTERNAL_ERROR, "protocol violation")
                )
                return False, "", conn
        except:
            secure_log_traceback()
            return False, "", conn

    def _process_upload(self, stream_ctx: StreamContext, fl_ctx: FLContext, **kwargs):
        conn_data = stream_ctx.get("conn_data")
        file_location = FileStreamer.get_file_location(stream_ctx)
        self.logger.debug(f"got upload from hci client: {conn_data=} {file_location=}")
        ok, command, conn = self._create_conn(conn_data)
        assert isinstance(conn, Connection)
        conn.set_prop(ConnProps.FILE_LOCATION, file_location)
        self.cmd_reg.process_command(conn, command)
        result = conn.close()
        self.logger.debug(f"upload result: {result}")
        return result

    def _process_admin_request(self, request: CellMessage) -> Union[None, CellMessage]:
        self.logger.debug(f"got admin_request: {request.payload}")
        ok, command, conn = self._create_conn(request.payload, request.headers)
        conn.set_prop(ConnProps.REQUEST, request)
        if ok:
            self.logger.debug(f"processing command {command}")
            self.cmd_reg.process_command(conn, command)
        else:
            self.logger.error(f"received invalid command: {request.headers}")
        payload = conn.close()
        return CellMessage(payload=payload)

    def stop(self):
        self.cmd_reg.close()
        logger.info("Admin Server is stopped!")

    def set_command_registry(self, cmd_reg: ServerCommandRegister):
        if cmd_reg:
            cmd_reg.finalize()

            if self.cmd_reg:
                self.cmd_reg.close()

            self.cmd_reg = cmd_reg

    def start(self):
        logger.info("Admin Server is started")
