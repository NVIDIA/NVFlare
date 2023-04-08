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

import copy
import logging

from nvflare.apis.fl_constant import FLContextKey, ServerCommandKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.utils.fl_context_utils import get_serializable_data
from nvflare.fuel.f3.cellnet.cell import Cell, MessageHeaderKey, ReturnCode, make_reply
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.fuel.utils import fobs
from nvflare.private.defs import CellChannel, CellMessageHeaderKeys, new_cell_message

from .server_commands import ServerCommands


class ServerCommandAgent(object):
    def __init__(self, engine, cell: Cell) -> None:
        """To init the CommandAgent.

        Args:
            listen_port: port to listen the command
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.asked_to_stop = False
        self.engine = engine
        self.cell = cell

    def start(self):
        self.cell.register_request_cb(
            channel=CellChannel.SERVER_COMMAND,
            topic="*",
            cb=self.execute_command,
        )
        self.cell.register_request_cb(
            channel=CellChannel.AUX_COMMUNICATION,
            topic="*",
            cb=self.aux_communicate,
        )
        self.logger.info(f"ServerCommandAgent cell register_request_cb: {self.cell.get_fqcn()}")

    def execute_command(self, request: CellMessage) -> CellMessage:

        if not isinstance(request, CellMessage):
            raise RuntimeError("request must be CellMessage but got {}".format(type(request)))

        command_name = request.get_header(MessageHeaderKey.TOPIC)
        data = fobs.loads(request.payload)

        token = request.get_header(CellMessageHeaderKeys.TOKEN, None)
        # client_name = request.get_header(CellMessageHeaderKeys.CLIENT_NAME, None)
        client = None
        if token:
            client = self._get_client(token)
            if client:
                data.set_header(ServerCommandKey.FL_CLIENT, client)

        command = ServerCommands.get_command(command_name)
        if command:
            if command_name in ServerCommands.client_request_commands_names:
                if not client:
                    return make_reply(
                        ReturnCode.AUTHENTICATION_ERROR,
                        "Request from client: missing client token",
                        fobs.dumps(None),
                    )

            with self.engine.new_context() as new_fl_ctx:
                if command_name in ServerCommands.client_request_commands_names:
                    state_check = command.get_state_check(new_fl_ctx)
                    error = self.engine.server.authentication_check(request, state_check)
                    if error:
                        return make_reply(ReturnCode.AUTHENTICATION_ERROR, error, fobs.dumps(None))

                reply = command.process(data=data, fl_ctx=new_fl_ctx)
                if reply is not None:
                    return_message = new_cell_message({}, fobs.dumps(reply))
                    return_message.set_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
                else:
                    return_message = make_reply(ReturnCode.PROCESS_EXCEPTION, "No process results", fobs.dumps(None))
                return return_message
        else:
            return make_reply(ReturnCode.INVALID_REQUEST, "No server command found", fobs.dumps(None))

    def _get_client(self, token):
        fl_server = self.engine.server
        client_manager = fl_server.client_manager
        clients = client_manager.clients
        return clients.get(token)

    def aux_communicate(self, request: CellMessage) -> CellMessage:

        assert isinstance(request, CellMessage), "request must be CellMessage but got {}".format(type(request))
        data = request.payload

        topic = request.get_header(MessageHeaderKey.TOPIC)
        with self.engine.new_context() as fl_ctx:
            server_state = self.engine.server.server_state
            state_check = server_state.aux_communicate(fl_ctx)
            error = self.engine.server.authentication_check(request, state_check)
            if error:
                make_reply(ReturnCode.AUTHENTICATION_ERROR, error, fobs.dumps(None))

            engine = fl_ctx.get_engine()
            reply = engine.dispatch(topic=topic, request=data, fl_ctx=fl_ctx)

            shared_fl_ctx = FLContext()
            shared_fl_ctx.set_public_props(copy.deepcopy(get_serializable_data(fl_ctx).get_all_public_props()))
            reply.set_header(key=FLContextKey.PEER_CONTEXT, value=shared_fl_ctx)

            if reply is not None:
                return_message = new_cell_message({}, reply)
                return_message.set_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
            else:
                return_message = new_cell_message({}, None)
            return return_message

    def shutdown(self):
        self.asked_to_stop = True
