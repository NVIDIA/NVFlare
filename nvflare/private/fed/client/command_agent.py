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

import copy
import logging

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.utils.fl_context_utils import get_serializable_data
from nvflare.fuel.f3.cellnet.cell import Message as CellMessage
from nvflare.fuel.f3.cellnet.cell import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.cell import make_reply as make_cellnet_reply
from nvflare.fuel.utils import fobs
from nvflare.private.defs import CellChannel, new_cell_message

from .admin_commands import AdminCommands


class CommandAgent(object):
    def __init__(self, federated_client) -> None:
        """To init the CommandAgent.

        Args:
            federated_client: FL client object
        """
        self.federated_client = federated_client
        self.thread = None
        self.asked_to_stop = False

        self.commands = AdminCommands.commands
        self.logger = logging.getLogger(self.__class__.__name__)

    def start(self, fl_ctx: FLContext):
        self.engine = fl_ctx.get_engine()
        self.register_cell_cb()

    def register_cell_cb(self):
        self.federated_client.cell.register_request_cb(
            channel=CellChannel.CLIENT_COMMAND,
            topic="*",
            cb=self.execute_command,
        )
        self.federated_client.cell.register_request_cb(
            channel=CellChannel.AUX_COMMUNICATION,
            topic="*",
            cb=self.aux_communication,
        )

    def execute_command(self, request: CellMessage) -> CellMessage:

        assert isinstance(request, CellMessage), "request must be CellMessage but got {}".format(type(request))

        command_name = request.get_header(MessageHeaderKey.TOPIC)
        data = fobs.loads(request.payload)

        command = AdminCommands.get_command(command_name)
        if command:
            with self.engine.new_context() as new_fl_ctx:
                reply = command.process(data=data, fl_ctx=new_fl_ctx)
                if reply is not None:
                    return_message = new_cell_message({}, fobs.dumps(reply))
                    return_message.set_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
                else:
                    return_message = new_cell_message({}, None)
                return return_message
        return make_cellnet_reply(ReturnCode.INVALID_REQUEST, "", None)

    def aux_communication(self, request: CellMessage) -> CellMessage:

        assert isinstance(request, CellMessage), "request must be CellMessage but got {}".format(type(request))
        shareable = request.payload

        with self.engine.new_context() as fl_ctx:
            topic = request.get_header(MessageHeaderKey.TOPIC)
            reply = self.engine.dispatch(topic=topic, request=shareable, fl_ctx=fl_ctx)

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
