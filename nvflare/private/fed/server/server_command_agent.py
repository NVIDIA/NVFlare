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

import logging
import threading

from nvflare.apis.fl_constant import ServerCommandKey
from nvflare.fuel.utils import fobs
from nvflare.private.fed.utils.fed_utils import listen_command
from nvflare.security.logging import secure_format_exception
from nvflare.fuel.f3.cellnet.cell import Cell, CellAgent, Message as CellMessage, MessageHeaderKey, ReturnCode
from nvflare.private.defs import RequestHeader, CellChannel, new_cell_message
from nvflare.private.admin_defs import Message, error_reply, ok_reply

from .server_commands import ServerCommands


class ServerCommandAgent(object):
    def __init__(self, engine, cell: Cell) -> None:
        """To init the CommandAgent.

        Args:
            listen_port: port to listen the command
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        # self.listen_port = int(listen_port)
        # self.thread = None
        self.asked_to_stop = False
        self.engine = engine
        self.cell = cell

    def start(self):
        # self.thread = threading.Thread(
        #     target=listen_command, args=[self.listen_port, engine, self.execute_command, self.logger]
        # )
        # self.thread.start()
        self.cell.register_request_cb(
            channel=CellChannel.SERVER_COMMAND,
            topic="*",
            cb=self.execute_command,
        )
        self.logger.info(f"ServerCommandAgent cell start: {self.cell.get_fqcn()}")

    def execute_command(self, request: CellMessage) -> CellMessage:

        # while not self.asked_to_stop:
        #     try:
        #         if conn.poll(1.0):
        #             msg = conn.recv()
        #             msg = fobs.loads(msg)
        #             command_name = msg.get(ServerCommandKey.COMMAND)
        #             data = msg.get(ServerCommandKey.DATA)
        #             command = ServerCommands.get_command(command_name)
        #             if command:
        #                 with engine.new_context() as new_fl_ctx:
        #                     reply = command.process(data=data, fl_ctx=new_fl_ctx)
        #                     if reply is not None:
        #                         conn.send(reply)
        #     except EOFError:
        #         self.logger.info("listener communication terminated.")
        #         break
        #     except Exception as e:
        #         self.logger.error(
        #             f"IPC Communication error on the port: {self.listen_port}: {secure_format_exception(e)}."
        #         )
        assert isinstance(request, CellMessage), "request must be CellMessage but got {}".format(type(request))
        req = request.payload

        # assert isinstance(req, Message), "request payload must be Message but got {}".format(type(req))
        # topic = req.topic

        msg = fobs.loads(req)
        command_name = msg.get(ServerCommandKey.COMMAND)
        data = msg.get(ServerCommandKey.DATA)
        command = ServerCommands.get_command(command_name)
        if command:
            with self.engine.new_context() as new_fl_ctx:
                reply = command.process(data=data, fl_ctx=new_fl_ctx)
                if reply is not None:
                    return_message = new_cell_message({}, fobs.dumps(reply))
                    return_message.set_header(MessageHeaderKey.RETURN_CODE, ReturnCode.OK)
                else:
                    return_message = new_cell_message({}, None)
                return return_message

    def shutdown(self):
        self.asked_to_stop = True

        # if self.thread and self.thread.is_alive():
        #     self.thread.join()
