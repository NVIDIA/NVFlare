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

from nvflare.apis.fl_context import FLContext
from nvflare.fuel.f3.cellnet import Cell, Message as CellMessage
from nvflare.private.defs import CellChannel, new_cell_message
from .admin_commands import AdminCommands


class CommandAgent(object):
    def __init__(self):
        """To init the CommandAgent.
        """
        self.commands = AdminCommands.commands
        self.logger = logging.getLogger(self.__class__.__name__)

    def start(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        cell = engine.get_cell()
        assert isinstance(cell, Cell)
        cell.register_request_cb(
            channel=CellChannel.COMMAND,
            topic="*",
            cb=self._execute_command,
            engine=engine
        )

    def _execute_command(
            self,
            cell: Cell,
            channel: str,
            topic: str,
            request: CellMessage,
            engine):
        # this runs in the job cell on the client side
        msg = request.payload
        command_name = msg.get("command")
        data = msg.get("data")
        command = AdminCommands.get_command(command_name)
        if command:
            with engine.new_context() as new_fl_ctx:
                result = command.process(data=data, fl_ctx=new_fl_ctx)
                if result:
                    return new_cell_message({}, result)
                else:
                    return None
        else:
            self.logger.error(f"unknown command '{command_name}'")
            return None

    def shutdown(self):
        pass
