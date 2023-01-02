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

from abc import ABC, abstractmethod

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply

from nvflare.fuel.f3.cellnet.cell import Cell, Message, MessageHeaderKey
from nvflare.private.defs import CellChannel
from nvflare.private.fed.cmi import CellMessageInterface


class CommandProcessor(ABC):
    """The CommandProcessor is responsible for processing a command from parent process."""

    @abstractmethod
    def get_command_name(self) -> str:
        """Gets the command name that this processor will handle.

        Returns:
            name of the command
        """
        pass

    @abstractmethod
    def process(self, data: Shareable, fl_ctx: FLContext):
        """Processes the data.

        Args:
            data: process data
            fl_ctx: FLContext

        Return:
            A reply message
        """
        pass


class CommandAgent(FLComponent):

    def __init__(self, engine, commands: list):
        """To init the CommandAgent.
        """
        FLComponent.__init__(self)
        self.processors = {}
        for c in commands:
            assert isinstance(c, CommandProcessor)
            self.processors[c.get_command_name()] = c

        self.engine = engine
        cell = engine.get_cell()
        assert isinstance(cell, Cell)
        cell.register_request_cb(
            channel=CellChannel.COMMAND,
            topic="*",
            cb=self._execute_command,
        )

    def _execute_command(
            self,
            request: Message):
        command_name = request.get_header(MessageHeaderKey.TOPIC)
        data = request.payload
        if data:
            assert isinstance(data, Shareable)
        fl_ctx = request.get_prop(CellMessageInterface.PROP_KEY_FL_CTX)
        processor = self.processors.get(command_name)
        if processor:
            assert isinstance(fl_ctx, FLContext)
            cmi = self.engine.get_cmi()
            assert isinstance(cmi, CellMessageInterface)
            reply = processor.process(data=data, fl_ctx=fl_ctx)
            if reply is not None:
                if not isinstance(reply, Shareable):
                    reply = make_reply(data=reply)
                return cmi.new_cmi_message(fl_ctx, payload=reply)
        else:
            self.log_error(fl_ctx, f"no processor for command '{command_name}'")
        return None
