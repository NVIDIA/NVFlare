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

from nvflare.apis.fl_context import FLContext

from ..utils.fed_utils import listen_command
from .admin_commands import AdminCommands


class CommandAgent(object):
    def __init__(self, federated_client, listen_port, client_runner) -> None:
        """To init the CommandAgent.

        Args:
            federated_client: FL client object
            listen_port: port to listen the command
            client_runner: ClientRunner object
        """
        self.federated_client = federated_client
        self.listen_port = int(listen_port)
        self.client_runner = client_runner
        self.thread = None
        self.asked_to_stop = False

        self.commands = AdminCommands.commands
        self.logger = logging.getLogger(self.__class__.__name__)

    def start(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        self.thread = threading.Thread(
            target=listen_command, args=[self.listen_port, engine, self.execute_command, self.logger]
        )
        self.thread.start()

    def execute_command(self, conn, engine):
        while not self.asked_to_stop:
            try:
                if conn.poll(1.0):
                    msg = conn.recv()
                    command_name = msg.get("command")
                    data = msg.get("data")
                    command = AdminCommands.get_command(command_name)
                    if command:
                        with engine.new_context() as new_fl_ctx:
                            reply = command.process(data=data, fl_ctx=new_fl_ctx)
                            if reply:
                                conn.send(reply)
            except EOFError:
                self.logger.info("listener communication terminated.")
                break
            except Exception as e:
                # traceback.print_exc()
                self.logger.error(f"Process communication error: {self.listen_port}: {e}.", exc_info=False)

    def shutdown(self):
        self.asked_to_stop = True

        if self.thread and self.thread.is_alive():
            self.thread.join()
