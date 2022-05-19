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
import pickle
import threading
from multiprocessing.connection import Listener

from nvflare.apis.fl_constant import ServerCommandKey

from .server_commands import ServerCommands


class ServerCommandAgent(object):
    def __init__(self, listen_port) -> None:
        """To init the CommandAgent.

        Args:
            listen_port: port to listen the command
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.listen_port = int(listen_port)
        self.thread = None
        self.asked_to_stop = False

        self.commands = ServerCommands.commands

    def start(self, engine):
        self.thread = threading.Thread(target=listen_command, args=[self, engine])
        self.thread.start()
        self.logger.info(f"ServerCommandAgent listening on port: {self.listen_port}")

    def listen_command(self, engine):
        try:
            address = ("localhost", self.listen_port)  # family is deduced to be 'AF_INET'
            listener = Listener(address, authkey="client process secret password".encode())
            conn = listener.accept()

            try:
                while not self.asked_to_stop:
                    if conn.poll(1.0):
                        msg = conn.recv()
                        msg = pickle.loads(msg)
                        command_name = msg.get(ServerCommandKey.COMMAND)
                        data = msg.get(ServerCommandKey.DATA)
                        command = ServerCommands.get_command(command_name)
                        if command:
                            with engine.new_context() as new_fl_ctx:
                                reply = command.process(data=data, fl_ctx=new_fl_ctx)
                                if reply:
                                    conn.send(reply)
            except Exception as e:
                self.logger.exception(
                    f"Process communication exception with listen port {self.listen_port}: {e}.", exc_info=True
                )
            finally:
                conn.close()

            listener.close()
        except Exception as e:
            self.logger.exception(
                f"Could not create the listener for this process on port: {self.listen_port}: {e}.", exc_info=True
            )

    def shutdown(self):
        self.asked_to_stop = True

        if self.thread and self.thread.is_alive():
            self.thread.join()


def listen_command(agent: ServerCommandAgent, engine):
    agent.listen_command(engine)
