# Copyright (c) 2021, NVIDIA CORPORATION.
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

import threading
from multiprocessing.connection import Listener

from nvflare.apis.fl_context import FLContext
from .admin_commands import AdminCommands


class CommandAgent(object):
    def __init__(self, federated_client, listen_port, client_runner) -> None:
        self.federated_client = federated_client
        self.listen_port = int(listen_port)
        self.client_runner = client_runner
        self.thread = None
        self.asked_to_stop = False

        self.commands = AdminCommands.commands

    def start(self, fl_ctx: FLContext):
        # self.thread = threading.Thread(target=listen_command, args=[federated_client, int(listen_port), client_runner])
        self.thread = threading.Thread(target=listen_command, args=[self, fl_ctx])
        self.thread.start()

        pass

    def listen_command(self, fl_ctx):
        try:
            address = ("localhost", self.listen_port)  # family is deduced to be 'AF_INET'
            listener = Listener(address, authkey="client process secret password".encode())
            conn = listener.accept()
            print(f"Created the listener on port: {self.listen_port}")

            try:
                while not self.asked_to_stop:
                    if conn.poll(1.0):
                        msg = conn.recv()
                        command_name = msg.get("command")
                        data = msg.get("data")
                        command = AdminCommands.get_command(command_name)
                        if command:
                            engine = fl_ctx.get_engine()
                            with engine.new_context() as new_fl_ctx:
                                reply = command.process(data=data, fl_ctx=new_fl_ctx)
                                if reply:
                                    conn.send(reply)
            except Exception as e:
                # traceback.print_exc()
                print(f"Process communication exception: {self.listen_port}.")
            finally:
                conn.close()

            listener.close()
        except Exception as e:
            print(f"Could not create the listener for this process on port: {self.listen_port}.")
            pass

    def shutdown(self):
        self.asked_to_stop = True

        if self.thread and self.thread.is_alive():
            self.thread.join()


def listen_command(agent: CommandAgent, fl_ctx: FLContext):
    agent.listen_command(fl_ctx)
