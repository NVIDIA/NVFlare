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

from nvflare.apis.collective_comm_constants import (
    CollectiveCommandKey,
    CollectiveCommHandleError,
    CollectiveCommShareableHeader,
)
from nvflare.apis.fl_constant import ServerCommandKey
from nvflare.apis.shareable import Shareable
from nvflare.fuel.utils import fobs
from nvflare.private.fed.utils.fed_utils import listen_command

from .server_commands import ServerCommands


class CollectiveCommandAgent:
    def __init__(self, listen_port) -> None:
        """To init the CollectiveCommandAgent.

        Args:
            listen_port: port to listen the command
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.listen_port = int(listen_port)
        self.thread = None
        self.asked_to_stop = False

        self._requests = []
        self._sequence_number = None
        self._world_size = None
        self._clients_received = []

    def start(self, engine):
        self.thread = threading.Thread(
            target=listen_command, args=[self.listen_port, engine, self._poll_command, self.logger]
        )
        self.thread.start()
        self.logger.info(f"CollectiveCommandAgent listening on port: {self.listen_port}")

    def _append_requests(self, request: Shareable):
        world_size = request.get_header(CollectiveCommShareableHeader.WORLD_SIZE)
        if world_size is None:
            raise CollectiveCommHandleError("missing world_size in incoming request")

        rank = request.get_header(CollectiveCommShareableHeader.RANK)
        if rank is None:
            raise CollectiveCommHandleError("missing rank in incoming request")

        sequence_number = request.get_header(CollectiveCommShareableHeader.SEQUENCE_NUMBER)
        if sequence_number is None:
            raise CollectiveCommHandleError("missing sequence_number in incoming request")

        # use the first sequence number as sequence number
        if self._sequence_number is None:
            self._sequence_number = sequence_number
            self._world_size = world_size
            self._requests = []

        if rank in self._clients_received:
            raise CollectiveCommHandleError(f"client {rank} already processed.")

        if sequence_number != self._sequence_number:
            raise CollectiveCommHandleError("sequence number does not match")

        if world_size != self._world_size:
            raise CollectiveCommHandleError("world size does not match.")

        self._clients_received.append(rank)
        self._requests.append(request)

    def _poll_command(self, conn, engine):
        while not self.asked_to_stop:
            try:
                if conn.poll(1.0):
                    msg = conn.recv()
                    msg = fobs.loads(msg)
                    command_name = msg.get(ServerCommandKey.COMMAND)
                    command = ServerCommands.get_command(command_name)
                    if not command:
                        raise Exception(f"server command {command_name} is not supported in collective command agent.")
                    timeout = msg.get(CollectiveCommandKey.TIMEOUT)
                    data = msg.get(ServerCommandKey.DATA)
                    request = data.get_header(ServerCommandKey.SHAREABLE)
                    if timeout:
                        request.set_header(CollectiveCommShareableHeader.TIMEOUT, True)
                        with engine.new_context() as new_fl_ctx:
                            reply = command.process(data=data, fl_ctx=new_fl_ctx)
                        if reply:
                            for i in range(len(self._clients_received)):
                                conn.send(reply)
                        self._reset()
                    else:
                        self._append_requests(request=request)

                        if len(self._requests) == self._world_size and len(self._clients_received) == self._world_size:
                            request.set_header(CollectiveCommShareableHeader.ALL_REQUESTS, self._requests)
                            with engine.new_context() as new_fl_ctx:
                                reply = command.process(data=data, fl_ctx=new_fl_ctx)

                            if reply:
                                for i in range(self._world_size):
                                    conn.send(reply)
                            self._reset()
            except CollectiveCommHandleError as e:
                self.logger.error(f"CollectiveCommHandleError: {e}")
                return
            except EOFError:
                self.logger.info("Listener communication terminated.")
                break
            except Exception as e:
                self.logger.error(f"Communication error on the port: {self.listen_port}: {e}.")

    def _reset(self):
        self._sequence_number = None
        self._requests = []
        self._clients_received = []
        self._world_size = None

    def shutdown(self):
        self.asked_to_stop = True

        if self.thread and self.thread.is_alive():
            self.thread.join()
