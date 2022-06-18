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
import socketserver
import ssl
import threading

from nvflare.fuel.hci.conn import Connection, receive_til_end
from nvflare.fuel.hci.proto import validate_proto
from nvflare.fuel.hci.security import get_certificate_common_name

from .reg import ServerCommandRegister

MAX_ADMIN_CONNECTIONS = 16


class _MsgHandler(socketserver.BaseRequestHandler):
    """Message handler.

    Used by the AdminServer to receive admin commands, validate, then process and do command through the
    ServerCommandRegister.
    """

    connections = 0
    lock = threading.Lock()

    def __init__(self, request, client_address, server):
        # handle() is called in the constructor so logger must be initialized first
        self.logger = logging.getLogger(self.__class__.__name__)
        super().__init__(request, client_address, server)

    def handle(self):
        try:
            with _MsgHandler.lock:
                _MsgHandler.connections += 1

            self.logger.debug(f"Concurrent admin connections: {_MsgHandler.connections}")
            if _MsgHandler.connections > MAX_ADMIN_CONNECTIONS:
                raise ConnectionRefusedError(f"Admin connection limit ({MAX_ADMIN_CONNECTIONS}) reached")

            conn = Connection(self.request, self.server)

            if self.server.use_ssl:
                cn = get_certificate_common_name(self.request.getpeercert())
                conn.set_prop("_client_cn", cn)
                valid = self.server.validate_client_cn(cn)
            else:
                valid = True

            if not valid:
                conn.append_error("authentication error")
            else:
                req = receive_til_end(self.request).strip()
                command = None
                req_json = validate_proto(req)
                conn.request = req_json
                if req_json is not None:
                    data = req_json["data"]
                    for item in data:
                        it = item["type"]
                        if it == "command":
                            command = item["data"]
                            break

                    if command is None:
                        conn.append_error("protocol violation")
                    else:
                        self.server.cmd_reg.process_command(conn, command)
                else:
                    # not json encoded
                    conn.append_error("protocol violation")

            if not conn.ended:
                conn.close()
        except BaseException as exc:
            self.logger.error(f"Admin connection terminated due to exception: {str(exc)}")
            if self.logger.getEffectiveLevel() <= logging.DEBUG:
                self.logger.exception("Admin connection error")
        finally:
            with _MsgHandler.lock:
                _MsgHandler.connections -= 1


def initialize_hci():
    socketserver.TCPServer.allow_reuse_address = True


class AdminServer(socketserver.ThreadingTCPServer):
    # faster re-binding
    allow_reuse_address = True

    # make this bigger than five
    request_queue_size = 10

    # kick connections when we exit
    daemon_threads = True

    def __init__(
        self,
        cmd_reg: ServerCommandRegister,
        host,
        port,
        ca_cert=None,
        server_cert=None,
        server_key=None,
        accepted_client_cns=None,
    ):
        """Base class of FedAdminServer to create a server that can receive commands.

        Args:
            cmd_reg: CommandRegister
            host: the IP address of the admin server
            port: port number of admin server
            ca_cert: the root CA's cert file name
            server_cert: server's cert, signed by the CA
            server_key: server's private key file
            accepted_client_cns: list of accepted Common Names from client, if specified
        """
        socketserver.TCPServer.__init__(self, (host, port), _MsgHandler, False)

        self.use_ssl = False
        if ca_cert and server_cert:
            if accepted_client_cns:
                assert isinstance(accepted_client_cns, list), "accepted_client_cns must be list but got {}.".format(
                    accepted_client_cns
                )

            ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ctx.verify_mode = ssl.CERT_REQUIRED
            ctx.load_verify_locations(ca_cert)
            ctx.load_cert_chain(certfile=server_cert, keyfile=server_key)

            # replace the socket with an SSL version of itself
            self.socket = ctx.wrap_socket(self.socket, server_side=True)
            self.use_ssl = True

        # bind the socket and start the server
        self.server_bind()
        self.server_activate()

        self._thread = None
        self.host = host
        self.port = port
        self.accepted_client_cns = accepted_client_cns
        self.cmd_reg = cmd_reg
        cmd_reg.finalize()
        self.logger = logging.getLogger(self.__class__.__name__)

    def validate_client_cn(self, cn):
        if self.accepted_client_cns:
            return cn in self.accepted_client_cns
        else:
            return True

    def stop(self):
        self.shutdown()
        self.cmd_reg.close()

        if self._thread.is_alive():
            self._thread.join()

        self.logger.info(f"Admin Server {self.host} on Port {self.port} shutdown!")

    def set_command_registry(self, cmd_reg: ServerCommandRegister):
        if cmd_reg:
            cmd_reg.finalize()

            if self.cmd_reg:
                self.cmd_reg.close()

            self.cmd_reg = cmd_reg

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, args=())

        if not self._thread.is_alive():
            self._thread.start()

    def _run(self):
        self.logger.info(f"Starting Admin Server {self.host} on Port {self.port}")
        self.serve_forever()
