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

import logging
import socketserver
import ssl
import threading

from nvflare.fuel.hci.binary_proto import CT_BINARY, receive_all
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import MetaKey, MetaStatusValue, ProtoKey, make_meta, validate_proto
from nvflare.fuel.hci.security import IdentityKey, get_identity_info
from nvflare.security.logging import secure_log_traceback

from .constants import ConnProps
from .reg import ServerCommandRegister

logger = logging.getLogger(__name__)


class _MsgHandler(socketserver.BaseRequestHandler):
    """Message handler.

    Used by the AdminServer to receive admin commands, validate, then process and do command through the
    ServerCommandRegister.
    """

    def handle(self):
        conn = None
        try:
            conn = Connection(self.request, self.server)
            conn.set_prop(ConnProps.CA_CERT, self.server.ca_cert)

            if self.server.extra_conn_props:
                conn.set_props(self.server.extra_conn_props)

            if self.server.cmd_reg.conn_props:
                conn.set_props(self.server.cmd_reg.conn_props)

            if self.server.use_ssl:
                identity = get_identity_info(self.request.getpeercert())
                conn.set_prop(ConnProps.CLIENT_IDENTITY, identity)
                valid = self.server.validate_client_cn(identity[IdentityKey.NAME])
            else:
                valid = True

            if not valid:
                conn.append_error(
                    "authentication error", meta=make_meta(MetaStatusValue.NOT_AUTHENTICATED, info="invalid credential")
                )
            else:
                ct, req, extra = receive_all(self.request)
                if ct == CT_BINARY and not extra:
                    conn.append_error(
                        "no data received from client",
                        meta=make_meta(MetaStatusValue.INTERNAL_ERROR, info="no data received"),
                    )
                else:
                    req = req.strip()
                    command = None
                    req_json = validate_proto(req)
                    conn.request = req_json
                    conn.content_type = ct
                    conn.extra = extra

                    if req_json is not None:
                        meta = req_json.get(ProtoKey.META, None)
                        if meta and isinstance(meta, dict):
                            cmd_timeout = meta.get(MetaKey.CMD_TIMEOUT)
                            if cmd_timeout:
                                conn.set_prop(ConnProps.CMD_TIMEOUT, cmd_timeout)

                            custom_props = meta.get(MetaKey.CUSTOM_PROPS)
                            if custom_props:
                                conn.set_prop(ConnProps.CUSTOM_PROPS, custom_props)

                            cmd_props = meta.get(MetaKey.CMD_PROPS)
                            if cmd_props:
                                conn.set_prop(ConnProps.CMD_PROPS, cmd_props)

                        data = req_json[ProtoKey.DATA]
                        for item in data:
                            it = item[ProtoKey.TYPE]
                            if it == ProtoKey.COMMAND:
                                command = item[ProtoKey.DATA]
                                break

                        if command is None:
                            conn.append_error(
                                "protocol violation",
                                meta=make_meta(MetaStatusValue.INTERNAL_ERROR, "protocol violation"),
                            )
                        else:
                            self.server.cmd_reg.process_command(conn, command)
                    else:
                        # not json encoded
                        conn.append_error(
                            "protocol violation", meta=make_meta(MetaStatusValue.INTERNAL_ERROR, "protocol violation")
                        )
        except:
            secure_log_traceback()
        if conn and not conn.ended:
            conn.close()


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
        extra_conn_props=None,
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
            extra_conn_props: a dict of extra conn props, if specified
        """
        if extra_conn_props is not None:
            assert isinstance(extra_conn_props, dict), "extra_conn_props must be dict but got {}".format(
                extra_conn_props
            )

        socketserver.TCPServer.__init__(self, ("0.0.0.0", port), _MsgHandler, False)

        self.use_ssl = False
        if ca_cert and server_cert:
            if accepted_client_cns:
                assert isinstance(accepted_client_cns, list), "accepted_client_cns must be list but got {}.".format(
                    accepted_client_cns
                )

            ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            # This feature is only supported on 3.7+
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ctx.verify_mode = ssl.CERT_REQUIRED
            ctx.load_verify_locations(ca_cert)
            ctx.load_cert_chain(certfile=server_cert, keyfile=server_key)

            # replace the socket with an ssl version of itself
            self.socket = ctx.wrap_socket(self.socket, server_side=True)
            self.use_ssl = True

        # bind the socket and start the server
        self.server_bind()
        self.server_activate()

        self._thread = None
        self.ca_cert = ca_cert
        self.host = host
        self.port = port
        self.accepted_client_cns = accepted_client_cns
        self.extra_conn_props = extra_conn_props
        self.cmd_reg = cmd_reg
        cmd_reg.finalize()

    def validate_client_cn(self, cn):
        if self.accepted_client_cns:
            return cn in self.accepted_client_cns
        else:
            return True

    def stop(self):
        self.shutdown()
        self.cmd_reg.close()
        logger.info(f"Admin Server {self.host} on Port {self.port} shutdown!")

    def set_command_registry(self, cmd_reg: ServerCommandRegister):
        if cmd_reg:
            cmd_reg.finalize()

            if self.cmd_reg:
                self.cmd_reg.close()

            self.cmd_reg = cmd_reg

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, args=())
            self._thread.daemon = True

        if not self._thread.is_alive():
            self._thread.start()

    def _run(self):
        logger.info(f"Starting Admin Server {self.host} on Port {self.port}")
        self.serve_forever()
