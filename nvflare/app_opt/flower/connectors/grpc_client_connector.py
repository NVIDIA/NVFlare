# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import flwr.proto.grpcadapter_pb2 as pb2
from flwr.proto.grpcadapter_pb2_grpc import GrpcAdapterServicer

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode
from nvflare.app_opt.flower.connectors.flower_connector import FlowerClientConnector
from nvflare.app_opt.flower.defs import Constant
from nvflare.app_opt.flower.grpc_server import GrpcServer
from nvflare.app_opt.flower.utils import msg_container_to_shareable, reply_should_exit, shareable_to_msg_container
from nvflare.fuel.f3.drivers.net_utils import get_open_tcp_port
from nvflare.security.logging import secure_format_exception


class GrpcClientConnector(FlowerClientConnector, GrpcAdapterServicer):
    def __init__(
        self,
        int_server_grpc_options=None,
        per_msg_timeout=2.0,
        tx_timeout=10.0,
        client_shutdown_timeout=5.0,
    ):
        """Constructor of GrpcClientConnector.
        GrpcClientConnector is used to connect Flare Client with the Flower Client App.

        Args:
            int_server_grpc_options: internal grpc server options
            per_msg_timeout: per-message timeout for using ReliableMessage
            tx_timeout: transaction timeout for using ReliableMessage
            client_shutdown_timeout: max time for shutting down Flare client
        """
        FlowerClientConnector.__init__(self, per_msg_timeout, tx_timeout)
        self.client_shutdown_timeout = client_shutdown_timeout
        self.int_server_grpc_options = int_server_grpc_options
        self.internal_grpc_server = None
        self.stopped = False
        self.internal_server_addr = None
        self._training_stopped = False
        self._client_name = None

    def initialize(self, fl_ctx: FLContext):
        super().initialize(fl_ctx)
        self._client_name = fl_ctx.get_identity_name()

    def _start_client(self, server_addr: str, fl_ctx: FLContext):
        app_ctx = {
            Constant.APP_CTX_CLIENT_NAME: self._client_name,
            Constant.APP_CTX_SERVER_ADDR: server_addr,
            Constant.APP_CTX_NUM_ROUNDS: self.num_rounds,
        }
        self.start_applet(app_ctx, fl_ctx)

    def _stop_client(self):
        self._training_stopped = True
        self.stop_applet(self.client_shutdown_timeout)

    def _is_stopped(self) -> (bool, int):
        applet_stopped, ec = self.is_applet_stopped()
        if applet_stopped:
            return applet_stopped, ec

        if self._training_stopped:
            return True, 0

        return False, 0

    def start(self, fl_ctx: FLContext):
        if not self.num_rounds:
            raise RuntimeError("cannot start - num_rounds is not set")

        # dynamically determine address on localhost
        port = get_open_tcp_port(resources={})
        if not port:
            raise RuntimeError("failed to get a port for Flower server")
        self.internal_server_addr = f"127.0.0.1:{port}"
        self.logger.info(f"Start internal server at {self.internal_server_addr}")
        self.internal_grpc_server = GrpcServer(self.internal_server_addr, 10, self.int_server_grpc_options, self)
        self.internal_grpc_server.start(no_blocking=True)
        self.logger.info(f"Started internal grpc server at {self.internal_server_addr}")
        self._start_client(self.internal_server_addr, fl_ctx)
        self.logger.info("Started external Flower grpc client")

    def stop(self, fl_ctx: FLContext):
        if self.stopped:
            return

        self.stopped = True
        self._stop_client()

        if self.internal_grpc_server:
            self.logger.info("Stop internal grpc Server")
            self.internal_grpc_server.shutdown()

    def _abort(self, reason: str):
        # stop the gRPC client (the target)
        self.abort_signal.trigger(True)

        # abort the FL client
        with self.engine.new_context() as fl_ctx:
            self.system_panic(reason, fl_ctx)

    def SendReceive(self, request: pb2.MessageContainer, context):
        """Process request received from a Flower client.

        This implements the SendReceive method required by Flower gRPC server (LGS on FLARE Client).
        1. convert the request to a Shareable object.
        2. send the Shareable request to FLARE server.
        3. convert received Shareable result to MessageContainer and return to the Flower client

        Args:
            request: the request received from the Flower client
            context: gRPC context

        Returns: the reply MessageContainer object

        """
        try:
            reply = self._send_flower_request(msg_container_to_shareable(request))
            rc = reply.get_return_code()
            if rc == ReturnCode.OK:
                return shareable_to_msg_container(reply)
            else:
                # server side already ended
                self.logger.warning(f"Flower server has stopped with RC {rc}")
                return reply_should_exit()
        except Exception as ex:
            self._abort(reason=f"_send_flower_request exception: {secure_format_exception(ex)}")
