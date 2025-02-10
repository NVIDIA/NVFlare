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
import threading
import time

import flwr.proto.grpcadapter_pb2 as pb2
from flwr.proto.grpcadapter_pb2_grpc import GrpcAdapterServicer

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode
from nvflare.app_opt.flower.connectors.flower_connector import FlowerClientConnector
from nvflare.app_opt.flower.defs import Constant
from nvflare.app_opt.flower.grpc_server import GrpcServer
from nvflare.app_opt.flower.utils import msg_container_to_shareable, reply_should_exit, shareable_to_msg_container
from nvflare.fuel.utils.network_utils import get_local_addresses
from nvflare.security.logging import secure_format_exception


class GrpcClientConnector(FlowerClientConnector, GrpcAdapterServicer):
    def __init__(
        self,
        int_server_grpc_options=None,
        per_msg_timeout=2.0,
        tx_timeout=10.0,
        client_shutdown_timeout=0.5,
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
        self._stopping = False
        self._exit_waiter = threading.Event()

    def initialize(self, fl_ctx: FLContext):
        super().initialize(fl_ctx)
        self._client_name = fl_ctx.get_identity_name()

    def _start_client(self, superlink_addr: str, clientapp_api_addr: str, fl_ctx: FLContext):
        app_ctx = {
            Constant.APP_CTX_CLIENT_NAME: self._client_name,
            Constant.APP_CTX_SUPERLINK_ADDR: superlink_addr,
            Constant.APP_CTX_CLIENTAPP_API_ADDR: clientapp_api_addr,
            Constant.APP_CTX_NUM_ROUNDS: self.num_rounds,
        }
        self.start_applet(app_ctx, fl_ctx)

    def _stop_client(self):
        self._training_stopped = True

        # do not stop the applet until should-exit is sent
        if not self._exit_waiter.wait(timeout=2.0):
            self.logger.warning("did not send should-exit before shutting down supernode")

        # give 1 sec for the supernode to quite gracefully
        self.logger.debug("about to stop applet")
        time.sleep(1.0)
        self.stop_applet(self.client_shutdown_timeout)

    def _is_stopped(self) -> (bool, int):
        applet_stopped, ec = self.is_applet_stopped()
        if applet_stopped:
            return applet_stopped, ec

        if self._training_stopped:
            return True, 0

        if self._stopping:
            self.stop(fl_ctx=None)
            return True, 0

        return False, 0

    def start(self, fl_ctx: FLContext):
        if not self.num_rounds:
            raise RuntimeError("cannot start - num_rounds is not set")

        # get addresses for flower supernode:
        # - superlink_addr for supernode to connect to superlink
        # - clientapp_api_addr for client app to connect to the supernode
        addresses = get_local_addresses(2)
        superlink_addr = addresses[0]
        clientapp_api_addr = addresses[1]

        self.internal_server_addr = superlink_addr
        self.logger.info(f"Start internal server at {self.internal_server_addr}")
        self.internal_grpc_server = GrpcServer(self.internal_server_addr, 10, self.int_server_grpc_options, self)
        self.internal_grpc_server.start(no_blocking=True)
        self.logger.info(f"Started internal grpc server at {self.internal_server_addr}")
        self._start_client(superlink_addr, clientapp_api_addr, fl_ctx)
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
            if self.stopped:
                self._stopping = True
                self._exit_waiter.set()
                self.logger.debug("asked supernode to exit_1!")
                return reply_should_exit()

            reply = self._send_flower_request(msg_container_to_shareable(request))
            rc = reply.get_return_code()
            if rc == ReturnCode.OK:
                return shareable_to_msg_container(reply)
            else:
                # server side already ended
                self.logger.warning(f"Flower server has stopped with RC {rc}")
                self._stopping = True
                self._exit_waiter.set()
                self.logger.debug("asked supernode to exit_2!")
                return reply_should_exit()
        except Exception as ex:
            self._abort(reason=f"_send_flower_request exception: {secure_format_exception(ex)}")
            self._stopping = True
            self._exit_waiter.set()
            self.logger.debug("asked supernode to exit_3!")
            return reply_should_exit()
