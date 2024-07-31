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

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.app_opt.flower.connectors.flower_connector import FlowerServerConnector
from nvflare.app_opt.flower.defs import Constant
from nvflare.app_opt.flower.grpc_client import GrpcClient
from nvflare.app_opt.flower.utils import msg_container_to_shareable, shareable_to_msg_container
from nvflare.fuel.f3.drivers.net_utils import get_open_tcp_port


class GrpcServerConnector(FlowerServerConnector):
    def __init__(
        self,
        int_client_grpc_options=None,
        flower_server_ready_timeout=Constant.FLOWER_SERVER_READY_TIMEOUT,
    ):
        FlowerServerConnector.__init__(self)
        self.int_client_grpc_options = int_client_grpc_options
        self.flower_server_ready_timeout = flower_server_ready_timeout
        self.internal_grpc_client = None
        self._server_stopped = False
        self._exit_code = 0

    def _start_server(self, addr: str, fl_ctx: FLContext):
        app_ctx = {
            Constant.APP_CTX_SERVER_ADDR: addr,
            Constant.APP_CTX_NUM_ROUNDS: self.num_rounds,
        }
        self.start_applet(app_ctx, fl_ctx)

    def _stop_server(self):
        self._server_stopped = True
        self._exit_code = self.stop_applet()

    def _is_stopped(self) -> (bool, int):
        runner_stopped, ec = self.is_applet_stopped()
        if runner_stopped:
            self.logger.info("applet is stopped!")
            return runner_stopped, ec

        if self._server_stopped:
            self.logger.info("Flower grpc server is stopped!")
            return True, self._exit_code

        return False, 0

    def start(self, fl_ctx: FLContext):
        # we dynamically create server address on localhost
        port = get_open_tcp_port(resources={})
        if not port:
            raise RuntimeError("failed to get a port for Flower grpc server")

        server_addr = f"127.0.0.1:{port}"
        self.log_info(fl_ctx, f"starting grpc connector: {server_addr=}")
        self._start_server(server_addr, fl_ctx)

        # start internal grpc client
        self.internal_grpc_client = GrpcClient(server_addr, self.int_client_grpc_options)
        self.internal_grpc_client.start(ready_timeout=self.flower_server_ready_timeout)

    def stop(self, fl_ctx: FLContext):
        client = self.internal_grpc_client
        self.internal_grpc_client = None
        if client:
            self.log_info(fl_ctx, "Stopping internal grpc client")
            client.stop()
        self._stop_server()

    def send_request_to_flower(self, request: Shareable, fl_ctx: FLContext) -> Shareable:
        """Send the request received from FL client to Flower server.

        This is done by:
        1. convert the request to Flower-defined MessageContainer object
        2. Send the MessageContainer object to Flower server via the internal GRPC client (LGC)
        3. Convert the reply MessageContainer object received from the Flower server to Shareable
        4. Return the reply Shareable object

        Args:
            request: the request received from FL client
            fl_ctx: FL context

        Returns: response from Flower server converted to Shareable

        """
        stopped, _ = self.is_applet_stopped()
        if stopped:
            self.log_warning(fl_ctx, "dropped app request since applet is already stopped")
            return make_reply(ReturnCode.SERVICE_UNAVAILABLE)

        result = self.internal_grpc_client.send_request(shareable_to_msg_container(request))

        if isinstance(result, pb2.MessageContainer):
            return msg_container_to_shareable(result)
        else:
            raise RuntimeError(f"bad result from Flower server: expect MessageContainer but got {type(result)}")
