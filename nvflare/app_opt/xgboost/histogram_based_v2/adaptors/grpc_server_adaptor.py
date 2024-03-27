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

import multiprocessing
import threading
from typing import Tuple

import nvflare.app_opt.xgboost.histogram_based_v2.proto.federated_pb2 as pb2
from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.xgboost.histogram_based_v2.adaptor import XGBServerAdaptor
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.app_opt.xgboost.histogram_based_v2.grpc.grpc_client import GrpcClient
from nvflare.fuel.f3.drivers.net_utils import get_open_tcp_port
from nvflare.security.logging import secure_format_exception


class GrpcServerAdaptor(XGBServerAdaptor):
    """Implementation of XGBServerAdaptor that uses an internal `GrpcClient`.

    The `GrpcServerAdaptor` class serves as an interface between the XGBoost
    federated client and federated server components.
    It employs its `XGBRunner` to initiate an XGBoost federated gRPC server
    and utilizes an internal `GrpcClient` to forward client requests/responses.

    The communication flow is as follows:

        1. XGBoost federated gRPC client talks to `GrpcClientAdaptor`, which
           encapsulates a `GrpcServer`.
           Requests are then forwarded to `GrpcServerAdaptor`, which internally
           manages a `GrpcClient` responsible for interacting with the XGBoost
           federated gRPC server.
        2. XGBoost federated gRPC server talks to `GrpcServerAdaptor`, which
           encapsulates a `GrpcClient`.
           Responses are then forwarded to `GrpcClientAdaptor`, which internally
           manages a `GrpcServer` responsible for interacting with the XGBoost
           federated gRPC client.
    """

    def __init__(
        self,
        int_client_grpc_options=None,
        in_process: bool = False,
        xgb_server_ready_timeout: float = Constant.XGB_SERVER_READY_TIMEOUT,
    ):
        """Constructor method to initialize the object.

        Args:
            int_client_grpc_options: An optional list of key-value pairs (`channel_arguments`
                in gRPC Core runtime) to configure the gRPC channel of internal `GrpcClient`.
            in_process (bool): Specifies whether to start the `XGBRunner` in the same process or not.
            xgb_server_ready_timeout (float): Duration for which the internal `GrpcClient`
                should wait for the XGBoost gRPC server before timing out.
        """
        XGBServerAdaptor.__init__(self)
        self.int_client_grpc_options = int_client_grpc_options
        self.xgb_server_ready_timeout = xgb_server_ready_timeout
        self.in_process = in_process
        self.internal_xgb_client = None
        self._process = None
        self._server_stopped = False

    def _try_start_server(self, addr: str, port: int):
        ctx = {
            Constant.RUNNER_CTX_SERVER_ADDR: addr,
            Constant.RUNNER_CTX_WORLD_SIZE: self.world_size,
            Constant.RUNNER_CTX_PORT: port,
        }
        self.logger.info(f"starting XGB server with {ctx=}")
        try:
            self.xgb_runner.run(ctx)
        except Exception as ex:
            self.logger.error(f"Exception running xgb_runner {ctx=}: {secure_format_exception(ex)}")
            raise ex

    def _start_server(self, addr: str, port: int):
        if self.in_process:
            self.logger.info("starting XGB server in another thread")
            t = threading.Thread(
                name="xgb_server_thread", target=self._try_start_server, args=(addr, port), daemon=True
            )
            t.start()
        else:
            self.logger.info("starting XGB server in another process")
            self._process = multiprocessing.Process(
                name="xgb_server_process", target=self._try_start_server, args=(addr, port), daemon=True
            )
            self._process.start()

    def _stop_server(self):
        self._server_stopped = True
        if self.in_process:
            if self.xgb_runner:
                self.xgb_runner.stop()
        else:
            if self._process:
                self._process.kill()
                self._process = None

    def _is_stopped(self) -> Tuple[bool, int]:
        if self._server_stopped:
            return True, 0

        if self.in_process:
            if self.xgb_runner:
                return self.xgb_runner.is_stopped()
            else:
                return True, 0
        else:
            if self._process:
                ec = self._process.exitcode
                if ec is None:
                    return False, 0
                else:
                    return True, ec
            else:
                return True, 0

    def start(self, fl_ctx: FLContext):
        # we dynamically create server address on localhost
        port = get_open_tcp_port(resources={})
        if not port:
            raise RuntimeError("failed to get a port for XGB server")

        server_addr = f"127.0.0.1:{port}"
        self._start_server(addr=server_addr, port=port)

        # start XGB client
        self.internal_xgb_client = GrpcClient(
            server_addr,
            self.int_client_grpc_options,
        )
        self.internal_xgb_client.start(ready_timeout=self.xgb_server_ready_timeout)

    def stop(self, fl_ctx: FLContext):
        client = self.internal_xgb_client
        self.internal_xgb_client = None
        if client:
            self.log_info(fl_ctx, "Stopping internal XGB client")
            client.stop()
        self._stop_server()

    def all_gather(self, rank: int, seq: int, send_buf: bytes, fl_ctx: FLContext) -> bytes:
        result = self.internal_xgb_client.send_allgather(seq_num=seq, rank=rank, data=send_buf)
        if isinstance(result, pb2.AllgatherReply):
            return result.receive_buffer
        else:
            raise RuntimeError(f"bad result from XGB server: expect AllgatherReply but got {type(result)}")

    def all_reduce(
        self,
        rank: int,
        seq: int,
        data_type: int,
        reduce_op: int,
        send_buf: bytes,
        fl_ctx: FLContext,
    ) -> bytes:
        result = self.internal_xgb_client.send_allreduce(
            seq_num=seq,
            rank=rank,
            data=send_buf,
            data_type=data_type,
            reduce_op=reduce_op,
        )
        if isinstance(result, pb2.AllreduceReply):
            return result.receive_buffer
        else:
            raise RuntimeError(f"bad result from XGB server: expect AllreduceReply but got {type(result)}")

    def broadcast(self, rank: int, seq: int, root: int, send_buf: bytes, fl_ctx: FLContext) -> bytes:
        result = self.internal_xgb_client.send_broadcast(seq_num=seq, rank=rank, data=send_buf, root=root)
        if isinstance(result, pb2.BroadcastReply):
            return result.receive_buffer
        else:
            raise RuntimeError(f"bad result from XGB server: expect BroadcastReply but got {type(result)}")
