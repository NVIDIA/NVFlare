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

from typing import Tuple

import nvflare.app_opt.xgboost.histogram_based_v2.proto.federated_pb2 as pb2
from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.xgboost.histogram_based_v2.adaptors.xgb_adaptor import XGBServerAdaptor
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.app_opt.xgboost.histogram_based_v2.grpc_client import GrpcClient
from nvflare.fuel.f3.drivers.net_utils import get_open_tcp_port


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
        xgb_server_ready_timeout=Constant.XGB_SERVER_READY_TIMEOUT,
        in_process=True,
    ):
        """Constructor method to initialize the object.

        Args:
            int_client_grpc_options: An optional list of key-value pairs (`channel_arguments`
                in gRPC Core runtime) to configure the gRPC channel of internal `GrpcClient`.
            in_process (bool): Specifies whether to call the `AppRunner.run()` in the same process or not.
            xgb_server_ready_timeout (float): Duration for which the internal `GrpcClient`
                should wait for the XGBoost gRPC server before timing out.
        """
        XGBServerAdaptor.__init__(self, in_process)
        self.int_client_grpc_options = int_client_grpc_options
        self.xgb_server_ready_timeout = xgb_server_ready_timeout
        self.in_process = in_process
        self.internal_xgb_client = None
        self._server_stopped = False
        self._exit_code = 0
        self._stopping = False

    def _start_server(self, addr: str, port: int, world_size: int, fl_ctx: FLContext):
        runner_ctx = {
            Constant.RUNNER_CTX_SERVER_ADDR: addr,
            Constant.RUNNER_CTX_WORLD_SIZE: world_size,
            Constant.RUNNER_CTX_PORT: port,
        }

        self.start_runner(runner_ctx, fl_ctx)

    def _stop_server(self):
        self._server_stopped = True
        self.stop_runner()

    def _is_stopped(self) -> Tuple[bool, int]:
        runner_stopped, ec = self.is_runner_stopped()
        if runner_stopped:
            return runner_stopped, ec

        if self._server_stopped:
            return True, self._exit_code

        return False, 0

    def start(self, fl_ctx: FLContext):
        # we dynamically create server address on localhost
        port = get_open_tcp_port(resources={})
        if not port:
            raise RuntimeError("failed to get a port for XGB server")

        server_addr = f"127.0.0.1:{port}"
        self._start_server(server_addr, port, self.world_size, fl_ctx)

        # start XGB client
        self.internal_xgb_client = GrpcClient(server_addr, self.int_client_grpc_options)
        self.internal_xgb_client.start(ready_timeout=self.xgb_server_ready_timeout)

    def stop(self, fl_ctx: FLContext):
        _stopping = True
        client = self.internal_xgb_client
        self.internal_xgb_client = None
        if client:
            self.log_info(fl_ctx, "Stopping internal XGB client")
            client.stop()
        self._stop_server()

    def all_gather(self, rank: int, seq: int, send_buf: bytes, fl_ctx: FLContext) -> bytes:
        try:
            result = self.internal_xgb_client.send_allgather(seq_num=seq, rank=rank, data=send_buf)
            if isinstance(result, pb2.AllgatherReply):
                return result.receive_buffer
            else:
                raise RuntimeError(f"bad result from XGB server: expect AllgatherReply but got {type(result)}")
        except Exception as ex:
            return self._handle_error(ex, "all_gather", rank, seq, send_buf)

    def all_gather_v(self, rank: int, seq: int, send_buf: bytes, fl_ctx: FLContext) -> bytes:
        try:
            result = self.internal_xgb_client.send_allgatherv(seq_num=seq, rank=rank, data=send_buf)
            if isinstance(result, pb2.AllgatherVReply):
                return result.receive_buffer
            else:
                raise RuntimeError(f"bad result from XGB server: expect AllgatherVReply but got {type(result)}")
        except Exception as ex:
            return self._handle_error(ex, "all_gather_v", rank, seq, send_buf)

    def all_reduce(
        self,
        rank: int,
        seq: int,
        data_type: int,
        reduce_op: int,
        send_buf: bytes,
        fl_ctx: FLContext,
    ) -> bytes:
        try:
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
        except Exception as ex:
            return self._handle_error(ex, "all_reduce", rank, seq, send_buf)

    def broadcast(self, rank: int, seq: int, root: int, send_buf: bytes, fl_ctx: FLContext) -> bytes:
        self.logger.debug(f"Sending broadcast: {rank=} {seq=} {root=} {len(send_buf)=}")
        try:
            result = self.internal_xgb_client.send_broadcast(seq_num=seq, rank=rank, data=send_buf, root=root)
            if isinstance(result, pb2.BroadcastReply):
                return result.receive_buffer
            else:
                raise RuntimeError(f"bad result from XGB server: expect BroadcastReply but got {type(result)}")
        except Exception as ex:
            return self._handle_error(ex, "broadcast", rank, seq, send_buf)

    def _handle_error(self, ex: Exception, op: str, rank: int, seq: int, send_buf: bytes) -> bytes:
        if self._stopping:
            self.logger.warning(f"Error while stopping ignored, " f"op={op} {rank=} {seq=} {len(send_buf)=}: {ex}")
            return bytes(0)
        else:
            raise ex
