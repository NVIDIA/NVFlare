# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import grpc

import nvflare.app_opt.xgboost.histogram_based_v2.proto.federated_pb2 as pb2
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.xgboost.histogram_based_v2.adaptors.xgb_adaptor import XGBClientAdaptor
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.app_opt.xgboost.histogram_based_v2.grpc_server import GrpcServer
from nvflare.app_opt.xgboost.histogram_based_v2.proto.federated_pb2_grpc import FederatedServicer
from nvflare.fuel.f3.drivers.net_utils import get_open_tcp_port
from nvflare.security.logging import secure_format_exception


class GrpcClientAdaptor(XGBClientAdaptor, FederatedServicer):
    def __init__(
        self,
        int_server_grpc_options=None,
        in_process=True,
    ):
        XGBClientAdaptor.__init__(self, in_process)
        self.int_server_grpc_options = int_server_grpc_options
        self.in_process = in_process
        self.internal_xgb_server = None
        self.stopped = False
        self.internal_server_addr = None
        self._training_stopped = False
        self._client_name = None
        self._app_dir = None
        self._workspace = None
        self._run_dir = None

    def initialize(self, fl_ctx: FLContext):
        self._client_name = fl_ctx.get_identity_name()
        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        self._app_dir = ws.get_app_dir(fl_ctx.get_job_id())
        self._workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        run_number = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
        self._run_dir = self._workspace.get_run_dir(run_number)
        self.engine = engine

    def _start_client(self, server_addr: str, fl_ctx: FLContext):
        """Start the XGB client runner in a separate thread or separate process based on config.
        Note that when starting runner in a separate process, we must not call a method defined in this
        class since the self object contains a sender that contains a Core Cell which cannot be sent to
        the new process. Instead, we use a small _ClientStarter object to run the process.

        Args:
            server_addr: the internal gRPC server address that the XGB client will connect to

        Returns: None

        """
        runner_ctx = {
            Constant.RUNNER_CTX_WORLD_SIZE: self.world_size,
            Constant.RUNNER_CTX_CLIENT_NAME: self._client_name,
            Constant.RUNNER_CTX_SERVER_ADDR: server_addr,
            Constant.RUNNER_CTX_RANK: self.rank,
            Constant.RUNNER_CTX_NUM_ROUNDS: self.num_rounds,
            Constant.RUNNER_CTX_MODEL_DIR: self._run_dir,
            Constant.RUNNER_CTX_TB_DIR: self._app_dir,
        }
        self.start_runner(runner_ctx, fl_ctx)

    def _stop_client(self):
        self._training_stopped = True
        self.stop_runner()

    def _is_stopped(self) -> (bool, int):
        runner_stopped, ec = self.is_runner_stopped()
        if runner_stopped:
            return runner_stopped, ec

        if self._training_stopped:
            return True, 0

        return False, 0

    def start(self, fl_ctx: FLContext):
        if self.rank is None:
            raise RuntimeError("cannot start - my rank is not set")

        if not self.num_rounds:
            raise RuntimeError("cannot start - num_rounds is not set")

        # dynamically determine address on localhost
        port = get_open_tcp_port(resources={})
        if not port:
            raise RuntimeError("failed to get a port for XGB server")
        self.internal_server_addr = f"127.0.0.1:{port}"
        self.logger.info(f"Start internal server at {self.internal_server_addr}")
        self.internal_xgb_server = GrpcServer(self.internal_server_addr, 10, self.int_server_grpc_options, self)
        self.internal_xgb_server.start(no_blocking=True)
        self.logger.info(f"Started internal server at {self.internal_server_addr}")
        self._start_client(self.internal_server_addr, fl_ctx)
        self.logger.info("Started external XGB Client")

    def stop(self, fl_ctx: FLContext):
        if self.stopped:
            return

        self.stopped = True
        self._stop_client()

        if self.internal_xgb_server:
            self.logger.info("Stop internal XGB Server")
            self.internal_xgb_server.shutdown()

    def _abort(self, reason: str):
        # stop the gRPC XGB client (the target)
        self.abort_signal.trigger(True)

        # abort the FL client
        with self.engine.new_context() as fl_ctx:
            self.system_panic(reason, fl_ctx)

    def Allgather(self, request: pb2.AllgatherRequest, context):
        try:
            rcv_buf, _ = self._send_all_gather(
                rank=request.rank,
                seq=request.sequence_number,
                send_buf=request.send_buffer,
            )
            return pb2.AllgatherReply(receive_buffer=rcv_buf)
        except Exception as ex:
            self._abort(reason=f"send_all_gather exception: {secure_format_exception(ex)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(ex))
            return pb2.AllgatherReply(receive_buffer=None)

    def AllgatherV(self, request: pb2.AllgatherVRequest, context):
        try:
            rcv_buf = self._do_all_gather_v(
                rank=request.rank,
                seq=request.sequence_number,
                send_buf=request.send_buffer,
            )
            return pb2.AllgatherVReply(receive_buffer=rcv_buf)
        except Exception as ex:
            self._abort(reason=f"send_all_gather_v exception: {secure_format_exception(ex)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(ex))
            return pb2.AllgatherVReply(receive_buffer=None)

    def Allreduce(self, request: pb2.AllreduceRequest, context):
        try:
            rcv_buf, _ = self._send_all_reduce(
                rank=request.rank,
                seq=request.sequence_number,
                data_type=request.data_type,
                reduce_op=request.reduce_operation,
                send_buf=request.send_buffer,
            )
            return pb2.AllreduceReply(receive_buffer=rcv_buf)
        except Exception as ex:
            self._abort(reason=f"send_all_reduce exception: {secure_format_exception(ex)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(ex))
            return pb2.AllreduceReply(receive_buffer=None)

    def Broadcast(self, request: pb2.BroadcastRequest, context):
        try:
            rcv_buf = self._do_broadcast(
                rank=request.rank,
                send_buf=request.send_buffer,
                seq=request.sequence_number,
                root=request.root,
            )
            return pb2.BroadcastReply(receive_buffer=rcv_buf)
        except Exception as ex:
            self._abort(reason=f"send_broadcast exception: {secure_format_exception(ex)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(ex))
            return pb2.BroadcastReply(receive_buffer=None)
