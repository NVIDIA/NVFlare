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

import threading
import time
from typing import Tuple

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

DUPLICATE_REQ_MAX_HOLD_TIME = 3600.0


class GrpcClientAdaptor(XGBClientAdaptor, FederatedServicer):
    """Implementation of XGBClientAdaptor that uses an internal `GrpcServer`.

    The `GrpcClientAdaptor` class serves as an interface between the XGBoost
    federated client and federated server components.
    It employs its `XGBRunner` to initiate an XGBoost federated gRPC client
    and utilizes an internal `GrpcServer` to forward client requests/responses.

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

    def __init__(self, int_server_grpc_options=None, in_process=True, per_msg_timeout=10.0, tx_timeout=100.0):
        """Constructor method to initialize the object.

        Args:
            int_server_grpc_options: An optional list of key-value pairs (`channel_arguments`
                in gRPC Core runtime) to configure the gRPC channel of internal `GrpcServer`.
            in_process (bool): Specifies whether to start the `XGBRunner` in the same process or not.
        """
        XGBClientAdaptor.__init__(self, in_process, per_msg_timeout, tx_timeout)
        self.int_server_grpc_options = int_server_grpc_options
        self.in_process = in_process
        self.internal_xgb_server = None
        self.stopped = False
        self.internal_server_addr = None
        self._training_stopped = False
        self._client_name = None
        self._workspace = None
        self._run_dir = None
        self._lock = threading.Lock()
        self._pending_req = {}

    def initialize(self, fl_ctx: FLContext):
        self._client_name = fl_ctx.get_identity_name()
        self._workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        run_number = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
        self._run_dir = self._workspace.get_run_dir(run_number)
        self.engine = fl_ctx.get_engine()

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
            Constant.RUNNER_CTX_DATA_SPLIT_MODE: self.data_split_mode,
            Constant.RUNNER_CTX_SECURE_TRAINING: self.secure_training,
            Constant.RUNNER_CTX_XGB_PARAMS: self.xgb_params,
            Constant.RUNNER_CTX_XGB_OPTIONS: self.xgb_options,
            Constant.RUNNER_CTX_MODEL_DIR: self._run_dir,
        }
        self.start_runner(runner_ctx, fl_ctx)

    def _stop_client(self):
        self._training_stopped = True
        self.stop_runner()

    def _is_stopped(self) -> Tuple[bool, int]:
        runner_stopped, ec = self.is_runner_stopped()
        if runner_stopped:
            return runner_stopped, ec

        if self._training_stopped:
            return True, 0

        return False, 0

    def start(self, fl_ctx: FLContext):
        if self.rank is None:
            raise RuntimeError("cannot start - my rank is not set")

        # dynamically determine address on localhost
        port = get_open_tcp_port(resources={})
        if not port:
            raise RuntimeError("failed to get a port for XGB server")
        self.internal_server_addr = f"127.0.0.1:{port}"
        self.log_info(fl_ctx, f"Start internal server at {self.internal_server_addr}")
        self.internal_xgb_server = GrpcServer(self.internal_server_addr, 10, self, self.int_server_grpc_options)
        self.internal_xgb_server.start(no_blocking=True)
        self.log_info(fl_ctx, f"Started internal server at {self.internal_server_addr}")
        self._start_client(self.internal_server_addr, fl_ctx)
        self.log_info(fl_ctx, "Started external XGB Client")

    def stop(self, fl_ctx: FLContext):
        if self.stopped:
            return

        self.stopped = True
        self._stop_client()

        if self.internal_xgb_server:
            self.log_info(fl_ctx, "Stop internal XGB Server")
            self.internal_xgb_server.shutdown()

    def _abort(self, reason: str):
        # stop the gRPC XGB client (the target)
        self.abort_signal.trigger(True)

        # abort the FL client
        with self.engine.new_context() as fl_ctx:
            self.system_panic(reason, fl_ctx)

    def Allgather(self, request: pb2.AllgatherRequest, context):
        try:
            if self._check_duplicate_seq("allgather", request.rank, request.sequence_number):
                return pb2.AllgatherReply(receive_buffer=bytes())

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
        finally:
            self._finish_pending_req("allgather", request.rank, request.sequence_number)

    def AllgatherV(self, request: pb2.AllgatherVRequest, context):
        try:
            if self._check_duplicate_seq("allgatherv", request.rank, request.sequence_number):
                return pb2.AllgatherVReply(receive_buffer=bytes())

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
        finally:
            self._finish_pending_req("allgatherv", request.rank, request.sequence_number)

    def Allreduce(self, request: pb2.AllreduceRequest, context):
        try:
            if self._check_duplicate_seq("allreduce", request.rank, request.sequence_number):
                return pb2.AllreduceReply(receive_buffer=bytes())

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
        finally:
            self._finish_pending_req("allreduce", request.rank, request.sequence_number)

    def Broadcast(self, request: pb2.BroadcastRequest, context):
        try:
            if self._check_duplicate_seq("broadcast", request.rank, request.sequence_number):
                return pb2.BroadcastReply(receive_buffer=bytes())

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
        finally:
            self._finish_pending_req("broadcast", request.rank, request.sequence_number)

    def _check_duplicate_seq(self, op: str, rank: int, seq: int):
        with self._lock:
            event = self._pending_req.get((rank, seq), None)
        if event:
            self.logger.info(f"Duplicate seq {op=} {rank=} {seq=}, wait till original req is done")
            event.wait(DUPLICATE_REQ_MAX_HOLD_TIME)
            time.sleep(1)  # To ensure the first request is returned first
            self.logger.info(f"Duplicate seq {op=} {rank=} {seq=} returned with empty buffer")
            return True

        with self._lock:
            self._pending_req[(rank, seq)] = threading.Event()
        return False

    def _finish_pending_req(self, op: str, rank: int, seq: int):
        with self._lock:
            event = self._pending_req.get((rank, seq), None)
            if not event:
                self.logger.error(f"No pending req {op=} {rank=} {seq=}")
                return

            event.set()
            del self._pending_req[(rank, seq)]
            self.logger.info(f"Request seq {op=} {rank=} {seq=} finished processing")
