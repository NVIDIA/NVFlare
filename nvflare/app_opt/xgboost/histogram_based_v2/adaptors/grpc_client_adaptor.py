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
import sys
import threading
from typing import Tuple

import nvflare.app_opt.xgboost.histogram_based_v2.proto.federated_pb2 as pb2
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.xgboost.histogram_based_v2.adaptor import XGBClientAdaptor
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.app_opt.xgboost.histogram_based_v2.grpc.grpc_server import GrpcServer
from nvflare.app_opt.xgboost.histogram_based_v2.proto.federated_pb2_grpc import FederatedServicer
from nvflare.fuel.f3.drivers.net_utils import get_open_tcp_port
from nvflare.security.logging import secure_format_exception, secure_log_traceback


class _ClientStarter:
    """This small class is used to start XGB client runner. It is used when running the runner in a thread
    or in a separate process.

    """

    def __init__(self, runner, in_process: bool):
        self.xgb_runner = runner
        self.in_process = in_process
        self.error = None
        self.started = True
        self.stopped = False
        self.exit_code = 0

    def start(self, ctx: dict):
        """Start the runner and wait for it to finish.

        Args:
            ctx:

        Returns:

        """
        try:
            self.xgb_runner.run(ctx)
            self.stopped = True
        except Exception as e:
            secure_log_traceback()
            self.error = f"Exception happens when running xgb train: {secure_format_exception(e)}"
            self.started = False
            self.exit_code = Constant.EXIT_CODE_CANT_START_XGB
            self.stopped = True
            if not self.in_process:
                # running in separate process - exit with error code for the monitor to report to server
                sys.exit(self.exit_code)


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

    def __init__(self, int_server_grpc_options=None, in_process=False, per_msg_timeout=10.0, tx_timeout=100.0):
        """Constructor method to initialize the object.

        Args:
            int_server_grpc_options: An optional list of key-value pairs (`channel_arguments`
                in gRPC Core runtime) to configure the gRPC channel of internal `GrpcServer`.
            in_process (bool): Specifies whether to start the `XGBRunner` in the same process or not.
            per_msg_timeout: Request per-msg timeout
            tx_timeout: timeout for the whole req transaction
        """
        XGBClientAdaptor.__init__(self, per_msg_timeout, tx_timeout)
        self.int_server_grpc_options = int_server_grpc_options
        self.in_process = in_process
        self.internal_xgb_server = None
        self.stopped = False
        self.internal_server_addr = None
        self._training_stopped = False
        self._client_name = None
        self._workspace = None
        self._run_dir = None
        self._process = None
        self._starter = None

    def initialize(self, fl_ctx: FLContext):
        self._client_name = fl_ctx.get_identity_name()
        self._workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        run_number = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
        self._run_dir = self._workspace.get_run_dir(run_number)

    def _start_client(self, server_addr: str):
        """Start the XGB client runner in a separate thread or separate process based on config.
        Note that when starting runner in a separate process, we must not call a method defined in this
        class since the self object contains a sender that contains a Core Cell which cannot be sent to
        the new process. Instead, we use a small _ClientStarter object to run the process.

        Args:
            server_addr: the internal gRPC server address that the XGB client will connect to

        Returns: None

        """
        ctx = {
            Constant.RUNNER_CTX_WORLD_SIZE: self.world_size,
            Constant.RUNNER_CTX_CLIENT_NAME: self._client_name,
            Constant.RUNNER_CTX_SERVER_ADDR: server_addr,
            Constant.RUNNER_CTX_RANK: self.rank,
            Constant.RUNNER_CTX_NUM_ROUNDS: self.num_rounds,
            Constant.RUNNER_CTX_MODEL_DIR: self._run_dir,
        }
        starter = _ClientStarter(self.xgb_runner, self.in_process)
        self.logger.info(f"starting XGB client with {ctx=}")
        if self.in_process:
            self.logger.info("starting XGB client in another thread")
            t = threading.Thread(
                target=starter.start,
                args=(ctx,),
                daemon=True,
                name="xgb_client_thread_runner",
            )
            t.start()
            self._starter = starter
            if not starter.started:
                self.logger.error(f"cannot start XGB client: {starter.error}")
                raise RuntimeError(starter.error)
        else:
            # start as a separate local process
            self.logger.info("starting XGB client in another process")
            self._process = multiprocessing.Process(
                target=starter.start,
                args=(ctx,),
                daemon=True,
                name="xgb_client_process_runner",
            )
            self._process.start()

    def _stop_client(self):
        self._training_stopped = True
        if self.in_process:
            if self.xgb_runner:
                self.xgb_runner.stop()
        else:
            if self._process:
                self._process.kill()

    def _is_stopped(self) -> Tuple[bool, int]:
        if self.in_process:
            if self._starter:
                if self._starter.stopped:
                    return True, self._starter.exit_code

            if self._training_stopped:
                return True, 0

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
        self.internal_xgb_server = GrpcServer(
            addr=self.internal_server_addr,
            max_workers=10,
            grpc_options=self.int_server_grpc_options,
            servicer=self,
        )
        self.internal_xgb_server.start(no_blocking=True)
        self.logger.info(f"Started internal server at {self.internal_server_addr}")
        self._start_client(self.internal_server_addr)
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
            rcv_buf = self._send_all_gather(
                rank=request.rank,
                seq=request.sequence_number,
                send_buf=request.send_buffer,
            )
            return pb2.AllgatherReply(receive_buffer=rcv_buf)
        except Exception as ex:
            self._abort(reason=f"send_all_gather exception: {secure_format_exception(ex)}")
            return None

    def Allreduce(self, request: pb2.AllreduceRequest, context):
        try:
            rcv_buf = self._send_all_reduce(
                rank=request.rank,
                seq=request.sequence_number,
                data_type=request.data_type,
                reduce_op=request.reduce_operation,
                send_buf=request.send_buffer,
            )
            return pb2.AllreduceReply(receive_buffer=rcv_buf)
        except Exception as ex:
            self._abort(reason=f"send_all_reduce exception: {secure_format_exception(ex)}")
            return None

    def Broadcast(self, request: pb2.BroadcastRequest, context):
        try:
            rcv_buf = self._send_broadcast(
                rank=request.rank,
                seq=request.sequence_number,
                root=request.root,
                send_buf=request.send_buffer,
            )
            return pb2.BroadcastReply(receive_buffer=rcv_buf)
        except Exception as ex:
            self._abort(reason=f"send_broadcast exception: {secure_format_exception(ex)}")
            return None
