from nvflare.app_common.xgb.bridge import XGBClientBridge
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.xgb.bridges.grpc.proto.federated_pb2_grpc import (
    FederatedServicer,
)
import nvflare.app_common.xgb.bridges.grpc.proto.federated_pb2 as pb2
from nvflare.fuel.f3.drivers.net_utils import get_open_tcp_port
from nvflare.app_common.xgb.bridges.grpc.server import XGBServer
from nvflare.app_common.xgb.process_manager import ProcessManager
from nvflare.security.logging import secure_format_exception


class GrpcClientBridge(XGBClientBridge, FederatedServicer):

    def __init__(
        self,
        run_xgb_client_cmd: str,
        internal_server_addr = None,
        grpc_options=None,
        req_timeout=10.0,
    ):
        XGBClientBridge.__init__(self, req_timeout)
        self.run_xgb_client_cmd = run_xgb_client_cmd
        self.internal_server_addr = internal_server_addr
        self.grpc_options = grpc_options
        self.internal_xgb_server = None
        self.client_manager = None
        self.stopped = False

    def start(self, fl_ctx: FLContext):
        if self.rank is None:
            raise RuntimeError("cannot start - my rank is not set")

        if not self.num_rounds:
            raise RuntimeError("cannot start - num_rounds is not set")

        if not self.internal_server_addr:
            # dynamically determine address on localhost
            port = get_open_tcp_port(resources={})
            if not port:
                raise RuntimeError("failed to get a port for XGB server")
            self.internal_server_addr = f"127.0.0.1:{port}"

        self.run_xgb_client_cmd = self.run_xgb_client_cmd.replace("$addr", self.internal_server_addr)
        self.run_xgb_client_cmd = self.run_xgb_client_cmd.replace("$rank", str(self.rank))
        self.run_xgb_client_cmd = self.run_xgb_client_cmd.replace("$num_rounds", str(self.num_rounds))

        self.logger.info(f"Start internal server at {self.internal_server_addr}")
        self.internal_xgb_server = XGBServer(self.internal_server_addr, 10, self.grpc_options, self)
        self.internal_xgb_server.start(no_blocking=True)
        self.logger.info(f"Started internal server at {self.internal_server_addr}")

        self.client_manager = ProcessManager(
            name="XGBClient",
            start_cmd=self.run_xgb_client_cmd,
        )
        self.client_manager.start()
        self.logger.info(f"Started external XGB Client")

    def stop(self, fl_ctx: FLContext):
        if self.stopped:
            return

        self.stopped = True
        if self.client_manager:
            self.logger.info("Stop external XGB client")
            self.client_manager.stop()

        if self.internal_xgb_server:
            self.logger.info("Stop internal XGB Server")
            self.internal_xgb_server.shutdown()

    def is_stopped(self) -> (bool, int):
        if self.client_manager:
            return self.client_manager.is_stopped()
        else:
            return True, 0

    def _abort(self, reason: str):
        # stop the gRPC XGB client (the target)
        self.abort_signal.trigger(True)

        # abort the FL client
        with self.engine.new_context() as fl_ctx:
            self.system_panic(reason, fl_ctx)

    def Allgather(self, request: pb2.AllgatherRequest, context):
        try:
            rcv_buf = self.send_all_gather(
                rank=request.rank,
                seq=request.sequence_number,
                send_buf=request.send_buffer,
            )
            return pb2.AllgatherReply(
                receive_buffer=rcv_buf
            )
        except Exception as ex:
            self._abort(reason=f"send_all_gather exception: {secure_format_exception(ex)}")
            return None

    def AllgatherV(self, request: pb2.AllgatherVRequest, context):
        try:
            rcv_buf = self.send_all_gather_v(
                rank=request.rank,
                seq=request.sequence_number,
                send_buf=request.send_buffer,
            )
            return pb2.AllgatherVReply(
                receive_buffer=rcv_buf
            )
        except Exception as ex:
            self._abort(reason=f"send_all_gather_v exception: {secure_format_exception(ex)}")
            return None

    def Allreduce(self, request: pb2.AllreduceRequest, context):
        try:
            rcv_buf = self.send_all_reduce(
                rank=request.rank,
                seq=request.sequence_number,
                data_type=request.data_type,
                reduce_op=request.reduce_operation,
                send_buf=request.send_buffer,
            )
            return pb2.AllreduceReply(
                receive_buffer=rcv_buf
            )
        except Exception as ex:
            self._abort(reason=f"send_all_reduce exception: {secure_format_exception(ex)}")
            return None

    def Broadcast(self, request: pb2.BroadcastRequest, context):
        try:
            rcv_buf = self.send_broadcast(
                rank=request.rank,
                seq=request.sequence_number,
                root=request.root,
                send_buf=request.send_buffer,
            )
            return pb2.BroadcastReply(
                receive_buffer=rcv_buf
            )
        except Exception as ex:
            self._abort(reason=f"send_broadcast exception: {secure_format_exception(ex)}")
            return None
