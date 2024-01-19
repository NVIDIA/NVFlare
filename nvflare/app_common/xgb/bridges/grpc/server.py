from nvflare.app_common.xgb.bridges.grpc.proto.federated_pb2_grpc import FederatedServicer, add_FederatedServicer_to_server
import nvflare.app_common.xgb.bridges.grpc.proto.federated_pb2 as pb2
import grpc
import concurrent.futures as futures
from nvflare.fuel.utils.obj_utils import get_logger


class EchoServicer(FederatedServicer):
    def __init__(self):
        self.logger = get_logger(self)

    def Allgather(self, request: pb2.AllgatherRequest, context):
        seq = request.sequence_number
        rank = request.rank
        data = request.send_buffer
        print(f"got Allgather: {seq=} {rank=} data_size={len(data)}")
        return pb2.AllgatherReply(receive_buffer=data)

    def AllgatherV(self, request: pb2.AllgatherVRequest, context):
        seq = request.sequence_number
        rank = request.rank
        data = request.send_buffer
        self.logger.info(f"got AllgatherV: {seq=} {rank=} data_size={len(data)}")
        return pb2.AllgatherVReply(receive_buffer=data)

    def Allreduce(self, request: pb2.AllreduceRequest, context):
        seq = request.sequence_number
        rank = request.rank
        data = request.send_buffer
        reduce_op = request.reduce_operation
        data_type = request.data_type
        self.logger.info(f"got Allreduce: {seq=} {rank=} {reduce_op=} {data_type=} data_size={len(data)}")
        return pb2.AllreduceReply(receive_buffer=data)

    def Broadcast(self, request: pb2.BroadcastRequest, context):
        seq = request.sequence_number
        rank = request.rank
        data = request.send_buffer
        root = request.root
        self.logger.info(f"got Broadcast: {seq=} {rank=} {root=} data_size={len(data)}")
        return pb2.BroadcastReply(receive_buffer=data)


class XGBServer:
    def __init__(self, addr, max_workers: int, options, servicer=None):
        if not servicer:
            servicer = EchoServicer()
        self.grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers), options=options)
        add_FederatedServicer_to_server(servicer, self.grpc_server)

        try:
            self.grpc_server.add_insecure_port(addr)
            print(f"XGBServer: added insecure port at {addr}")
        except Exception as ex:
            print(f"cannot listen on {addr}: {type(ex)}")

    def start(self, no_blocking=False):
        print("starting server")
        self.grpc_server.start()
        if no_blocking:
            print("no blocking")
            return
        else:
            print("wait_for_termination ")
            self.grpc_server.wait_for_termination()
            print("server terminated")

    def shutdown(self):
        self.grpc_server.stop(grace=0.5)
        self.grpc_server = None
