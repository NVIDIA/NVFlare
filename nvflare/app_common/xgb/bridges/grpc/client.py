from nvflare.app_common.xgb.bridges.grpc.proto.federated_pb2_grpc import FederatedStub
import nvflare.app_common.xgb.bridges.grpc.proto.federated_pb2 as pb2
from nvflare.app_common.xgb.defs import Constant
from nvflare.fuel.utils.obj_utils import get_logger

import grpc


class XGBClient:

    def __init__(self, server_addr, grpc_options=None):
        self.stub = None
        self.channel = None
        self.server_addr = server_addr
        self.grpc_options = grpc_options
        self.started = False
        self.logger = get_logger(self)
        self.op_table = None

    def start(self, ready_timeout=10):
        if self.started:
            return

        self.started = True
        self.channel = grpc.insecure_channel(self.server_addr, options=self.grpc_options)
        self.stub = FederatedStub(self.channel)

        self.op_table = {
            Constant.OP_ALL_GATHER: (pb2.AllgatherRequest, pb2.AllgatherReply, self.stub.Allgather),
            Constant.OP_ALL_GATHER_V: (pb2.AllgatherVRequest, pb2.AllgatherVReply, self.stub.AllgatherV),
            Constant.OP_ALL_REDUCE: (pb2.AllreduceRequest, pb2.AllreduceReply, self.stub.Allreduce),
            Constant.OP_BROADCAST: (pb2.BroadcastRequest, pb2.BroadcastReply, self.stub.Broadcast)
        }

        # wait for channel ready
        try:
            grpc.channel_ready_future(self.channel).result(timeout=ready_timeout)
        except grpc.FutureTimeoutError:
            raise RuntimeError(f"cannot connect to server after {ready_timeout} seconds")

    def send_allgather(self, seq_num, rank, data: bytes):
        req = pb2.AllgatherRequest(
            sequence_number=seq_num,
            rank=rank,
            send_buffer=data,
        )

        assert isinstance(self.stub, FederatedStub)
        result = self.stub.Allgather(req)
        if not isinstance(result, pb2.AllgatherReply):
            self.logger.error(f"expect reply to be pb2.AllgatherReply but got {type(result)}")
            return None
        return result

    def send_allgatherv(self, seq_num, rank, data: bytes):
        req = pb2.AllgatherVRequest(
            sequence_number=seq_num,
            rank=rank,
            send_buffer=data,
        )

        assert isinstance(self.stub, FederatedStub)
        result = self.stub.AllgatherV(req)
        if not isinstance(result, pb2.AllgatherVReply):
            self.logger.error(f"expect reply to be pb2.AllgatherVReply but got {type(result)}")
            return None
        return result

    def send_allreduce(self, seq_num, rank, data: bytes, data_type, reduce_op):
        req = pb2.AllreduceRequest(
            sequence_number=seq_num,
            rank=rank,
            send_buffer=data,
            data_type=data_type,
            reduce_operation=reduce_op,
        )

        assert isinstance(self.stub, FederatedStub)
        result = self.stub.Allreduce(req)
        if not isinstance(result, pb2.AllreduceReply):
            self.logger.error(f"expect reply to be pb2.AllreduceReply but got {type(result)}")
            return None
        return result

    def send_broadcast(self, seq_num, rank, data: bytes, root):
        req = pb2.BroadcastRequest(
            sequence_number=seq_num,
            rank=rank,
            send_buffer=data,
            root=root,
        )

        assert isinstance(self.stub, FederatedStub)
        result = self.stub.Broadcast(req)
        if not isinstance(result, pb2.BroadcastReply):
            self.logger.error(f"expect reply to be pb2.BroadcastReply but got {type(result)}")
            return None
        return result

    def forward_request(self, op: str, serialized_xgb_req: bytes):
        op_info = self.op_table.get(op)
        if not op_info:
            self.logger.error(f"no operator defined for '{op}'")
            return None

        req_cls, reply_cls, send_f = op_info
        req = req_cls.FromString(serialized_xgb_req)
        result = send_f(req)
        if isinstance(result, reply_cls):
            return result.SerializeToString()
        else:
            self.logger.error(f"expect result to be {reply_cls} but got {type(result)}")
            return None

    def stop(self):
        if self.channel:
            try:
                self.channel.close()
            except:
                pass
