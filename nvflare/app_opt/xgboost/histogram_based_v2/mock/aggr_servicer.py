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

import nvflare.app_opt.xgboost.histogram_based_v2.proto.federated_pb2 as pb2
from nvflare.app_opt.xgboost.histogram_based_v2.proto.federated_pb2_grpc import FederatedServicer
from nvflare.fuel.utils.obj_utils import get_logger


class ReqWaiter:
    def __init__(self, exp_num_clients: int, exp_seq: int, exp_op):
        self.exp_num_clients = exp_num_clients
        self.exp_seq = exp_seq
        self.exp_op = exp_op
        self.reqs = {}
        self.result = {}
        self.waiter = threading.Event()

    def add_request(self, op: str, rank, seq, req):
        if seq != self.exp_seq:
            raise RuntimeError(f"expecting seq {self.exp_seq} from {rank=} but got {seq}")

        if op != self.exp_op:
            raise RuntimeError(f"expecting op {self.exp_op} from {rank=} but got {op}")

        if rank in self.reqs:
            raise RuntimeError(f"duplicate request from {op=} {rank=} {seq=}")

        self.reqs[rank] = req

        if isinstance(req, pb2.AllgatherRequest):
            reply = pb2.AllgatherReply(receive_buffer=req.send_buffer)
        elif isinstance(req, pb2.AllgatherVRequest):
            reply = pb2.AllgatherVReply(receive_buffer=req.send_buffer)
        elif isinstance(req, pb2.AllreduceRequest):
            reply = pb2.AllreduceReply(receive_buffer=req.send_buffer)
        elif isinstance(req, pb2.BroadcastRequest):
            reply = pb2.BroadcastReply(receive_buffer=req.send_buffer)
        else:
            raise RuntimeError(f"unknown request type {type(req)}")
        self.result[rank] = reply
        if len(self.reqs) == self.exp_num_clients:
            self.waiter.set()

    def wait(self, timeout):
        return self.waiter.wait(timeout)


class AggrServicer(FederatedServicer):
    def __init__(self, num_clients, aggr_timeout=10.0):
        self.logger = get_logger(self)
        self.num_clients = num_clients
        self.aggr_timeout = aggr_timeout
        self.req_lock = threading.Lock()
        self.req_waiter = None

    def _wait_for_result(self, op, rank, seq, request):
        with self.req_lock:
            if not self.req_waiter:
                self.logger.info(f"setting new waiter for {self.aggr_timeout} secs: {seq=} {op=}")
                self.req_waiter = ReqWaiter(
                    exp_num_clients=self.num_clients,
                    exp_seq=seq,
                    exp_op=op,
                )
            self.req_waiter.add_request(op, rank, seq, request)
        if not self.req_waiter.wait(self.aggr_timeout):
            self.logger.error(f"results not received from all ranks after {self.aggr_timeout} seconds")
        self.logger.info(f"for {rank=}: results remaining: {self.req_waiter.result.keys()}")
        with self.req_lock:
            result = self.req_waiter.result.pop(rank, None)
            if len(self.req_waiter.result) == 0:
                self.logger.info("all results are retrieved - reset req_waiter to None")
                self.req_waiter = None
        return result

    def Allgather(self, request: pb2.AllgatherRequest, context):
        seq = request.sequence_number
        rank = request.rank
        data = request.send_buffer
        op = "Allgather"
        self.logger.info(f"got {op}: {seq=} {rank=} data_size={len(data)}")
        return self._wait_for_result(op, rank, seq, request)

    def AllgatherV(self, request: pb2.AllgatherVRequest, context):
        seq = request.sequence_number
        rank = request.rank
        data = request.send_buffer
        op = "AllgatherV"
        self.logger.info(f"got {op}: {seq=} {rank=} data_size={len(data)}")
        return self._wait_for_result(op, rank, seq, request)

    def Allreduce(self, request: pb2.AllreduceRequest, context):
        seq = request.sequence_number
        rank = request.rank
        data = request.send_buffer
        reduce_op = request.reduce_operation
        data_type = request.data_type
        op = "Allreduce"
        self.logger.info(f"got {op}: {seq=} {rank=} {reduce_op=} {data_type=} data_size={len(data)}")
        return self._wait_for_result(op, rank, seq, request)

    def Broadcast(self, request: pb2.BroadcastRequest, context):
        seq = request.sequence_number
        rank = request.rank
        data = request.send_buffer
        root = request.root
        op = "Broadcast"
        self.logger.info(f"got {op}: {seq=} {rank=} {root=} data_size={len(data)}")
        return self._wait_for_result(op, rank, seq, request)
