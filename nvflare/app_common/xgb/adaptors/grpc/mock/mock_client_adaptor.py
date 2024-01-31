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

import os
import threading
import time

import nvflare.app_common.xgb.adaptors.grpc.proto.federated_pb2 as pb2
from nvflare.app_common.xgb.adaptors.grpc.client import XGBClient
from nvflare.app_common.xgb.adaptors.grpc.client_adaptor import GrpcClientAdaptor


class MockClientAdaptor(GrpcClientAdaptor):
    def __init__(
        self,
        grpc_options=None,
        req_timeout=10.0,
    ):
        GrpcClientAdaptor.__init__(self, grpc_options, req_timeout)
        self.training_stopped = False
        self.asked_to_stop = False

    def start_client(self, server_addr: str, port: int):
        t = threading.Thread(target=self._do_start_client, args=(server_addr,), daemon=True)
        t.start()

    def _do_start_client(self, server_addr: str):
        client = XGBClient(server_addr=server_addr)
        client.start()

        rank = self.rank
        seq = 0
        total_time = 0
        total_reqs = 0
        for i in range(self.num_rounds):
            if self.abort_signal.triggered or self.asked_to_stop:
                self.logger.info("training aborted")
                self.training_stopped = True
                return

            self.logger.info(f"Test round {i}")
            data = os.urandom(1000000)

            self.logger.info("sending allgather")
            start = time.time()
            result = client.send_allgather(seq_num=seq + 1, rank=rank, data=data)
            total_reqs += 1
            total_time += time.time() - start
            if not isinstance(result, pb2.AllgatherReply):
                self.logger.error(f"expect reply to be pb2.AllgatherReply but got {type(result)}")
            elif result.receive_buffer != data:
                self.logger.error("allgather result does not match request")
            else:
                self.logger.info("OK: allgather result matches request!")

            self.logger.info("sending allgatherV")
            start = time.time()
            result = client.send_allgatherv(seq_num=seq + 2, rank=rank, data=data)
            total_reqs += 1
            total_time += time.time() - start
            if not isinstance(result, pb2.AllgatherVReply):
                self.logger.error(f"expect reply to be pb2.AllgatherVReply but got {type(result)}")
            elif result.receive_buffer != data:
                self.logger.error("allgatherV result does not match request")
            else:
                self.logger.info("OK: allgatherV result matches request!")

            self.logger.info("sending allreduce")
            start = time.time()
            result = client.send_allreduce(
                seq_num=seq + 3,
                rank=rank,
                data=data,
                reduce_op=2,
                data_type=2,
            )
            total_reqs += 1
            total_time += time.time() - start
            if not isinstance(result, pb2.AllreduceReply):
                self.logger.error(f"expect reply to be pb2.AllreduceReply but got {type(result)}")
            elif result.receive_buffer != data:
                self.logger.error("allreduce result does not match request")
            else:
                self.logger.info("OK: allreduce result matches request!")

            self.logger.info("sending broadcast")
            start = time.time()
            result = client.send_broadcast(
                seq_num=seq + 4,
                rank=rank,
                data=data,
                root=3,
            )
            total_reqs += 1
            total_time += time.time() - start
            if not isinstance(result, pb2.BroadcastReply):
                self.logger.error(f"expect reply to be pb2.BroadcastReply but got {type(result)}")
            elif result.receive_buffer != data:
                self.logger.error("ERROR: broadcast result does not match request")
            else:
                self.logger.info("OK: broadcast result matches request!")

            seq += 4
            time.sleep(1.0)

        time_per_req = total_time / total_reqs
        self.logger.info(f"DONE: {total_reqs=} {total_time=} {time_per_req=}")
        self.training_stopped = True

    def stop_client(self):
        self.asked_to_stop = True

    def is_client_stopped(self) -> (bool, int):
        return self.training_stopped, 0
