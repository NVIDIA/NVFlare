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
import json
import time

import nvflare.app_opt.xgboost.histogram_based_v2.proto.federated_pb2 as pb2
from nvflare.apis.fl_component import FLComponent
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.app_opt.xgboost.histogram_based_v2.grpc_client import GrpcClient
from nvflare.app_opt.xgboost.histogram_based_v2.runners.xgb_runner import AppRunner


def encode_msg(msg: dict):
    return bytes(json.dumps(msg), "utf-8")


class MockSecureClientRunner(AppRunner, FLComponent):
    def __init__(self, sample_size=1000):
        FLComponent.__init__(self)
        self.training_stopped = False
        self.asked_to_stop = False
        self.sample_size = sample_size

    def run(self, ctx: dict):
        self.logger.info("START TRAINING")
        server_addr = ctx.get(Constant.RUNNER_CTX_SERVER_ADDR)
        rank = ctx.get(Constant.RUNNER_CTX_RANK)
        num_rounds = ctx.get(Constant.RUNNER_CTX_NUM_ROUNDS)

        client = GrpcClient(server_addr=server_addr)
        client.start()

        rank = rank
        seq = 0
        total_time = 0
        total_reqs = 0
        for i in range(num_rounds):
            if self.asked_to_stop:
                self.logger.info("training aborted")
                self.training_stopped = True
                return

            # fake bcst
            data = {
                "op": "none",
            }
            msg_data = encode_msg(data)
            self.logger.info("sending non-GH broadcast")
            start = time.time()
            seq += 1
            result = client.send_broadcast(
                seq_num=seq,
                rank=rank,
                data=msg_data,
                root=0,
            )
            total_reqs += 1
            total_time += time.time() - start
            if not isinstance(result, pb2.BroadcastReply):
                self.logger.error(f"expect reply to be pb2.BroadcastReply but got {type(result)}")
            elif result.receive_buffer != msg_data:
                self.logger.error("ERROR: broadcast result does not match request")
            else:
                self.logger.info("OK: broadcast result matched!")

            # gh bcst
            data = {
                "op": "gh",
                "size": self.sample_size,
            }

            self.logger.info("sending broadcast")
            start = time.time()
            seq += 1
            result = client.send_broadcast(
                seq_num=seq,
                rank=rank,
                data=encode_msg(data),
                root=0,
            )
            total_reqs += 1
            total_time += time.time() - start
            if not isinstance(result, pb2.BroadcastReply):
                self.logger.error(f"expect reply to be pb2.BroadcastReply but got {type(result)}")
            else:
                self.logger.info("OK: broadcast result received!")

            self.logger.info("sending allgatherV")
            start = time.time()
            seq += 1
            data = {"op": "aggr", "groups": None}
            result = client.send_allgatherv(seq_num=seq, rank=rank, data=encode_msg(data))
            total_reqs += 1
            total_time += time.time() - start
            if not isinstance(result, pb2.AllgatherVReply):
                self.logger.error(f"expect reply to be pb2.AllgatherVReply but got {type(result)}")
            else:
                self.logger.info("OK: allgatherV result received!")

            time.sleep(1.0)

        time_per_req = total_time / total_reqs
        self.logger.info(f"DONE: {total_reqs=} {total_time=} {time_per_req=}")
        print(f"DONE: {total_reqs=} {total_time=} {time_per_req=}")
        self.training_stopped = True

    def stop(self):
        self.asked_to_stop = True

    def is_stopped(self) -> (bool, int):
        return self.training_stopped, 0
