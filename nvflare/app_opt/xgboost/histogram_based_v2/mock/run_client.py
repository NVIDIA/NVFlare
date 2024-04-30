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

import argparse
import os
import time

import nvflare.app_opt.xgboost.histogram_based_v2.proto.federated_pb2 as pb2
from nvflare.app_opt.xgboost.histogram_based_v2.grpc_client import GrpcClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", "-a", type=str, help="server address", required=True)
    parser.add_argument("--rank", "-r", type=int, help="client rank", required=True)
    parser.add_argument("--num_rounds", "-n", type=int, help="number of rounds", required=True)

    args = parser.parse_args()
    client = GrpcClient(server_addr=args.addr)
    client.start()

    rank = args.rank
    seq = 0
    total_time = 0
    total_reqs = 0
    for i in range(args.num_rounds):
        print(f"Test round {i}")
        data = os.urandom(1000000)

        print("sending allgather")
        start = time.time()
        result = client.send_allgather(seq_num=seq + 1, rank=rank, data=data)
        total_reqs += 1
        total_time += time.time() - start
        if not isinstance(result, pb2.AllgatherReply):
            print(f"expect reply to be pb2.AllgatherReply but got {type(result)}")
        elif result.receive_buffer != data:
            print("ERROR: allgather result does not match request")
        else:
            print("OK: allgather result matches request!")

        print("sending allgatherV")
        start = time.time()
        result = client.send_allgatherv(seq_num=seq + 2, rank=rank, data=data)
        total_reqs += 1
        total_time += time.time() - start
        if not isinstance(result, pb2.AllgatherVReply):
            print(f"expect reply to be pb2.AllgatherVReply but got {type(result)}")
        elif result.receive_buffer != data:
            print("ERROR: allgatherV result does not match request")
        else:
            print("OK: allgatherV result matches request!")

        print("sending allreduce")
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
            print(f"expect reply to be pb2.AllreduceReply but got {type(result)}")
        elif result.receive_buffer != data:
            print("ERROR: allreduce result does not match request")
        else:
            print("OK: allreduce result matches request!")

        print("sending broadcast")
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
            print(f"expect reply to be pb2.BroadcastReply but got {type(result)}")
        elif result.receive_buffer != data:
            print("ERROR: broadcast result does not match request")
        else:
            print("OK: broadcast result matches request!")

        seq += 4
        time.sleep(1.0)

    time_per_req = total_time / total_reqs
    print(f"DONE: {total_reqs=} {total_time=} {time_per_req=}")


if __name__ == "__main__":
    main()
