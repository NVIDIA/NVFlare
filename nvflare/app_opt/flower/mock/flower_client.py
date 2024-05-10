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

import os
import time

import nvflare.app_opt.flower.proto.fleet_pb2 as pb2
from nvflare.app_opt.flower.defs import Constant
from nvflare.app_opt.flower.grpc_client import GrpcClient


def main():
    env = os.environ
    addr = env.get(Constant.APP_CTX_SERVER_ADDR)
    if not addr:
        raise RuntimeError(f"missing {Constant.APP_CTX_SERVER_ADDR} in env")

    num_rounds = env.get(Constant.APP_CTX_NUM_ROUNDS)
    if not num_rounds:
        raise RuntimeError(f"missing {Constant.APP_CTX_NUM_ROUNDS} in env")

    client_name = env.get(Constant.APP_CTX_CLIENT_NAME)

    num_rounds = int(num_rounds)

    print(f"starting client {client_name} to connect to server at {addr}")

    client = GrpcClient(server_addr=addr)
    client.start()

    total_time = 0
    total_reqs = 0
    for i in range(num_rounds):
        print(f"Test round {i}")
        data = os.urandom(10)

        headers = {
            "target": "server",
            "round": str(i),
            "origin": client_name,
        }
        req = pb2.MessageContainer(
            grpc_message_name="abc",
            grpc_message_content=data,
        )
        req.metadata.update(headers)

        start = time.time()
        result = client.send_request(req)
        total_reqs += 1
        total_time += time.time() - start
        if not isinstance(result, pb2.MessageContainer):
            print(f"expect reply to be pb2.MessageContainer but got {type(result)}")
        elif result.metadata != req.metadata:
            print("ERROR: metadata does not match request")
        elif result.grpc_message_name != req.grpc_message_name:
            print("ERROR: msg_name does not match request")
        elif result.grpc_message_content != data:
            print("ERROR: result does not match request")
        else:
            print("OK: result matches request!")
        time.sleep(1.0)

    time_per_req = total_time / total_reqs
    print(f"DONE: {total_reqs=} {total_time=} {time_per_req=}")


if __name__ == "__main__":
    main()
