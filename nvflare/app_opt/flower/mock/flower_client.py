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

import argparse
import logging
import os
import sys
import time

import flwr.proto.grpcadapter_pb2 as pb2

from nvflare.app_opt.flower.grpc_client import GrpcClient
from nvflare.fuel.utils.time_utils import time_to_string


def log(msg: str):
    for i in range(5):
        print(f"\r{i}", end=" ")
        sys.stdout.flush()
    print("\nend")
    print(f"{time_to_string(time.time())}: {msg}")
    sys.stdout.flush()


def train(server_addr, client_name):
    log(f"starting client {client_name} to connect to server at {server_addr}")
    client = GrpcClient(server_addr=server_addr)
    client.start()

    total_time = 0
    total_reqs = 0
    next_round = 0
    while True:
        log(f"Test round {next_round}")
        data = os.urandom(10)

        headers = {
            "target": "server",
            "round": str(next_round),
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
            log(f"expect reply to be pb2.MessageContainer but got {type(result)}")
        elif result.grpc_message_name != req.grpc_message_name:
            log("ERROR: msg_name does not match request")
        elif result.grpc_message_content != data:
            log("ERROR: result does not match request")
        else:
            log("OK: result matches request!")

        result_headers = result.metadata
        should_exit = result_headers.get("should-exit")
        if should_exit:
            log("got should-exit!")
            break

        next_round = result_headers.get("round")
        time.sleep(1.0)

    time_per_req = total_time / total_reqs
    log(f"DONE: {total_reqs=} {total_time=} {time_per_req=}")


def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", "-a", type=str, help="server address", required=True)
    parser.add_argument("--client_name", "-c", type=str, help="client name", required=True)
    parser.add_argument("--num_rounds", "-n", type=int, help="number of rounds", required=True)
    args = parser.parse_args()

    if not args.addr:
        raise RuntimeError("missing server address '--addr/-a' in command")

    if not args.num_rounds:
        raise RuntimeError("missing num rounds '--num_rounds/-n' in command")

    if args.num_rounds <= 0:
        raise RuntimeError("bad num rounds '--num_rounds/-n' in command: must be > 0")

    train(args.addr, args.client_name)


if __name__ == "__main__":
    main()
