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

from nvflare.app_opt.flower.grpc_server import GrpcServer
from nvflare.app_opt.flower.mock.echo_servicer import EchoServicer


def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_workers", "-w", type=int, help="max number of workers", required=False, default=20)
    parser.add_argument("--addr", "-a", type=str, help="server address", required=True)
    parser.add_argument("--num_rounds", "-n", type=int, help="number of rounds", required=True)
    args = parser.parse_args()

    if not args.addr:
        raise RuntimeError("missing server address '--addr/-a' in command")

    print(f"starting server: {args.addr=} {args.max_workers=} {args.num_rounds=}")
    servicer = EchoServicer(args.num_rounds)
    server = GrpcServer(
        args.addr,
        max_workers=args.max_workers,
        grpc_options=None,
        servicer=servicer,
    )
    servicer.set_server(server)
    server.start()


if __name__ == "__main__":
    main()
