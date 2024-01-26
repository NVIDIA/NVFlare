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
import logging

from nvflare.app_common.xgb.adaptors.grpc.mock.aggr_servicer import AggrServicer
from nvflare.app_common.xgb.adaptors.grpc.server import XGBServer


def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", "-a", type=str, help="server address", required=True)
    parser.add_argument("--num_clients", "-c", type=int, help="number of clients", required=True)
    parser.add_argument("--max_workers", "-w", type=int, help="max number of workers", required=False, default=20)

    args = parser.parse_args()
    print(f"starting server at {args.addr} max_workers={args.max_workers}")
    server = XGBServer(
        args.addr,
        max_workers=args.max_workers,
        options=None,
        servicer=AggrServicer(num_clients=args.num_clients),
    )
    server.start()


if __name__ == "__main__":
    main()
