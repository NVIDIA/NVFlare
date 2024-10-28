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
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", "-c", type=int, help="number of clients", required=False, default=2)
    parser.add_argument("--num_jobs", "-j", type=int, help="number of jobs", required=False, default=1)
    parser.add_argument("--scheme", "-s", type=str, help="scheme of the root url", required=False, default="grpc")
    args = parser.parse_args()

    num_clients = args.num_clients
    if num_clients <= 0:
        print(f"invalid num_clients {num_clients}: must be > 0")

    num_jobs = args.num_jobs
    if num_jobs <= 0:
        print(f"invalid num_jobs {num_jobs}: must be > 0")

    clients = [f"c{i + 1}" for i in range(num_clients)]
    jobs = [f"j{i + 1}" for i in range(num_jobs)]
    server_jobs = [f"s_{j}" for j in jobs]

    config = {
        "root_url": f"{args.scheme}://localhost:8002",
        "admin": {"host": "localhost", "port": "8003"},
        "server": {"children": server_jobs, "clients": clients},
    }

    for c in clients:
        cc = {c: {"children": [f"{c}_{j}" for j in jobs]}}
        config.update(cc)

    file_name = f"net_config_c{num_clients}_j{num_jobs}.json"
    json_object = json.dumps(config, indent=4)

    with open(file_name, "w") as outfile:
        outfile.write(json_object)
    print(f"Config file created: {file_name}")


if __name__ == "__main__":
    main()
