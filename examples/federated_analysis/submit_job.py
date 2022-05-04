# Copyright (c) 2021, NVIDIA CORPORATION.
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
import os
import time

from nvflare.fuel.hci.client.fl_admin_api_runner import FLAdminAPIRunner, api_command_wrapper


def read_json(filename):
    assert os.path.isfile(filename), f"{filename} does not exist!"

    with open(filename, "r") as f:
        return json.load(f)


def write_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--admin_dir", type=str, default="./admin/", help="Path to admin directory.")
    parser.add_argument("--username", type=str, default="admin@nvidia.com", help="Admin username.")
    parser.add_argument("--job", type=str, default="cifar10_fedavg", help="Path to job config.")
    parser.add_argument("--poc", action='store_true', help="Whether admin uses POC mode.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Value controls the degree of heterogeneity. "
                                                                 "Lower values of alpha means higher heterogeneity.")
    args = parser.parse_args()

    assert 0.0 <= args.alpha <= 1.0, f"Alpha values should be between 0.0 and 1.0 but is {args.alpha}."

    assert os.path.isdir(args.admin_dir), f"admin directory does not exist at {args.admin_dir}"

    # Initialize the runner
    runner = FLAdminAPIRunner(
        username=args.username,
        admin_dir=args.admin_dir,
        poc=args.poc,
        debug=False,
    )

    # update alpha and split data dir
    # TODO: use full path in args.job
    client_config_filename = os.path.join("/home/hroth/Code2/nvflare/dev-2.1-cifar10/examples/cifar10/workspaces/poc_workspace/admin/transfer", args.job, args.job, "config", "config_fed_client.json")
    server_config_filename = os.path.join("/home/hroth/Code2/nvflare/dev-2.1-cifar10/examples/cifar10/workspaces/poc_workspace/admin/transfer", args.job, args.job, "config", "config_fed_server.json")

    job_name = os.path.basename(args.job)
    client_config = read_json(client_config_filename)
    server_config = read_json(server_config_filename)
    print(f"Set alpha to {args.alpha}")
    server_config["alpha"] = args.alpha
    split_dir = os.path.join(server_config["TRAIN_SPLIT_ROOT"], job_name + f"_alpha{args.alpha}")
    print(f"Set train split root to {split_dir}")
    server_config["TRAIN_SPLIT_ROOT"] = split_dir
    client_config["TRAIN_SPLIT_ROOT"] = split_dir
    write_json(client_config, client_config_filename)
    write_json(server_config, server_config_filename)

    # Submit job
    api_command_wrapper(runner.api.submit_job(args.job))


if __name__ == "__main__":
    main()
