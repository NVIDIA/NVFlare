# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.fuel.hci.client.fl_admin_api_runner import FLAdminAPIRunner, api_command_wrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--admin_dir", type=str, default="./admin/", help="Path to admin directory.")
    parser.add_argument("--username", type=str, default="admin", help="Admin username.")
    parser.add_argument("--job", type=str, default="prostate_fedavg", help="Path to job config.")
    args = parser.parse_args()

    assert os.path.isdir(args.admin_dir), f"admin directory does not exist at {args.admin_dir}"

    # Initialize the runner
    runner = FLAdminAPIRunner(
        username=args.username,
        admin_dir=args.admin_dir,
        poc=True,
        debug=False,
    )

    # Submit job
    api_command_wrapper(runner.api.submit_job(args.job))

    # finish
    runner.api.logout()
    runner.api.overseer_agent.end()


if __name__ == "__main__":
    main()
