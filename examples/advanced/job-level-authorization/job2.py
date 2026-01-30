# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""
This job demonstrates job-level authorization with ProdEnv.
Job2 has the name "FL-Demo-Job2" which is BLOCKED by site_a's CustomSecurityHandler.

This script must be run from the job-level-authorization directory so that
the client.py script can be found and included in the job package.
"""
import argparse

from nvflare.app_common.np.recipes.fedavg import NumpyFedAvgRecipe
from nvflare.recipe import ProdEnv


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--startup_kit_location",
        type=str,
        default="/tmp/nvflare/poc/job-level-authorization/prod_00/super@a.org",
        help="Path to the admin startup kit directory",
    )
    parser.add_argument("--username", type=str, default="super@a.org", help="Username for authentication")

    return parser.parse_args()


def main():
    args = define_parser()

    # Create recipe with job name "FL-Demo-Job2" - this is BLOCKED by site_a
    recipe = NumpyFedAvgRecipe(
        name="FL-Demo-Job2",  # This name is BLOCKED by site_a's security handler
        min_clients=1,
        num_rounds=1,
        initial_model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        train_script="client.py",
    )

    print("Submitting job with name: 'FL-Demo-Job2' (BLOCKED by site_a)")
    print(f"Using startup kit at: {args.startup_kit_location}")
    print(f"Authenticating as: {args.username}")

    # Use ProdEnv to submit to the production environment
    env = ProdEnv(startup_kit_location=args.startup_kit_location, username=args.username)

    print("Submitting job to production environment...")
    run = recipe.execute(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in:", run.get_result())
    print()
    print("EXPECTED BEHAVIOR: Job 'FL-Demo-Job2' should be REJECTED by site_a's security handler")
    print("site_a will block this job, but site_b (without security handler) will accept it")


if __name__ == "__main__":
    main()
