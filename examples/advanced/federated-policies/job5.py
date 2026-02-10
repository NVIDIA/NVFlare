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
This job demonstrates federated-policy with ProdEnv.
Job5 has scope "foo."
"""
import argparse

from nvflare.app_common.np.recipes.fedavg import NumpyFedAvgRecipe
from nvflare.recipe import ProdEnv


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--startup_kit_location",
        type=str,
        default="./workspace/fed_policy/prod_00/trainer@b.org",
        help="Path to the admin startup kit directory",
    )
    parser.add_argument("--username", type=str, default="trainer@b.org", help="Username for authentication")

    return parser.parse_args()


def main():
    args = define_parser()

    recipe = NumpyFedAvgRecipe(
        name="hello-numpy",
        min_clients=1,
        num_rounds=1,
        initial_model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        train_script="client.py",
    )
    recipe.job.job.meta_props = {"scope": "foo"}
    print(f"Using startup kit at: {args.startup_kit_location}")
    print(f"Authenticating as: {args.username}")

    # Use ProdEnv to submit to the production environment
    env = ProdEnv(startup_kit_location=args.startup_kit_location, username=args.username)

    print("Submitting job to production environment...")
    run = recipe.execute(env)
    print()
    print("Result can be found in:", run.get_result())
    print("Job Status is:", run.get_status())


if __name__ == "__main__":
    main()
