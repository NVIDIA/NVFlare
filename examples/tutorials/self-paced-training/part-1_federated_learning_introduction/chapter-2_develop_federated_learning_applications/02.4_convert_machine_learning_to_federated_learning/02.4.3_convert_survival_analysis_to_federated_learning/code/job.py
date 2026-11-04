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

import argparse

from recipe import KaplanMeierRecipe

from nvflare.recipe import SimEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encryption", action="store_true")
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--workspace_dir", default="/tmp/nvflare/chapter2/survival")
    args = parser.parse_args()
    recipe = KaplanMeierRecipe(num_clients=args.num_clients, encryption=args.encryption)
    run = recipe.execute(SimEnv(num_clients=args.num_clients, workspace_root=args.workspace_dir, log_config="progress"))
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()
