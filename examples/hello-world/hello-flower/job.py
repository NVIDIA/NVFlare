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

from argparse import ArgumentParser

from nvflare.app_opt.flower.recipe import FlowerRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking


def main():
    parser = ArgumentParser()
    parser.add_argument("--job_name", type=str, required=True)
    parser.add_argument("--content_dir", type=str, required=True)
    parser.add_argument("--stream_metrics", action="store_true")
    parser.add_argument("--export_job", action="store_true")
    parser.add_argument("--export_dir", type=str, default="jobs")
    parser.add_argument("--workdir", type=str, default="/tmp/nvflare/hello-flower")
    args = parser.parse_args()

    num_of_clients = 2

    recipe = FlowerRecipe(
        name=args.job_name,
        flower_content=args.content_dir,
        min_clients=num_of_clients,
    )

    if args.stream_metrics:
        add_experiment_tracking(recipe, tracking_type="tensorboard")

    if args.export_job:
        recipe.export(args.export_dir)
        print(f"Job exported to {args.export_dir}")
    else:
        env = SimEnv(num_clients=num_of_clients, workspace_root=args.workdir)
        run = recipe.execute(env)
        print()
        print("Result can be found in :", run.get_result())
        print("Job Status is:", run.get_status())
        print()


if __name__ == "__main__":
    main()
