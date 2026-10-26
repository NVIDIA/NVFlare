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
from pathlib import Path

from nvflare.app_common.np.recipes.fedavg import NumpyFedAvgRecipe
from nvflare.recipe import SimEnv


def make_recipe(manifest, source="gen3", rounds=3):
    root = Path(__file__).parent
    recipe = NumpyFedAvgRecipe(
        name=f"gen3-{source}",
        min_clients=2,
        num_rounds=rounds,
        model=[0.0, 0.0, 0.0],
        train_script=str(root / "client.py"),
        train_args=["--manifest", Path(manifest).name, "--source", source],
        key_metric="mse",
        key_metric_mode="min",
    )
    # Place data beside the bundled trainer, regardless of the submitter's import
    # paths. Recipe currently exposes no file-source destination argument.
    script_path = recipe._job.job._get_relative_script(str(root / "client.py"))
    recipe._job.add_file_to_clients(
        str(Path(manifest).resolve()), dest_dir=str(Path(script_path).parent), app_folder_type="custom"
    )
    return recipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export the Gen3 example or run the pre-staged simulator control")
    parser.add_argument("--manifest", default="data/manifest.json")
    parser.add_argument("--source", choices=("gen3", "local"), default="gen3")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--job-dir", metavar="DIRECTORY", help="Export a deployable job to this directory")
    parser.add_argument("--workspace", default="/tmp/nvflare-gen3-simulation")
    args = parser.parse_args()
    recipe = make_recipe(args.manifest, args.source, args.rounds)
    if args.job_dir:
        recipe.export(args.job_dir)
    elif args.source == "local":
        run = recipe.execute(SimEnv(num_clients=2, workspace_root=args.workspace))
        print(run.get_status())
        print(run.get_result())
    else:
        parser.error("Use --job-dir for Gen3 jobs; their credentials belong to each deployed client site")
