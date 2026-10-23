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

"""Export two-site, two-round job/task controls for a provisioned Process site."""

import argparse
import json
from pathlib import Path

import torch

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import add_final_global_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lifetime", choices=("job", "task"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-oom", action="store_true", help="fail site-1 task training with a real CUDA OOM")
    args = parser.parse_args()
    if args.gpu_oom and (args.device != "cuda" or args.lifetime != "task"):
        parser.error("--gpu-oom requires --device cuda --lifetime task")
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    name = f"process-{args.device}-{args.lifetime}" + ("-oom" if args.gpu_oom else "")
    recipe = FedAvgRecipe(
        name=name,
        model=model,
        min_clients=2,
        num_rounds=2,
        train_script=str(Path(__file__).with_name("train.py")),
        train_args=["--device", args.device] + (["--gpu-oom"] if args.gpu_oom else []),
        execution_lifetime=args.lifetime,
    )
    add_final_global_evaluation(recipe, participating_clients=["site-1", "site-2"])
    recipe.export(args.output)
    # Build explicit submitted-job metadata in this test fixture. Task GPU
    # counts belong to the task launcher; resident controls use the CP manager.
    meta_path = Path(args.output) / name / "meta.json"
    meta = json.loads(meta_path.read_text())
    if args.device == "cuda":
        if args.lifetime == "task":
            meta["launcher_spec"] = {"default": {"process": {"num_of_gpus": 1}}}
        else:
            meta["resource_spec"] = {site: {"num_of_gpus": 1, "mem_per_gpu_in_GiB": 1} for site in ("site-1", "site-2")}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(meta_path.parent)


if __name__ == "__main__":
    main()
