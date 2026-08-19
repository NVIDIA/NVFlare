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

"""FedAvg Recipe for federated classifier-logit completion."""

import argparse
import json
import os
import sys
from pathlib import Path

from src.features import load_cache_split

PROJECT_DIR = Path(__file__).resolve().parent


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FedCoRe completion operator with NVFlare FedAvg.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workspace", default="/tmp/nvflare/fedcore")
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--task-weight", type=float, default=1.0)
    parser.add_argument("--effect-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
    from nvflare.recipe import SimEnv, set_per_site_config

    args = define_parser().parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sites = ["site-1", "site-2", "site-3"]
    input_dim = int(load_cache_split(cache_dir, "site-1", "train")["missing_features"].shape[1])
    per_site_config = {}
    for site in sites:
        train_args = (
            f"--cache-dir {cache_dir} --output-dir {output_dir} --site {site} --input-dim {input_dim} "
            f"--hidden-dim {args.hidden_dim} --dropout {args.dropout} --local-epochs {args.local_epochs} "
            f"--batch-size {args.batch_size} --learning-rate {args.learning_rate} --task-weight {args.task_weight} "
            f"--effect-weight {args.effect_weight} --seed {args.seed}"
        )
        per_site_config[site] = {"train_args": train_args, "command": f"{sys.executable} -u"}

    model = {
        "class_path": "model.LogitCompletionModel",
        "args": {
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "seed": args.seed,
        },
    }
    previous_cwd = Path.cwd()
    os.chdir(PROJECT_DIR)
    try:
        recipe = FedAvgRecipe(
            name="fedcore-image-completion",
            model=model,
            min_clients=len(sites),
            num_rounds=args.num_rounds,
            train_script="client.py",
            launch_external_process=True,
            server_expected_format="pytorch",
            key_metric="",
        )
        set_per_site_config(recipe, per_site_config)
    finally:
        os.chdir(previous_cwd)
    recipe.add_client_file(str(PROJECT_DIR / "client.py"), clients=sites)
    recipe.add_server_file(str(PROJECT_DIR / "model.py"))
    recipe.add_client_config(
        {"max_resends": 3, "tensor_min_download_timeout": 600},
        clients=sites,
    )
    recipe.add_server_config({"streaming_per_request_timeout": 600, "tensor_min_download_timeout": 600})
    env = SimEnv(clients=sites, num_threads=len(sites), workspace_root=str(workspace))
    run = recipe.execute(env)
    status = run.get_status()
    result = {
        "status": str(status) if status is not None else "completed",
        "result": str(run.get_result()),
        "workspace": str(workspace),
        "output_dir": str(output_dir),
        "input_dim": input_dim,
    }
    with (output_dir / "job_result.json").open("w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Job status: {result['status']}")
    print(f"Result: {result['result']}")
    print(f"Machine-readable job result: {output_dir / 'job_result.json'}")


if __name__ == "__main__":
    main()
