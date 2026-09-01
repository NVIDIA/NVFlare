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
import shlex
import shutil
import sys
import tempfile
from pathlib import Path

from src.features import load_cache_split

PROJECT_DIR = Path(__file__).resolve().parent
SITES = ["site-1", "site-2", "site-3"]


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FedCoRe completion operator with NVFlare FedAvg.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workspace", default="/tmp/nvflare/fedcore")
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--task-weight", type=float, default=4.0)
    parser.add_argument("--effect-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _build_train_args(args, cache_dir: Path, output_dir: Path, site: str, input_dim: int) -> str:
    return shlex.join(
        [
            "--cache-dir",
            str(cache_dir),
            "--output-dir",
            str(output_dir),
            "--site",
            site,
            "--input-dim",
            str(input_dim),
            "--hidden-dim",
            str(args.hidden_dim),
            "--dropout",
            str(args.dropout),
            "--local-epochs",
            str(args.local_epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--task-weight",
            str(args.task_weight),
            "--effect-weight",
            str(args.effect_weight),
            "--seed",
            str(args.seed),
        ]
    )


def _stage_client_runtime(destination: Path) -> Path:
    """Build the minimal directory tree copied into each client's custom directory."""

    runtime_dir = destination / "runtime"
    shutil.copytree(
        PROJECT_DIR / "src",
        runtime_dir / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    return runtime_dir


def build_recipe(args, cache_dir: Path, output_dir: Path, client_runtime_dir: Path):
    from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
    from nvflare.recipe import set_per_site_config

    input_dim = int(load_cache_split(cache_dir, "site-1", "train")["missing_features"].shape[1])
    per_site_config = {}
    for site in SITES:
        train_args = _build_train_args(args, cache_dir, output_dir, site, input_dim)
        per_site_config[site] = {"train_args": train_args, "command": shlex.join([sys.executable, "-u"])}

    model = {
        "class_path": "model.LogitCompletionModel",
        "args": {
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "seed": args.seed,
        },
    }
    recipe = FedAvgRecipe(
        name="fedcore-image-completion",
        model=model,
        min_clients=len(SITES),
        num_rounds=args.num_rounds,
        train_script=str(PROJECT_DIR / "client.py"),
        launch_external_process=True,
        server_expected_format="pytorch",
        key_metric="",
    )
    set_per_site_config(recipe, per_site_config)
    recipe.add_client_file(str(client_runtime_dir), clients=SITES)
    recipe.add_server_file(str(PROJECT_DIR / "model.py"))
    return recipe, input_dim


def main() -> None:
    from nvflare.recipe import SimEnv

    args = define_parser().parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fedcore-client-runtime-") as temporary_dir:
        client_runtime_dir = _stage_client_runtime(Path(temporary_dir))
        recipe, input_dim = build_recipe(args, cache_dir, output_dir, client_runtime_dir)
        env = SimEnv(clients=SITES, num_threads=len(SITES), workspace_root=str(workspace))
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
