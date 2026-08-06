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

"""Run two-party CIFAR-10 SplitNN training with a CollabRecipe."""

import argparse
import logging
from pathlib import Path

from data import validate_prepared_data
from prepare_data import DEFAULT_DATA_ROOT
from trainer import SplitNNTrainer
from workflow import SplitNNWorkflow

from nvflare.collab import CollabRecipe, simple_logging
from nvflare.recipe import SimEnv

JOB_NAME = "collab_pt_splitnn"
EXAMPLE_DIR = Path(__file__).resolve().parent


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CIFAR-10 SplitNN training with the Collab API")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Prepared data root from prepare_data.py")
    parser.add_argument("--num-steps", type=int, default=15_625)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--validation-frequency", type=int, default=1000)
    parser.add_argument("--log-frequency", type=int, default=100)
    parser.add_argument("--call-timeout", type=float, default=600.0)
    parser.add_argument("--run-timeout", type=float, default=7200.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--fp32", action="store_true", help="Exchange float32 instead of float16 tensors")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workspace-root", default="/tmp/nvflare/collab")
    return parser


def make_recipe(args, manifest: dict) -> CollabRecipe:
    if args.num_steps < 1 or args.batch_size < 1:
        raise ValueError("--num-steps and --batch-size must be >= 1")
    if args.learning_rate <= 0 or args.call_timeout <= 0 or args.run_timeout <= 0:
        raise ValueError("--learning-rate, --call-timeout, and --run-timeout must be > 0")
    if args.validation_frequency < 0 or args.log_frequency < 1:
        raise ValueError("--validation-frequency must be >= 0 and --log-frequency must be >= 1")

    device = None if args.device == "auto" else args.device
    recipe = CollabRecipe(
        job_name=JOB_NAME,
        server=SplitNNWorkflow(
            train_size=manifest["overlap"],
            test_size=manifest["test_size"],
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            validation_frequency=args.validation_frequency,
            run_timeout=args.run_timeout,
            call_timeout=args.call_timeout,
            seed=args.seed,
            log_frequency=args.log_frequency,
        ),
        client=SplitNNTrainer(
            data_root=args.data_root,
            learning_rate=args.learning_rate,
            fp16=not args.fp32,
            device=device,
            seed=args.seed,
        ),
        min_clients=2,
        sync_task_timeout=args.call_timeout,
    )
    recipe.set_client_prop("data_root", args.data_root)
    recipe.set_per_site_config(
        {
            "site-1": {"role": "image"},
            "site-2": {"role": "label"},
        }
    )
    recipe.add_client_file(str(EXAMPLE_DIR / "data.py"))
    recipe.add_client_file(str(EXAMPLE_DIR / "model.py"))
    return recipe


def main():
    args = define_parser().parse_args()
    args.data_root = str(Path(args.data_root).expanduser().resolve())
    manifest = validate_prepared_data(args.data_root)
    simple_logging(logging.INFO)
    recipe = make_recipe(args, manifest)

    print("=" * 80)
    print("CIFAR-10 COLLAB SPLITNN")
    print("  Image/model-bottom site: site-1")
    print("  Label/model-top site: site-2")
    print(f"  Aligned training samples: {manifest['overlap']}")
    print(f"  Steps: {args.num_steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Data root: {args.data_root}")
    print("=" * 80)

    env = SimEnv(clients=recipe.configured_sites(), workspace_root=args.workspace_root)
    run = recipe.execute(env)
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()
