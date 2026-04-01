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
FedAvgRecipe-based job for federated BLIP-VQA fine-tuning.

Run with the NVFlare simulator (single machine):
    python job.py
"""

import argparse

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking


def _parse_args():
    p = argparse.ArgumentParser(description="Run a FedUMM simulation")
    # --- FL ---
    p.add_argument("--num_clients", type=int, default=2)
    p.add_argument("--num_rounds", type=int, default=3)
    p.add_argument("--local_epochs", type=int, default=1)
    # --- training ---
    p.add_argument("--model_name_or_path", type=str, default="", help="HF model id (uses backend default if empty).")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--max_train_samples", type=int, default=-1)
    p.add_argument("--max_eval_samples", type=int, default=-1)
    p.add_argument("--data_path", type=str, default="")
    p.add_argument(
        "--dirichlet_alpha",
        type=float,
        default=0.0,
        help="Dirichlet concentration for non-IID data partition. 0 = IID, 0.1/0.5/1.0 = non-IID levels from paper.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=str, default="0", help="GPU IDs for simulator clients, comma-separated.")
    p.add_argument(
        "--workspace",
        type=str,
        default="/tmp/nvflare/workspaces/fedumm",
        help="SimEnv workspace root. The recipe name (K{n}_alpha{a}) is used as the job subfolder.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    client_names = [f"site-{i + 1}" for i in range(args.num_clients)]

    # Build script args (same for all clients; site-specific sharding is handled
    # inside client.py via flare.get_site_name())
    script_args = (
        f"--num_clients {args.num_clients}"
        f" --local_epochs {args.local_epochs}"
        f" --batch_size {args.batch_size}"
        f" --grad_accum {args.grad_accum}"
        f" --lr {args.lr}"
        f" --lora_r {args.lora_r}"
        f" --lora_alpha {args.lora_alpha}"
        f" --max_train_samples {args.max_train_samples}"
        f" --max_eval_samples {args.max_eval_samples}"
        f" --dirichlet_alpha {args.dirichlet_alpha}"
        f" --seed {args.seed}"
    )
    if args.model_name_or_path:
        script_args += f" --model_name_or_path {args.model_name_or_path}"
    if args.data_path:
        script_args += f" --data_path {args.data_path}"

    per_site_config = {site: {"train_args": script_args} for site in client_names}

    model = {
        "class_path": "src.blip_backend.BLIPLoRAModel",
        "args": {
            "model_name_or_path": args.model_name_or_path,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
        },
    }

    recipe = FedAvgRecipe(
        name=f"K{args.num_clients}_alpha{args.dirichlet_alpha}",
        model=model,
        min_clients=args.num_clients,
        num_rounds=args.num_rounds,
        train_script="client.py",
        per_site_config=per_site_config,
        launch_external_process=False,
        key_metric="val_accuracy",
    )

    add_experiment_tracking(recipe, tracking_type="tensorboard")

    env = SimEnv(
        clients=client_names,
        num_threads=args.num_clients,
        gpu_config=args.gpu,
        workspace_root=args.workspace,
    )
    run = recipe.execute(env)
    print()
    print("Job status:", run.get_status())
    print("Results in:", run.get_result())


if __name__ == "__main__":
    main()
