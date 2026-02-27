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
NVFlare job recipe for federated Qwen3-VL fine-tuning with PubMedVision.
Uses FedAvg with 3 clients; each client gets a site-specific data path.
Requires a Qwen3-VL base model when using the Qwen3-VL repo's train_qwen.py.
"""

import argparse
import os

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking


def define_parser():
    parser = argparse.ArgumentParser(description="Federated Qwen3-VL SFT (3 clients) via Qwen3-VL train_qwen.py")
    parser.add_argument("--n_clients", type=int, default=3, help="Number of clients (default 3)")
    parser.add_argument("--num_rounds", type=int, default=5, help="FL rounds (default 5)")
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help='GPU IDs per client, e.g. "[0],[1],[2]" for 3 clients on GPUs 0,1,2 (default: one GPU per client, [0],[1],...)',
    )
    parser.add_argument(
        "--nproc_per_client",
        type=int,
        default=1,
        help="Number of torchrun processes (GPUs) per client (default 1).",
    )
    parser.add_argument("--data_dir", type=str, default="./data", help="Root dir for site-1, site-2, site-3 data")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="HuggingFace model ID (must be Qwen3-VL when using Qwen3-VL repo train_qwen.py)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Max steps per round (omit to train one epoch per round)",
    )
    parser.add_argument(
        "--learning_rate",
        type=str,
        default="5e-7",
        help="Peak learning rate for training (default 5e-7; try 1e-6 for faster convergence)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases experiment tracking (optional).",
    )
    return parser.parse_args()


def main():
    args = define_parser()
    n_clients = args.n_clients
    data_dir = os.path.abspath(args.data_dir)
    qwen_root = os.environ.get("QWEN3VL_ROOT", "")

    client_names = [f"site-{i}" for i in range(1, n_clients + 1)]
    per_site_config = {}
    for idx, site_name in enumerate(client_names):
        site_data_path = os.path.join(data_dir, site_name)
        step_or_epoch = f"--max_steps {args.max_steps} " if args.max_steps is not None else "--num_train_epochs 1 "
        train_args = (
            f"--data_path {site_data_path} "
            f"--dataset_use fl_site "
            f"--model_name_or_path {args.model_name_or_path} "
            f"{step_or_epoch}"
            f"--learning_rate {args.learning_rate}"
        )
        if qwen_root:
            train_args = f"--qwen_root {qwen_root} " + train_args
        # Per-site torchrun so Qwen train_qwen gets a proper distributed env (unique master_port per client)
        master_port = 29500 + (idx + 1)
        command = f"torchrun --nproc_per_node={args.nproc_per_client} --nnodes=1 --master_port {master_port}"
        per_site_config[site_name] = {"train_args": train_args, "command": command}

    # Initial model: dict-based config (same pattern as llm_hf); model loaded from path/args.
    model = {
        "class_path": "model.Qwen3VLModel",
        "args": {"model_name_or_path": args.model_name_or_path},
    }

    # Use native PyTorch tensor exchange (like llm_hf message_mode=tensor) so BFloat16
    # weights are not converted to numpy (numpy does not support BFloat16).
    recipe = FedAvgRecipe(
        name="qwen3-vl",
        min_clients=n_clients,
        num_rounds=args.num_rounds,
        model=model,
        train_script="client.py",
        train_args="",  # overridden by per_site_config
        per_site_config=per_site_config,
        launch_external_process=True,
        server_expected_format="pytorch",
        key_metric="loss",
    )

    # Client script is the only custom file; it runs Qwen train in-process

    # Client timeouts: get_task_timeout for receiving the next task; submit_task_result_timeout for
    # sending results (when unset, framework uses communication_timeout default 300s).
    recipe.add_client_config(
        {"get_task_timeout": 1200, "submit_task_result_timeout": 1200},
        clients=client_names,
    )

    # Optional: add experiment tracking with Weights & Biases
    if args.wandb:
        add_experiment_tracking(
            recipe,
            tracking_type="wandb",
            tracking_config={
                "wandb_args": {
                    "name": "qwen3-vl-fedavg",
                    "project": "nvflare",
                    "group": "nvidia",
                    "job_type": "training",
                },
                "mode": "online",  # optional, default "offline"
            },
        )

    if args.gpu is not None:
        gpu_config = args.gpu
    else:
        site_gpu_groups = []
        for i in range(n_clients):
            start = i * args.nproc_per_client
            gpus = ",".join(str(start + j) for j in range(args.nproc_per_client))
            site_gpu_groups.append(f"[{gpus}]")
        gpu_config = ",".join(site_gpu_groups)
    env = SimEnv(clients=client_names, num_threads=n_clients, gpu_config=gpu_config)
    run = recipe.execute(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in:", run.get_result())
    print()


if __name__ == "__main__":
    main()
