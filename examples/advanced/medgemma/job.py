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
NVFlare job recipe for federated MedGemma LoRA fine-tuning.
"""

from __future__ import annotations

import argparse
import os
import re

from data_utils import DEFAULT_MODEL_NAME_OR_PATH

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking


def define_parser():
    parser = argparse.ArgumentParser(description="Federated MedGemma QLoRA fine-tuning with FedAvg.")
    parser.add_argument("--n_clients", type=int, default=3, help="Number of federated clients (default: 3).")
    parser.add_argument("--num_rounds", type=int, default=3, help="Federated rounds (default: 3).")
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help='GPU IDs per client, e.g. "[0],[1],[2]". When omitted, one GPU is assigned per client automatically.',
    )
    parser.add_argument("--data_dir", type=str, default="./data", help="Root directory containing site-1, site-2, ...")
    parser.add_argument(
        "--image_root",
        type=str,
        default="./NCT-CRC-HE-100K",
        help="Dataset root used to resolve image paths (default: ./NCT-CRC-HE-100K).",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=DEFAULT_MODEL_NAME_OR_PATH,
        help="MedGemma Hugging Face model ID or local path.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional max steps per client round. If omitted, each round trains one local epoch.",
    )
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Peak learning rate (default: 2e-4).")
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Training batch size per device (default: 4).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Evaluation batch size per device (default: 4).",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable optional Weights & Biases tracking.",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="/tmp/nvflare/simulation",
        help="Simulation workspace root (default: /tmp/nvflare/simulation).",
    )
    return parser.parse_args()


def _parse_gpu_string(gpu_str: str) -> list[str]:
    if not gpu_str or not gpu_str.strip():
        return []
    return re.findall(r"\[[^\]]*\]", gpu_str.strip())


def _configure_timeouts(
    recipe, client_names, task_timeout: int = 1800, tensor_timeout: int = 600, max_resends: int = 3
):
    recipe.add_client_config(
        {
            "get_task_timeout": task_timeout,
            "submit_task_result_timeout": task_timeout,
            "tensor_min_download_timeout": tensor_timeout,
            "max_resends": max_resends,
        },
        clients=client_names,
    )
    recipe.add_server_config(
        {
            "streaming_per_request_timeout": tensor_timeout,
            "tensor_min_download_timeout": tensor_timeout,
        }
    )


def main():
    args = define_parser()
    n_clients = args.n_clients
    client_names = [f"site-{idx}" for idx in range(1, n_clients + 1)]
    data_dir = os.path.abspath(args.data_dir)
    image_root = os.path.abspath(args.image_root)

    per_site_config = {}
    report_to = "wandb" if args.wandb else "none"
    for site_name in client_names:
        site_data_path = os.path.join(data_dir, site_name)
        step_or_epoch = f"--max_steps {args.max_steps} " if args.max_steps is not None else "--num_train_epochs 1 "
        train_args = (
            f"--data_path {site_data_path} "
            f"--image_root {image_root} "
            f"--model_name_or_path {args.model_name_or_path} "
            f"{step_or_epoch}"
            f"--learning_rate {args.learning_rate} "
            f"--per_device_train_batch_size {args.per_device_train_batch_size} "
            f"--per_device_eval_batch_size {args.per_device_eval_batch_size} "
            f"--gradient_accumulation_steps {args.gradient_accumulation_steps} "
            f"--report_to {report_to}"
        )
        per_site_config[site_name] = {"train_args": train_args}

    model = {
        "class_path": "model.MedGemmaLoRAModel",
        "args": {"model_name_or_path": args.model_name_or_path},
    }

    recipe = FedAvgRecipe(
        name="medgemma",
        min_clients=n_clients,
        num_rounds=args.num_rounds,
        model=model,
        train_script="client.py",
        per_site_config=per_site_config,
        launch_external_process=True,
        server_expected_format="pytorch",
        key_metric="",
    )
    _configure_timeouts(recipe, client_names)

    if args.wandb:
        add_experiment_tracking(
            recipe,
            tracking_type="wandb",
            tracking_config={
                "wandb_args": {
                    "name": "medgemma-fedavg",
                    "project": "nvflare",
                    "group": "nvidia",
                    "job_type": "training",
                },
                "mode": "online",
            },
        )

    if args.gpu is not None:
        gpu_groups = _parse_gpu_string(args.gpu)
        if len(gpu_groups) != n_clients:
            raise ValueError(
                f"--gpu has {len(gpu_groups)} group(s) but --n_clients={n_clients}. "
                f'Expected one group per client, e.g. "[0],[1],[2]".'
            )
        gpu_config = args.gpu
    else:
        gpu_config = ",".join(f"[{idx}]" for idx in range(n_clients))

    env = SimEnv(
        clients=client_names,
        num_threads=n_clients,
        gpu_config=gpu_config,
        workspace_root=os.path.abspath(args.workspace),
    )
    run = recipe.execute(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in:", run.get_result())
    print()


if __name__ == "__main__":
    main()
