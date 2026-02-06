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
NVFlare job recipe for federated Qwen2.5-VL fine-tuning with PubMedVision.
Uses FedAvg with 3 clients; each client gets a site-specific data path.
"""
import argparse
import os

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking


def define_parser():
    parser = argparse.ArgumentParser(description="Federated Qwen2.5-VL SFT (3 clients) via Qwen3-VL train_qwen.py")
    parser.add_argument("--n_clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--num_rounds", type=int, default=2, help="FL rounds")
    parser.add_argument("--data_dir", type=str, default="./data", help="Root dir for site-1, site-2, site-3 data")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="HuggingFace model ID (same as llm in Qwen3-VL sft config)",
    )
    parser.add_argument("--max_steps", type=int, default=50, help="Max steps per round (passed to train_qwen.py)")
    return parser.parse_args()


def main():
    args = define_parser()
    n_clients = args.n_clients
    data_dir = os.path.abspath(args.data_dir)
    qwen_root = os.environ.get("QWEN3VL_ROOT", "")

    client_names = [f"site-{i}" for i in range(1, n_clients + 1)]
    per_site_config = {}
    for site_name in client_names:
        site_data_path = os.path.join(data_dir, site_name)
        train_args = (
            f"--data_path {site_data_path} "
            f"--dataset_use fl_site "
            f"--model_name_or_path {args.model_name_or_path} "
            f"--max_steps {args.max_steps}"
        )
        if qwen_root:
            train_args = f"--qwen_root {qwen_root} " + train_args
        per_site_config[site_name] = {
            "train_args": train_args,
            "command": "bash custom/client_wrapper.sh",
        }

    # Initial model: dict-based config (same pattern as llm_hf); model loaded from path/args.
    initial_model = {
        "path": "model.Qwen3VLModel",
        "args": {"model_name_or_path": args.model_name_or_path},
    }

    recipe = FedAvgRecipe(
        name="qwen3-vl",
        min_clients=n_clients,
        num_rounds=args.num_rounds,
        initial_model=initial_model,
        train_script="client_sft_runner.py",
        train_args="",  # overridden by per_site_config
        per_site_config=per_site_config,
        launch_external_process=True,
        key_metric="loss",
    )

    for site_name in client_names:
        recipe.job.to("client_wrapper.sh", site_name)

    # Increase timeouts so clients can finish train_qwen.py (50 steps of VL can take 10+ min per client)
    for site_name in client_names:
        client_params = {"get_task_timeout": 600, "submit_task_result_timeout": 900}
        recipe.job.to(client_params, site_name)

    # Add experiment tracking with Weights & Biases
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

    env = SimEnv(clients=client_names, num_threads=n_clients)
    run = recipe.execute(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in:", run.get_result())
    print()


if __name__ == "__main__":
    main()
