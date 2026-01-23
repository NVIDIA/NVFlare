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
Federated BraTS18 segmentation using NVFlare Job Recipe API.
"""
import argparse

from model import BratsSegResNet

from nvflare.app_common.filters.svt_privacy import SVTPrivacy
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.client.config import TransferType
from nvflare.recipe import SimEnv, add_experiment_tracking

DEFAULT_NUM_ROUNDS = 600
DEFAULT_HEARTBEAT_TIMEOUT = 600


def main():
    parser = argparse.ArgumentParser(description="BraTS18 segmentation with NVFlare Recipe API.")
    parser.add_argument("--n_clients", type=int, default=4, help="Number of FL clients.")
    parser.add_argument(
        "--job_name",
        type=str,
        default=None,
        help="Optional job name to avoid reusing an existing workspace.",
    )
    parser.add_argument("--enable_dp", action="store_true", help="Enable SVT privacy filter.")
    parser.add_argument(
        "--dp_fraction", type=float, default=0.9, help="DP: fraction of weights to share (default from paper: 0.9)"
    )
    parser.add_argument(
        "--dp_epsilon", type=float, default=0.001, help="DP: privacy budget (default from paper: 0.001)"
    )
    parser.add_argument("--dp_noise_var", type=float, default=1.0, help="DP: noise variance (default from paper: 1.0)")
    parser.add_argument(
        "--dp_gamma", type=float, default=1e-4, help="DP: clipping threshold (default from paper: 1e-4)"
    )
    parser.add_argument("--num_rounds", type=int, default=DEFAULT_NUM_ROUNDS, help="Number of FL rounds.")
    parser.add_argument("--aggregation_epochs", type=int, default=1, help="Local epochs per round.")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--fedproxloss_mu", type=float, default=0.0)
    parser.add_argument("--cache_dataset", type=float, default=0.0)
    parser.add_argument("--dataset_base_dir", type=str, required=True)
    parser.add_argument("--datalist_json_path", type=str, required=True)
    parser.add_argument(
        "--roi_size",
        type=int,
        nargs=3,
        default=(224, 224, 144),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--infer_roi_size",
        type=int,
        nargs=3,
        default=(240, 240, 160),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Simulation workspace root.",
    )
    parser.add_argument("--threads", type=int, default=None, help="Number of simulator threads.")
    parser.add_argument("--gpu", type=str, default=None, help="GPU config for SimEnv (e.g., 0,1,2,3 or 0).")
    args = parser.parse_args()

    # Add --centralized flag for single client (centralized training)
    centralized_flag = "--centralized" if args.n_clients == 1 else ""
    train_args = (
        f"--aggregation_epochs {args.aggregation_epochs} "
        f"--learning_rate {args.learning_rate} "
        f"--fedproxloss_mu {args.fedproxloss_mu} "
        f"--cache_dataset {args.cache_dataset} "
        f"--dataset_base_dir {args.dataset_base_dir} "
        f"--datalist_json_path {args.datalist_json_path} "
        f"--roi_size {args.roi_size[0]} {args.roi_size[1]} {args.roi_size[2]} "
        f"--infer_roi_size {args.infer_roi_size[0]} {args.infer_roi_size[1]} {args.infer_roi_size[2]} "
        f"{centralized_flag}"
    ).strip()
    if args.job_name:
        recipe_name = args.job_name
    else:
        dp_suffix = "_dp" if args.enable_dp else ""
        recipe_name = f"brats18_{args.n_clients}{dp_suffix}"
    recipe = FedAvgRecipe(
        name=recipe_name,
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        initial_model=BratsSegResNet(),
        train_script="client.py",
        train_args=train_args,
        key_metric="val_dice",
        # Use WEIGHT_DIFF for efficiency and consistency (required for SVTPrivacy filter)
        params_transfer_type=TransferType.DIFF,
    )
    recipe.job.to_server({"server": {"heart_beat_timeout": DEFAULT_HEARTBEAT_TIMEOUT}})

    # Enable TensorBoard tracking
    add_experiment_tracking(recipe, tracking_type="tensorboard")
    if args.enable_dp:
        recipe.add_client_output_filter(
            SVTPrivacy(
                fraction=args.dp_fraction,
                epsilon=args.dp_epsilon,
                noise_var=args.dp_noise_var,
                gamma=args.dp_gamma,
            ),
            tasks=["train"],
        )

    # Create simulation environment
    sim_env_kwargs = {
        "num_clients": args.n_clients,
        "num_threads": args.threads,
        "gpu_config": args.gpu,
    }
    if args.workspace:
        sim_env_kwargs["workspace_root"] = args.workspace
    env = SimEnv(**sim_env_kwargs)

    run = recipe.execute(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in:", run.get_result())
    print()


if __name__ == "__main__":
    main()
