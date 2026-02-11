# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
This code shows how to use NVIDIA FLARE Job Recipe to train a model with Differential Privacy
"""
import argparse

from model import TabularMLP

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_script", type=str, default="client.py")
    parser.add_argument(
        "--target_epsilon",
        type=float,
        default=1.0,
        help="Target epsilon for differential privacy (lower = more private)",
    )

    return parser.parse_args()


def main():
    args = define_parser()

    n_clients = args.n_clients
    num_rounds = args.num_rounds
    batch_size = args.batch_size
    epochs = args.epochs
    # Create FedAvg recipe with tabular MLP model for fraud detection
    recipe = FedAvgRecipe(
        name="hello-dp",
        min_clients=n_clients,
        num_rounds=num_rounds,
        # Model can be specified as class instance or dict config:
        model=TabularMLP(input_dim=29, hidden_dims=[64, 32], output_dim=2),  # Credit card fraud: 29 features, 2 classes
        # Alternative: model={"path": "model.TabularMLP", "args": {"input_dim": 29, "hidden_dims": [64, 32], "output_dim": 2}},
        # For pre-trained weights: initial_ckpt="/server/path/to/pretrained.pt",
        train_script=args.train_script,
        train_args=f"--batch_size {batch_size} --epochs {epochs} --target_epsilon {args.target_epsilon} --n_clients {n_clients}",
    )
    add_experiment_tracking(recipe, tracking_type="tensorboard")

    # Run FL simulation
    env = SimEnv(num_clients=n_clients)
    run = recipe.execute(env)
    print()
    print("Job Status is:", run.get_status())
    print("Result can be found in:", run.get_result())
    print()


if __name__ == "__main__":
    main()
