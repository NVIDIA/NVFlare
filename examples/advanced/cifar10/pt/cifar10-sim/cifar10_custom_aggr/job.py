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
This code demonstrates how to use custom aggregators with NVIDIA FLARE's FedAvgRecipe.

Available aggregators:
1. 'weighted' - WeightedAggregator: Aggregates based on client data size
2. 'median' - MedianAggregator: Uses median aggregation for Byzantine robustness
3. 'default' - Uses the default FedAvg aggregator (InTimeAccumulateWeightedAggregator)

Example usage:
    python job.py --aggregator weighted --n_clients 8 --num_rounds 50 --alpha 0.1 --seed 0
    python job.py --aggregator median --n_clients 8 --num_rounds 50 --alpha 0.1 --seed 0
    python job.py --aggregator default --n_clients 8 --num_rounds 50 --alpha 0.1 --seed 0

Note: Use the same --seed value across experiments for reproducible model initialization!
"""
import argparse
import os

from custom_aggregators import MedianAggregator, WeightedAggregator
from data.cifar10_data_split import split_and_save
from model import ModerateCNN

from nvflare.apis.dxo import DataKind
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking


def define_parser():
    parser = argparse.ArgumentParser(
        description="Run FedAvg with custom aggregators on CIFAR-10",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with weighted aggregator
  python job.py --aggregator weighted --n_clients 8 --num_rounds 50 --alpha 0.1 --seed 0

  # Run with median aggregator  
  python job.py --aggregator median --n_clients 8 --num_rounds 50 --alpha 0.1 --seed 0

  # Run with default aggregator
  python job.py --aggregator default --n_clients 8 --num_rounds 50 --alpha 0.1 --seed 0

Note: Use the same --seed for reproducible model initialization!
        """,
    )

    # Custom aggregator selection
    parser.add_argument(
        "--aggregator",
        type=str,
        default="weighted",
        choices=["weighted", "median", "default"],
        help="Type of aggregator to use: 'weighted' (weight by data size), "
        "'median' (robust to Byzantine clients), or 'default' (standard FedAvg)",
    )

    # Standard federated learning parameters
    parser.add_argument("--n_clients", type=int, default=8, help="Number of federated learning clients to simulate")
    parser.add_argument(
        "--num_rounds", type=int, default=50, help="Number of federated learning rounds (global aggregation iterations)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Number of worker processes for data loading (0 = main process only)"
    )
    parser.add_argument("--lr", type=float, default=5e-2, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size for each client")
    parser.add_argument(
        "--aggregation_epochs", type=int, default=4, help="Number of local training epochs per client and FL round"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet distribution parameter (controls data heterogeneity: "
        "lower values create more heterogeneous distributions)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for model initialization and reproducibility (default: 0)"
    )
    parser.add_argument("--name", type=str, default=None, help="Custom name for the recipe (overrides default naming)")

    return parser.parse_args()


def get_aggregator(aggregator_type: str):
    """
    Factory function to create the appropriate aggregator based on user selection.

    Args:
        aggregator_type: One of 'weighted', 'median', or 'default'

    Returns:
        Aggregator instance or None (for default)
    """
    if aggregator_type == "weighted":
        print("Using WeightedAggregator (weights by client data size)")
        return WeightedAggregator()
    elif aggregator_type == "median":
        print("Using MedianAggregator (Byzantine-robust median aggregation)")
        return MedianAggregator()
    elif aggregator_type == "default":
        print("Using default FedAvg aggregator")
        return None  # FedAvgRecipe will use its default aggregator
    else:
        raise ValueError(f"Unknown aggregator type: {aggregator_type}")


def main():
    args = define_parser()

    n_clients = args.n_clients
    num_rounds = args.num_rounds
    alpha = args.alpha
    seed = args.seed
    num_workers = args.num_workers
    lr = args.lr
    batch_size = args.batch_size
    aggregation_epochs = args.aggregation_epochs
    aggregator_type = args.aggregator

    # Generate job name
    job_name = args.name if args.name else f"cifar10_custom_{aggregator_type}_alpha{alpha}"

    print("=" * 80)
    print("Running FedAvg with Custom Aggregator")
    print("=" * 80)
    print(f"Aggregator Type: {aggregator_type}")
    print(f"Number of Rounds: {num_rounds}")
    print(f"Alpha (heterogeneity): {alpha}")
    print(f"Random Seed: {seed}")
    print(f"Number of Clients: {n_clients}")
    print(f"Local Epochs per Round: {aggregation_epochs}")
    print(f"Job Name: {job_name}")
    print("=" * 80)

    # Prepare data split
    if alpha > 0.0:
        print(f"\nPreparing CIFAR10 and doing data split with alpha = {alpha}")
        train_idx_root = split_and_save(
            num_sites=n_clients, alpha=alpha, split_dir_prefix=f"/tmp/cifar10_splits/{job_name}"
        )
    else:
        raise ValueError("Alpha must be greater than 0 for federated settings")

    # Get the appropriate aggregator
    custom_aggregator = get_aggregator(aggregator_type)

    # Create recipe with or without custom aggregator
    recipe = FedAvgRecipe(
        name=job_name,
        min_clients=n_clients,
        num_rounds=num_rounds,
        model=ModerateCNN(seed=seed),  # Use seed for reproducible initialization
        train_script=os.path.join(os.path.dirname(__file__), "client.py"),
        train_args=f"--train_idx_root {train_idx_root} --num_workers {num_workers} --lr {lr} --batch_size {batch_size} --aggregation_epochs {aggregation_epochs}",
        aggregator_data_kind=DataKind.WEIGHT_DIFF,
        aggregator=custom_aggregator,
    )
    add_experiment_tracking(recipe, tracking_type="tensorboard")

    # Run simulation
    print("\nStarting simulation...")
    env = SimEnv(num_clients=n_clients)
    run = recipe.execute(env)

    # Print results
    print()
    print("=" * 80)
    print("Simulation Complete!")
    print("=" * 80)
    print("Job Status is:", run.get_status())
    print("Result can be found in:", run.get_result())
    print()
    print("View results with TensorBoard:")
    print(f"  tensorboard --logdir={run.get_result()}")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
