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
MONAI Spleen CT Statistics Collection

This example shows how to collect federated statistics from MONAI bundle datasets
using FedStatsRecipe.
"""

import argparse
import os

from client import MonaiBundleStatistics

from nvflare.recipe import SimEnv
from nvflare.recipe.fedstats import FedStatsRecipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle_root", type=str, default="bundles/spleen_ct_segmentation", help="Path to MONAI bundle"
    )
    parser.add_argument("--n_clients", type=int, default=2, help="Number of simulated clients")
    parser.add_argument(
        "--workspace", type=str, default="/tmp/nvflare/simulation", help="Workspace directory for simulation"
    )
    args = parser.parse_args()

    # Statistics configuration
    statistic_configs = {"count": {}, "histogram": {"*": {"bins": 8, "range": [-200, 300]}}}

    # Create statistics generator
    stats_generator = MonaiBundleStatistics(bundle_root=os.path.join(os.getcwd(), args.bundle_root))

    sites = [f"site-{i + 1}" for i in range(args.n_clients)]

    # Create FedStatsRecipe
    recipe = FedStatsRecipe(
        name="spleen_bundle_stats",
        stats_output_path="statistics/image_statistics.json",
        sites=sites,
        statistic_configs=statistic_configs,
        stats_generator=stats_generator,
        min_count=5,
        max_bins_percent=30,
    )

    # Setup simulation environment
    env = SimEnv(clients=sites, workspace_root=args.workspace)

    # Execute the recipe
    run = recipe.execute(env)

    print()
    print("Job Status:", run.get_status())
    print("Results at:", run.get_result())
    print()


if __name__ == "__main__":
    main()
