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

import argparse

from split_data import SplitMethod, split_data

from nvflare.app_opt.sklearn.recipes import KMeansFedAvgRecipe
from nvflare.recipe import SimEnv, set_per_site_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/tmp/nvflare/dataset/sklearn_iris.csv")
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--split_mode", choices=[method.value for method in SplitMethod], default="uniform")
    parser.add_argument("--workspace_dir", default="/tmp/nvflare/chapter2/kmeans")
    args = parser.parse_args()
    if args.num_clients < 1:
        parser.error("--num_clients must be positive")
    n_clusters = 3
    indices = split_data(args.data_path, args.num_clients, valid_frac=1, split_method=SplitMethod(args.split_mode))
    undersized = [
        f"site-{i} ({indices[i]['end'] - indices[i]['start']} rows)"
        for i in range(1, args.num_clients + 1)
        if indices[i]["end"] - indices[i]["start"] < n_clusters
    ]
    if undersized:
        parser.error(
            f"Each training split needs at least {n_clusters} rows for K-means initialization; "
            f"undersized: {', '.join(undersized)}. Use fewer clients or a less imbalanced --split_mode."
        )
    recipe = KMeansFedAvgRecipe(
        name="kmeans_iris",
        min_clients=args.num_clients,
        num_rounds=args.num_rounds,
        n_clusters=n_clusters,
        train_script="client.py",
    )
    set_per_site_config(
        recipe,
        {
            f"site-{i}": {
                "train_args": [
                    "--data_path",
                    args.data_path,
                    "--train_start",
                    str(indices[i]["start"]),
                    "--train_end",
                    str(indices[i]["end"]),
                    "--valid_start",
                    str(indices["valid"]["start"]),
                    "--valid_end",
                    str(indices["valid"]["end"]),
                ]
            }
            for i in range(1, args.num_clients + 1)
        },
    )
    run = recipe.execute(
        SimEnv(clients=recipe.configured_sites(), workspace_root=args.workspace_dir, log_config="progress")
    )
    print("Results at:", run.get_result())


if __name__ == "__main__":
    main()
