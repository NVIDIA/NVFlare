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

import argparse

from sklearn.cluster import KMeans, MiniBatchKMeans, kmeans_plusplus
from sklearn.metrics import homogeneity_score
from torch.utils.tensorboard import SummaryWriter

import nvflare.client as flare
from nvflare.app_opt.sklearn.data_loader import load_data_for_range


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/tmp/nvflare/dataset/sklearn_iris.csv",
        help="data directory, default to '/tmp/nvflare/dataset/sklearn_iris.csv'",
    )
    parser.add_argument(
        "--train_start",
        type=int,
        help="start index of training data",
    )
    parser.add_argument(
        "--train_end",
        type=int,
        help="end index of training data",
    )
    parser.add_argument(
        "--valid_start",
        type=int,
        help="start index of validation data",
    )
    parser.add_argument(
        "--valid_end",
        type=int,
        help="end index of validation data",
    )
    args = parser.parse_args()
    max_iter = 1
    n_init = 1
    reassignment_ratio = 0
    n_clusters = 0
    writer = SummaryWriter(log_dir="./logs")

    # Load data
    train_data = load_data_for_range(args.data_path, args.train_start, args.train_end)
    x_train = train_data[0]
    n_samples = train_data[2]
    valid_data = load_data_for_range(args.data_path, args.valid_start, args.valid_end)

    # initializes NVFlare client API
    flare.init()
    # Train federated rounds
    # start with global model at the beginning of each round
    while flare.is_running():
        # receives FLModel from NVFlare
        input_model = flare.receive()
        curr_round = input_model.current_round
        global_param = input_model.params
        print(f"current_round={curr_round}")

        if curr_round == 0:
            # first round, compute initial center with kmeans++ method
            # model will be None for this round
            n_clusters = global_param["n_clusters"]
            center_local, _ = kmeans_plusplus(x_train, n_clusters=n_clusters, random_state=0)
            params = {"center": center_local, "count": None}
            homo = 0
        else:
            center_global = global_param["center"]

            # local validation with global center
            # fit a standalone KMeans with just the given center
            kmeans_global = KMeans(n_clusters=n_clusters, init=center_global, n_init=1)
            kmeans_global.fit(center_global)
            # get validation data, both x and y will be used
            (x_valid, y_valid, valid_size) = valid_data
            y_pred = kmeans_global.predict(x_valid)
            homo = homogeneity_score(y_valid, y_pred)
            print(f"Homogeneity {homo:.4f}")

            # local training starting from global center
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=n_samples,
                max_iter=max_iter,
                init=center_global,
                n_init=n_init,
                reassignment_ratio=reassignment_ratio,
                random_state=0,
            )
            kmeans.fit(x_train)
            center_local = kmeans.cluster_centers_
            count_local = kmeans._counts
            params = {"center": center_local, "count": count_local}

        # log metric
        writer.add_scalar("Homogeneity", homo, curr_round)

        # construct trained FL model
        output_model = flare.FLModel(
            params=params,
            metrics={"metrics": homo},
            meta={"NUM_STEPS_CURRENT_ROUND": n_samples},
        )
        # send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    main()
