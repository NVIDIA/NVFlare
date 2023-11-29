# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import csv
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans, kmeans_plusplus
from sklearn.metrics import homogeneity_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# (1) import nvflare client API
from nvflare import client as flare


def to_dataset_tuple(data: dict):
    dataset_tuples = {}
    for dataset_name, dataset in data.items():
        dataset_tuples[dataset_name] = _to_data_tuple(dataset)
    return dataset_tuples


def _to_data_tuple(data):
    data_num = data.shape[0]
    # split to feature and label
    x = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    return x.to_numpy(), y.to_numpy(), data_num


def load_features(feature_data_path: str) -> List:
    try:
        features = []
        with open(feature_data_path, "r") as file:
            # Create a CSV reader object
            csv_reader = csv.reader(file)
            line_list = next(csv_reader)
            features = line_list
        return features
    except Exception as e:
        raise Exception(f"Load header for path'{feature_data_path} failed! {e}")


def load_data(
    data_path: str, data_features: List, random_state: int, test_size: float, skip_rows=None
) -> Dict[str, pd.DataFrame]:
    try:
        df: pd.DataFrame = pd.read_csv(
            data_path, names=data_features, sep=r"\s*,\s*", engine="python", na_values="?", skiprows=skip_rows
        )

        train, test = train_test_split(df, test_size=test_size, random_state=random_state)

        return {"train": train, "test": test}

    except Exception as e:
        raise Exception(f"Load data for path '{data_path}' failed! {e}")


def transform_data(data: Dict[str, Tuple]) -> Dict[str, Tuple]:
    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    scaled_datasets = {}
    for dataset_name, (x_data, y_data, data_num) in data.items():
        x_scaled = scaler.fit_transform(x_data)
        scaled_datasets[dataset_name] = (x_scaled, y_data, data_num)
    return scaled_datasets


def main():
    parser = define_args_parser()
    args = parser.parse_args()
    data_root_dir = args.data_root_dir
    random_state = args.random_state
    test_size = args.test_size
    skip_rows = args.skip_rows

    # (2) initializes NVFlare client API
    flare.init()

    site_name = flare.get_site_name()
    feature_data_path = f"{data_root_dir}/{site_name}_header.csv"
    features = load_features(feature_data_path)
    n_features = len(features) - 1  # remove label

    data_path = f"{data_root_dir}/{site_name}.csv"
    data = load_data(
        data_path=data_path, data_features=features, random_state=random_state, test_size=test_size, skip_rows=skip_rows
    )

    data = to_dataset_tuple(data)
    dataset = transform_data(data)
    x_train, y_train, train_size = dataset["train"]
    x_test, y_test, test_size = dataset["test"]

    model = None
    n_clusters = 0
    while flare.is_running():
        # (3) receives FLModel from NVFlare
        input_model = flare.receive()
        global_params = input_model.params
        curr_round = input_model.current_round

        print(f"current_round={curr_round}")
        if curr_round == 0:
            # (4) first round, initialize centers with kmeans++
            n_clusters = global_params["n_clusters"]
            center_local, _ = kmeans_plusplus(x_train, n_clusters=n_clusters, random_state=random_state)
            params = {"center": center_local, "count": None}
            homo = 0.0
        else:
            # (5) following rounds, starting from global centers
            center_global = global_params["center"]
            model = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=train_size,
                max_iter=1,
                init=center_global,
                n_init=1,
                reassignment_ratio=0,
                random_state=random_state,
            )
            # train model
            model.fit(x_train)
            center_local = model.cluster_centers_
            count_local = model._counts
            params = {"center": center_local, "count": count_local}

            # (6) evaluate global center
            model_eval = KMeans(n_clusters=n_clusters, init=center_global, n_init=1)
            model_eval.fit(center_global)
            homo = evaluate_model(x_test, model_eval, y_test)
            # Print the results
            print(f"{site_name}: global model homogeneity_score: {homo:.4f}")

        # (7) construct trained FL model
        metrics = {"accuracy": homo}
        output_model = flare.FLModel(params=params, metrics=metrics)

        # (8) send model back to NVFlare
        flare.send(output_model)


def evaluate_model(x_test, model, y_test):
    # Make predictions on the testing set
    y_pred = model.predict(x_test)

    # Evaluate the model
    homo = homogeneity_score(y_test, y_pred)
    return homo


def define_args_parser():
    parser = argparse.ArgumentParser(description="scikit learn linear model with SGD")
    parser.add_argument("--data_root_dir", type=str, help="root directory path to csv data file")
    parser.add_argument("--random_state", type=int, default=0, help="random state")
    parser.add_argument("--test_size", type=float, default=0.2, help="test ratio, default to 20%")
    parser.add_argument(
        "--skip_rows",
        type=str,
        default=None,
        help="""If skip_rows = N, the first N rows will be skipped, 
       if skiprows=[0, 1, 4], the rows will be skip by row indices such as row 0,1,4 will be skipped. """,
    )
    return parser


if __name__ == "__main__":
    main()
