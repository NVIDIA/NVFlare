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
import os
import time

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


def xgboost_args_parser():
    parser = argparse.ArgumentParser(description="Centralized XGBoost training with random forest option")
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="./dataset/train.csv",
        help="folder to training dataset file",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="./dataset/test.csv",
        help="folder to testing dataset file",
    )
    parser.add_argument("--valid_ratio", type=float, default=0.1, help="ratio of validation split")
    parser.add_argument("--num_rounds", type=int, default=100, help="number of boosting rounds")
    parser.add_argument("--num_parallel_tree", type=int, default=1, help="number of parallel tree")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./workspaces/xgboost_workspace_centralized",
        help="model output folder",
    )
    return parser


def prepare_data(data_path: str):
    df = pd.read_csv(data_path)
    print(df.info())
    print(df.head())
    total_data_num = df.shape[0]
    print(f"Total data count: {total_data_num}")
    # Split to feature and label
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    print(y.value_counts())
    return X, y


def get_training_parameters(args):
    # use logistic regression loss for binary classification
    # use auc as metric
    param = {
        "objective": "binary:logistic",
        "eta": 0.1,
        "max_depth": 8,
        "eval_metric": "auc",
        "nthread": 16,
        "num_parallel_tree": args.num_parallel_tree,
    }
    return param


def main():
    parser = xgboost_args_parser()
    args = parser.parse_args()

    train_data_path = args.train_data_path
    valid_ratio = args.valid_ratio
    num_rounds = args.num_rounds
    output_folder = args.output_folder

    # Set mode file paths
    model_path = os.path.join(output_folder, "model_centralized.json")

    # Load data
    start = time.time()
    X, y = prepare_data(train_data_path)

    # Split to training and validation
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_ratio, random_state=77)
    print(
        f"TRAINING: X_train: {X_train.shape}, y_train: {y_train.shape}, Fraudulant transaction: {y_train.value_counts()[1]}"
    )
    print(
        f"VALIDATION: X_validate: {X_valid.shape}, y_validate: {y_valid.shape}, Fraudulant transaction: {y_valid.value_counts()[1]}"
    )

    # construct xgboost DMatrix
    dmat_train = xgb.DMatrix(X_train, label=y_train)
    dmat_valid = xgb.DMatrix(X_valid, label=y_valid)

    end = time.time()
    lapse_time = end - start
    print(f"Data loading time: {lapse_time}")

    # xgboost training
    start = time.time()
    xgb_params = get_training_parameters(args)
    bst = xgb.train(
        xgb_params,
        dmat_train,
        num_boost_round=num_rounds,
        evals=[(dmat_valid, "validate"), (dmat_train, "train")],
    )
    end = time.time()
    lapse_time = end - start
    print(f"Training time: {lapse_time}")

    # save model
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    bst.save_model(model_path)


if __name__ == "__main__":
    main()
