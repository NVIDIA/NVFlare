# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score


def random_forest_args_parser():
    parser = argparse.ArgumentParser(description="Centralized random forest training")
    parser.add_argument(
        "--data_path", type=str, default="/media/ziyuexu/Data/HIGGS/HIGGS_UCI.csv", help="path to dataset file"
    )
    parser.add_argument("--num_parallel_tree", type=int, default=100, help="number of parallel trees")
    parser.add_argument("--num_sites", type=int, default=20, help="number of sites")
    parser.add_argument("--subsample", type=float, default=0.8, help="data subsample rate")
    parser.add_argument("--split_option", type=str, default="uniform", help="data split method for clients")
    parser.add_argument("--bagging_option", type=str, default="uniform", help="bagging method among clients")
    parser.add_argument("--models_root", type=str, default="./models/", help="model ouput folder root")
    return parser


def split_num_proportion(n, site_num, option: str):
    split = []
    if option == "uniform":
        ratio_vec = np.ones(site_num)
    elif option == "linear":
        ratio_vec = np.linspace(1, site_num, num=site_num)
    elif option == "square":
        ratio_vec = np.square(np.linspace(1, site_num, num=site_num))
    elif option == "exponential":
        ratio_vec = np.exp(np.linspace(1, site_num, num=site_num))
    else:
        raise ValueError("Split method not implemented!")

    total = sum(ratio_vec)
    left = n
    for site in range(site_num - 1):
        x = int(n * ratio_vec[site] / total)
        left = left - x
        split.append(x)
    split.append(left)
    return split


def prepare_higgs(data_path: str, valid_num: int):
    higgs = pd.read_csv(data_path, header=None)
    # print(higgs.info())
    # print(higgs.head())
    total_data_num = higgs.shape[0]
    # print(f"Total data count: {total_data_num}")
    # split to feature and label
    X_higgs_valid = higgs.iloc[0:valid_num, 1:]
    y_higgs_valid = higgs.iloc[0:valid_num, 0]
    X_higgs_train = higgs.iloc[valid_num:, 1:]
    y_higgs_train = higgs.iloc[valid_num:, 0]
    # print(y_higgs_valid.value_counts())
    # print(y_higgs_train.value_counts())
    return X_higgs_valid, y_higgs_valid, X_higgs_train, y_higgs_train, total_data_num


def main():
    parser = random_forest_args_parser()
    args = parser.parse_args()

    # Prepare data
    data_path = args.data_path
    valid_num = 1000000
    start = time.time()
    X_higgs_valid, y_higgs_valid, X_higgs_train, y_higgs_train, total_data_num = prepare_higgs(data_path, valid_num)
    end = time.time()
    lapse_time = end - start
    print(f"Data loading time: {lapse_time}")

    # Set record paths
    mode = (
        str(args.num_parallel_tree)
        + "_"
        + str(args.subsample)
        + "_"
        + str(args.num_sites)
        + "_"
        + args.split_option
        + "_"
        + args.bagging_option
    )
    model_path = args.models_root + "xgb_bagging_" + mode

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Set mode file paths
    model_path_full = model_path + "/bagging.json"
    model_path_sub_pre = model_path + "/bagging_site_temp_"

    # construct xgboost DMatrix
    # split to validation and multi-site training
    dmat_valid = xgb.DMatrix(X_higgs_valid, label=y_higgs_valid)
    dmat_train = []
    model_path_sub = []

    # split to multi_site data with random sizes
    site_size = split_num_proportion(total_data_num - valid_num, args.num_sites, args.split_option)
    eta_weight = [item / sum(site_size) for item in site_size]

    for site in range(args.num_sites):
        idx_start = sum(site_size[:site])
        idx_end = sum(site_size[: site + 1])
        # print(f"Assign {idx_start}:{idx_end} to Site {site}")
        X_higgs_train_site = X_higgs_train.iloc[idx_start:idx_end, :]
        y_higgs_train_site = y_higgs_train.iloc[idx_start:idx_end]
        dmat_train.append(xgb.DMatrix(X_higgs_train_site, label=y_higgs_train_site))
        model_path_sub.append(model_path_sub_pre + str(site) + ".json")

    # setup parameters for xgboost
    # use logistic regression loss for binary classification
    # learning rate 0.1 max_depth 5
    # use auc as metric
    param = {
        "objective": "binary:logistic",
        "eta": 1,
        "max_depth": 8,
        "eval_metric": "auc",
        "nthread": 16,
        "num_parallel_tree": int(args.num_parallel_tree / args.num_sites),
        "subsample": args.subsample,
        "tree_method": "hist",
    }

    param_bagging = {
        "objective": "binary:logistic",
        "eta": 1,
        "max_depth": 8,
        "eval_metric": "auc",
        "nthread": 16,
        "num_parallel_tree": args.num_parallel_tree,
        "subsample": args.subsample,
        "tree_method": "hist",
    }

    # xgboost training
    start = time.time()
    # Train individual sites' models
    # Saving to individual files
    for site in range(args.num_sites):
        print(f"Site {site} ", end="")
        # Boost a tree under tree param setting
        if args.bagging_option == "uniform":
            param["eta"] = 1 / args.num_sites
        elif args.bagging_option == "weighted":
            param["eta"] = eta_weight[site]
        else:
            raise ValueError("Bagging option not implemented!")
        bst = xgb.train(
            param, dmat_train[site], num_boost_round=1, evals=[(dmat_valid, "validate"), (dmat_train[site], "train")]
        )
        bst.save_model(model_path_sub[site])

    # Combine all sub-models to RF model
    # Initial, copy from tree 1
    shutil.copy(model_path_sub[0], model_path_full)
    # Clear the tree info
    with open(model_path_full) as f:
        json_bagging = json.load(f)
    json_bagging["learner"]["gradient_booster"]["model"]["trees"] = []
    json_bagging["learner"]["gradient_booster"]["model"]["tree_info"] = []
    with open(model_path_full, "w") as f:
        json.dump(json_bagging, f, separators=(",", ":"))
    # Add models
    with open(model_path_full) as f:
        json_bagging = json.load(f)
    # Append this round's trees to global model tree list
    tree_ct = 0
    for site in range(args.num_sites):
        with open(model_path_sub[site]) as f:
            json_single = json.load(f)
        for parallel_id in range(int(args.num_parallel_tree / args.num_sites)):
            append_info = json_single["learner"]["gradient_booster"]["model"]["trees"][parallel_id]
            append_info["id"] = tree_ct
            json_bagging["learner"]["gradient_booster"]["model"]["trees"].append(append_info)
            json_bagging["learner"]["gradient_booster"]["model"]["tree_info"].append(0)
            tree_ct = tree_ct + 1
    json_bagging["learner"]["attributes"]["best_iteration"] = str(0)
    json_bagging["learner"]["attributes"]["best_ntree_limit"] = str(args.num_parallel_tree)
    json_bagging["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"] = str(
        args.num_parallel_tree
    )
    json_bagging["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_parallel_tree"] = str(
        args.num_parallel_tree
    )
    # Save the global bagging model
    with open(model_path_full, "w") as f:
        json.dump(json_bagging, f, separators=(",", ":"))

    end = time.time()
    lapse_time = end - start

    print(mode)
    print(f"Training time: {lapse_time}")

    # test model
    bst_bagging = xgb.Booster(param_bagging, model_file=model_path_full)
    y_pred = bst_bagging.predict(dmat_valid)
    roc_bagging = roc_auc_score(y_higgs_valid, y_pred)
    print(f"Bagging model: {roc_bagging}")


if __name__ == "__main__":
    main()
