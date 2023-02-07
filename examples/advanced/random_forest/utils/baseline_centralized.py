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
import os
import pickle
import time

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def random_forest_args_parser():
    parser = argparse.ArgumentParser(description="Centralized random forest training")
    parser.add_argument(
        "--data_path", type=str, default="/media/ziyuexu/Data/HIGGS/HIGGS_UCI.csv", help="path to dataset file"
    )
    parser.add_argument("--num_parallel_tree", type=int, default=100, help="number of parallel trees")
    parser.add_argument("--subsample", type=float, default=0.8, help="data subsample rate")
    parser.add_argument("--models_root", type=str, default="./models", help="model ouput folder root")
    parser.add_argument("--backend_method", type=str, default="sklearn", help="use xgboost or sklearn as backend")
    return parser


def prepare_higgs(data_path: str, valid_num: int):
    higgs = pd.read_csv(data_path, header=None)
    print(higgs.info())
    print(higgs.head())
    total_data_num = higgs.shape[0]
    print(f"Total data count: {total_data_num}")
    # split to feature and label
    X_higgs_valid = higgs.iloc[0:valid_num, 1:]
    y_higgs_valid = higgs.iloc[0:valid_num, 0]
    X_higgs_train = higgs.iloc[valid_num:, 1:]
    y_higgs_train = higgs.iloc[valid_num:, 0]
    print(y_higgs_valid.value_counts())
    print(y_higgs_train.value_counts())
    return X_higgs_valid, y_higgs_valid, X_higgs_train, y_higgs_train


def get_training_parameters(args):
    # use logistic regression loss for binary classification
    # use auc as metric
    param_xgboost = {
        "objective": "binary:logistic",
        "eta": 0.1,
        "max_depth": 8,
        "eval_metric": "auc",
        "nthread": 16,
        "num_parallel_tree": args.num_parallel_tree,
        "subsample": args.subsample,
        "tree_method": "hist",
    }
    # use same parameter for sklearn random forest
    param_sklearn = {"n_estimators": args.num_parallel_tree, "max_depth": 8, "max_samples": args.subsample}
    return param_xgboost, param_sklearn


def main():
    parser = random_forest_args_parser()
    args = parser.parse_args()

    # Prepare data
    data_path = args.data_path
    valid_num = 1000000
    start = time.time()
    X_higgs_valid, y_higgs_valid, X_higgs_train, y_higgs_train = prepare_higgs(data_path, valid_num)
    end = time.time()
    lapse_time = end - start
    print(f"Data loading time: {lapse_time}")

    # Specify training params
    param_xgboost, param_sklearn = get_training_parameters(args)

    # training
    start = time.time()
    if args.backend_method == "xgboost":
        # construct training and validation xgboost DMatrix
        dmat_valid = xgb.DMatrix(X_higgs_valid, label=y_higgs_valid)
        dmat_train = xgb.DMatrix(X_higgs_train, label=y_higgs_train)
        # set mode file paths
        model_name = "xgboost_" + str(args.num_parallel_tree) + "_" + str(args.subsample)
        exp_root = os.path.join(args.workspace_root, model_name)
        os.makedirs(exp_root)
        model_path = os.path.join(exp_root, "model.json")
        # train model with xgboost backend
        bst = xgb.train(
            param_xgboost, dmat_train, num_boost_round=1, evals=[(dmat_valid, "validate"), (dmat_train, "train")]
        )
        bst.save_model(model_path)
    elif args.backend_method == "sklearn":
        # set mode file paths
        model_name = "sklearn_" + str(args.num_parallel_tree) + "_" + str(args.subsample)
        exp_root = os.path.join(args.workspace_root, model_name)
        if not os.path.exists(exp_root):
            os.makedirs(exp_root)
        model_path = os.path.join(exp_root, "model.pkl")
        # train model with sklearn backend
        clf = RandomForestClassifier(
            n_estimators=param_sklearn["n_estimators"],
            max_depth=param_sklearn["max_depth"],
            bootstrap=True,
            max_samples=param_sklearn["max_samples"],
        )
        clf = clf.fit(X_higgs_train, y_higgs_train)
        with open(model_path, "wb") as file:
            pickle.dump(clf, file)

    end = time.time()
    lapse_time = end - start
    print(f"Training time: {lapse_time}")

    # test model
    if args.backend_method == "xgboost":
        bst = xgb.Booster(param_xgboost, model_file=model_path)
        y_pred = bst.predict(dmat_valid)
        roc = roc_auc_score(y_higgs_valid, y_pred)
        print(f"Model AUC: {roc}")
    elif args.backend_method == "sklearn":
        with open(model_path, "rb") as file:
            clf = pickle.load(file)
        y_pred = clf.predict(X_higgs_valid)
        roc = roc_auc_score(y_higgs_valid, y_pred)
        print(f"Model AUC: {roc}")


if __name__ == "__main__":
    main()
