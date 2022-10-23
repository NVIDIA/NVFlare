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
import time
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def random_forest_args_parser():
    parser = argparse.ArgumentParser(description="Centralized random forest training")
    parser.add_argument("--data_path", type=str, default="/media/ziyuexu/Data/HIGGS/HIGGS_UCI.csv",
                        help="path to dataset file")
    parser.add_argument("--model_path", type=str, default="/home/ziyuexu/Desktop/Experiment/NVFlare/Exp_RandomForest/workspaces/xgboost_workspace_5_uniform_split_uniform_lr/simulate_job/app_server/xgboost_model.json", help="path to json model")
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
        "num_parallel_tree": 100,
        "tree_method": "hist",
    }
    return param_xgboost


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
    param_xgboost = get_training_parameters(args)

    # test model
    dmat_valid = xgb.DMatrix(X_higgs_valid, label=y_higgs_valid)
    bst = xgb.Booster(param_xgboost, model_file=args.model_path)
    y_pred = bst.predict(dmat_valid)
    roc = roc_auc_score(y_higgs_valid, y_pred)
    print(f"Model AUC: {roc}")

if __name__ == "__main__":
    main()
