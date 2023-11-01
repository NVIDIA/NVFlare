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

import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score


def xgboost_args_parser():
    parser = argparse.ArgumentParser(description="Test XGBoost models")
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="./dataset/test.csv",
        help="testing dataset file path",
    )
    parser.add_argument("--model_path", type=str, help="model path")
    return parser


def prepare_data(data_path: str):
    df = pd.read_csv(data_path)
    # Split to feature and label
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
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
    }
    return param


def main():
    parser = xgboost_args_parser()
    args = parser.parse_args()

    test_data_path = args.test_data_path
    model_path = args.model_path

    # Load data
    X_test, y_test = prepare_data(test_data_path)

    # construct xgboost DMatrix
    dmat_test = xgb.DMatrix(X_test, label=y_test)

    # test model
    xgb_params = get_training_parameters(args)
    bst = xgb.Booster(xgb_params, model_file=model_path)
    y_pred = bst.predict(dmat_test)
    print("AUC score: ", roc_auc_score(y_test, y_pred))


if __name__ == "__main__":
    main()
