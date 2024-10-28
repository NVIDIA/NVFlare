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
import tempfile
import time

import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter


def xgboost_args_parser():
    parser = argparse.ArgumentParser(description="Centralized XGBoost training with random forest options")
    parser.add_argument("--data_path", type=str, default="./dataset/HIGGS_UCI.csv", help="path to dataset file")
    parser.add_argument("--num_parallel_tree", type=int, default=1, help="num_parallel_tree for random forest setting")
    parser.add_argument("--subsample", type=float, default=1, help="subsample for random forest setting")
    parser.add_argument("--num_rounds", type=int, default=100, help="number of boosting rounds")
    parser.add_argument("--workspace_root", type=str, default="workspaces", help="workspaces root")
    parser.add_argument("--tree_method", type=str, default="hist", help="tree_method")
    parser.add_argument("--train_in_one_session", action="store_true", help="whether to train in one session")
    return parser


def prepare_higgs(data_path: str):
    higgs = pd.read_csv(data_path, header=None)
    print(higgs.info())
    print(higgs.head())
    total_data_num = higgs.shape[0]
    print(f"Total data count: {total_data_num}")
    # split to feature and label
    X_higgs = higgs.iloc[:, 1:]
    y_higgs = higgs.iloc[:, 0]
    print(y_higgs.value_counts())
    return X_higgs, y_higgs


def train_one_by_one(train_data, val_data, xgb_params, num_rounds, val_label, writer: SummaryWriter):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_model_path = os.path.join(tmp_dir, "model.json")
        # Round 0
        print("Round: 0 Base ", end="")
        bst = xgb.train(
            xgb_params, train_data, num_boost_round=1, evals=[(val_data, "validate"), (train_data, "train")]
        )
        bst.save_model(tmp_model_path)
        for r in range(1, num_rounds):
            # Validate the last round's model
            bst_last = xgb.Booster(xgb_params, model_file=tmp_model_path)
            y_pred = bst_last.predict(val_data)
            roc = roc_auc_score(val_label, y_pred)
            print(f"Round: {bst_last.num_boosted_rounds()} model testing AUC {roc}")
            writer.add_scalar("AUC", roc, r - 1)
            # Train new model
            print(f"Round: {r} Base ", end="")
            bst = xgb.train(
                xgb_params,
                train_data,
                num_boost_round=1,
                xgb_model=tmp_model_path,
                evals=[(val_data, "validate"), (train_data, "train")],
            )
            bst.save_model(tmp_model_path)
        return bst


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
        "subsample": args.subsample,
        "tree_method": args.tree_method,
    }
    return param


def main():
    parser = xgboost_args_parser()
    args = parser.parse_args()

    # Specify training params
    if args.train_in_one_session:
        model_name = "centralized_simple_" + str(args.num_parallel_tree) + "_" + str(args.subsample)
    else:
        model_name = "centralized_" + str(args.num_parallel_tree) + "_" + str(args.subsample)
    data_path = args.data_path
    num_rounds = args.num_rounds
    valid_num = 1000000

    exp_root = os.path.join(args.workspace_root, model_name)
    # Set mode file paths
    model_path = os.path.join(exp_root, "model.json")
    # Set tensorboard output
    writer = SummaryWriter(exp_root)

    # Load data
    start = time.time()
    X_higgs, y_higgs = prepare_higgs(data_path)
    end = time.time()
    lapse_time = end - start
    print(f"Data loading time: {lapse_time}")

    # construct training and validation xgboost DMatrix
    dmat_higgs = xgb.DMatrix(X_higgs, label=y_higgs)
    dmat_valid = dmat_higgs.slice(X_higgs.index[0:valid_num])
    dmat_train = dmat_higgs.slice(X_higgs.index[valid_num:])

    # setup parameters for xgboost
    xgb_params = get_training_parameters(args)

    # xgboost training
    start = time.time()
    if args.train_in_one_session:
        bst = xgb.train(
            xgb_params, dmat_train, num_boost_round=num_rounds, evals=[(dmat_valid, "validate"), (dmat_train, "train")]
        )
    else:
        bst = train_one_by_one(
            train_data=dmat_train,
            val_data=dmat_valid,
            xgb_params=xgb_params,
            num_rounds=num_rounds,
            val_label=y_higgs[0:valid_num],
            writer=writer,
        )
    bst.save_model(model_path)
    end = time.time()
    lapse_time = end - start
    print(f"Training time: {lapse_time}")

    # test model
    bst = xgb.Booster(xgb_params, model_file=model_path)
    y_pred = bst.predict(dmat_valid)
    roc = roc_auc_score(y_higgs[0:valid_num], y_pred)
    print(f"Base model: {roc}")
    writer.add_scalar("AUC", roc, num_rounds - 1)
    writer.close()


if __name__ == "__main__":
    main()
