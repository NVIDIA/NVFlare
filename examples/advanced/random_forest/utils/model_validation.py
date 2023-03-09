#!/usr/bin/env python

import argparse

import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score


def model_validation_args_parser():
    parser = argparse.ArgumentParser(description="Validate model performance")
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to dataset file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to model file",
    )
    parser.add_argument(
        "--size_valid", type=int, help="Validation size, the first N instances to be treated as validation data"
    )
    parser.add_argument(
        "--num_trees",
        type=int,
        help="Total number of trees",
    )
    parser.add_argument(
        "--tree_method", type=str, default="hist", help="tree_method for xgboost - use hist or gpu_hist for best perf"
    )
    return parser


def main():
    parser = model_validation_args_parser()
    args = parser.parse_args()
    data_path = args.data_path
    model_path = args.model_path
    num_trees = args.num_trees
    param = {}
    param["objective"] = "binary:logistic"
    param["eta"] = 0.1
    param["max_depth"] = 8
    param["eval_metric"] = "auc"
    param["nthread"] = 16
    param["num_parallel_tree"] = num_trees

    # get validation data
    size_valid = args.size_valid
    data = pd.read_csv(data_path, header=None, nrows=size_valid)
    # split to feature and label
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    dmat = xgb.DMatrix(X, label=y)

    # validate model performance
    bst = xgb.Booster(param, model_file=model_path)
    y_pred = bst.predict(dmat)
    auc = roc_auc_score(y, y_pred)
    print(f"AUC over first {size_valid} instances is: {auc}")


if __name__ == "__main__":
    main()
