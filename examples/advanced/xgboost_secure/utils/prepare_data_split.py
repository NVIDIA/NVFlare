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

import pandas as pd
from sklearn.model_selection import train_test_split


def data_split_args_parser():
    parser = argparse.ArgumentParser(description="Generate training/testing split for dataset")
    parser.add_argument("--data_path", type=str, help="Path to data file")
    parser.add_argument("--row_ratio", type=float, help="Ratio of subsample row set")
    parser.add_argument("--test_ratio", type=float, help="Ratio of testing set")
    parser.add_argument(
        "--out_folder",
        type=str,
        default="~/dataset",
        help="Output folder for training/testing data",
    )
    return parser


def main():
    parser = data_split_args_parser()
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    df_pos = df[df.Class == 1]
    df_neg = df[df.Class == 0]
    # print the number of positive and negative samples
    print("Number of positive samples: ", len(df_pos))
    print("Number of negative samples: ", len(df_neg))

    # specify negtive ratio
    pos_ratio = args.row_ratio
    neg_ratio = pos_ratio * len(df_pos) / len(df_neg)

    df_pos = df_pos.sample(frac=pos_ratio)
    df_neg = df_neg.sample(frac=neg_ratio)

    # Split data into training and testing sets
    X_pos = df_pos.drop(["Class","Time","Amount","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28"], axis=1)
    y_pos = df_pos.Class
    X_neg = df_neg.drop(["Class","Time","Amount","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28"], axis=1)
    y_neg = df_neg.Class
    X = pd.concat([X_pos, X_neg])
    y = pd.concat([y_pos, y_neg])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_ratio, random_state=77)
    df_train = pd.concat([y_train, X_train], axis=1)
    df_test = pd.concat([y_test, X_test], axis=1)

    # Save training and testing sets
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder, exist_ok=True)
    df_train.to_csv(path_or_buf=os.path.join(args.out_folder, "train_lite.csv"), index=False, header=False)
    df_test.to_csv(path_or_buf=os.path.join(args.out_folder, "test_lite.csv"), index=False, header=False)


if __name__ == "__main__":
    main()
