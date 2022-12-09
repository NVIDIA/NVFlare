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

import numpy as np
import pandas as pd
from sklearn import datasets


def sklearn_dataset_args_parser():
    parser = argparse.ArgumentParser(description="Load sklearn data and save to csv")
    parser.add_argument(
        "--dataset_name", type=str, choices=["iris", "cancer"], help="Dataset name"
    )
    parser.add_argument(
        "--randomize", type=int, help="Whether to randomize data sequence"
    )
    parser.add_argument("--out_path", type=str, help="Path to output data file")
    return parser


def main():
    parser = sklearn_dataset_args_parser()
    args = parser.parse_args()
    # Load data
    if args.dataset_name == "iris":
        dataset = datasets.load_iris()
    elif args.dataset_name == "cancer":
        dataset = datasets.load_breast_cancer()
    else:
        raise ValueError("Dataset unknown!")
    X = dataset.data
    y = dataset.target

    if args.randomize != 0:
        np.random.seed(0)
        idx_random = np.random.permutation(len(y))
        X = X[idx_random, :]
        y = y[idx_random]

    data = np.column_stack((y, X))
    df = pd.DataFrame(data=data)

    # Check if the target folder exists,
    # If not, create
    tgt_folder = os.path.dirname(args.out_path)
    if not os.path.exists(tgt_folder):
        os.makedirs(tgt_folder)

    # Save to csv file
    df.to_csv(args.out_path, sep=",", index=False, header=False)


if __name__ == "__main__":
    main()
