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
from typing import Optional

import numpy as np
import pandas as pd
from sklearn import datasets


def sklearn_dataset_args_parser():
    parser = argparse.ArgumentParser(description="Load sklearn data and save to csv")
    parser.add_argument("--dataset_name", type=str, choices=["iris", "cancer"], help="Dataset name")
    parser.add_argument("--randomize", type=int, help="Whether to randomize data sequence")
    parser.add_argument("--out_path", type=str, help="Path to output data file")
    return parser


def load_data(dataset_name: str = "iris"):
    if dataset_name == "iris":
        dataset = datasets.load_iris()
    elif dataset_name == "cancer":
        dataset = datasets.load_breast_cancer()
    else:
        raise ValueError("Dataset unknown!")
    return dataset


def download_data(
    output_dir: str,
    dataset_name: str = "iris",
    randomize: bool = False,
    filename: Optional[str] = None,
    file_format="csv",
):
    # Load data
    dataset = load_data(dataset_name)
    x = dataset.data
    y = dataset.target
    if randomize:
        np.random.seed(0)
        idx_random = np.random.permutation(len(y))
        x = x[idx_random, :]
        y = y[idx_random]

    data = np.column_stack((y, x))
    df = pd.DataFrame(data=data)

    # Check if the target folder exists,
    # If not, create

    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        os.rmdir(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Save to csv file
    filename = filename if filename else f"{dataset_name}.csv"
    if file_format == "csv":
        file_path = os.path.join(output_dir, filename)

        df.to_csv(file_path, sep=",", index=False, header=False)
    else:
        raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description="Load sklearn data and save to csv")
    parser.add_argument("--dataset_name", type=str, choices=["iris", "cancer"], help="Dataset name")
    parser.add_argument("--randomize", type=int, help="Whether to randomize data sequence")
    parser.add_argument("--out_path", type=str, help="Path to output data file")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.out_path)
    filename = os.path.basename(args.out_path)
    download_data(output_dir, args.dataset_name, args.randomize, filename)


if __name__ == "__main__":
    main()
