# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import shutil

import numpy as np
import pandas as pd


def data_split_args_parser():
    parser = argparse.ArgumentParser(description="Generate data split for dataset")
    parser.add_argument("--data_path", type=str, help="Path to data file")
    parser.add_argument("--site_num", type=int, default=3, help="Total number of sites")
    parser.add_argument(
        "--out_path",
        type=str,
        default="./dataset",
        help="Output path for the data split file",
    )
    return parser


def split_num_proportion(n, site_num):
    split = []
    ratio_vec = np.ones(site_num)
    total = sum(ratio_vec)
    left = n
    for site in range(site_num - 1):
        x = int(n * ratio_vec[site] / total)
        left = left - x
        split.append(x)
    split.append(left)
    return split


def main():
    parser = data_split_args_parser()
    args = parser.parse_args()

    df = pd.read_csv(args.data_path, header=None)

    rows_total, cols_total = df.shape[0], df.shape[1]

    print(f"site_num: {args.site_num}")
    print(f"rows_total: {rows_total}, cols_total: {cols_total}")

    # split row
    site_row_size = split_num_proportion(int(0.8 * rows_total), args.site_num)
    print(f"site_row_size: {site_row_size}")

    if os.path.exists(args.out_path):
        shutil.rmtree(args.out_path)

    # assign first 80% rows to train
    df_train = df.iloc[: int(0.8 * rows_total), :]
    # assign last 20% rows to valid
    df_valid = df.iloc[int(0.8 * rows_total) :, :]

    for site in range(args.site_num):
        row_start = sum(site_row_size[:site])
        row_end = sum(site_row_size[: site + 1])

        df_split = df_train.iloc[row_start:row_end, :]
        print(f"site-{site + 1} split rows [{row_start}:{row_end}]")

        data_path = os.path.join(args.out_path, f"site-{site + 1}")
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)

        # save train and valid data
        df_split.to_csv(path_or_buf=os.path.join(data_path, "train.csv"), index=False, header=False)
        df_valid.to_csv(path_or_buf=os.path.join(data_path, "valid.csv"), index=False, header=False)


if __name__ == "__main__":
    main()
