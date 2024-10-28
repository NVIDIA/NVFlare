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
import shutil

import numpy as np
import pandas as pd


def data_split_args_parser():
    parser = argparse.ArgumentParser(description="Generate data split for dataset")
    parser.add_argument("--data_path", type=str, help="Path to data file")
    parser.add_argument("--site_num", type=int, default=2, help="Total number of sites")
    parser.add_argument(
        "--rows_total_percentage",
        type=float,
        default=1.0,
        help="Percentage of dataset_rows_total to use for rows_total",
    )
    parser.add_argument(
        "--rows_overlap_percentage",
        type=float,
        default=1.0,
        help="Percentage of rows_total to use for rows_overlap between sites",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="./dataset",
        help="Output path for the data split file",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="data.csv",
        help="Output file name for the data split file",
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

    dataset_rows_total, cols_total = df.shape[0], df.shape[1]
    rows_total = int(dataset_rows_total * args.rows_total_percentage)
    rows_overlap = int(rows_total * args.rows_overlap_percentage)

    print(f"site_num: {args.site_num}")
    print(
        f"dataset_num_rows: {dataset_rows_total}, rows_total_percentage: {args.rows_total_percentage}, rows_total: {rows_total}"
    )
    print(f"rows_overlap_percentage: {args.rows_overlap_percentage}, rows_overlap: {rows_overlap}")
    print(f"cols_total: {cols_total}")

    df["uid"] = df.index.to_series().map(lambda x: "uid_" + str(x))

    site_col_size = split_num_proportion(cols_total, args.site_num)
    site_row_size = split_num_proportion(rows_total - rows_overlap, args.site_num)

    if os.path.exists(args.out_path):
        shutil.rmtree(args.out_path)

    for site in range(args.site_num):
        col_start = sum(site_col_size[:site])
        col_end = sum(site_col_size[: site + 1])

        row_start = sum(site_row_size[:site])
        row_end = sum(site_row_size[: site + 1])

        df_split = pd.concat(
            [
                df.iloc[row_start:row_end, np.r_[col_start:col_end, cols_total]],
                df.iloc[
                    rows_total - rows_overlap : rows_total,
                    np.r_[col_start:col_end, cols_total],
                ],
            ]
        )
        df_split = df_split.sample(frac=1)
        print(f"site-{site + 1} split rows [{row_start}:{row_end}],[{rows_total - rows_overlap}:{rows_total}]")
        print(f"site-{site + 1} split cols [{col_start}:{col_end}]")

        data_path = os.path.join(args.out_path, f"site-{site + 1}")
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)

        df_split.to_csv(path_or_buf=os.path.join(data_path, args.out_file), index=False)


if __name__ == "__main__":
    main()
