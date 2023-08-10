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
    parser.add_argument("--site_num", type=int, help="Total number of sites")
    parser.add_argument("--site_name_prefix", type=str, default="site-", help="Site name prefix")
    parser.add_argument("--cols_total", type=int, help="Total number of columns")
    parser.add_argument("--rows_overlap_size", type=int, help="Size of the data overlap between sites")
    parser.add_argument("--rows_total", type=int, help="Total number of instances")
    parser.add_argument("--out_path", type=str, default="~/dataset", help="Output path for the data split json file")
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
    print(f"site_num: {args.site_num}")
    print(f"rows_total: {args.rows_total}")
    print(f"rows_overlap_size: {args.rows_overlap_size}")
    print(f"cols_total: {args.cols_total}")

    site_col_size = split_num_proportion(args.cols_total, args.site_num)
    site_row_size = split_num_proportion(args.rows_total - args.rows_overlap_size, args.site_num)
    print(f"column splits: {site_col_size}")
    print(f"non-overlap row splits: {site_row_size}")

    df = pd.read_csv(args.data_path, header=None)
    df["uid"] = df.index.to_series().map(lambda x: "uid_" + str(x))

    if os.path.exists(args.out_path):
        shutil.rmtree(args.out_path)

    for site in range(args.site_num):
        site_id = args.site_name_prefix + str(site + 1)

        col_start = sum(site_col_size[:site])
        col_end = sum(site_col_size[: site + 1])

        row_start = sum(site_row_size[:site])
        row_end = sum(site_row_size[: site + 1])

        df_split = pd.concat(
            [
                df.iloc[row_start:row_end, np.r_[col_start:col_end, args.cols_total]],
                df.iloc[
                    args.rows_total - args.rows_overlap_size : args.rows_total,
                    np.r_[col_start:col_end, args.cols_total],
                ],
            ]
        )
        print(
            f"site-{site+1} split rows ({row_start}:{row_end}),({args.rows_total - args.rows_overlap_size}:{args.rows_total})"
        )
        print(f"site-{site+1} split cols ({col_start}:{col_end})")

        data_path = os.path.join(args.out_path, f"{args.site_name_prefix}{site + 1}")
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)

        df_split.to_csv(path_or_buf=os.path.join(data_path, "higgs.data.csv"), index=False)


if __name__ == "__main__":
    main()
