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

import pandas as pd


def load_data(input_file_path) -> pd.DataFrame:
    # Read the CSV file into a pandas DataFrame
    return pd.read_csv(input_file_path, header=None)


def split_csv(input_file_path, output_dir, num_parts, part_name, sample_rate):
    df = load_data(input_file_path)

    # Calculate the number of rows per part
    total_size = int(len(df) * sample_rate)
    rows_per_part = total_size // num_parts

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Split the DataFrame into N parts
    for i in range(num_parts):
        start_index = i * rows_per_part
        end_index = (i + 1) * rows_per_part if i < num_parts - 1 else total_size
        print(f"{part_name}{i + 1}=", f"{start_index=}", f"{end_index=}")
        part_df = df.iloc[start_index:end_index]

        # Save each part to a separate CSV file
        output_file = os.path.join(output_dir, f"{part_name}{i + 1}.csv")
        part_df.to_csv(output_file, header=False, index=False)


def distribute_header_file(input_header_file: str, output_dir: str, num_parts: int, part_name: str):
    source_file = input_header_file

    # Split the DataFrame into N parts
    for i in range(num_parts):
        output_file = os.path.join(output_dir, f"{part_name}{i + 1}_header.csv")
        shutil.copy(source_file, output_file)
        print(f"File copied to {output_file}")


def define_args_parser():
    parser = argparse.ArgumentParser(description="csv data split")
    parser.add_argument("--input_data_path", type=str, help="input path to csv data file")
    parser.add_argument("--input_header_path", type=str, help="input path to csv header file")
    parser.add_argument("--site_num", type=int, help="Total number of sites or clients")
    parser.add_argument("--site_name_prefix", type=str, default="site-", help="Site name prefix")
    parser.add_argument("--output_dir", type=str, default="/tmp/nvflare/dataset/output", help="Output directory")
    parser.add_argument(
        "--sample_rate", type=float, default="1.0", help="percent of the data will be used. default 1.0 for 100%"
    )
    return parser


def main():
    parser = define_args_parser()
    args = parser.parse_args()
    input_file = args.input_data_path
    output_directory = args.output_dir
    num_parts = args.site_num
    site_name_prefix = args.site_name_prefix
    sample_rate = args.sample_rate
    split_csv(input_file, output_directory, num_parts, site_name_prefix, sample_rate)
    distribute_header_file(args.input_header_path, output_directory, num_parts, site_name_prefix)


if __name__ == "__main__":
    main()
