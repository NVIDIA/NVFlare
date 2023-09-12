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


def data_args():
    parser = argparse.ArgumentParser(description="Combine a list of jsonl files")
    parser.add_argument("--file_list", nargs="+", required=True, help="Path to input file list")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output file")
    args = parser.parse_args()
    return args


def main():
    args = data_args()
    # load training data
    file_list = args.file_list
    data_combined = pd.DataFrame()
    for file in file_list:
        data = pd.read_json(file, lines=True)
        data_combined = pd.concat([data_combined, data])
    # randomize the order of the data
    data_combined = data_combined.sample(frac=1, random_state=0).reset_index(drop=True)
    # save the combined data
    output_path = args.output_path
    with open(output_path, "w") as f:
        f.write(data_combined.to_json(orient="records", lines=True))


if __name__ == "__main__":
    main()
