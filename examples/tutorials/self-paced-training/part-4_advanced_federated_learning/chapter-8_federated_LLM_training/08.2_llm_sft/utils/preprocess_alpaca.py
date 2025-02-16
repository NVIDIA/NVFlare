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
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def data_args():
    parser = argparse.ArgumentParser(description="Preprocess data to train and validation files in jsonl format")
    parser.add_argument("--training_file", type=str, required=True, help="Path to training set")
    parser.add_argument("--validation_file", type=str, help="Path to validation set, if given, append to training data")
    parser.add_argument("--validation_ratio", type=float, default=0.1, help="Ratio of validation set, defult to 10%")
    parser.add_argument("--testing_ratio", type=float, default=0.1, help="Ratio of testing set, defult to 10%")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output folder")
    args = parser.parse_args()
    return args


def split_to_jsonl(data, output_dir, validation_ratio, testing_ratio):
    print("Preprocessing data to NeMo_SFT jsonl format...")
    output_path_tra = os.path.join(output_dir, "training.jsonl")
    output_path_val = os.path.join(output_dir, "validation.jsonl")
    output_path_tst = os.path.join(output_dir, "testing.jsonl")

    data_ct = len(data)
    val_threshold = int(data_ct * validation_ratio)
    test_threshold = int(data_ct * testing_ratio)

    with open(output_path_val, "w") as g, open(output_path_tst, "w") as h, open(output_path_tra, "w") as i:
        for index, item in data.iterrows():
            input = item["input"].strip()
            if input != "":
                # Randomize input and instruction order.
                input_first = np.random.randint(0, 2) == 0
                if input_first:
                    instruction = item["instruction"].strip()
                    assert instruction != ""
                    input = f"{input}\n\n{instruction}"
                    output = item["output"]
                else:
                    instruction = item["instruction"].strip()
                    assert instruction != ""
                    input = f"{instruction}\n\n{input}"
                    output = item["output"]
            else:
                input = item["instruction"]
                output = item["output"]
            # write to jsonl file according to index
            if index < val_threshold:
                h.write(json.dumps({"input": input, "output": output}) + "\n")
            elif index < val_threshold + test_threshold:
                g.write(json.dumps({"input": input, "output": output}) + "\n")
            else:
                i.write(json.dumps({"input": input, "output": output}) + "\n")
    print(f"{index + 1} out of {data_ct} Data was successfully preprocessed and saved.")


def main():
    args = data_args()
    # load training data
    path_to_train = args.training_file
    ds = pq.read_table(path_to_train)
    train = ds.to_pandas()
    # load validation data if provided and append to training data
    if args.validation_file:
        path_to_val = args.validation_file
        ds = pq.read_table(path_to_val)
        val = ds.to_pandas()
        train = pd.concat([train, val])
    # randomize the order of the data
    data_full = train.sample(frac=1, random_state=0).reset_index(drop=True)
    # split data into training, validation and testing
    val_ratio = args.validation_ratio
    test_ratio = args.testing_ratio
    output_dir = args.output_dir
    split_to_jsonl(data_full, output_dir, val_ratio, test_ratio)


if __name__ == "__main__":
    main()
