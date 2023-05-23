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

import numpy as np
import pandas as pd


def split_data(data_path, out_dir, num_clients, site_name_prefix, seed):
    # use pandas to read jsonl format
    train_data = pd.read_json(data_path, lines=True)
    assert len(train_data) > 0, f"No data loaded from {data_path}"
    print(f"Loaded training data with {len(train_data)} entries")

    # shuffle the data
    train_data = train_data.sample(frac=1, random_state=seed)

    train_data_splits = np.array_split(train_data, num_clients)

    for idx, split in enumerate(train_data_splits):
        df = pd.DataFrame(split)

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        out_file = os.path.join(out_dir, f"{site_name_prefix}{idx+1}.jsonl")

        df.to_json(out_file, orient="records", lines=True)
        print(f"Save split {idx+1} of {len(train_data_splits)} with {len(split)} entries to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data split for dataset")
    parser.add_argument("--data_path", type=str, help="Path to data file")
    parser.add_argument("--out_dir", type=str, help="Path to output directory", default=".")
    parser.add_argument("--num_clients", type=int, help="Total number of clients", default=3)
    parser.add_argument("--random_seed", type=int, help="Random seed", default=0)
    parser.add_argument("--site_name_prefix", type=str, help="Site name prefix", default="site-")
    args = parser.parse_args()

    split_data(
        data_path=args.data_path,
        out_dir=args.out_dir,
        num_clients=args.num_clients,
        site_name_prefix=args.site_name_prefix,
        seed=args.random_seed,
    )
