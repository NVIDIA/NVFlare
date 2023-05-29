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

import os
import argparse
import numpy as np
import pandas as pd
import json


def data_split_args():
    parser = argparse.ArgumentParser(description="Generate data split for dataset")
    parser.add_argument("--data_path", type=str, help="Path to data file")
    parser.add_argument("--num_clients", type=int, default=2, help="Total number of clients")
    parser.add_argument("--testing_ratio", type=float, default=0.1, help="Ratio of testing set")
    parser.add_argument("--validation_ratio", type=float, default=0.1, help="Ratio of validation set")
    parser.add_argument("--site_name_prefix", type=str, default="site-", help="Site name prefix")
    args = parser.parse_args()
    return args


def split_ratio(ratio_tra, ratio_val, num_clients):
    splits = [0.0]
    # append training
    for i in range(num_clients):
        splits.append(ratio_tra / num_clients * (i + 1))
    # append validation
    for i in range(num_clients):
        splits.append(ratio_tra + ratio_val / num_clients * (i + 1))
    # append testing
    splits.append(1.0)
    return np.array(splits)


def split_data(data_path,
               num_clients,
               site_name_prefix,
               split_points):
    # use pandas to read jsonl format
    data = pd.read_json(data_path, lines=True)
    total_count = len(data)
    assert total_count > 0, f"No data loaded from {data_path}"
    print(f"Loaded training data with {total_count} entries")
    split_points = np.rint(split_points * total_count)
    # shuffle the data
    # data = data.sample(frac=1, random_state=random_seed)
    # use the given split to split the data
    # raw data needs preprocessing, visit items one by one
    print(f"Preprocessing data to NeMo jsonl format...")
    # base file name
    path_to_data = f"{data_path.split('.')[0]}"
    output_path_list = [None] * num_clients * 2
    out_dir = f"{path_to_data}/{num_clients}-clients/"
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for i in range(num_clients):
        output_path_list[i] = f"{out_dir}/{site_name_prefix}{i}-training.jsonl"
        output_path_list[num_clients + i] = f"{out_dir}/{site_name_prefix}{i}-validation.jsonl"
    output_path_list.append(f"{out_dir}/testing.jsonl")
    # split the data and save to files
    file_ct = 0
    f = open(output_path_list[file_ct], "w")
    for ct in range(total_count):
        item = data.iloc[ct]
        # precrocess the data item
        context = item['context'].strip()
        if context != "":
            # Randomize context and instruction order.
            context_first = np.random.randint(0, 2) == 0
            if context_first:
                instruction = item['instruction'].strip()
                assert instruction != ""
                input = f"{context}\n\n{instruction}"
                output = item['response']
            else:
                instruction = item['instruction'].strip()
                assert instruction != ""
                input = f"{instruction}\n\n{context}"
                output = item['response']
        else:
            input = item['instruction']
            output = item['response']
        if ct >= split_points[file_ct] and ct < split_points[file_ct+1]:
            f.write(json.dumps({'input': input, 'output': output}) + '\n')
        if ct == split_points[file_ct+1]:
            # close previous file and open the next one
            file_ct += 1
            f.close()
            f = open(output_path_list[file_ct], "w")
    print(f"Data was successfully preprocessed and saved.")


def main():
    args = data_split_args()
    # parse and check the ratios
    num_clients = args.num_clients
    ratio_tst = args.testing_ratio
    assert ratio_tst < 1 and ratio_tst >= 0, f"Invalid testing ratio!"
    ratio_val = args.validation_ratio
    assert ratio_val < 1 and ratio_val >= 0, f"Invalid validation ratio!"
    ratio_tra = 1 - (ratio_tst + ratio_val)
    assert ratio_tra > 0, f"""Invalid ratio: training ratio {ratio_tra}, testing ratio {ratio_tst}, validation ratio {ratio_val}"""

    # generate split: training_0...training_k, validation_0...validation_k, testing
    # testing set is the same for all clients
    split_points = split_ratio(ratio_tra, ratio_val, num_clients)

    # data split
    split_data(
        data_path=args.data_path,
        num_clients=num_clients,
        site_name_prefix=args.site_name_prefix,
        split_points=split_points
    )


if __name__ == "__main__":
    main()
