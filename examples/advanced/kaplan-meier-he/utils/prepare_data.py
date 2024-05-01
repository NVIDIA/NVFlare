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

import numpy as np
import pandas as pd
from sksurv.datasets import load_veterans_lung_cancer

np.random.seed(77)


def data_split_args_parser():
    parser = argparse.ArgumentParser(description="Generate data split for dataset")
    parser.add_argument("--site_num", type=int, default=5, help="Total number of sites, default is 5")
    parser.add_argument(
        "--site_name_prefix",
        type=str,
        default="site-",
        help="Site name prefix, default is site-",
    )
    parser.add_argument("--bin_days", type=int, default=1, help="Bin days for categorizing data")
    parser.add_argument("--out_path", type=str, help="Output root path for split data files")
    return parser


def prepare_data(data, site_num, bin_days):
    # Get total data count
    total_data_num = data.shape[0]
    print(f"Total data count: {total_data_num}")
    # Get event and time
    event = data["Status"]
    time = data["Survival_in_days"]
    # Categorize data to a bin, default is a week (7 days)
    time = np.ceil(time / bin_days).astype(int) * bin_days
    # Shuffle data
    idx = np.random.permutation(total_data_num)
    # Split data to clients
    event_clients = {}
    time_clients = {}
    for i in range(site_num):
        start = int(i * total_data_num / site_num)
        end = int((i + 1) * total_data_num / site_num)
        event_i = event[idx[start:end]]
        time_i = time[idx[start:end]]
        event_clients["site-" + str(i + 1)] = event_i
        time_clients["site-" + str(i + 1)] = time_i
    return event_clients, time_clients


def main():
    parser = data_split_args_parser()
    args = parser.parse_args()

    # Load data
    # For this KM analysis, we use full timeline and event label only
    _, data = load_veterans_lung_cancer()

    # Prepare data
    event_clients, time_clients = prepare_data(data=data, site_num=args.site_num, bin_days=args.bin_days)

    # Save data to csv files
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)
    for site in range(args.site_num):
        output_file = os.path.join(args.out_path, f"{args.site_name_prefix}{site + 1}.csv")
        df = pd.DataFrame(
            {
                "event": event_clients["site-" + str(site + 1)],
                "time": time_clients["site-" + str(site + 1)],
            }
        )
        df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
