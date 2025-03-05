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

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# (1) import nvflare client API
import nvflare.client as flare

dataset_names = ["train", "test"]
datasets = {}


def main():
    print("\n pre-process starts \n ")
    args = define_parser()
    input_dir = args.input_dir
    output_dir = args.output_dir

    flare.init()
    site_name = flare.get_site_name()

    # receives global message from NVFlare
    etl_task = flare.receive()

    print("\n receive task \n ")
    processed_dfs = process_dataset(input_dir, site_name)

    save_normalized_files(output_dir, processed_dfs, site_name)

    print("end task")
    # send message back the controller indicating end.
    etl_task.meta["status"] = "done"
    flare.send(etl_task)


def save_normalized_files(output_dir, processed_dfs, site_name):
    for name in processed_dfs:
        print(f"\n dataset {name=} \n ")
        site_dir = os.path.join(output_dir, site_name)
        os.makedirs(site_dir, exist_ok=True)

        enrich_file_name = os.path.join(site_dir, f"{name}_normalized.csv")
        print("save to = ", enrich_file_name)
        processed_dfs[name].to_csv(enrich_file_name)


def process_dataset(input_dir, site_name):
    processed_dfs = {}
    numerical_columns = [
        "Timestamp",
        "Class",
        "Amount",
        "trans_volume",
        "total_amount",
        "average_amount",
        "hist_trans_volume",
        "hist_total_amount",
        "hist_average_amount",
        "x2_y1",
        "x3_y2",
    ]
    category_columns = ["Currency_Country", "Beneficiary_BIC", "Currency", "UETR", "Receiver_BIC", "Sender_BIC"]
    for ds_name in dataset_names:
        file_name = os.path.join(input_dir, site_name, f"{ds_name}_enrichment.csv")
        df = pd.read_csv(file_name)
        datasets[ds_name] = df

        # Convert 'Time' column to datetime
        df["Time"] = pd.to_datetime(df["Time"])
        # Convert datetime to Unix timestamp
        df["Timestamp"] = df["Time"].astype(int) / 10**9  # convert to seconds

        # Separate numerical and categorical features
        numerical_features = df[numerical_columns]
        categorical_features = df[category_columns]

        # Initialize the MinMaxScaler (or StandardScaler)
        scaler = MinMaxScaler()

        # Fit and transform the numerical data
        numerical_normalized = pd.DataFrame(
            scaler.fit_transform(numerical_features), columns=numerical_features.columns
        )

        # Combine the normalized numerical features with the categorical features
        df_combined = pd.concat([categorical_features, numerical_normalized], axis=1)

        # one-hot encoding -- skip this step, this will increase the file size from 11M => 2.x GB
        # df_combined = pd.get_dummies(df_combined, columns=category_columns)

        print("Combined DataFrame with Normalized Numerical Features:")
        print(df_combined)

        processed_dfs[ds_name] = df_combined

    return processed_dfs


def define_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        nargs="?",
        default="/tmp/dataset/credit_data",
        help="input directory where csv files for each site are expected, default to /tmp/dataset/credit_data",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        nargs="?",
        default="/tmp/dataset/credit_data",
        help="output directory, default to '/tmp/dataset/credit_data'",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
