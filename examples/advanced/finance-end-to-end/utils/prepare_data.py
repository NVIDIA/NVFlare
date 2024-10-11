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
import random
import shutil
import string

import pandas as pd

# List of example BICs for demonstration
from sklearn.model_selection import train_test_split

bic_list = {
    "ZHSZUS33": "United States",  # Bank 1
    "SHSHKHH1": "Hong Kong",  # bank 2
    "YXRXGB22": "United Kingdom",  # bank 3
    "WPUWDEFF": "Germany",  # bank 4
    "YMNYFRPP": "France",  # bank 5
    "FBSFCHZH": "Switzerland",  # Bank 6
    "YSYCESMM": "Spain",  # bank 7
    "ZNZZAU3M": "Australia",  # Bank 8
    "HCBHSGSG": "Singapore",  # bank 9
    "XITXUS33": "United States",  # bank 10
}

# List of currencies and their respective countries
currencies = {
    "USD": "United States",
    "EUR": "Eurozone",
    "GBP": "United Kingdom",
    "JPY": "Japan",
    "AUD": "Australia",
    "CHF": "Switzerland",
    "SGD": "Singapore",
}

# BIC to Bank Name mapping
bic_to_bank = {
    "ZHSZUS33": "Bank_1",
    "SHSHKHH1": "Bank_2",
    "YXRXGB22": "Bank_3",
    "WPUWDEFF": "Bank_4",
    "YMNYFRPP": "Bank_5",
    "FBSFCHZH": "Bank_6",
    "YSYCESMM": "Bank_7",
    "ZNZZAU3M": "Bank_8",
    "HCBHSGSG": "Bank_9",
    "XITXUS33": "Bank_10",
}


# Function to generate random UETR
def generate_random_uetr(length=22):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


# Function to generate random BICs and currency details
def generate_random_details(df):
    # Ensure the currency and beneficiary BIC match
    def match_currency_and_bic():
        while True:
            currency = random.choice(list(currencies.keys()))
            country = currencies[currency]
            matching_bics = [bic for bic, bic_country in bic_list.items() if bic_country == country]
            if matching_bics:
                return currency, random.choice(matching_bics)

    df["Sender_BIC"] = [random.choice(list(bic_list.keys())) for _ in range(len(df))]
    df["Receiver_BIC"] = [random.choice(list(bic_list.keys())) for _ in range(len(df))]
    df["UETR"] = [generate_random_uetr() for _ in range(len(df))]

    df["Currency"], df["Beneficiary_BIC"] = zip(*[match_currency_and_bic() for _ in range(len(df))])
    df["Currency_Country"] = df["Currency"].map(currencies)

    return df


def split_datasets(df, out_folder: str, hist_ratio=0.55, train_ratio=0.35):
    # Sort the DataFrame by the Time column
    df = df.sort_values(by="Time").reset_index(drop=True)

    # Calculate the number of samples for each split
    total_size = len(df)
    historical_size = int(total_size * hist_ratio)
    train_size = int(total_size * train_ratio)
    test_size = total_size - historical_size - train_size

    # Split into historical and remaining data
    df_history = df.iloc[:historical_size]
    remaining_df = df.iloc[historical_size:]
    y = remaining_df.Class

    ds = remaining_df.drop("Class", axis=1)
    # Split the remaining data into train and test
    x_train, x_test, y_train, y_test = train_test_split(
        ds, y, test_size=test_size / (train_size + test_size), random_state=42
    )

    df_train = pd.concat([y_train, x_train], axis=1)
    df_test = pd.concat([y_test, x_test], axis=1)

    # Display sizes of each dataset
    print(f"Historical DataFrame size: {len(df_history)}")
    print(f"Training DataFrame size: {len(df_train)}")
    print(f"Testing DataFrame size: {len(df_test)}")

    # Save training and testing sets
    os.makedirs(out_folder, exist_ok=True)

    df_train.to_csv(path_or_buf=os.path.join(out_folder, "train.csv"), index=False)
    df_test.to_csv(path_or_buf=os.path.join(out_folder, "test.csv"), index=False)
    df_history.to_csv(path_or_buf=os.path.join(out_folder, "history.csv"), index=False)


def split_site_datasets(out_folder):
    files = ["history", "train", "test"]
    client_names = set()

    for f in files:
        file_path = os.path.join(out_folder, f + ".csv")
        df = pd.read_csv(file_path)
        # Group the DataFrame by 'Sender_BIC'
        grouped = df.groupby("Sender_BIC")
        # Save each group to a separate file
        for name, group in grouped:
            bank_name = bic_to_bank[name].replace(" ", "_")
            client_name = f"{name}_{bank_name}"
            client_names.add(client_name)
            site_dir = os.path.join(out_folder, client_name)
            os.makedirs(site_dir, exist_ok=True)

            filename = os.path.join(site_dir, f"{f}.csv")
            group.to_csv(filename, index=False)
            print(f"Saved {name} {f} transactions to {filename}")

    print(client_names)


def replicate_dataset(data_path):
    # expand original data and generate a 2-plus year data
    origin_df = pd.read_csv(data_path)
    n = 4
    df_temp = origin_df[["Time", "Amount", "Class"]].copy()
    df_temp["Time"] = df_temp["Time"] * 100

    # Find the maximum value in the 'Time' column
    max_time = df_temp["Time"].max()
    df = df_temp

    for i in range(1, n):
        # Create a duplicate of the DataFrame with incremental 'Time' values

        df_duplicate = df_temp.copy()
        df_duplicate["Time"] = df_duplicate["Time"] + max_time * i

        # Combine the original DataFrame with the duplicated DataFrame
        df = pd.concat([df, df_duplicate], ignore_index=True)

    min_time = df["Time"].min()
    max_time = df["Time"].max()

    min_months = min_time / 3600 / 24 / 30
    max_months = max_time / 3600 / 24 / 30
    # Try to generate a 2-plus year data
    print(f"{min_months=}, {max_months=}")

    return df


def main():
    args = define_parser()

    input_data_path = args.input_data_path
    out_folder = args.output_dir

    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)

    df = replicate_dataset(input_data_path)

    # Add random BIC and currency details to the DataFrame
    df = generate_random_details(df)

    split_datasets(df, out_folder=out_folder, hist_ratio=0.55, train_ratio=0.35)
    split_site_datasets(out_folder)


def define_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input_data_path",
        type=str,
        nargs="?",
        default="creditcard.csv",
        help="input data path for credit car csv file path, default to creditcard.csv'",
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
