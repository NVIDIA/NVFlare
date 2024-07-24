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
import csv
import os
import random


def parse_args(prog_name: str):
    _parser = argparse.ArgumentParser(description=prog_name)
    _parser.add_argument(
        "--prepare-data",
        dest="prepare_data",
        action="store_const",
        const=prepare_data,
        help="prepare data based on configuration",
    )
    return _parser, _parser.parse_args()


def prepare_data():
    # Set variables
    NUM_UNIVERSITIES = 7

    DATASET_PATH = "/tmp/nvflare/data/hierarchical_stats/"
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)

    print(f"Preparing data at data directory `{DATASET_PATH}`...\n")

    # Generate the entries for 7 different universities and copy it to each client data directory
    for n in range(1, NUM_UNIVERSITIES + 1):
        output_file = f"university-{n}.csv"
        with open(output_file, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            # Create and write the header to the CSV file
            csvwriter.writerow(["Pass", "Fail", "Percentage"])
            num_entries = random.randint(1000, 2000)
            for _ in range(num_entries):
                # Randomly decide pass or fail (0 or 1)
                pass_ = random.randint(0, 1)
                if pass_ == 1:
                    fail = 0
                    percentage = round(random.uniform(50.01, 99.99), 2)
                else:
                    fail = 1
                    percentage = round(random.uniform(20.00, 49.99), 2)
                csvwriter.writerow([pass_, fail, percentage])

        client_path = os.path.join(DATASET_PATH, f"university-{n}")
        if not os.path.exists(client_path):
            os.makedirs(client_path)
        os.rename(output_file, os.path.join(client_path, output_file))
        print(
            f"CSV file `{output_file}` is generated with {num_entries} entries for client `university-{n}` at {client_path}."
        )

    print("\nDone preparing data.")
