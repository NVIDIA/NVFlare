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
import csv
import os
import shutil


def parse_args(prog_name: str):
    _parser = argparse.ArgumentParser(description=prog_name)
    _parser.add_argument(
        "--prepare-data",
        dest="prepare_data",
        action="store_const",
        const=prepare_data,
        help="prepare data based on configuration",
    )
    _parser.add_argument(
        "-d",
        "--dest",
        type=str,
        nargs="?",
        default="",
        help="destination directory where the data to download to",
    )
    return _parser, _parser.parse_args()


def get_data_url() -> dict:
    client_data = {
        "site-1": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "site-2": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    }
    return client_data


def prepare_data(data_root_dir: str):
    print(f"prepare data for data directory {data_root_dir}")
    client_data_urls = get_data_url()
    for client in client_data_urls:
        client_data_dir = os.path.join(data_root_dir, client)
        if not os.path.exists(client_data_dir):
            os.makedirs(client_data_dir, exist_ok=True)

        dest = os.path.join(client_data_dir, "data.csv")
        if os.path.exists(dest):
            print(f"\nremove existing data at {dest}")
            shutil.rmtree(dest, ignore_errors=True)

        print(f"\ndownload to {dest}")
        url = client_data_urls[client]
        import requests

        with open(dest, "w") as f:
            writer = csv.writer(f)
            r = requests.get(url, allow_redirects=True)
            for line in r.iter_lines():
                if line:
                    writer.writerow(line.decode("utf-8").split(","))
                else:
                    print("skip empty line\n")
    print("\ndone with prepare data")


def main():
    prog_name = "data_utils"
    parser, args = parse_args(prog_name)

    if args.prepare_data:
        prepare_data(args.dest)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
