# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import requests

# Veterans' Lung Cancer Trial dataset from the R survival package,
# hosted by the Rdatasets collection (MIT-licensed mirror of public R datasets).
VETERAN_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/survival/veteran.csv"
DEFAULT_DATA_PATH = "/tmp/nvflare/dataset/veteran/veteran.csv"


def download_veteran_data(output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    r = requests.get(VETERAN_URL, timeout=30)
    r.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(r.content)
    print(f"Saved Veterans' Lung Cancer dataset to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download the Veterans' Lung Cancer dataset")
    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Path to save the dataset CSV. Default: {DEFAULT_DATA_PATH}",
    )
    args = parser.parse_args()
    download_veteran_data(args.output_path)


if __name__ == "__main__":
    main()
