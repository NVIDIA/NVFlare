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

from monai.apps.utils import download_and_extract


def download_spleen_dataset(filepath, output_dir):
    url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
    download_and_extract(url=url, filepath=filepath, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath",
        "-f",
        type=str,
        help="the file path of the downloaded compressed file.",
        default="./data/Task09_Spleen.tar",
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, help="target directory to save extracted files.", default="./data"
    )
    args = parser.parse_args()
    download_spleen_dataset(args.filepath, args.output_dir)
