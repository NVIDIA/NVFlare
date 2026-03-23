# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile

TRAIN_ARCHIVE_URL = "https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K.zip"
TRAIN_ARCHIVE_NAME = "NCT-CRC-HE-100K.zip"
EXTRACTED_DIR_NAME = "NCT-CRC-HE-100K"


def download_file(url: str, output_path: str) -> None:
    with urllib.request.urlopen(url) as response, open(output_path, "wb") as output_file:
        shutil.copyfileobj(response, output_file)


def main():
    parser = argparse.ArgumentParser(description="Download and extract NCT-CRC-HE-100K for the MedGemma example.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to place the downloaded archive and extracted dataset (default: current directory).",
    )
    parser.add_argument(
        "--skip_extract",
        action="store_true",
        help="Download the archive but skip extraction.",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    archive_path = os.path.join(output_dir, TRAIN_ARCHIVE_NAME)
    extracted_dir = os.path.join(output_dir, EXTRACTED_DIR_NAME)

    if not os.path.isfile(archive_path):
        print(f"Downloading {TRAIN_ARCHIVE_URL} -> {archive_path}")
        download_file(TRAIN_ARCHIVE_URL, archive_path)
    else:
        print(f"Archive already exists: {archive_path}")

    if args.skip_extract:
        print("Skipping extraction (--skip_extract).")
        return 0

    if os.path.isdir(extracted_dir):
        print(f"Dataset already extracted at {extracted_dir}")
        return 0

    print(f"Extracting {archive_path} into {output_dir}")
    with zipfile.ZipFile(archive_path, "r") as zip_file:
        zip_file.extractall(output_dir)

    print(f"Done. Dataset available at {extracted_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
