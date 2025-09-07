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

import os
import zipfile

import requests


def download_file(url: str, outdir: str) -> str:
    """Download a file from a URL to a specified directory."""
    os.makedirs(outdir, exist_ok=True)
    filename = url.split("/")[-1]
    filepath = os.path.join(outdir, filename)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print(f"Downloaded: {filepath}")
    return filepath


def extract_files(zip_path: str, extract_to: str, pattern: str):
    """Extract files from a zip archive that match a given pattern."""
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.startswith(pattern) and file_info.filename.endswith(".data"):
                zip_ref.extract(file_info, extract_to)
                print(f"Extracted: {file_info.filename}")


def main():
    DATA_DIR = "/tmp/flare/dataset/heart_disease_data"
    URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"

    # Download the dataset
    zip_file_path = download_file(URL, DATA_DIR)

    # Extract specific files from the zip archive
    extract_files(zip_file_path, DATA_DIR, "processed")


if __name__ == "__main__":
    main()
