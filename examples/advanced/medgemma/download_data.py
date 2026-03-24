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
import sys
import urllib.request
import zipfile

TRAIN_ARCHIVE_URL = "https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K.zip"
TRAIN_ARCHIVE_NAME = "NCT-CRC-HE-100K.zip"
EXTRACTED_DIR_NAME = "NCT-CRC-HE-100K"


def _format_bytes(n: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    idx = 0
    while n >= 1024.0 and idx < len(units) - 1:
        n /= 1024.0
        idx += 1
    unit = units[idx]
    return f"{int(n)} {unit}" if unit == "B" else f"{n:.1f} {unit}"


def download_file(url: str, output_path: str, chunk_size: int = 1024 * 1024) -> None:
    with urllib.request.urlopen(url) as response, open(output_path, "wb") as output_file:
        total_header = response.headers.get("Content-Length")
        total = None
        if total_header:
            try:
                total = int(total_header.strip())
            except ValueError:
                total = None
        downloaded = 0
        bar_width = 40

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            output_file.write(chunk)
            downloaded += len(chunk)

            if total is not None:
                ratio = min(downloaded / total, 1.0)
                filled = int(bar_width * ratio)
                bar = "=" * filled + "-" * (bar_width - filled)
                pct = 100.0 * ratio
                sys.stdout.write(f"\r[{bar}] {pct:5.1f}%  {_format_bytes(downloaded)} / {_format_bytes(total)}")
            else:
                sys.stdout.write(f"\rDownloaded {_format_bytes(downloaded)}")
            sys.stdout.flush()

        sys.stdout.write("\n")
        sys.stdout.flush()


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
