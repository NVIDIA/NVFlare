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
EVAL_ARCHIVE_URL = "https://zenodo.org/records/1214456/files/CRC-VAL-HE-7K.zip"
EVAL_ARCHIVE_NAME = "CRC-VAL-HE-7K.zip"
EVAL_EXTRACTED_DIR_NAME = "CRC-VAL-HE-7K"

# Plain text progress with flush=True so it is visible when streams are block-buffered (non-TTY, logs, pipes).
_DOWNLOAD_REPORT_BYTES = 16 * 1024 * 1024


def archive_is_valid_zip(path: str) -> bool:
    """Return True if path is a non-empty file that opens as a zip (central directory readable)."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    try:
        with zipfile.ZipFile(path, "r"):
            pass
        return True
    except zipfile.BadZipFile:
        return False


def _log_progress(msg: str) -> None:
    print(msg, flush=True)


def resolve_extraction_path(output_dir: str, member_name: str) -> str:
    output_dir = os.path.realpath(output_dir)
    dest_path = os.path.realpath(os.path.join(output_dir, member_name))
    if os.path.commonpath([output_dir, dest_path]) != output_dir:
        raise ValueError(f"Refusing to extract unsafe path outside {output_dir}: {member_name}")
    return dest_path


def download_file(url: str, output_path: str) -> None:
    # Write to .partial first so an interrupted download does not leave a corrupt .zip in place.
    partial_path = output_path + ".partial"
    try:
        _log_progress("Connecting to server (this can take a while for large files)...")
        with urllib.request.urlopen(url) as response:
            content_length = response.headers.get("Content-Length")
            try:
                total = int(content_length) if content_length else None
            except ValueError:
                total = None
            if total is not None:
                _log_progress(f"Expected download size: {total / (1024 ** 2):.1f} MiB")
            else:
                _log_progress("Expected download size: unknown (streaming); reporting bytes received.")

            block_size = 1024 * 1024
            downloaded = 0
            next_report_at = _DOWNLOAD_REPORT_BYTES
            first_chunk = True
            with open(partial_path, "wb") as output_file:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    output_file.write(chunk)
                    downloaded += len(chunk)
                    if first_chunk:
                        _log_progress("Receiving data...")
                        first_chunk = False
                    if downloaded >= next_report_at:
                        if total is not None:
                            pct = min(100.0, 100.0 * downloaded / total)
                            _log_progress(
                                f"Downloaded {downloaded / (1024 ** 2):.1f} MiB of "
                                f"{total / (1024 ** 2):.1f} MiB ({pct:.1f}%)"
                            )
                        else:
                            _log_progress(f"Downloaded {downloaded / (1024 ** 2):.1f} MiB so far")
                        next_report_at += _DOWNLOAD_REPORT_BYTES

            _log_progress(f"Download finished: {downloaded / (1024 ** 2):.1f} MiB written.")
        os.replace(partial_path, output_path)
    except BaseException:
        if os.path.isfile(partial_path):
            try:
                os.remove(partial_path)
            except OSError:
                pass
        raise


def download_and_extract_dataset(
    *,
    url: str,
    archive_name: str,
    extracted_dir_name: str,
    output_dir: str,
    skip_extract: bool,
) -> None:
    archive_path = os.path.join(output_dir, archive_name)
    extracted_dir = os.path.join(output_dir, extracted_dir_name)

    if archive_is_valid_zip(archive_path):
        print(f"Archive already exists: {archive_path}")
    else:
        if os.path.isfile(archive_path):
            print(
                f"Existing file is missing or not a valid zip (corrupt or incomplete); "
                f"re-downloading:\n  {archive_path}"
            )
            os.remove(archive_path)
        print(f"Downloading {url} -> {archive_path}", flush=True)
        download_file(url, archive_path)

    if skip_extract:
        print(f"Skipping extraction for {archive_name} (--skip_extract).")
        return

    if os.path.isdir(extracted_dir):
        print(f"Dataset already extracted at {extracted_dir}")
        return

    print(f"Extracting {archive_path} into {output_dir}", flush=True)
    with zipfile.ZipFile(archive_path, "r") as zip_file:
        members = zip_file.infolist()
        n = len(members)
        report_every = max(2000, n // 40)
        for i, member in enumerate(members, start=1):
            resolve_extraction_path(output_dir, member.filename)
            zip_file.extract(member, output_dir)
            if i % report_every == 0 or i == n:
                _log_progress(f"Extracted {i}/{n} zip entries ({100.0 * i / n:.1f}%)")


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
    parser.add_argument(
        "--include_eval",
        action="store_true",
        help="Also download/extract CRC-VAL-HE-7K for accuracy evaluation.",
    )
    args = parser.parse_args()

    # Reduce block-buffering on stderr/stdout so progress lines show up under nohup, pipes, and some batch systems.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(line_buffering=True)
            except (OSError, ValueError, AttributeError):
                pass

    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    download_and_extract_dataset(
        url=TRAIN_ARCHIVE_URL,
        archive_name=TRAIN_ARCHIVE_NAME,
        extracted_dir_name=EXTRACTED_DIR_NAME,
        output_dir=output_dir,
        skip_extract=args.skip_extract,
    )
    if args.include_eval:
        download_and_extract_dataset(
            url=EVAL_ARCHIVE_URL,
            archive_name=EVAL_ARCHIVE_NAME,
            extracted_dir_name=EVAL_EXTRACTED_DIR_NAME,
            output_dir=output_dir,
            skip_extract=args.skip_extract,
        )

    print(f"Done. Training dataset available at {os.path.join(output_dir, EXTRACTED_DIR_NAME)}")
    if args.include_eval:
        print(f"Done. Evaluation dataset available at {os.path.join(output_dir, EVAL_EXTRACTED_DIR_NAME)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
