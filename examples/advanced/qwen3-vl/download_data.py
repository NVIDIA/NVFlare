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

"""
Download and unpack the PubMedVision dataset for federated Qwen3-VL.

Run from this directory. Standard workflow:

  python download_data.py
  python prepare_data.py --data_file PubMedVision/PubMedVision_InstructionTuning_VQA.json --output_dir ./data
  python job.py --data_dir ./data

Requires: git and Git LFS (https://git-lfs.com/) for cloning the dataset repo.
"""

import argparse
import os
import subprocess
import sys

PUBMEDVISION_REPO = "https://huggingface.co/datasets/FreedomIntelligence/PubMedVision"
NUM_IMAGE_ARCHIVES = 20


def main():
    parser = argparse.ArgumentParser(description="Download PubMedVision and unzip image archives for Qwen3-VL example.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="PubMedVision",
        help="Directory to clone the dataset into (default: PubMedVision). Created if missing.",
    )
    parser.add_argument(
        "--skip_unzip",
        action="store_true",
        help="Only clone; do not unzip image archives (e.g. for testing).",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))

    # Clone (skip if repo already has expected content)
    json_name = "PubMedVision_InstructionTuning_VQA.json"
    json_path = os.path.join(output_dir, json_name)
    if os.path.isfile(json_path):
        print(f"Dataset already present at {output_dir} ({json_name} found). Skipping clone.")
    else:
        parent = os.path.dirname(output_dir)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        if os.path.isdir(output_dir) and os.listdir(output_dir):
            print(
                f"Directory {output_dir} exists but does not contain {json_name}. Remove it or use a different --output_dir.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Cloning {PUBMEDVISION_REPO} into {output_dir} ...")
        try:
            subprocess.run(
                ["git", "clone", PUBMEDVISION_REPO, output_dir],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            print("Error: git not found. Install git and Git LFS (https://git-lfs.com/).", file=sys.stderr)
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"git clone failed: {e.stderr or e}", file=sys.stderr)
            sys.exit(1)
        print("Clone done.")

    if args.skip_unzip:
        print("Skipping unzip (--skip_unzip).")
        return 0

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    for i in range(NUM_IMAGE_ARCHIVES):
        zip_name = f"images_{i}.zip"
        zip_path = os.path.join(output_dir, zip_name)
        if not os.path.isfile(zip_path):
            print(f"Warning: {zip_path} not found; skipping.", file=sys.stderr)
            continue
        print(f"Unzipping {zip_name} ({i + 1}/{NUM_IMAGE_ARCHIVES}) ...")
        try:
            subprocess.run(
                ["unzip", "-q", "-o", "-j", zip_path, "-d", images_dir],
                check=True,
                capture_output=True,
            )
        except FileNotFoundError:
            print("Error: unzip not found.", file=sys.stderr)
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"unzip {zip_path} failed: {e.stderr or e}", file=sys.stderr)
            sys.exit(1)

    print(f"Done. Dataset at {output_dir}. Next: python prepare_data.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
