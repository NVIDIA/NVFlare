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

import shutil
from pathlib import Path

import kagglehub


def download(input_path: str, output_path: str, overwrite: bool = False):
    # Download latest version
    path = kagglehub.dataset_download(input_path)
    print("Path to dataset files:", path)

    # Move downloaded data to output path
    output = Path(output_path)

    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output path '{output_path}' already exists. "
                f"Use overwrite=True to replace it, or manually remove the directory."
            )

        # Safety: Only remove if it looks like a previous kaggle download
        # Check for expected structure/marker file
        if not output.is_dir():
            raise ValueError(f"{output_path} exists but is not a directory")

        # Check for marker file we created in previous runs
        marker_file = output / ".kaggle_download_marker"
        if not marker_file.exists():
            raise ValueError(
                f"{output_path} exists but doesn't appear to be a previous "
                f"kaggle download. Remove it manually to proceed."
            )

        shutil.rmtree(output)
        print(f"Removed previous download: {output}")

    shutil.move(path, output_path)

    # Create marker for future safety checks
    (Path(output_path) / ".kaggle_download_marker").touch()

    print(f"Dataset moved to: {output_path}")
