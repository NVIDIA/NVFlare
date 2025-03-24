#!/usr/bin/env python3

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

import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from zipfile import ZipFile


def print_zip_tree(zip_path: Path, exclude_patterns: Optional[str] = None):
    """Print zip contents in a tree structure using system tree command.

    Args:
        zip_path: Path to zip file
        exclude_patterns: Optional patterns to exclude (e.g. "__pycache__|*.pyc")
    """
    print("\n" + "=" * 50)
    print(f"Zip file contents: {zip_path}")
    print("=" * 50)

    # Create temp directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Extract zip
        with ZipFile(zip_path) as zf:
            zf.extractall(temp_path)

        print_tree(temp_path, exclude_patterns)


def print_tree(temp_path: Path, exclude_patterns: Optional[str] = None):
    """Print zip contents in a tree structure using system tree command.

    Args:
        zip_path: Path to zip file
        exclude_patterns: Optional patterns to exclude (e.g. "__pycache__|*.pyc")
    """
    print("\n" + "=" * 50)
    print(f"path file contents: {temp_path}")
    print("=" * 50)

    # Use tree command
    try:
        cmd = ["tree", "-a", "--dirsfirst"]
        if exclude_patterns:
            cmd.extend(["-I", exclude_patterns])
        cmd.append(str(temp_path))

        subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print("Note: 'tree' command not found. Install it for better directory visualization.")
        # Simple fallback
        for item in sorted(temp_path.rglob("*")):
            rel_path = item.relative_to(temp_path)
            indent = "    " * (len(rel_path.parts) - 1)
            print(f"{indent}{'├── ' if item.is_file() else '└── '}{item.name}")
