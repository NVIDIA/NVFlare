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

import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile


def define_prepare_parser(cmd_name: str, sub_cmd):
    """Define parser for prepare command."""
    parser = sub_cmd.add_parser(cmd_name)
    parser.add_argument("-j", "--job", required=True, help="Job folder path (e.g., jobs/fedavg)")
    parser.add_argument(
        "-o", "--output", required=True, default="/tmp/application/prepare", help="Output directory for application.zip"
    )
    parser.add_argument("-s", "--shared", help="Optional shared library folder")
    parser.add_argument("-r", "--requirements", type=Path, help="Optional requirements.txt file")
    parser.add_argument("-debug", "--debug", action="store_true", help="debug is on")

    return parser


def prepare_app_code(job_folder: Path, output_dir: Path, shared_lib: Path = None, requirements: Path = None) -> None:
    """Package NVFLARE application code for pre-installation.

    Args:
        job_folder: Path to job folder (e.g., jobs/fedavg)
        output_dir: Output directory for application.zip
        shared_lib: Optional path to shared library folder
        requirements: Optional path to requirements.txt file
    """
    if not job_folder.exists():
        raise FileNotFoundError(f"Job folder not found: {job_folder}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        app_dir = temp_path / "application"
        share_dir = temp_path / "application-share"

        app_dir.mkdir()
        share_dir.mkdir()
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Packaging application from {job_folder}...")

        # Copy job files
        shutil.copytree(job_folder, app_dir / job_folder.name)

        # Copy shared lib if specified
        if shared_lib:
            if not shared_lib.exists():
                raise FileNotFoundError(f"Shared library folder not found: {shared_lib}")
            print(f"Including shared library from {shared_lib}...")
            shutil.copytree(shared_lib, share_dir / shared_lib.name)

        # Copy requirements.txt if specified
        if requirements:
            if not requirements.exists():
                raise FileNotFoundError(f"Requirements file not found: {requirements}")
            print(f"Including requirements from {requirements}...")
            shutil.copy2(requirements, temp_path / "requirements.txt")

        # Create zip file
        zip_file = output_dir / "application.zip"
        with ZipFile(zip_file, "w") as zf:
            # Add application directory
            for item in app_dir.rglob("*"):
                if item.is_file():
                    zf.write(item, item.relative_to(temp_path))

            # Add application-share directory
            for item in share_dir.rglob("*"):
                if item.is_file():
                    zf.write(item, item.relative_to(temp_path))

            # Add requirements.txt if present
            req_file = temp_path / "requirements.txt"
            if req_file.exists():
                zf.write(req_file, req_file.relative_to(temp_path))

        print(f"Application package created: {zip_file}")


def prepare(args):
    """Run prepare command."""
    try:
        prepare_app_code(
            Path(args.job),
            Path(args.output),
            Path(args.shared) if args.shared else None,
            Path(args.requirements) if args.requirements else None,
        )
    except Exception as e:
        if args.debug:
            import traceback

            traceback.print_exc()
        raise RuntimeError(f"Failed to package application: {str(e)}")
