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

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict
from zipfile import ZipFile

from nvflare.tool.code_pre_installer.constants import (
    APPLICATION_CODE_DIR,
    APPLICATION_SHARED_CODE_DIR,
    CUSTOM_DIR_NAME,
    DEFAULT_APPLICATION_INSTALL_DIR,
    PYTHON_PATH_SHARED_DIR,
)


def define_pre_install_parser(cmd_name: str, sub_cmd):
    """Define parser for install command."""
    parser = sub_cmd.add_parser(cmd_name)
    parser.add_argument("-a", "--application", required=True, help="Path to application code zip file")
    parser.add_argument(
        "-p",
        "--install-prefix",
        default=DEFAULT_APPLICATION_INSTALL_DIR,
        help=f"Installation prefix (default: {DEFAULT_APPLICATION_INSTALL_DIR})",
    )
    parser.add_argument("-s", "--site-name", required=True, help="Target site name (e.g., site-1, server)")
    parser.add_argument(
        "-ts",
        "--target_shared_dir",
        default=PYTHON_PATH_SHARED_DIR,
        help=f"Target share path (default: {PYTHON_PATH_SHARED_DIR})",
    )
    parser.add_argument("-debug", "--debug", action="store_true", help="debug is on")
    parser.add_argument("-d", "--delete", action="store_true", help="delete the zip file after installation")
    return parser


def install_requirements(requirements_file: Path):
    """Install Python packages from requirements.txt."""
    if not requirements_file.exists():
        print("No requirements.txt found, skipping package installation")
        return

    print(f"Installing packages from {requirements_file}...")

    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], check=True)
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Failed to install requirements: {e}")


def _process_meta_json(meta_file: Path, site_name: str, base_dir: Path, job_name: str = "app") -> Dict[str, Path]:
    """Process meta.json file to find matching app directories.

    Args:
        meta_file: Path to meta.json file
        site_name: Target site name
        base_dir: Base directory containing app directories
        job_name: Name to use for the job (default: "app")

    Returns:
        Dictionary mapping job names to their app directory paths
    """
    matched_apps = {}

    try:
        with open(meta_file) as f:
            meta = json.load(f)

        deploy_map = meta.get("deploy_map", {})
        if deploy_map:
            for app_name, sites in deploy_map.items():
                site_app_dir = base_dir / app_name
                if site_name in sites and site_app_dir.exists():
                    matched_apps[job_name] = site_app_dir
                elif "@ALL" in sites:
                    matched_apps[job_name] = site_app_dir

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Error reading {meta_file}: {str(e)}")

    return matched_apps


def _find_app_dirs(application_dir: Path, site_name: str) -> Dict[str, Path]:
    """Find all appropriate app directories based on meta.json deployment maps.

    Args:
        application_dir: Base application directory (containing app directories)
        site_name: Target site name to find app directories for

    Returns:
        Dictionary mapping job names to their app directory paths

    Raises:
        ValueError: If no matching app directories found for site
    """
    matched_apps = {}
    print(f"Searching for app directories in {application_dir}...")

    for job_dir in [d for d in application_dir.iterdir() if d.is_dir()]:
        meta_file = job_dir / "meta.json"
        if meta_file.exists():
            matched_apps.update(_process_meta_json(meta_file, site_name, job_dir, job_dir.name))

    if not matched_apps:
        raise ValueError(f"No application directories found for site {site_name}")

    return matched_apps


def _install_site_specific_code(application_dir: Path, site_name: str, install_prefix: Path):
    """Find and install site-specific custom code directories under application_dir.

    Args:
        application_dir (Path): Root application directory containing site apps.
        site_name (str): Site name to filter app directories.
        install_prefix (Path): Destination prefix path for installation.
    """
    app_dirs = _find_app_dirs(application_dir, site_name)

    for job_name, site_app_dir in app_dirs.items():
        custom_dir = site_app_dir / CUSTOM_DIR_NAME

        if not custom_dir.exists() or not any(custom_dir.iterdir()):
            continue

        install_dir = install_prefix / job_name
        install_dir.mkdir(parents=True, exist_ok=True)

        for item in custom_dir.iterdir():
            dest = install_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)


def _install_shared_code(shared_dir: Path, target_shared_dir: Path):
    """Install shared application code from shared_dir to target_shared_dir.

    Args:
        shared_dir (Path): Source directory for shared code.
        target_shared_dir (Path): Destination directory for shared code.
    """
    if not shared_dir.exists() or not any(shared_dir.iterdir()):
        return  # Nothing to install

    target_dir = target_shared_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    for item in shared_dir.iterdir():
        dest = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)


def install_app_code(
    app_code_zip: Path, install_prefix: Path, site_name: str, target_shared_dir: str, delete: bool
) -> None:
    """Install application code from zip file.

    Args:
        app_code_zip: Path to application code zip file
        install_prefix: Installation prefix directory
        site_name: Target site name
        target_shared_dir: Target shared directory path
    """
    if not app_code_zip.exists():
        raise FileNotFoundError(f"Application code zip not found: {app_code_zip}")

    # Create temp directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Extract zip
        with ZipFile(app_code_zip) as zf:
            zf.extractall(temp_path)
        # Verify structure
        application_dir = temp_path / APPLICATION_CODE_DIR
        shared_dir = temp_path / APPLICATION_SHARED_CODE_DIR
        if not application_dir.exists() and not shared_dir.exists():
            raise ValueError(
                f"Invalid application code zip: Missing both {APPLICATION_CODE_DIR} and {APPLICATION_SHARED_CODE_DIR} directory."
            )

        # Install site specific code if present
        if application_dir.exists() and any(application_dir.iterdir()):
            _install_site_specific_code(application_dir, site_name, install_prefix)

        # Install shared code if present
        if shared_dir.exists() and any(shared_dir.iterdir()):
            _install_shared_code(shared_dir, Path(target_shared_dir))

        # Install requirements if present
        requirements = temp_path / "requirements.txt"
        if requirements.exists():
            install_requirements(requirements)

    # Cleanup
    print(f"Deleting {app_code_zip} after installation: {delete}")
    if delete:
        app_code_zip.unlink()


def install(args):
    """Run install command."""
    try:
        install_app_code(
            Path(args.application), Path(args.install_prefix), args.site_name, args.target_shared_dir, args.delete
        )
    except Exception as e:
        if args.debug:
            import traceback

            traceback.print_exc()
        raise RuntimeError(f"Failed to install application: {str(e)}")
