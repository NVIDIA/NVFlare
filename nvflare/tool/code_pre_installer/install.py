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
import tempfile
from pathlib import Path
from typing import Dict
from zipfile import ZipFile

from nvflare.tool.code_pre_installer.constants import (
    CUSTOM_DIR_NAME,
    DEFAULT_APPLICATION_INSTALL_DIR,
    PYTHON_PATH_SHARED_DIR,
)


def define_pre_install_parser(cmd_name, sub_cmd):
    parser = sub_cmd.add_parser(cmd_name)
    parser.add_argument("-a", "--application", required=True, help="Path to application code zip file")
    parser.add_argument(
        "-p",
        "--install-prefix",
        default=DEFAULT_APPLICATION_INSTALL_DIR,
        help="Installation prefix (default: /opt/nvflare/apps)",
    )
    parser.add_argument("-s", "--site-name", required=True, help="Target site name (e.g., site-1, server)")
    parser.add_argument(
        "-ts",
        "--target_shared_dir",
        default=PYTHON_PATH_SHARED_DIR,
        help=f"Target share path (default: {PYTHON_PATH_SHARED_DIR})",
    )
    parser.add_argument("-debug", "--debug", action="store_true", help="debug is on")
    return parser


def install_requirements(requirements_file: Path):
    """Install Python packages from requirements.txt."""
    if not requirements_file.exists():
        print("No requirements.txt found, skipping package installation")
        return

    print(f"Installing packages from {requirements_file}...")
    import subprocess

    try:
        subprocess.run(["pip", "install", "-r", str(requirements_file)], check=True)
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Failed to install requirements: {e}")


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
    # Find all app directories that have meta.json
    job_dirs = [d for d in application_dir.iterdir() if d.is_dir() and (d / "meta.json").exists()]
    if not job_dirs:
        raise ValueError("No application directories with meta.json found")

    # Dictionary to store job_name -> app_dir mappings
    matched_apps = {}

    # Search through all app directories for matching deployments
    for job_dir in job_dirs:
        try:
            with open(job_dir / "meta.json") as f:
                meta = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue  # Skip invalid meta.json

        deploy_map = meta.get("deploy_map", {})
        if not deploy_map:
            continue  # Skip if no deploy map

        # Case 1: Site-specific format (app_site-1 -> [site-1, ...])
        for app_name, sites in deploy_map.items():
            if site_name in sites:
                site_app_dir = job_dir / app_name
                if site_app_dir.exists():
                    matched_apps[job_dir.name] = site_app_dir

        # Case 2: Default format (any_app_name -> [@ALL])
        for app_name, sites in deploy_map.items():
            if "@ALL" in sites:
                default_app_dir = job_dir / app_name
                if default_app_dir.exists():
                    matched_apps[job_dir.name] = default_app_dir

    if not matched_apps:
        raise ValueError(f"No application directories found for site {site_name}")

    return matched_apps


def install_app_code(app_code_zip: Path, install_prefix: Path, site_name: str, target_shared_dir: str) -> None:
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
        application_dir = temp_path / "application"
        shared_dir = temp_path / "application-share"
        if not application_dir.exists() or not any(application_dir.iterdir()):
            raise ValueError("Invalid application code: Missing application or application-share directory")

        # Find all appropriate app directories based on site name
        app_dirs = _find_app_dirs(application_dir, site_name)

        # Install site-specific code for each app
        for job_name, site_app_dir in app_dirs.items():
            custom_dir = site_app_dir / CUSTOM_DIR_NAME
            if custom_dir.exists():
                # Create install directory
                install_dir = install_prefix / job_name
                install_dir.mkdir(parents=True, exist_ok=True)

                # Copy custom code
                for item in custom_dir.iterdir():
                    dest = install_dir / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, dest)

        # Install shared code if present
        if shared_dir.exists() and any(shared_dir.iterdir()):
            target_dir = Path(target_shared_dir)
            target_dir.mkdir(parents=True, exist_ok=True)

            for item in shared_dir.iterdir():
                dest = target_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)

        # Install requirements if present
        requirements = temp_path / "requirements.txt"
        if requirements.exists():
            install_requirements(requirements)

    # Cleanup
    app_code_zip.unlink()
