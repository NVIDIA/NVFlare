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

from nvflare.tool.code_pre_installer.constants import (
    CUSTOM_DIR_NAME,
    DEFAULT_APPLICATION_INSTALL_DIR,
    DEFAULT_GENERIC_APP_NAME,
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


def install_app_code(application: Path, install_prefix: Path, site_name: str, target_shared_dir: Path):
    """Install NVFLARE application code for a specific site."""
    CUSTOM_DIR = Path(target_shared_dir)

    # Verify input zip file exists
    if not application.exists():
        raise FileNotFoundError(f"Application code zip not found: {application}")

    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Extract zip
        print("Extracting application code...")
        with ZipFile(application) as zf:
            zf.extractall(temp_path)

        # Install requirements if present
        requirements_file = temp_path / "requirements.txt"
        if requirements_file.exists():
            install_requirements(requirements_file)

        # Validate structure
        application_dir = temp_path / "application"
        application_share = temp_path / "application-share"

        if not (application_dir.exists() and application_share.exists()):
            raise ValueError("Invalid application code: Missing application or application-share directory")

        # Find all application directories under application/
        app_dirs = [d for d in application_dir.iterdir() if d.is_dir()]
        if not app_dirs:
            raise ValueError("No application directory found under application/")

        # Process each application directory
        for app_dir in app_dirs:
            app_name = app_dir.name

            # Read meta.json
            meta_file = app_dir / "meta.json"
            if not meta_file.exists():
                raise ValueError(f"meta.json not found in {app_name} directory")

            # Create installation directories
            install_dir = install_prefix / app_name
            install_dir.mkdir(parents=True, exist_ok=True)
            CUSTOM_DIR.mkdir(parents=True, exist_ok=True)

            # Install site-specific custom code
            default_site_dir = app_dir / DEFAULT_GENERIC_APP_NAME
            site_dir = app_dir / f"app_{site_name}"
            if not site_dir.exists():
                site_dir = default_site_dir

            if not site_dir.exists():
                raise ValueError(f"Site directory not found for {site_name}")

            custom_dir = site_dir / CUSTOM_DIR_NAME
            if custom_dir.exists():
                print(f"Installing custom code for site {site_name}...")
                for item in custom_dir.iterdir():
                    if item.is_file():
                        shutil.copy2(item, install_dir)
                    else:
                        shutil.copytree(item, install_dir / item.name, dirs_exist_ok=True)

            # Install shared resources
            if application_share.exists() and any(application_share.iterdir()):
                print("Installing shared resources...")
                for item in application_share.iterdir():
                    if item.is_file():
                        shutil.copy2(item, CUSTOM_DIR)
                    else:
                        shutil.copytree(item, CUSTOM_DIR / item.name, dirs_exist_ok=True)

            print("\nInstallation completed successfully:")
            print(f"- Application files installed to: {install_dir}")
            print(f"- Shared files installed to: {CUSTOM_DIR}")

            # Delete the original zip file after successful installation
            application.unlink()
            print(f"- Cleaned up application code zip: {application}")
