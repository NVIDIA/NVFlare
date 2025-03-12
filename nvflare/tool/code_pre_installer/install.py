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


import argparse
import json
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile


def parse_args():
    parser = argparse.ArgumentParser(description="pre-Install application code and libs")
    parser.add_argument("--app-code", required=True, help="Path to application code zip file")
    parser.add_argument(
        "--install-prefix", default="/opt/nvflare/apps", help="Installation prefix (default: /opt/nvflare/apps)"
    )
    parser.add_argument("--site-name", required=True, help="Target site name (e.g., site-1, server)")
    return parser.parse_args()


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


def install_app_code(app_code: Path, install_prefix: Path, site_name: str):
    """Install NVFLARE application code for a specific site."""
    CUSTOM_DIR = Path("/local/custom")

    # Verify input zip file exists
    if not app_code.exists():
        raise FileNotFoundError(f"Application code zip not found: {app_code}")

    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Extract zip
        print("Extracting application code...")
        with ZipFile(app_code) as zf:
            zf.extractall(temp_path)

        # Install requirements if present
        requirements_file = temp_path / "requirements.txt"
        if requirements_file.exists():
            install_requirements(requirements_file)

        # Validate structure
        app_code_dir = temp_path / "app_code"
        app_share = temp_path / "app_share"

        if not (app_code_dir.exists() and app_share.exists()):
            raise ValueError("Invalid application code: Missing app_code or app_share directory")

        # Read meta.json
        meta_file = app_code_dir / "meta.json"
        if not meta_file.exists():
            raise ValueError("meta.json not found in app_code")

        with open(meta_file) as f:
            meta = json.load(f)
            app_name = meta.get("name")

        if not app_name:
            raise ValueError("Application name not found in meta.json")

        # Create installation directories
        app_dir = install_prefix / app_name
        app_dir.mkdir(parents=True, exist_ok=True)
        CUSTOM_DIR.mkdir(parents=True, exist_ok=True)

        # Install site-specific custom code
        default_site_dir = app_code_dir / "apps"
        site_dir = app_code_dir / f"app_{site_name}"
        if not site_dir.exists():
            site_dir = default_site_dir

        if not site_dir.exists():
            raise ValueError(f"Site directory not found for {site_name}")

        custom_dir = site_dir / "custom"
        if custom_dir.exists():
            print(f"Installing custom code for site {site_name}...")
            for item in custom_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, app_dir)
                else:
                    shutil.copytree(item, app_dir / item.name, dirs_exist_ok=True)

        # Install shared resources
        if app_share.exists() and any(app_share.iterdir()):
            print("Installing shared resources...")
            for item in app_share.iterdir():
                if item.is_file():
                    shutil.copy2(item, CUSTOM_DIR)
                else:
                    shutil.copytree(item, CUSTOM_DIR / item.name, dirs_exist_ok=True)

        print("\nInstallation completed successfully:")
        print(f"- Application files installed to: {app_dir}")
        print(f"- Shared files installed to: {CUSTOM_DIR}")

        # Delete the original zip file after successful installation
        app_code.unlink()
        print(f"- Cleaned up application code zip: {app_code}")


def main():
    args = parse_args()
    try:
        install_app_code(Path(args.app_code), Path(args.install_prefix), args.site_name)
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
