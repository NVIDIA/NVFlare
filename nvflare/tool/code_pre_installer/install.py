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
import site
import tempfile
from pathlib import Path
from zipfile import ZipFile


def parse_args():
    parser = argparse.ArgumentParser(description="Install NVFLARE job structure")
    parser.add_argument("--job-structure", required=True, help="Path to job structure zip file")
    parser.add_argument(
        "--install-prefix", default="/opt/nvflare/jobs", help="Installation prefix (default: /opt/nvflare/jobs)"
    )
    parser.add_argument(
        "--share-location", default="/opt/nvflare/share", help="Shared resources location (default: /opt/nvflare/share)"
    )
    parser.add_argument("--site-name", required=True, help="Target site name (e.g., site-1, server)")
    return parser.parse_args()


def check_python_path(share_location: Path) -> bool:
    """Check if share_location is already in Python path."""
    site_packages = site.getsitepackages()[0]
    pth_file = Path(site_packages) / "nvflare_shared.pth"

    if pth_file.exists():
        current_path = pth_file.read_text().strip()
        if Path(current_path) == share_location:
            return True
    return False


def setup_python_path(share_location: Path):
    """Setup Python path to include shared packages if not already set."""
    if check_python_path(share_location):
        print(f"Python path already includes {share_location}")
        return

    site_packages = site.getsitepackages()[0]
    pth_file = Path(site_packages) / "nvflare_shared.pth"

    print(f"Adding {share_location} to Python path...")
    with open(pth_file, "w") as f:
        f.write(str(share_location))


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


def install_job_structure(job_structure: Path, install_prefix: Path, share_location: Path, site_name: str):
    """Install NVFLARE job structure for a specific site."""

    # Check Python path once at the start
    needs_python_setup = not check_python_path(share_location)

    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Extract zip
        print("Extracting job structure...")
        with ZipFile(job_structure) as zf:
            zf.extractall(temp_path)

        # Install requirements if present
        requirements_file = temp_path / "requirements.txt"
        if requirements_file.exists():
            install_requirements(requirements_file)

        # Validate structure
        job_config = temp_path / "job_config"
        job_share = temp_path / "job_share"

        if not (job_config.exists() and job_share.exists()):
            raise ValueError("Invalid job structure: Missing job_config or job_share directory")

        # Read meta.json
        meta_file = job_config / "meta.json"
        if not meta_file.exists():
            raise ValueError("meta.json not found in job_config")

        with open(meta_file) as f:
            meta = json.load(f)
            job_name = meta.get("name")

        if not job_name:
            raise ValueError("Job name not found in meta.json")

        # Create installation directories
        job_dir = install_prefix / job_name
        job_dir.mkdir(parents=True, exist_ok=True)
        share_location.mkdir(parents=True, exist_ok=True)

        # Install site-specific custom code
        default_site_dir = job_config / "apps"
        site_dir = job_config / f"app_{site_name}"
        if not site_dir.exists():
            site_dir = default_site_dir

        if not site_dir.exists():
            raise ValueError(f"Site directory not found for {site_name}")

        custom_dir = site_dir / "custom"
        if custom_dir.exists():
            print(f"Installing custom code for site {site_name}...")
            for item in custom_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, job_dir)
                else:
                    shutil.copytree(item, job_dir / item.name, dirs_exist_ok=True)

        # Install shared resources
        if job_share.exists() and any(job_share.iterdir()):
            print("Installing shared resources...")
            for item in job_share.iterdir():
                if item.is_file():
                    shutil.copy2(item, share_location)
                else:
                    shutil.copytree(item, share_location / item.name, dirs_exist_ok=True)

            # Setup Python path only if needed
            if needs_python_setup:
                setup_python_path(share_location)

        print("\nInstallation completed successfully:")
        print(f"- Job files installed to: {job_dir}")
        print(f"- Shared files installed to: {share_location}")


def main():
    args = parse_args()
    try:
        install_job_structure(
            Path(args.job_structure), Path(args.install_prefix), Path(args.share_location), args.site_name
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
