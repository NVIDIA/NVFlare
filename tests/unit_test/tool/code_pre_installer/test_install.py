import json
import site
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

import pytest

from nvflare.tool.code_pre_installer.install import check_python_path, install_job_structure, setup_python_path

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


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace with job structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create job structure
        job_config = temp_path / "job_config"
        job_share = temp_path / "job_share"
        job_config.mkdir()
        job_share.mkdir()

        # Create meta.json
        meta = {"name": "test_job"}
        with open(job_config / "meta.json", "w") as f:
            json.dump(meta, f)

        # Create site directories
        site_dir = job_config / "app_site-1"
        site_dir.mkdir()
        custom_dir = site_dir / "custom"
        custom_dir.mkdir()

        # Create some test files
        (custom_dir / "test.py").write_text("print('test')")
        (custom_dir / "src").mkdir()
        (custom_dir / "src/client.py").write_text("class Client: pass")

        # Create shared resources
        shared_pkg = job_share / "pt"
        shared_pkg.mkdir()
        (shared_pkg / "__init__.py").touch()
        (shared_pkg / "utils.py").write_text("def helper(): pass")

        # Create zip file
        zip_path = temp_path / "job_structure.zip"
        with ZipFile(zip_path, "w") as zf:
            for item in job_config.rglob("*"):
                if item.is_file():
                    zf.write(item, item.relative_to(temp_path))
            for item in job_share.rglob("*"):
                if item.is_file():
                    zf.write(item, item.relative_to(temp_path))

        yield {"zip_path": zip_path, "temp_dir": temp_path, "job_name": "test_job"}


@pytest.fixture
def temp_install_dirs():
    """Create temporary installation directories."""
    with tempfile.TemporaryDirectory() as install_dir, tempfile.TemporaryDirectory() as share_dir:
        yield {"install_prefix": Path(install_dir), "share_location": Path(share_dir)}


@pytest.fixture
def temp_site_packages():
    """Create a temporary site-packages directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_site = Path(temp_dir)
        # Add temp directory to sys.path temporarily
        sys.path.insert(0, str(temp_site))
        # Mock site.getsitepackages to return our temp directory
        original_getsitepackages = site.getsitepackages
        site.getsitepackages = lambda: [str(temp_site)]

        yield temp_site

        # Cleanup
        sys.path.remove(str(temp_site))
        site.getsitepackages = original_getsitepackages


def test_install_job_structure(temp_workspace, temp_install_dirs):
    """Test basic job structure installation."""
    install_job_structure(
        temp_workspace["zip_path"], temp_install_dirs["install_prefix"], temp_install_dirs["share_location"], "site-1"
    )

    # Check job files
    job_dir = temp_install_dirs["install_prefix"] / temp_workspace["job_name"]
    assert (job_dir / "test.py").exists()
    assert (job_dir / "src/client.py").exists()

    # Check shared files
    share_dir = temp_install_dirs["share_location"]
    assert (share_dir / "pt/__init__.py").exists()
    assert (share_dir / "pt/utils.py").exists()


def test_install_with_default_app(temp_workspace, temp_install_dirs):
    """Test installation using default apps directory."""
    # Create default app structure
    job_config = temp_workspace["temp_dir"] / "job_config"
    default_app = job_config / "apps"
    default_app.mkdir()
    custom_dir = default_app / "custom"
    custom_dir.mkdir()
    (custom_dir / "default.py").write_text("print('default')")

    # Update zip file
    with ZipFile(temp_workspace["zip_path"], "w") as zf:
        # Include both job_config and job_share
        for item in temp_workspace["temp_dir"].rglob("*"):
            if item.is_file():
                zf.write(item, item.relative_to(temp_workspace["temp_dir"]))

    install_job_structure(
        temp_workspace["zip_path"],
        temp_install_dirs["install_prefix"],
        temp_install_dirs["share_location"],
        "nonexistent-site",  # Should fall back to default app
    )

    job_dir = temp_install_dirs["install_prefix"] / temp_workspace["job_name"]
    assert (job_dir / "default.py").exists()


def test_python_path_setup(temp_install_dirs, temp_site_packages):
    """Test Python path setup."""
    share_location = temp_install_dirs["share_location"]

    # Initial setup
    assert not check_python_path(share_location)
    setup_python_path(share_location)
    assert check_python_path(share_location)

    # Verify .pth file was created in our temp site-packages
    pth_file = temp_site_packages / "nvflare_shared.pth"
    assert pth_file.exists()
    assert pth_file.read_text().strip() == str(share_location)

    # Repeated setup should not change anything
    setup_python_path(share_location)
    assert check_python_path(share_location)


def test_invalid_job_structure(temp_workspace, temp_install_dirs):
    """Test handling of invalid job structure."""
    # Create invalid zip without required directories
    invalid_zip = temp_workspace["temp_dir"] / "invalid.zip"
    with ZipFile(invalid_zip, "w") as zf:
        zf.writestr("dummy.txt", "test")

    with pytest.raises(ValueError, match="Invalid job structure"):
        install_job_structure(
            invalid_zip, temp_install_dirs["install_prefix"], temp_install_dirs["share_location"], "site-1"
        )


def test_missing_meta_json(temp_workspace, temp_install_dirs):
    """Test handling of missing meta.json."""
    # Remove meta.json from the structure
    job_config = temp_workspace["temp_dir"] / "job_config"
    (job_config / "meta.json").unlink()

    # Update zip file
    with ZipFile(temp_workspace["zip_path"], "w") as zf:
        # Include both job_config and job_share
        for item in temp_workspace["temp_dir"].rglob("*"):
            if item.is_file():
                zf.write(item, item.relative_to(temp_workspace["temp_dir"]))

    with pytest.raises(ValueError, match="meta.json not found in job_config"):
        install_job_structure(
            temp_workspace["zip_path"],
            temp_install_dirs["install_prefix"],
            temp_install_dirs["share_location"],
            "site-1",
        )
