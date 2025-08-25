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
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

import pytest

from nvflare.cli import CMD_PRE_INSTALL
from nvflare.tool.code_pre_installer.constants import (
    CUSTOM_DIR_NAME,
    DEFAULT_APPLICATION_INSTALL_DIR,
    PYTHON_PATH_SHARED_DIR,
)
from nvflare.tool.code_pre_installer.install import (
    _find_app_dirs,
    _process_meta_json,
    define_pre_install_parser,
    install_app_code,
    install_requirements,
)
from nvflare.tool.code_pre_installer.prepare import prepare_app_code
from nvflare.utils.zip_utils import print_zip_tree


@pytest.fixture
def mock_custom_dir(tmp_path):
    """Mock /local/custom directory for testing."""
    custom_dir = tmp_path / "local" / "custom"
    custom_dir.mkdir(parents=True)

    # Mock Path("/local/custom") to return our test directory
    def mock_path(*args, **kwargs):
        if args == ("/local/custom",):
            return custom_dir
        return Path(*args, **kwargs)

    with patch("nvflare.tool.code_pre_installer.install.Path", mock_path):
        yield custom_dir


def create_test_code(tmp_path):
    """Create a test application code package."""
    # Create directories
    application = tmp_path / "application"
    app_dir = application / "test_app"
    application_share = tmp_path / "application-share"
    application.mkdir()
    app_dir.mkdir()
    application_share.mkdir()

    # Create meta.json with proper deploy_map
    meta = {"deploy_map": {"app_site-1": ["site-1"]}}  # Add deploy_map
    with open(app_dir / "meta.json", "w") as f:
        f.write(json.dumps(meta))

    # Create site directories
    site_dir = app_dir / "app_site-1"
    site_dir.mkdir()
    custom_dir = site_dir / "custom"
    custom_dir.mkdir()

    # Create test files
    (custom_dir / "test.py").write_text("print('test')")
    (application_share / "shared.py").write_text("print('shared')")

    # Create zip file
    zip_path = tmp_path / "test_code.zip"
    with ZipFile(zip_path, "w") as zf:
        for path in tmp_path.rglob("*"):
            if path != zip_path:
                zf.write(path, path.relative_to(tmp_path))

    return zip_path


def create_test_app_structure(base_path: Path, meta_content: dict):
    """Helper to create test application structure."""
    app_dir = base_path / "application"
    app_dir.mkdir(parents=True, exist_ok=True)

    job_dir = app_dir / meta_content["name"]
    job_dir.mkdir(exist_ok=True)

    # Write meta.json
    with open(job_dir / "meta.json", "w") as f:
        json.dump(meta_content, f)

    # Create app directories from deploy_map
    for app_name in meta_content["deploy_map"]:
        app_path = job_dir / app_name
        app_path.mkdir(exist_ok=True)
        custom_dir = app_path / CUSTOM_DIR_NAME
        custom_dir.mkdir(exist_ok=True)
        (custom_dir / "test.py").write_text("test code")

    return job_dir


def test_install_app_code(tmp_path, mock_custom_dir):
    """Test application code installation."""
    zip_path = create_test_code(tmp_path)
    install_prefix = tmp_path / "install"
    install_app_code(zip_path, install_prefix, "site-1", mock_custom_dir, False)


def test_cleanup_nonexistent_zip(tmp_path):
    """Test cleanup of nonexistent zip file."""
    with pytest.raises(FileNotFoundError):
        install_app_code(
            tmp_path / "nonexistent.zip",
            tmp_path / "install",
            "site-1",
            str(tmp_path / "shared"),
            True,  # Add delete parameter
        )


def test_invalid_structure(tmp_path):
    """Test invalid application structure."""
    zip_path = tmp_path / "test.zip"
    with ZipFile(zip_path, "w") as zf:
        zf.writestr("test.txt", "test")

    with pytest.raises(ValueError):
        install_app_code(
            zip_path, tmp_path / "install", "site-1", str(tmp_path / "shared"), False  # Add delete parameter
        )


def test_missing_meta(tmp_path, mock_custom_dir):
    """Test installation with missing meta.json."""
    # Create structure without meta.json
    app_dir = tmp_path / "application" / "test_app"
    app_dir.mkdir(parents=True)
    shared_dir = tmp_path / "application-share"
    shared_dir.mkdir()

    zip_path = tmp_path / "test.zip"
    with ZipFile(zip_path, "w") as zf:
        for path in tmp_path.rglob("*"):
            if path != zip_path:
                zf.write(path, path.relative_to(tmp_path))

    with pytest.raises(ValueError, match="No application directories found for site"):
        install_app_code(zip_path, tmp_path, "site-1", mock_custom_dir, False)


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def mock_zip_structure(temp_dir):
    # Create a mock zip structure
    app_dir = temp_dir / "application"
    app_share = temp_dir / "application-share"
    app_dir.mkdir()
    app_share.mkdir()

    # Create test app directory
    test_app = app_dir / "test_app"
    test_app.mkdir()

    # Create meta.json
    with open(test_app / "meta.json", "w") as f:
        f.write("{}")

    # Create site directories
    site_specific = test_app / "app_site-1"
    site_specific.mkdir()
    custom_dir = site_specific / CUSTOM_DIR_NAME
    custom_dir.mkdir()

    # Create some test files
    (custom_dir / "test_file.py").write_text("test content")
    (app_share / "shared_file.py").write_text("shared content")

    return temp_dir


def test_parse_args():
    """Test argument parsing for pre-install command."""
    mock_sub_cmd = MagicMock()
    mock_parser = MagicMock()
    mock_sub_cmd.add_parser.return_value = mock_parser

    parser = define_pre_install_parser(CMD_PRE_INSTALL, mock_sub_cmd)

    # Verify that the parser was created with correct arguments
    mock_sub_cmd.add_parser.assert_called_once_with(CMD_PRE_INSTALL)

    # Verify that all required arguments were added
    mock_parser.add_argument.assert_any_call(
        "-a", "--application", required=True, help="Path to application code zip file"
    )
    mock_parser.add_argument.assert_any_call(
        "-p",
        "--install-prefix",
        default=DEFAULT_APPLICATION_INSTALL_DIR,
        help="Installation prefix (default: /opt/nvflare/apps)",
    )
    mock_parser.add_argument.assert_any_call(
        "-s", "--site-name", required=True, help="Target site name (e.g., site-1, server)"
    )
    mock_parser.add_argument.assert_any_call(
        "-ts",
        "--target_shared_dir",
        default=PYTHON_PATH_SHARED_DIR,
        help=f"Target share path (default: {PYTHON_PATH_SHARED_DIR})",
    )
    mock_parser.add_argument.assert_any_call("-debug", "--debug", action="store_true", help="debug is on")


def test_install_requirements_no_file(temp_dir):
    requirements_file = temp_dir / "requirements.txt"
    install_requirements(requirements_file)
    # Should not raise any exception


@patch("subprocess.run")
def test_install_requirements_success(mock_run, temp_dir):
    requirements_file = temp_dir / "requirements.txt"
    requirements_file.write_text("pytest==7.0.0")

    install_requirements(requirements_file)
    mock_run.assert_called_once_with([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], check=True)


@patch("subprocess.run")
def test_install_requirements_failure(mock_run, temp_dir):
    requirements_file = temp_dir / "requirements.txt"
    requirements_file.write_text("invalid-package")

    mock_run.side_effect = subprocess.CalledProcessError(1, "pip install")

    with pytest.raises(ValueError, match="Failed to install requirements"):
        install_requirements(requirements_file)


def test_install_app_code_missing_zip(tmp_path, mock_custom_dir):
    """Test installation with missing zip file."""
    with pytest.raises(FileNotFoundError):
        install_app_code(tmp_path / "nonexistent.zip", tmp_path / "install", "site-1", mock_custom_dir, False)


class TestInstall:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def create_meta_json(self, path: Path, meta_json: dict = None):
        with open(path / "meta.json", "w") as f:
            # Create meta.json in the correct location
            meta_content = {
                "name": "fedavg",
                "min_clients": 1,
                "deploy_map": {"app_server": ["server"], "app_site-1": ["site-1"]},
            }
            if not meta_json:
                meta_json = meta_content
            json.dump(meta_json, f)

    def test_process_meta_json(self, temp_dir):
        # Setup
        base_dir = temp_dir / "app"
        base_dir.mkdir()
        app_dir = base_dir / "app_site-1"
        app_dir.mkdir()
        self.create_meta_json(base_dir)

        # Test
        result = _process_meta_json(base_dir / "meta.json", "site-1", base_dir)
        assert "app" in result
        assert result["app"] == app_dir

    def test_find_app_dirs_case1(self, temp_dir):
        """Test meta.json directly under application directory."""
        # Setup
        app_dir = temp_dir / "application"
        app_dir.mkdir(exist_ok=True)
        job_dir = app_dir / "fedavg"
        job_dir.mkdir(exist_ok=True)
        app_site_dir = job_dir / "app_site-1"
        app_site_dir.mkdir(exist_ok=True)
        self.create_meta_json(job_dir)

        # Test
        result = _find_app_dirs(app_dir, "site-1")
        assert "fedavg" in result
        assert result["fedavg"] == app_site_dir

    def test_find_app_dirs_case2(self, temp_dir):
        """Test meta.json in subdirectories."""
        # Setup
        app_dir = temp_dir / "application"
        app_dir.mkdir(exist_ok=True)
        job_dir = app_dir / "fedavg"
        job_dir.mkdir(exist_ok=True)
        site_dir = job_dir / "app_site-1"
        site_dir.mkdir(exist_ok=True)
        self.create_meta_json(job_dir)

        # Test
        result = _find_app_dirs(app_dir, "site-1")
        assert "fedavg" in result
        assert result["fedavg"] == site_dir

    def test_find_app_dirs_case3(self, temp_dir):
        """Test meta.json in subdirectories."""
        # Setup
        app_dir = temp_dir / "application"
        app_dir.mkdir(exist_ok=True)
        job_dir1 = app_dir / "fedavg"
        job_dir1.mkdir(exist_ok=True)
        site_dir1 = job_dir1 / "app_site-1"
        site_dir1.mkdir(exist_ok=True)
        self.create_meta_json(job_dir1)

        job_dir2 = app_dir / "fedavg2"
        job_dir2.mkdir(exist_ok=True)
        site_dir2 = job_dir2 / "app_site-1"
        site_dir2.mkdir(exist_ok=True)
        self.create_meta_json(job_dir2)

        # Test
        result = _find_app_dirs(app_dir, "site-1")
        assert "fedavg" in result
        assert result["fedavg"] == site_dir1

        result = _find_app_dirs(app_dir, "site-1")
        assert "fedavg2" in result
        assert result["fedavg2"] == site_dir2

    def test_find_app_dirs_all_sites(self, temp_dir):
        """Test @ALL in deployment map."""
        # Setup
        app_dir = temp_dir / "application"
        app_dir.mkdir()
        job_dir = app_dir / "fedavg"
        job_dir.mkdir()
        site_dir = job_dir / "app"
        site_dir.mkdir()
        # Create meta.json in the correct location
        meta_content = {"name": "fedavg", "min_clients": 1, "deploy_map": {"app": ["@ALL"]}}
        self.create_meta_json(job_dir, meta_content)

        # Test
        result = _find_app_dirs(app_dir, "any-site")
        assert "fedavg" in result
        assert result["fedavg"] == site_dir

    def test_find_app_dirs_no_match(self, temp_dir):
        """Test no matching directories found."""
        # Setup
        app_dir = temp_dir / "application"
        app_dir.mkdir()
        self.create_meta_json(app_dir)

        # Test
        with pytest.raises(ValueError, match="No application directories found for site"):
            _find_app_dirs(app_dir, "site-1")

    def test_install_app_code(self, temp_dir):
        """Test full installation process."""
        # Setup source structure
        app_dir = temp_dir / "src/application"
        app_dir.mkdir(parents=True)
        job_dir = app_dir / "fedavg"
        job_dir.mkdir()
        site_dir = job_dir / "app_site-1"
        site_dir.mkdir()
        custom_dir = site_dir / "custom"
        custom_dir.mkdir()
        (custom_dir / "test.py").write_text("test")
        self.create_meta_json(job_dir)

        # Create zip file
        zip_path = temp_dir / "application.zip"
        prepare_app_code(job_dir, temp_dir)
        print_zip_tree(zip_path)
        # Setup install paths
        install_prefix = temp_dir / "install"
        shared_dir = temp_dir / "shared"

        # Test installation
        install_app_code(zip_path, install_prefix, "site-1", str(shared_dir), False)  # Add delete parameter

        # Verify installation
        assert (install_prefix / "fedavg/test.py").exists()
        assert (install_prefix / "fedavg/test.py").read_text() == "test"

    def test_install_app_code_with_requirements(self, temp_dir):
        """Test installation with requirements.txt."""
        # Setup job directory
        jobs_dir = temp_dir / "jobs"
        jobs_dir.mkdir(parents=True)
        job_dir = jobs_dir / "fedavg"
        job_dir.mkdir()
        # Create app structure
        app_dir = job_dir / "app_site-1"
        app_dir.mkdir()
        custom_dir = app_dir / "custom"
        custom_dir.mkdir()
        (custom_dir / "test.py").write_text("test")

        self.create_meta_json(job_dir)

        # Create requirements.txt
        req_file = jobs_dir / "requirements.txt"
        req_file.write_text("pytest")

        # Use prepare_app_code to create package
        output_dir = temp_dir / "prepare"
        prepare_app_code(job_dir, output_dir, None, req_file)  # No shared lib
        # Print zip contents
        print("Printing zip contents at ", output_dir / "application.zip")
        print_zip_tree(output_dir / "application.zip")

        # Test installation
        install_app_code(
            output_dir / "application.zip", temp_dir / "install", "site-1", str(temp_dir / "shared"), False
        )
        assert (temp_dir / "install/fedavg/test.py").exists()

    def test_install_app_code_success(self, temp_dir):
        """Test successful installation."""
        # Setup job directory
        jobs_dir = temp_dir / "jobs"
        jobs_dir.mkdir(parents=True)
        job_dir = jobs_dir / "fedavg"
        job_dir.mkdir()
        # Create app structure
        app_dir = job_dir / "app_site-1"
        app_dir.mkdir()
        custom_dir = app_dir / "custom"
        custom_dir.mkdir()
        (custom_dir / "test.py").write_text("test")

        self.create_meta_json(job_dir)

        # Use prepare_app_code to create package
        output_dir = temp_dir / "prepare"
        output_dir.mkdir(parents=True, exist_ok=True)

        prepare_app_code(job_dir, output_dir, None)  # No shared lib

        # Verify zip structure
        with ZipFile(output_dir / "application.zip") as zf:
            files = zf.namelist()
            assert "application/fedavg/meta.json" in files
            assert "application/fedavg/app_site-1/custom/test.py" in files

        # Test installation
        install_prefix = temp_dir / "install"
        install_app_code(output_dir / "application.zip", install_prefix, "site-1", str(temp_dir / "shared"), False)

        assert (install_prefix / "fedavg/test.py").exists()

    def test_install_app_code_invalid_structure(self, temp_dir):
        """Test installation with invalid structure."""
        # Create invalid structure (missing application directory)
        temp_path = temp_dir / "temp"
        temp_path.mkdir()

        # Creates application-code
        code_dir = temp_path / "application-code"
        code_dir.mkdir()

        zip_path = temp_dir / "test.zip"
        with ZipFile(zip_path, "w") as zf:
            for path in temp_path.rglob("*"):
                zf.write(path, path.relative_to(temp_path))

        with pytest.raises(
            ValueError, match="Invalid application code zip: Missing both application and application-share directory."
        ):
            install_app_code(zip_path, temp_dir, "site-1", str(temp_dir / "shared"), False)

    def test_install_app_code_missing_site(self, temp_dir):
        """Test installation with missing site directory."""
        # Create application directory with meta.json for different site
        app_dir = temp_dir / "application"
        app_dir.mkdir(parents=True)
        meta_content = {"deploy_map": {"app_site-2": ["site-2"]}}  # Different site
        self.create_meta_json(app_dir, meta_content)

        # Create zip
        zip_path = temp_dir / "test.zip"
        with ZipFile(zip_path, "w") as zf:
            # Add application directory
            for item in app_dir.rglob("*"):
                if item.is_file():
                    zf.write(item, item.relative_to(temp_dir))

        # Test installation - should fail because site-1 not in deploy_map
        with pytest.raises(ValueError, match="No application directories found for site site-1"):
            install_app_code(zip_path, temp_dir / "install", "site-1", str(temp_dir / "shared"), False)


@pytest.fixture(autouse=True)
def cleanup():
    # Setup
    yield
    # Cleanup
    if Path(PYTHON_PATH_SHARED_DIR).exists():
        shutil.rmtree(PYTHON_PATH_SHARED_DIR)


def test_find_app_dirs_site_specific(tmp_path):
    """Test finding app directories with site-specific deployment map."""
    meta_content = {
        "name": "job1",
        "deploy_map": {"app_server": ["server"], "app_site1": ["site-1"], "app_site2": ["site-2"]},
    }
    create_test_app_structure(tmp_path, meta_content)

    app_dirs = _find_app_dirs(tmp_path / "application", "site-1")
    assert len(app_dirs) == 1
    assert "job1" in app_dirs
    assert app_dirs["job1"].name == "app_site1"


def test_find_app_dirs_all_sites(tmp_path):
    """Test finding app directories with @ALL deployment map."""
    meta_content = {"name": "job1", "deploy_map": {"custom_app": ["@ALL"]}}
    create_test_app_structure(tmp_path, meta_content)

    app_dirs = _find_app_dirs(tmp_path / "application", "any-site")
    assert len(app_dirs) == 1
    assert "job1" in app_dirs
    assert app_dirs["job1"].name == "custom_app"


def test_find_app_dirs_multiple_jobs(tmp_path):
    """Test finding app directories with multiple jobs."""
    # Create first job
    meta1 = {"name": "job1", "deploy_map": {"app_site1": ["site-1"]}}
    create_test_app_structure(tmp_path, meta1)

    # Create second job
    meta2 = {"name": "job2", "deploy_map": {"app": ["@ALL"]}}
    create_test_app_structure(tmp_path, meta2)

    app_dirs = _find_app_dirs(tmp_path / "application", "site-1")
    assert len(app_dirs) == 2
    assert "job1" in app_dirs
    assert "job2" in app_dirs
    assert app_dirs["job1"].name == "app_site1"
    assert app_dirs["job2"].name == "app"


def test_install_app_code_multiple_jobs(tmp_path, mock_custom_dir):
    """Test installing app code from multiple jobs."""
    # Create application structure
    temp_path = tmp_path / "temp"
    temp_path.mkdir()

    # Create first job
    meta1 = {"name": "job1", "deploy_map": {"app_site1": ["site-1"]}}
    create_test_app_structure(temp_path, meta1)

    # Create second job
    meta2 = {"name": "job2", "deploy_map": {"custom_app": ["@ALL"]}}
    create_test_app_structure(temp_path, meta2)

    # Create shared directory
    shared_dir = temp_path / "application-share"
    shared_dir.mkdir()
    (shared_dir / "shared.py").write_text("# Shared code")

    # Create zip file
    zip_path = tmp_path / "test.zip"
    with ZipFile(zip_path, "w") as zf:
        for path in temp_path.rglob("*"):
            zf.write(path, path.relative_to(temp_path))

    # Install app code
    install_dir = tmp_path / "install"
    install_app_code(zip_path, install_dir, "site-1", str(mock_custom_dir), False)

    # Verify installations
    assert (install_dir / "job1" / "test.py").exists()
    assert (install_dir / "job2" / "test.py").exists()
    assert (Path(mock_custom_dir) / "shared.py").exists()
