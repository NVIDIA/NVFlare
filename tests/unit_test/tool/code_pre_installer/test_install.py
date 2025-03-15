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
from nvflare.tool.code_pre_installer.install import define_pre_install_parser, install_app_code, install_requirements


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

    # Create meta.json
    meta = {"name": "test_app"}
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


def test_install_app_code(tmp_path, mock_custom_dir):
    """Test application code installation."""
    zip_path = create_test_code(tmp_path)
    install_prefix = tmp_path / "install"

    install_app_code(zip_path, install_prefix, "site-1", mock_custom_dir)

    assert (install_prefix / "test_app").exists()
    assert (install_prefix / "test_app" / "test.py").exists()
    assert (mock_custom_dir / "shared.py").exists()
    # Verify zip file is deleted after installation
    assert not zip_path.exists(), "Zip file should be deleted after successful installation"


def test_cleanup_nonexistent_zip(tmp_path, mock_custom_dir):
    """Test cleanup handling when zip file is already deleted."""
    zip_path = create_test_code(tmp_path)
    install_prefix = tmp_path / "install"

    # Delete zip file before installation
    zip_path.unlink()

    # Should handle missing zip file gracefully
    with pytest.raises(FileNotFoundError):
        install_app_code(zip_path, install_prefix, "site-1", mock_custom_dir)


def test_invalid_structure(tmp_path, mock_custom_dir):
    """Test installation with invalid structure."""
    invalid_zip = tmp_path / "invalid.zip"
    with ZipFile(invalid_zip, "w") as zf:
        zf.writestr("dummy.txt", "not a valid structure")

    with pytest.raises(
        ValueError, match="Invalid application code: Missing application or application-share directory"
    ):
        install_app_code(invalid_zip, tmp_path, "site-1", mock_custom_dir)


def test_missing_meta(tmp_path, mock_custom_dir):
    """Test installation with missing meta.json."""
    with ZipFile(tmp_path / "invalid.zip", "w") as zf:
        zf.writestr("application/test_app/empty", "")
        zf.writestr("application-share/empty", "")

    with pytest.raises(ValueError, match="meta.json not found"):
        install_app_code(tmp_path / "invalid.zip", tmp_path, "site-1", mock_custom_dir)


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
    mock_run.assert_called_once_with(["pip", "install", "-r", str(requirements_file)], check=True)


@patch("subprocess.run")
def test_install_requirements_failure(mock_run, temp_dir):
    requirements_file = temp_dir / "requirements.txt"
    requirements_file.write_text("invalid-package")

    mock_run.side_effect = subprocess.CalledProcessError(1, "pip install")

    with pytest.raises(ValueError, match="Failed to install requirements"):
        install_requirements(requirements_file)


def test_install_app_code_missing_zip(temp_dir, mock_custom_dir):
    with pytest.raises(FileNotFoundError):
        install_app_code(Path(temp_dir) / "nonexistent.zip", Path(temp_dir) / "install", "site-1", mock_custom_dir)


def test_install_app_code_success(tmp_path, mock_custom_dir):
    """Test successful application code installation."""
    # Create test zip
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        application = temp_path / "application"
        app_dir = application / "test_app"
        application_share = temp_path / "application-share"
        application.mkdir()
        app_dir.mkdir()
        application_share.mkdir()

        # Create meta.json
        meta = {"name": "test_app"}
        with open(app_dir / "meta.json", "w") as f:
            json.dump(meta, f)

        # Create site directories
        site_dir = app_dir / "app_site-1"
        site_dir.mkdir(parents=True)
        custom_dir = site_dir / "custom"
        custom_dir.mkdir()

        # Create test files
        (custom_dir / "test.py").write_text("print('test')")
        (application_share / "shared.py").write_text("print('shared')")

        # Create zip file
        zip_path = tmp_path / "test.zip"
        with ZipFile(zip_path, "w") as zf:
            for path in temp_path.rglob("*"):
                zf.write(path, path.relative_to(temp_path))

    # Test installation
    install_app_code(zip_path, tmp_path, "site-1", mock_custom_dir)

    # Verify installation
    assert (tmp_path / "test_app").exists()
    assert (tmp_path / "test_app" / "test.py").exists()
    assert (mock_custom_dir / "shared.py").exists()
    assert not zip_path.exists()


def test_install_app_code_invalid_structure(tmp_path, mock_custom_dir):
    """Test installation with invalid structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        application = temp_path / "application"
        application_share = temp_path / "application-share"
        application.mkdir()
        application_share.mkdir()

        # Create zip file without meta.json
        zip_path = tmp_path / "test.zip"
        with ZipFile(zip_path, "w") as zf:
            for path in temp_path.rglob("*"):
                zf.write(path, path.relative_to(temp_path))

    with pytest.raises(ValueError, match="No application directory found under application"):
        install_app_code(zip_path, tmp_path, "site-1", mock_custom_dir)


def test_install_app_code_missing_site(tmp_path, mock_custom_dir):
    """Test installation with missing site directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        application = temp_path / "application"
        app_dir = application / "test_app"
        application_share = temp_path / "application-share"
        application.mkdir()
        app_dir.mkdir()
        application_share.mkdir()

        # Create meta.json
        meta = {"name": "test_app"}
        with open(app_dir / "meta.json", "w") as f:
            json.dump(meta, f)

        # Create zip file without site directory
        zip_path = tmp_path / "test.zip"
        with ZipFile(zip_path, "w") as zf:
            for path in temp_path.rglob("*"):
                zf.write(path, path.relative_to(temp_path))

    with pytest.raises(ValueError, match="Site directory not found for site-1"):
        install_app_code(zip_path, tmp_path, "site-1", mock_custom_dir)


def test_define_pre_install_parser():
    """Test the argument parser definition."""
    mock_sub_cmd = MagicMock()
    mock_parser = MagicMock()
    mock_args = MagicMock()

    # Set up the mock parser
    mock_sub_cmd.add_parser.return_value = mock_parser
    mock_parser.parse_args.return_value = mock_args
    mock_args.application = "test.zip"
    mock_args.install_prefix = "/opt/nvflare/apps"
    mock_args.site_name = "site-1"
    mock_args.target_shared_dir = "/tmp/local/custom"

    parser = define_pre_install_parser("pre-install", mock_sub_cmd)
    args = parser.parse_args(["-a", "test.zip", "-p", "/opt/nvflare/apps", "-s", "site-1", "-ts", "/tmp/local/custom"])

    assert args.application == "test.zip"
    assert args.install_prefix == "/opt/nvflare/apps"
    assert args.site_name == "site-1"
    assert args.target_shared_dir == "/tmp/local/custom"


@pytest.fixture(autouse=True)
def cleanup():
    # Setup
    yield
    # Cleanup
    if Path(PYTHON_PATH_SHARED_DIR).exists():
        shutil.rmtree(PYTHON_PATH_SHARED_DIR)
