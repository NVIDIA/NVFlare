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
from pathlib import Path
from zipfile import ZipFile

import pytest

from nvflare.tool.code_pre_installer.install import install_app_code


@pytest.fixture
def mock_custom_dir(monkeypatch, tmp_path):
    """Mock /local/custom directory for testing."""
    custom_dir = tmp_path / "local" / "custom"
    custom_dir.mkdir(parents=True)

    # Mock Path("/local/custom") to return our test directory
    def mock_path(*args, **kwargs):
        if args == ("/local/custom",):
            return custom_dir
        return Path(*args, **kwargs)

    monkeypatch.setattr("nvflare.tool.code_pre_installer.install.Path", mock_path)
    return custom_dir


def create_test_code(tmp_path):
    """Create a test application code package."""
    # Create directories
    app_code = tmp_path / "app_code"
    app_share = tmp_path / "app_share"
    app_code.mkdir()
    app_share.mkdir()

    # Create meta.json
    meta = {"name": "test_app"}
    with open(app_code / "meta.json", "w") as f:
        f.write(json.dumps(meta))

    # Create site directories
    site_dir = app_code / "app_site-1"
    site_dir.mkdir()
    custom_dir = site_dir / "custom"
    custom_dir.mkdir()

    # Create test files
    (custom_dir / "test.py").write_text("print('test')")
    (app_share / "shared.py").write_text("print('shared')")

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

    install_app_code(zip_path, install_prefix, "site-1")

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
        install_app_code(zip_path, install_prefix, "site-1")


def test_invalid_structure(tmp_path):
    """Test installation with invalid structure."""
    # Create an invalid zip file
    invalid_zip = tmp_path / "invalid.zip"
    with ZipFile(invalid_zip, "w") as zf:
        zf.writestr("dummy.txt", "not a valid structure")

    with pytest.raises(ValueError, match="Invalid application code"):
        install_app_code(invalid_zip, tmp_path, "site-1")
    # Verify zip file still exists after failed installation
    assert invalid_zip.exists(), "Zip file should not be deleted after failed installation"


def test_missing_meta(tmp_path):
    """Test installation with missing meta.json."""
    with ZipFile(tmp_path / "invalid.zip", "w") as zf:
        zf.writestr("app_code/empty", "")
        zf.writestr("app_share/empty", "")

    with pytest.raises(ValueError, match="meta.json not found"):
        install_app_code(tmp_path / "invalid.zip", tmp_path, "site-1")
