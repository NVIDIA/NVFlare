# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import os

import pytest

import pytest

from nvflare.app_opt.flower.utils import (
    validate_flower_app_path,
    validate_flower_app_path_no_symlinks,
)


class TestValidateFlowerAppPath:
    """Direct unit tests for the validation helper function."""

    @pytest.mark.parametrize(
        "valid_path",
        [
            "local/custom/",
            "local/custom/app",
            "local/custom/deep/nested/path",
        ],
    )
    def test_valid_paths(self, valid_path):
        """Valid paths should not raise."""
        # Should not raise
        validate_flower_app_path(valid_path)

    @pytest.mark.parametrize(
        "invalid_path,error_pattern",
        [
            ("/absolute/path", "must start with 'local/custom/'"),
            ("app/path", "must start with 'local/custom/'"),
            ("local/apps/app", "must start with 'local/custom/'"),
            ("", "must start with 'local/custom/'"),
            ("../etc/passwd", "must start with 'local/custom/'"),
            ("local/custom/../../../etc", "invalid path traversal"),
            ("local/custom/..\\..\\windows", "invalid path traversal"),
            ("local/custom/app/..", "invalid path traversal"),
            ("C:\\windows\\system32", "must start with 'local/custom/'"),
        ],
    )
    def test_invalid_paths(self, invalid_path, error_pattern):
        """Invalid paths should raise ValueError with appropriate message."""
        with pytest.raises(ValueError, match=error_pattern):
            validate_flower_app_path(invalid_path)

    def test_symlink_check_in_separate_validator(self, tmp_path, monkeypatch):
        """Symlink checking is done by validate_flower_app_path_no_symlinks, not validate_flower_app_path."""
        monkeypatch.chdir(tmp_path)
        real_dir = tmp_path / "local" / "custom" / "real_app"
        real_dir.mkdir(parents=True)
        link_dir = tmp_path / "local" / "custom" / "link_app"
        link_dir.symlink_to(real_dir)

        link_path = "local/custom/link_app"
        # validate_flower_app_path only checks format, not symlinks
        validate_flower_app_path(link_path)

    def test_symlink_rejected_on_resolved_path(self, tmp_path, monkeypatch):
        """Symlinks should be rejected when validating the resolved absolute path."""
        monkeypatch.chdir(tmp_path)
        real_dir = tmp_path / "workspace" / "local" / "custom" / "real_app"
        real_dir.mkdir(parents=True)
        link_dir = tmp_path / "workspace" / "local" / "custom" / "link_app"
        link_dir.symlink_to(real_dir)

        with pytest.raises(RuntimeError, match="resolves to a symbolic link"):
            validate_flower_app_path_no_symlinks(str(link_dir))

    def test_real_directory_accepted(self, tmp_path, monkeypatch):
        """Real directories should be accepted."""
        monkeypatch.chdir(tmp_path)
        real_dir = tmp_path / "workspace" / "local" / "custom" / "real_app"
        real_dir.mkdir(parents=True)

        validate_flower_app_path_no_symlinks(str(real_dir))
