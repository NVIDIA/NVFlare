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

import pytest

from nvflare.app_opt.flower.utils import validate_flower_app_path


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

    def test_symlink_rejected_when_allow_symlinks_false(self, tmp_path, monkeypatch):
        """Symlinks should be rejected when allow_symlinks=False."""
        monkeypatch.chdir(tmp_path)
        real_dir = tmp_path / "local" / "custom" / "real_app"
        real_dir.mkdir(parents=True)
        link_dir = tmp_path / "local" / "custom" / "link_app"
        link_dir.symlink_to(real_dir)

        link_path = "local/custom/link_app"
        with pytest.raises(ValueError, match="is a symbolic link"):
            validate_flower_app_path(link_path, allow_symlinks=False)

    def test_symlink_allowed_when_allow_symlinks_true(self, tmp_path, monkeypatch):
        """Symlinks should be allowed when allow_symlinks=True (default)."""
        monkeypatch.chdir(tmp_path)
        real_dir = tmp_path / "local" / "custom" / "real_app"
        real_dir.mkdir(parents=True)
        link_dir = tmp_path / "local" / "custom" / "link_app"
        link_dir.symlink_to(real_dir)

        link_path = "local/custom/link_app"
        validate_flower_app_path(link_path, allow_symlinks=True)

    def test_real_directory_not_rejected(self, tmp_path, monkeypatch):
        """Real directories should not be rejected regardless of allow_symlinks setting."""
        monkeypatch.chdir(tmp_path)
        real_dir = tmp_path / "local" / "custom" / "real_app"
        real_dir.mkdir(parents=True)

        real_path = "local/custom/real_app"
        validate_flower_app_path(real_path, allow_symlinks=False)
        validate_flower_app_path(real_path, allow_symlinks=True)
