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
import tempfile
from unittest.mock import MagicMock

import pytest

from nvflare.recipe.utils import prepare_initial_ckpt, validate_initial_ckpt


@pytest.fixture
def temp_workdir():
    """Fixture that creates a temp directory and changes into it, restoring cwd after."""
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        yield tmpdir
        os.chdir(original_cwd)


class TestValidateInitialCkpt:
    """Tests for validate_initial_ckpt function."""

    def test_none_ckpt(self):
        """None should pass validation."""
        validate_initial_ckpt(None)  # Should not raise

    def test_absolute_path_not_exists(self):
        """Absolute path that doesn't exist should pass (server-side path)."""
        validate_initial_ckpt("/server/path/to/checkpoint.pt")  # Should not raise

    def test_absolute_path_exists(self):
        """Absolute path that exists locally should pass."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            validate_initial_ckpt(ckpt_path)  # Should not raise
        finally:
            os.unlink(ckpt_path)

    def test_relative_path_exists(self, temp_workdir):
        """Relative path that exists locally should pass."""
        ckpt_file = "checkpoint.pt"
        open(ckpt_file, "w").close()
        validate_initial_ckpt(ckpt_file)  # Should not raise

    def test_relative_path_not_exists(self):
        """Relative path that doesn't exist should raise ValueError."""
        with pytest.raises(ValueError, match="does not exist locally"):
            validate_initial_ckpt("non_existent_checkpoint.pt")

    def test_relative_path_subdirectory_exists(self, temp_workdir):
        """Relative path in subdirectory that exists should pass."""
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_file = "checkpoints/model.pt"
        open(ckpt_file, "w").close()
        validate_initial_ckpt(ckpt_file)  # Should not raise

    def test_relative_path_subdirectory_not_exists(self):
        """Relative path in subdirectory that doesn't exist should raise."""
        with pytest.raises(ValueError, match="does not exist locally"):
            validate_initial_ckpt("checkpoints/non_existent.pt")


class TestPrepareInitialCkpt:
    """Tests for prepare_initial_ckpt function."""

    def test_none_ckpt(self):
        """None should return None."""
        job = MagicMock()
        result = prepare_initial_ckpt(None, job)
        assert result is None
        job.add_file_to_server.assert_not_called()

    def test_absolute_path_server_side(self):
        """Absolute path should be returned as-is (server-side path)."""
        job = MagicMock()
        abs_path = "/workspace/models/checkpoint.pt"
        result = prepare_initial_ckpt(abs_path, job)
        assert result == abs_path
        job.add_file_to_server.assert_not_called()

    def test_absolute_path_local_file(self):
        """Absolute path to local file should still be returned as-is (user intent)."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            job = MagicMock()
            result = prepare_initial_ckpt(ckpt_path, job)
            # Absolute paths are treated as server-side, not bundled
            assert result == ckpt_path
            job.add_file_to_server.assert_not_called()
        finally:
            os.unlink(ckpt_path)

    def test_relative_path_bundled(self, temp_workdir):
        """Relative path should be bundled and basename returned."""
        ckpt_file = "checkpoint.pt"
        open(ckpt_file, "w").close()

        job = MagicMock()
        result = prepare_initial_ckpt(ckpt_file, job)

        # Should bundle the file
        job.add_file_to_server.assert_called_once_with(ckpt_file)
        # Should return basename
        assert result == "checkpoint.pt"

    def test_relative_path_subdirectory_bundled(self, temp_workdir):
        """Relative path in subdirectory should be bundled and basename returned."""
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_file = "checkpoints/model.pt"
        open(ckpt_file, "w").close()

        job = MagicMock()
        result = prepare_initial_ckpt(ckpt_file, job)

        # Should bundle the full relative path
        job.add_file_to_server.assert_called_once_with(ckpt_file)
        # Should return basename only
        assert result == "model.pt"

    def test_multiple_calls_different_files(self, temp_workdir):
        """Multiple calls with different files should bundle each."""
        ckpt1 = "ckpt1.pt"
        ckpt2 = "ckpt2.pt"
        open(ckpt1, "w").close()
        open(ckpt2, "w").close()

        job = MagicMock()

        result1 = prepare_initial_ckpt(ckpt1, job)
        assert result1 == "ckpt1.pt"

        result2 = prepare_initial_ckpt(ckpt2, job)
        assert result2 == "ckpt2.pt"

        assert job.add_file_to_server.call_count == 2
