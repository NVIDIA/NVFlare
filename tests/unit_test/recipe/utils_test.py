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

from nvflare.recipe.utils import (
    extract_persistor_id,
    prepare_initial_ckpt,
    resolve_initial_ckpt,
    setup_custom_persistor,
    validate_ckpt,
)


@pytest.fixture
def temp_workdir():
    """Fixture that creates a temp directory and changes into it, restoring cwd after."""
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        yield tmpdir
        os.chdir(original_cwd)


class TestValidateCkpt:
    """Tests for validate_ckpt function."""

    def test_none_ckpt(self):
        """None should pass validation."""
        validate_ckpt(None)  # Should not raise

    def test_absolute_path_not_exists(self):
        """Absolute path that doesn't exist should pass (server-side path)."""
        validate_ckpt("/server/path/to/checkpoint.pt")  # Should not raise

    def test_absolute_path_exists(self):
        """Absolute path that exists locally should pass."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
        try:
            validate_ckpt(ckpt_path)  # Should not raise
        finally:
            os.unlink(ckpt_path)

    def test_relative_path_exists(self, temp_workdir):
        """Relative path that exists locally should pass."""
        ckpt_file = "checkpoint.pt"
        open(ckpt_file, "w").close()
        validate_ckpt(ckpt_file)  # Should not raise

    def test_relative_path_not_exists(self):
        """Relative path that doesn't exist should raise ValueError."""
        with pytest.raises(ValueError, match="does not exist locally"):
            validate_ckpt("non_existent_checkpoint.pt")

    def test_relative_path_subdirectory_exists(self, temp_workdir):
        """Relative path in subdirectory that exists should pass."""
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_file = "checkpoints/model.pt"
        open(ckpt_file, "w").close()
        validate_ckpt(ckpt_file)  # Should not raise

    def test_relative_path_subdirectory_not_exists(self):
        """Relative path in subdirectory that doesn't exist should raise."""
        with pytest.raises(ValueError, match="does not exist locally"):
            validate_ckpt("checkpoints/non_existent.pt")


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


class TestPersistorUtils:
    """Tests for persistor utility helpers."""

    def test_extract_persistor_id(self):
        assert extract_persistor_id({"persistor_id": "persistor_a"}) == "persistor_a"
        assert extract_persistor_id({"persistor_id": 123}) == ""
        assert extract_persistor_id("persistor_b") == "persistor_b"
        assert extract_persistor_id(None) == ""

    def test_setup_custom_persistor_returns_empty_when_not_provided(self):
        job = MagicMock()

        result = setup_custom_persistor(job=job, model_persistor=None)

        assert result == ""
        job.to_server.assert_not_called()

    def test_setup_custom_persistor_registers_component(self):
        job = MagicMock()
        custom_persistor = object()
        job.to_server.return_value = "custom_persistor"

        result = setup_custom_persistor(job=job, model_persistor=custom_persistor)

        assert result == "custom_persistor"
        job.to_server.assert_called_once_with(custom_persistor, id="persistor")

    def test_setup_custom_persistor_extracts_dict_result(self):
        job = MagicMock()
        custom_persistor = object()
        job.to_server.return_value = {"persistor_id": "custom_from_dict"}

        result = setup_custom_persistor(job=job, model_persistor=custom_persistor)

        assert result == "custom_from_dict"

    def test_resolve_initial_ckpt_prefers_prepared_value(self):
        job = MagicMock()

        result = resolve_initial_ckpt(
            initial_ckpt="relative/path/model.pt",
            prepared_initial_ckpt="already_prepared.pt",
            job=job,
        )

        assert result == "already_prepared.pt"
        job.add_file_to_server.assert_not_called()

    def test_resolve_initial_ckpt_uses_prepare_when_prepared_missing(self, monkeypatch):
        calls = {}

        def fake_prepare(initial_ckpt, job):
            calls["initial_ckpt"] = initial_ckpt
            calls["job"] = job
            return "prepared_by_helper.pt"

        monkeypatch.setattr("nvflare.recipe.utils.prepare_initial_ckpt", fake_prepare)
        job = MagicMock()

        result = resolve_initial_ckpt(
            initial_ckpt="relative/path/model.pt",
            prepared_initial_ckpt=None,
            job=job,
        )

        assert result == "prepared_by_helper.pt"
        assert calls["initial_ckpt"] == "relative/path/model.pt"
        assert calls["job"] is job


class TestRecipePackageExports:
    """Tests for public API exports from nvflare.recipe."""

    def test_add_cross_site_evaluation_importable_from_recipe(self):
        """add_cross_site_evaluation must be importable from the top-level nvflare.recipe package."""
        from nvflare.recipe import add_cross_site_evaluation

        assert callable(add_cross_site_evaluation)


class TestCrossSiteEvalIdempotency:
    """Tests for resilient idempotency in add_cross_site_evaluation."""

    def test_idempotency_survives_missing_flag(self):
        from nvflare.app_common.np.recipes.fedavg import NumpyFedAvgRecipe
        from nvflare.recipe.utils import add_cross_site_evaluation

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("# dummy train script\n")
            train_script = f.name

        try:
            recipe = NumpyFedAvgRecipe(
                name="test_cse_idempotency",
                model=[1.0, 2.0],
                min_clients=2,
                num_rounds=2,
                train_script=train_script,
            )

            add_cross_site_evaluation(recipe)
            assert getattr(recipe, "_cse_added", False) is True

            # Simulate transient attribute loss (e.g. serialization boundary).
            del recipe._cse_added
            assert not hasattr(recipe, "_cse_added")

            with pytest.raises(RuntimeError, match="already been added"):
                add_cross_site_evaluation(recipe)
        finally:
            os.unlink(train_script)
