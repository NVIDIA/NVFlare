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
import sys
import tempfile
import types
from unittest.mock import MagicMock

import pytest

from nvflare.client.config import ExchangeFormat
from nvflare.fuel.utils.constants import FrameworkType
from nvflare.recipe.utils import (
    _extract_persistor_id,
    prepare_initial_ckpt,
    setup_framework_model_persistor,
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
    """Tests for extract_persistor_id and setup_framework_model_persistor."""

    def test__extract_persistor_id(self):
        assert _extract_persistor_id({"persistor_id": "persistor_a"}) == "persistor_a"
        assert _extract_persistor_id({"persistor_id": 123}) == ""
        assert _extract_persistor_id("persistor_b") == "persistor_b"
        assert _extract_persistor_id(None) == ""

    def test_setup_uses_custom_model_persistor_first(self):
        job = MagicMock()
        custom_persistor = object()
        job.to_server.return_value = "custom_persistor"

        result = setup_framework_model_persistor(
            job=job,
            framework=FrameworkType.PYTORCH,
            model=None,
            initial_ckpt=None,
            server_expected_format=ExchangeFormat.NUMPY,
            model_persistor=custom_persistor,
            recipe_name="TestRecipe",
        )

        assert result == "custom_persistor"
        job.to_server.assert_called_once_with(custom_persistor, id="persistor")
        job.add_file_to_server.assert_not_called()

    def test_setup_pytorch_ckpt_without_model_raises(self):
        job = MagicMock()
        with pytest.raises(ValueError, match="requires 'model'"):
            setup_framework_model_persistor(
                job=job,
                framework=FrameworkType.PYTORCH,
                model=None,
                initial_ckpt="/abs/path/model.pt",
                server_expected_format=ExchangeFormat.NUMPY,
                recipe_name="FedAvgRecipe",
            )

    def test_setup_pytorch_with_model_uses_pt_model(self, monkeypatch):
        fake_mod = types.ModuleType("fake_pt_model")

        class FakePTModel:
            def __init__(self, model, initial_ckpt, allow_numpy_conversion):
                self.model = model
                self.initial_ckpt = initial_ckpt
                self.allow_numpy_conversion = allow_numpy_conversion

        fake_mod.PTModel = FakePTModel
        monkeypatch.setitem(sys.modules, "nvflare.app_opt.pt.job_config.model", fake_mod)

        job = MagicMock()
        job.to_server.return_value = "pt_persistor"

        result = setup_framework_model_persistor(
            job=job,
            framework=FrameworkType.PYTORCH,
            model={"path": "m.Model", "args": {}},
            initial_ckpt=None,
            prepared_initial_ckpt="bundled.pt",
            server_expected_format=ExchangeFormat.PYTORCH,
            recipe_name="FedAvgRecipe",
        )

        assert result == "pt_persistor"
        sent_obj = job.to_server.call_args.args[0]
        assert isinstance(sent_obj, FakePTModel)
        assert sent_obj.initial_ckpt == "bundled.pt"
        assert sent_obj.allow_numpy_conversion is False

    def test_setup_tensorflow_with_ckpt_only_uses_tf_model(self, monkeypatch):
        fake_mod = types.ModuleType("fake_tf_model")

        class FakeTFModel:
            def __init__(self, model, initial_ckpt):
                self.model = model
                self.initial_ckpt = initial_ckpt

        fake_mod.TFModel = FakeTFModel
        monkeypatch.setitem(sys.modules, "nvflare.app_opt.tf.job_config.model", fake_mod)

        job = MagicMock()
        job.to_server.return_value = {"persistor_id": "tf_persistor"}

        result = setup_framework_model_persistor(
            job=job,
            framework=FrameworkType.TENSORFLOW,
            model=None,
            initial_ckpt=None,
            prepared_initial_ckpt="server/model.h5",
            server_expected_format=ExchangeFormat.NUMPY,
            recipe_name="FedAvgRecipe",
        )

        assert result == "tf_persistor"
        sent_obj = job.to_server.call_args.args[0]
        assert isinstance(sent_obj, FakeTFModel)
        assert sent_obj.model is None
        assert sent_obj.initial_ckpt == "server/model.h5"

    def test_setup_numpy_with_bad_model_type_raises(self):
        job = MagicMock()
        with pytest.raises(TypeError, match="model must be list or np.ndarray"):
            setup_framework_model_persistor(
                job=job,
                framework=FrameworkType.NUMPY,
                model={"bad": "type"},
                initial_ckpt=None,
                server_expected_format=ExchangeFormat.NUMPY,
                support_numpy=True,
                recipe_name="FedAvgRecipe",
            )

    def test_setup_numpy_with_list_model(self, monkeypatch):
        fake_mod = types.ModuleType("fake_np_persistor")

        class FakeNPModelPersistor:
            def __init__(self, model, source_ckpt_file_full_name):
                self.model = model
                self.source_ckpt_file_full_name = source_ckpt_file_full_name

        fake_mod.NPModelPersistor = FakeNPModelPersistor
        monkeypatch.setitem(sys.modules, "nvflare.app_common.np.np_model_persistor", fake_mod)

        job = MagicMock()
        job.to_server.return_value = "np_persistor"

        result = setup_framework_model_persistor(
            job=job,
            framework=FrameworkType.NUMPY,
            model=[1.0, 2.0],
            initial_ckpt=None,
            prepared_initial_ckpt="model.npy",
            server_expected_format=ExchangeFormat.NUMPY,
            support_numpy=True,
            recipe_name="FedAvgRecipe",
        )

        assert result == "np_persistor"
        sent_obj = job.to_server.call_args.args[0]
        assert isinstance(sent_obj, FakeNPModelPersistor)
        assert sent_obj.model == [1.0, 2.0]
        assert sent_obj.source_ckpt_file_full_name == "model.npy"

    def test_setup_unsupported_framework_returns_empty_by_default(self):
        job = MagicMock()
        result = setup_framework_model_persistor(
            job=job,
            framework=FrameworkType.RAW,
            model=None,
            initial_ckpt=None,
            server_expected_format=ExchangeFormat.NUMPY,
            recipe_name="SomeRecipe",
        )
        assert result == ""

    def test_setup_unsupported_framework_raises_with_custom_message(self):
        job = MagicMock()
        with pytest.raises(ValueError, match="custom unsupported message"):
            setup_framework_model_persistor(
                job=job,
                framework=FrameworkType.RAW,
                model=None,
                initial_ckpt=None,
                server_expected_format=ExchangeFormat.NUMPY,
                raise_on_unsupported=True,
                unsupported_framework_message="custom unsupported message",
                recipe_name="SomeRecipe",
            )


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
