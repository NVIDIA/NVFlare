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

"""Tests for Swarm Learning recipes."""

import json
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def mock_file_system():
    """Mock file system operations for all tests."""
    with (
        patch("os.path.isfile", return_value=True),
        patch("os.path.isdir", return_value=True),
        patch("os.path.exists", return_value=True),
    ):
        yield


@pytest.fixture
def simple_pt_model():
    """Create a simple PyTorch model for testing."""
    import torch.nn as nn

    return nn.Linear(10, 2)


class TestSimpleSwarmLearningRecipe:
    """Test cases for SimpleSwarmLearningRecipe."""

    def test_import_from_new_location(self, mock_file_system, simple_pt_model):
        """Test importing from new location (app_opt/pt/recipes)."""
        from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

        recipe = SimpleSwarmLearningRecipe(
            name="test_swarm",
            model=simple_pt_model,
            num_rounds=5,
            train_script="train.py",
            min_clients=2,
        )

        assert recipe.job is not None

    def test_import_from_old_location_backward_compat(self, mock_file_system, simple_pt_model):
        """Test importing from old location (backward compatibility)."""
        from nvflare.app_common.ccwf.recipes.swarm import SimpleSwarmLearningRecipe

        recipe = SimpleSwarmLearningRecipe(
            name="test_swarm",
            model=simple_pt_model,
            num_rounds=5,
            train_script="train.py",
            min_clients=2,
        )

        assert recipe.job is not None

    def test_initial_ckpt_accepted(self, mock_file_system, simple_pt_model):
        """Test that initial_ckpt parameter is accepted."""
        from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

        recipe = SimpleSwarmLearningRecipe(
            name="test_swarm_ckpt",
            model=simple_pt_model,
            num_rounds=5,
            train_script="train.py",
            min_clients=2,
            initial_ckpt="/abs/path/to/model.pt",
        )

        assert recipe.job is not None

    def test_relative_path_accepted_if_exists(self, mock_file_system, simple_pt_model):
        """Test that existing relative paths are accepted and bundled."""
        from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

        # This should not raise since relative paths are now supported
        recipe = SimpleSwarmLearningRecipe(
            name="test_swarm",
            model=simple_pt_model,
            num_rounds=5,
            train_script="train.py",
            min_clients=2,
            initial_ckpt="relative/path/model.pt",
        )
        assert recipe is not None

    def test_cross_site_eval_option(self, mock_file_system, simple_pt_model):
        """Test with cross-site evaluation enabled."""
        from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

        recipe = SimpleSwarmLearningRecipe(
            name="test_swarm_cse",
            model=simple_pt_model,
            num_rounds=5,
            train_script="train.py",
            min_clients=2,
            do_cross_site_eval=True,
            cross_site_eval_timeout=600,
        )

        assert recipe.job is not None

    def test_dict_model_config_accepted(self, mock_file_system):
        """Test that dict model config is accepted."""
        from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

        recipe = SimpleSwarmLearningRecipe(
            name="test_swarm_dict",
            model={"class_path": "torch.nn.Linear", "args": {"in_features": 10, "out_features": 2}},
            num_rounds=5,
            train_script="train.py",
            min_clients=2,
        )

        assert recipe.job is not None

    def test_dict_model_config_with_ckpt(self, mock_file_system):
        """Test dict model config with initial checkpoint."""
        from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

        recipe = SimpleSwarmLearningRecipe(
            name="test_swarm_dict_ckpt",
            model={"class_path": "torch.nn.Linear", "args": {"in_features": 10, "out_features": 2}},
            num_rounds=5,
            train_script="train.py",
            min_clients=2,
            initial_ckpt="/abs/path/to/model.pt",
        )

        assert recipe.job is not None

    def test_dict_model_missing_path_rejected(self, mock_file_system):
        """Test that dict model without 'class_path' key is rejected."""
        from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

        with pytest.raises(ValueError, match="must have 'class_path' key"):
            SimpleSwarmLearningRecipe(
                name="test_swarm_bad_dict",
                model={"args": {"in_features": 10}},  # Missing 'path'
                num_rounds=5,
                train_script="train.py",
                min_clients=2,
            )

    def test_train_args_reserved_keys_rejected(self, mock_file_system, simple_pt_model):
        """Test that train_args with reserved keys are rejected."""
        from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

        with pytest.raises(ValueError, match="reserved keys"):
            SimpleSwarmLearningRecipe(
                name="test_swarm_bad_args",
                model=simple_pt_model,
                num_rounds=5,
                train_script="train.py",
                min_clients=2,
                train_args={"script": "other.py"},  # 'script' is reserved
            )

    def test_train_args_valid_keys_accepted(self, mock_file_system, simple_pt_model):
        """Test that valid train_args are accepted."""
        from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

        recipe = SimpleSwarmLearningRecipe(
            name="test_swarm_args",
            model=simple_pt_model,
            num_rounds=5,
            train_script="train.py",
            min_clients=2,
            train_args={"script_args": "--batch_size 32"},  # valid key
        )

        assert recipe.job is not None

    def test_min_clients_accepted(self, mock_file_system, simple_pt_model):
        """Test that min_clients is a required parameter and is passed to the job."""
        import inspect

        from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

        sig = inspect.signature(SimpleSwarmLearningRecipe.__init__)
        assert "min_clients" in sig.parameters
        assert sig.parameters["min_clients"].default is inspect.Parameter.empty  # required, no default

        recipe = SimpleSwarmLearningRecipe(
            name="test_swarm_min_clients",
            model=simple_pt_model,
            num_rounds=5,
            train_script="train.py",
            min_clients=3,
        )

        assert recipe.job is not None

    def test_launch_external_process_accepted(self, mock_file_system, simple_pt_model):
        """Test that launch_external_process=True is accepted."""
        from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

        recipe = SimpleSwarmLearningRecipe(
            name="test_swarm_ext",
            model=simple_pt_model,
            num_rounds=5,
            train_script="train.py",
            min_clients=2,
            launch_external_process=True,
        )

        assert recipe.job is not None

    def test_command_accepted(self, mock_file_system, simple_pt_model):
        """Test that command is accepted alongside launch_external_process."""
        from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

        recipe = SimpleSwarmLearningRecipe(
            name="test_swarm_cmd",
            model=simple_pt_model,
            num_rounds=5,
            train_script="train.py",
            min_clients=2,
            launch_external_process=True,
            command="python3 -u",
        )

        assert recipe.job is not None


class TestSimpleSwarmLearningRecipeMemoryGC:
    """Test memory GC parameters on SimpleSwarmLearningRecipe."""

    def test_default_memory_gc_rounds_is_one(self):
        """Default memory_gc_rounds=1 for backward compatibility with legacy GC behavior."""
        import inspect

        from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

        sig = inspect.signature(SimpleSwarmLearningRecipe.__init__)
        assert sig.parameters["memory_gc_rounds"].default == 1

    def test_old_param_name_rejected(self, mock_file_system, simple_pt_model):
        """client_memory_gc_rounds (old name) is no longer accepted."""
        from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

        with pytest.raises(TypeError, match="client_memory_gc_rounds"):
            SimpleSwarmLearningRecipe(
                name="test_swarm",
                model=simple_pt_model,
                num_rounds=5,
                train_script="train.py",
                client_memory_gc_rounds=2,
            )

    def test_memory_gc_rounds_custom_accepted(self, mock_file_system, simple_pt_model):
        """Custom memory_gc_rounds is accepted."""
        from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

        recipe = SimpleSwarmLearningRecipe(
            name="test_swarm",
            model=simple_pt_model,
            num_rounds=5,
            train_script="train.py",
            min_clients=2,
            memory_gc_rounds=2,
        )
        assert recipe.job is not None

    def test_memory_gc_disabled_accepted(self, mock_file_system, simple_pt_model):
        """memory_gc_rounds=0 disables GC."""
        from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

        recipe = SimpleSwarmLearningRecipe(
            name="test_swarm",
            model=simple_pt_model,
            num_rounds=5,
            train_script="train.py",
            min_clients=2,
            memory_gc_rounds=0,
        )
        assert recipe.job is not None

    def test_cuda_empty_cache_accepted(self, mock_file_system, simple_pt_model):
        """cuda_empty_cache=True is accepted and wired through."""
        from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

        recipe = SimpleSwarmLearningRecipe(
            name="test_swarm",
            model=simple_pt_model,
            num_rounds=5,
            train_script="train.py",
            min_clients=2,
            cuda_empty_cache=True,
        )
        assert recipe.job is not None


class TestSimpleSwarmLearningRecipeExport:
    """Export behavior tests for SimpleSwarmLearningRecipe."""

    def test_export_preserves_dict_model_args_in_client_config(self, tmp_path):
        """Regression: exported client config keeps dict model args for PTFileModelPersistor."""
        from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

        train_script = tmp_path / "driver.py"
        train_script.write_text("print('train')\n")

        model_name_or_path = "meta-llama/Llama-3.1-8B"
        model = {
            "class_path": "hf_sft_model.CausalLMModel",
            "args": {"model_name_or_path": model_name_or_path},
        }
        job_name = "swarm_issue_reproducer"

        recipe = SimpleSwarmLearningRecipe(
            name=job_name,
            model=model,
            num_rounds=3,
            train_script=str(train_script),
            min_clients=2,
        )

        export_dir = tmp_path / "job"
        recipe.export(str(export_dir))

        config_path = export_dir / job_name / "app" / "config" / "config_fed_client.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        persistor = None
        for comp in config.get("components", []):
            if comp.get("id") == "persistor":
                persistor = comp
                break

        assert persistor is not None, "Persistor component not found in client config"
        model_cfg = persistor.get("args", {}).get("model")
        assert model_cfg is not None, "Persistor model config is missing"
        assert model_cfg.get("path") == "hf_sft_model.CausalLMModel"
        assert model_cfg.get("args", {}).get("model_name_or_path") == model_name_or_path
