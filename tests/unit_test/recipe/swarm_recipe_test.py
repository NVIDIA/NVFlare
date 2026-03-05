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


class TestSwarmLearningRecipe:
    """Test cases for SimpleSwarmLearningRecipe."""

    def test_import_from_new_location(self, mock_file_system, simple_pt_model):
        """Test importing from new location (app_opt/pt/recipes)."""
        from nvflare.app_opt.pt.recipes.swarm import SwarmLearningRecipe

        recipe = SwarmLearningRecipe(
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
        from nvflare.app_opt.pt.recipes.swarm import SwarmLearningRecipe

        recipe = SwarmLearningRecipe(
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
        from nvflare.app_opt.pt.recipes.swarm import SwarmLearningRecipe

        # This should not raise since relative paths are now supported
        recipe = SwarmLearningRecipe(
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
        from nvflare.app_opt.pt.recipes.swarm import SwarmLearningRecipe

        recipe = SwarmLearningRecipe(
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
        from nvflare.app_opt.pt.recipes.swarm import SwarmLearningRecipe

        recipe = SwarmLearningRecipe(
            name="test_swarm_dict",
            model={"class_path": "torch.nn.Linear", "args": {"in_features": 10, "out_features": 2}},
            num_rounds=5,
            train_script="train.py",
            min_clients=2,
        )

        assert recipe.job is not None

    def test_dict_model_config_with_ckpt(self, mock_file_system):
        """Test dict model config with initial checkpoint."""
        from nvflare.app_opt.pt.recipes.swarm import SwarmLearningRecipe

        recipe = SwarmLearningRecipe(
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
        from nvflare.app_opt.pt.recipes.swarm import SwarmLearningRecipe

        with pytest.raises(ValueError, match="must have 'class_path' key"):
            SwarmLearningRecipe(
                name="test_swarm_bad_dict",
                model={"args": {"in_features": 10}},  # Missing 'path'
                num_rounds=5,
                train_script="train.py",
                min_clients=2,
            )

    def test_train_args_reserved_keys_rejected(self, mock_file_system, simple_pt_model):
        """Test that train_args with reserved keys are rejected."""
        from nvflare.app_opt.pt.recipes.swarm import SwarmLearningRecipe

        with pytest.raises(ValueError, match="reserved keys"):
            SwarmLearningRecipe(
                name="test_swarm_bad_args",
                model=simple_pt_model,
                num_rounds=5,
                train_script="train.py",
                min_clients=2,
                train_args={"script": "other.py"},  # 'script' is reserved
            )

    def test_train_args_valid_keys_accepted(self, mock_file_system, simple_pt_model):
        """Test that valid train_args are accepted."""
        from nvflare.app_opt.pt.recipes.swarm import SwarmLearningRecipe

        recipe = SwarmLearningRecipe(
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

        from nvflare.app_opt.pt.recipes.swarm import SwarmLearningRecipe

        sig = inspect.signature(SwarmLearningRecipe.__init__)
        assert "min_clients" in sig.parameters
        assert sig.parameters["min_clients"].default is inspect.Parameter.empty  # required, no default

        recipe = SwarmLearningRecipe(
            name="test_swarm_min_clients",
            model=simple_pt_model,
            num_rounds=5,
            train_script="train.py",
            min_clients=3,
        )

        assert recipe.job is not None

    def test_launch_external_process_accepted(self, mock_file_system, simple_pt_model):
        """Test that launch_external_process=True is accepted."""
        from nvflare.app_opt.pt.recipes.swarm import SwarmLearningRecipe

        recipe = SwarmLearningRecipe(
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
        from nvflare.app_opt.pt.recipes.swarm import SwarmLearningRecipe

        recipe = SwarmLearningRecipe(
            name="test_swarm_cmd",
            model=simple_pt_model,
            num_rounds=5,
            train_script="train.py",
            min_clients=2,
            launch_external_process=True,
            command="python3 -u",
        )

        assert recipe.job is not None


class TestSwarmLearningRecipeMemoryGC:
    """Test memory GC parameters on SimpleSwarmLearningRecipe."""

    def test_default_memory_gc_rounds_is_one(self):
        """Default memory_gc_rounds=1 for backward compatibility with legacy GC behavior."""
        import inspect

        from nvflare.app_opt.pt.recipes.swarm import SwarmLearningRecipe

        sig = inspect.signature(SwarmLearningRecipe.__init__)
        assert sig.parameters["memory_gc_rounds"].default == 1

    def test_old_param_name_rejected(self, mock_file_system, simple_pt_model):
        """client_memory_gc_rounds (old name) is no longer accepted."""
        from nvflare.app_opt.pt.recipes.swarm import SwarmLearningRecipe

        with pytest.raises(TypeError, match="client_memory_gc_rounds"):
            SwarmLearningRecipe(
                name="test_swarm",
                model=simple_pt_model,
                num_rounds=5,
                train_script="train.py",
                client_memory_gc_rounds=2,
            )

    def test_memory_gc_rounds_custom_accepted(self, mock_file_system, simple_pt_model):
        """Custom memory_gc_rounds is accepted."""
        from nvflare.app_opt.pt.recipes.swarm import SwarmLearningRecipe

        recipe = SwarmLearningRecipe(
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
        from nvflare.app_opt.pt.recipes.swarm import SwarmLearningRecipe

        recipe = SwarmLearningRecipe(
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
        from nvflare.app_opt.pt.recipes.swarm import SwarmLearningRecipe

        recipe = SwarmLearningRecipe(
            name="test_swarm",
            model=simple_pt_model,
            num_rounds=5,
            train_script="train.py",
            min_clients=2,
            cuda_empty_cache=True,
        )
        assert recipe.job is not None


class TestSwarmLearningRecipePipeType:
    """Tests for pipe_type and pipe_root_path parameters."""

    def _capture_task_pipe(self, recipe_kwargs):
        """Helper: build recipe and return the task_pipe passed to ScriptRunner."""
        import torch.nn as nn

        from nvflare.app_opt.pt.recipes.swarm import SwarmLearningRecipe
        from nvflare.job_config.script_runner import ScriptRunner

        captured = {}
        orig = ScriptRunner.__init__

        def _capture(self, *a, **kw):
            captured["task_pipe"] = kw.get("task_pipe")
            orig(self, *a, **kw)

        model = nn.Linear(2, 2)
        defaults = dict(name="t", model=model, num_rounds=1, train_script="t.py", min_clients=2)
        defaults.update(recipe_kwargs)

        with (
            patch("os.path.isfile", return_value=True),
            patch("os.path.isdir", return_value=True),
            patch("os.path.exists", return_value=True),
            patch.object(ScriptRunner, "__init__", _capture),
        ):
            SwarmLearningRecipe(**defaults)

        return captured.get("task_pipe")

    def test_default_cell_pipe_passes_task_pipe_none(self):
        """Default pipe_type='cell_pipe' must pass task_pipe=None to ScriptRunner
        so ScriptRunner creates a CellPipe via its own _create_cell_pipe()."""
        task_pipe = self._capture_task_pipe({})
        assert task_pipe is None

    def test_file_pipe_passes_filepipe_instance(self):
        """pipe_type='file_pipe' must pass a FilePipe instance to ScriptRunner."""
        from nvflare.fuel.utils.pipe.file_pipe import FilePipe

        task_pipe = self._capture_task_pipe({"pipe_type": "file_pipe"})
        assert isinstance(task_pipe, FilePipe)

    def test_file_pipe_default_root_path_uses_workspace_template(self):
        """pipe_type='file_pipe' with no pipe_root_path must use the workspace
        template variable so the path is resolved at runtime on each client."""
        from nvflare.apis.fl_constant import SystemVarName

        task_pipe = self._capture_task_pipe({"pipe_type": "file_pipe"})
        expected_fragment = "{" + SystemVarName.WORKSPACE + "}"
        assert expected_fragment in task_pipe.root_path

    def test_file_pipe_custom_root_path_forwarded(self, tmp_path):
        """pipe_type='file_pipe' with a valid pipe_root_path must use that path."""
        from nvflare.fuel.utils.pipe.file_pipe import FilePipe

        custom_path = str(tmp_path)
        task_pipe = self._capture_task_pipe({"pipe_type": "file_pipe", "pipe_root_path": custom_path})
        assert isinstance(task_pipe, FilePipe)
        assert task_pipe.root_path == custom_path

    def test_invalid_pipe_type_raises(self, mock_file_system, simple_pt_model):
        """An unrecognised pipe_type must raise ValueError immediately."""
        from nvflare.app_opt.pt.recipes.swarm import SwarmLearningRecipe

        with pytest.raises(ValueError, match="pipe_type must be one of"):
            SwarmLearningRecipe(
                name="t",
                model=simple_pt_model,
                num_rounds=1,
                train_script="t.py",
                min_clients=2,
                pipe_type="grpc_pipe",
            )

    def test_cell_pipe_with_root_path_warns(self, mock_file_system, simple_pt_model, caplog):
        """pipe_root_path set alongside pipe_type='cell_pipe' must emit a warning."""
        import logging

        from nvflare.app_opt.pt.recipes.swarm import SwarmLearningRecipe

        with caplog.at_level(logging.WARNING, logger="nvflare.app_opt.pt.recipes.swarm"):
            SwarmLearningRecipe(
                name="t",
                model=simple_pt_model,
                num_rounds=1,
                train_script="t.py",
                min_clients=2,
                pipe_type="cell_pipe",
                pipe_root_path="/some/path",
            )
        assert any("pipe_root_path" in r.message and "ignored" in r.message for r in caplog.records)

    def test_file_pipe_with_in_process_warns(self, mock_file_system, simple_pt_model, caplog):
        """pipe_type='file_pipe' + launch_external_process=False must warn the user."""
        import logging

        from nvflare.app_opt.pt.recipes.swarm import SwarmLearningRecipe

        with caplog.at_level(logging.WARNING, logger="nvflare.app_opt.pt.recipes.swarm"):
            SwarmLearningRecipe(
                name="t",
                model=simple_pt_model,
                num_rounds=1,
                train_script="t.py",
                min_clients=2,
                pipe_type="file_pipe",
                launch_external_process=False,
            )
        assert any("has no effect" in r.message for r in caplog.records)

    def test_pipe_root_path_relative_raises(self, mock_file_system, simple_pt_model):
        """A relative pipe_root_path must raise ValueError."""
        from nvflare.app_opt.pt.recipes.swarm import SwarmLearningRecipe

        with pytest.raises(ValueError, match="absolute path"):
            SwarmLearningRecipe(
                name="t",
                model=simple_pt_model,
                num_rounds=1,
                train_script="t.py",
                min_clients=2,
                pipe_type="file_pipe",
                pipe_root_path="relative/path",
            )

    def test_pipe_root_path_nonexistent_raises(self, simple_pt_model):
        """A non-existent pipe_root_path must raise ValueError."""
        from nvflare.app_opt.pt.recipes.swarm import SwarmLearningRecipe

        with (
            patch("os.path.isfile", return_value=True),
            patch("os.path.exists", return_value=True),
            patch("os.path.isabs", return_value=True),
            patch("os.path.isdir", return_value=False),
        ):
            with pytest.raises(ValueError, match="does not exist or is not a directory"):
                SwarmLearningRecipe(
                    name="t",
                    model=simple_pt_model,
                    num_rounds=1,
                    train_script="t.py",
                    min_clients=2,
                    pipe_type="file_pipe",
                    pipe_root_path="/nonexistent/path",
                )


class TestSwarmLearningRecipeExport:
    """Export behavior tests for SimpleSwarmLearningRecipe."""

    def test_export_preserves_dict_model_args_in_client_config(self, tmp_path):
        """Regression: exported client config keeps dict model args for PTFileModelPersistor."""
        from nvflare.app_opt.pt.recipes.swarm import SwarmLearningRecipe

        train_script = tmp_path / "driver.py"
        train_script.write_text("print('train')\n")

        model_name_or_path = "meta-llama/Llama-3.1-8B"
        model = {
            "class_path": "hf_sft_model.CausalLMModel",
            "args": {"model_name_or_path": model_name_or_path},
        }
        job_name = "swarm_issue_reproducer"

        recipe = SwarmLearningRecipe(
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
