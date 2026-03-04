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

"""Tests for Edge FedBuff recipes."""

import importlib.util
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

executorch_available = importlib.util.find_spec("executorch") is not None


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


class TestEdgeFedBuffRecipe:
    """Test cases for EdgeFedBuffRecipe."""

    @pytest.fixture
    def model_manager_config(self):
        """Create ModelManagerConfig for testing."""
        from nvflare.edge.tools.edge_fed_buff_recipe import ModelManagerConfig

        return ModelManagerConfig(
            max_model_version=10,
            num_updates_for_model=5,
        )

    @pytest.fixture
    def device_manager_config(self):
        """Create DeviceManagerConfig for testing."""
        from nvflare.edge.tools.edge_fed_buff_recipe import DeviceManagerConfig

        return DeviceManagerConfig(
            device_selection_size=10,
        )

    def test_basic_initialization(self, mock_file_system, simple_pt_model, model_manager_config, device_manager_config):
        """Test EdgeFedBuffRecipe basic initialization."""
        from nvflare.edge.tools.edge_fed_buff_recipe import EdgeFedBuffRecipe

        recipe = EdgeFedBuffRecipe(
            job_name="test_edge",
            model=simple_pt_model,
            model_manager_config=model_manager_config,
            device_manager_config=device_manager_config,
        )

        assert recipe.job is not None

    def test_initial_ckpt_accepted(
        self, mock_file_system, simple_pt_model, model_manager_config, device_manager_config
    ):
        """Test that initial_ckpt parameter is accepted."""
        from nvflare.edge.tools.edge_fed_buff_recipe import EdgeFedBuffRecipe

        recipe = EdgeFedBuffRecipe(
            job_name="test_edge_ckpt",
            model=simple_pt_model,
            model_manager_config=model_manager_config,
            device_manager_config=device_manager_config,
            initial_ckpt="/abs/path/to/model.pt",
        )

        assert recipe.job is not None
        assert recipe.initial_ckpt == "/abs/path/to/model.pt"

    def test_dict_model_config_accepted(self, mock_file_system, model_manager_config, device_manager_config):
        """Test that dict model config is accepted."""
        from nvflare.edge.tools.edge_fed_buff_recipe import EdgeFedBuffRecipe

        recipe = EdgeFedBuffRecipe(
            job_name="test_edge_dict",
            model={"class_path": "torch.nn.Linear", "args": {"in_features": 10, "out_features": 2}},
            model_manager_config=model_manager_config,
            device_manager_config=device_manager_config,
        )

        assert recipe.job is not None

    def test_dict_model_config_with_evaluator(self, mock_file_system, model_manager_config, device_manager_config):
        """Test that dict model config works with evaluator_config.

        This verifies that GlobalEvaluator correctly handles dict model config.
        """
        from nvflare.edge.tools.edge_fed_buff_recipe import EdgeFedBuffRecipe, EvaluatorConfig

        evaluator_config = EvaluatorConfig(
            custom_dataset={"data": [[0, 0], [1, 1]], "label": [0, 1]},
        )

        recipe = EdgeFedBuffRecipe(
            job_name="test_edge_dict_eval",
            model={"class_path": "torch.nn.Linear", "args": {"in_features": 10, "out_features": 2}},
            model_manager_config=model_manager_config,
            device_manager_config=device_manager_config,
            evaluator_config=evaluator_config,
        )

        assert recipe.job is not None
        # Verify model is stored as dict
        assert isinstance(recipe.model, dict)
        assert recipe.model["path"] == "torch.nn.Linear"

    def test_relative_path_accepted_if_exists(
        self, mock_file_system, simple_pt_model, model_manager_config, device_manager_config
    ):
        """Test that existing relative paths are accepted and bundled."""
        from nvflare.edge.tools.edge_fed_buff_recipe import EdgeFedBuffRecipe

        # This should not raise since relative paths are now supported
        recipe = EdgeFedBuffRecipe(
            job_name="test_edge",
            model=simple_pt_model,
            model_manager_config=model_manager_config,
            device_manager_config=device_manager_config,
            initial_ckpt="relative/path/model.pt",
        )
        assert recipe is not None


@pytest.mark.skipif(not executorch_available, reason="executorch not installed")
class TestETFedBuffRecipeSimBasic:
    """Tests for ETFedBuffRecipe — skipped when executorch is absent."""

    @pytest.fixture
    def simple_device_model(self):
        import torch.nn as nn

        from nvflare.edge.models.model import DeviceModel

        return DeviceModel(nn.Linear(10, 2))

    @pytest.fixture
    def model_manager_config(self):
        from nvflare.edge.tools.edge_fed_buff_recipe import ModelManagerConfig

        return ModelManagerConfig(max_model_version=10, num_updates_for_model=5)

    @pytest.fixture
    def device_manager_config(self):
        from nvflare.edge.tools.edge_fed_buff_recipe import DeviceManagerConfig

        return DeviceManagerConfig(device_selection_size=10)

    def test_sim_basic(self, mock_file_system, simple_device_model, model_manager_config, device_manager_config):
        """Basic ETFedBuffRecipe initialization with required args."""
        from nvflare.edge.tools.et_fed_buff_recipe import ETFedBuffRecipe

        recipe = ETFedBuffRecipe(
            job_name="test_et_fedbuff",
            device_model=simple_device_model,
            input_shape=(1, 10),
            output_shape=(1,),
            model_manager_config=model_manager_config,
            device_manager_config=device_manager_config,
        )
        assert recipe.job is not None


class TestETFedBuffRecipeWithoutExecutorch:
    """Negative tests for ETFedBuffRecipe — verify graceful error when executorch is absent."""

    def test_raises_import_error_without_executorch(self, mock_file_system):
        """ETFedBuffRecipe.__init__ raises ImportError with install link when executorch is missing.

        Uses mock so this test always runs regardless of whether executorch is installed.
        """
        import torch.nn as nn

        from nvflare.edge.models.model import DeviceModel
        from nvflare.edge.tools.edge_fed_buff_recipe import DeviceManagerConfig, ModelManagerConfig
        from nvflare.edge.tools.et_fed_buff_recipe import ETFedBuffRecipe

        device_model = DeviceModel(nn.Linear(10, 2))

        with patch("nvflare.edge.tools.et_fed_buff_recipe.importlib.util.find_spec", return_value=None):
            with pytest.raises(ImportError, match="executorch"):
                ETFedBuffRecipe(
                    job_name="test_no_et",
                    device_model=device_model,
                    input_shape=(1, 10),
                    output_shape=(1,),
                    model_manager_config=ModelManagerConfig(max_model_version=10, num_updates_for_model=5),
                    device_manager_config=DeviceManagerConfig(device_selection_size=10),
                )
