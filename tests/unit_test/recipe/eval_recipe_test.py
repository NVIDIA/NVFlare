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

"""Tests for Numpy CrossSiteEval recipe.

FedEvalRecipe (PT) is tested in tests/unit_test/app_opt/pt/recipes/fed_eval_recipe_test.py.
"""

from unittest.mock import patch

import pytest

from nvflare.fuel.utils.secret_utils import PotentialSecretWarning


@pytest.fixture
def mock_file_system():
    """Mock file system operations for all tests."""
    with (
        patch("os.path.isfile", return_value=True),
        patch("os.path.isdir", return_value=True),
        patch("os.path.exists", return_value=True),
    ):
        yield


class TestNumpyCrossSiteEvalRecipe:
    """Test cases for NumpyCrossSiteEvalRecipe."""

    def test_warns_on_secret_in_eval_args(self, mock_file_system):
        from nvflare.app_common.np.recipes.cross_site_eval import NumpyCrossSiteEvalRecipe

        with pytest.warns(PotentialSecretWarning, match="eval_args"):
            NumpyCrossSiteEvalRecipe(
                name="secret_cse",
                min_clients=2,
                eval_script="evaluate.py",
                eval_args="--password hunter22x",
            )

    def test_warns_on_secret_assignment_in_external_command(self, mock_file_system):
        from nvflare.app_common.np.recipes.cross_site_eval import NumpyCrossSiteEvalRecipe

        with pytest.warns(PotentialSecretWarning, match="command") as record:
            NumpyCrossSiteEvalRecipe(
                name="secret_command_cse",
                min_clients=2,
                eval_script="evaluate.py",
                launch_external_process=True,
                command="env API_PASSWORD=hunter22x python3 -u",
            )

        assert all("hunter22x" not in str(warning.message) for warning in record)

    def test_basic_initialization(self, mock_file_system):
        """Test NumpyCrossSiteEvalRecipe basic initialization."""
        from nvflare.app_common.np.recipes.cross_site_eval import NumpyCrossSiteEvalRecipe

        recipe = NumpyCrossSiteEvalRecipe(
            name="test_cse",
            min_clients=2,
        )

        assert recipe.job is not None

    def test_initial_ckpt_accepted(self, mock_file_system):
        """Test that initial_ckpt parameter is accepted."""
        from nvflare.app_common.np.recipes.cross_site_eval import NumpyCrossSiteEvalRecipe

        recipe = NumpyCrossSiteEvalRecipe(
            name="test_cse_ckpt",
            min_clients=2,
            initial_ckpt="/abs/path/to/model.npy",
        )

        assert recipe.job is not None

    def test_with_model_dir(self, mock_file_system):
        """Test with model_dir specified."""
        from nvflare.app_common.np.recipes.cross_site_eval import NumpyCrossSiteEvalRecipe

        recipe = NumpyCrossSiteEvalRecipe(
            name="test_cse_dir",
            min_clients=2,
            model_dir="models",
            model_name={"server": "best_model.npy"},
        )

        assert recipe.job is not None

    def test_with_eval_script(self, mock_file_system):
        """Test with custom eval_script."""
        from nvflare.app_common.np.recipes.cross_site_eval import NumpyCrossSiteEvalRecipe

        recipe = NumpyCrossSiteEvalRecipe(
            name="test_cse_script",
            min_clients=2,
            eval_script="evaluate.py",
            eval_args="--data_root /tmp/data",
            initial_ckpt="/abs/path/to/model.npy",
        )

        assert recipe.job is not None

    def test_with_eval_script_external_process(self, mock_file_system):
        """Test with eval_script in external process mode."""
        from nvflare.app_common.np.recipes.cross_site_eval import NumpyCrossSiteEvalRecipe

        recipe = NumpyCrossSiteEvalRecipe(
            name="test_cse_external",
            min_clients=2,
            eval_script="evaluate.py",
            launch_external_process=True,
            command="python3 -u",
        )

        assert recipe.job is not None
