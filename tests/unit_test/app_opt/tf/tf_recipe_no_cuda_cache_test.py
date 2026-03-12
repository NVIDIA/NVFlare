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

"""Tests that TF recipes do not expose cuda_empty_cache.

TF GPU memory is managed differently from PyTorch; torch.cuda.empty_cache()
is PyTorch-only and must not be offered on TF recipes.
"""

from unittest.mock import patch

import pytest

# All four TF recipes live under app_opt/tf and import TF components at the
# module level, so skip the whole file if TF is not installed.
pytest.importorskip("tensorflow", reason="TensorFlow not available")


@pytest.fixture
def mock_file_system():
    with (
        patch("os.path.isfile", return_value=True),
        patch("os.path.isdir", return_value=True),
        patch("os.path.exists", return_value=True),
    ):
        yield


class TestTFRecipesNoCudaEmptyCache:
    """Verify that TF recipes reject cuda_empty_cache as a parameter."""

    def test_tf_fedavg_rejects_cuda_empty_cache(self, mock_file_system):
        from nvflare.app_opt.tf.recipes.fedavg import FedAvgRecipe

        with pytest.raises(TypeError, match="cuda_empty_cache"):
            FedAvgRecipe(
                min_clients=2,
                num_rounds=2,
                train_script="train.py",
                cuda_empty_cache=True,
            )

    def test_tf_cyclic_rejects_cuda_empty_cache(self, mock_file_system):
        from nvflare.app_opt.tf.recipes.cyclic import CyclicRecipe

        with pytest.raises(TypeError, match="cuda_empty_cache"):
            CyclicRecipe(
                min_clients=2,
                num_rounds=2,
                train_script="train.py",
                cuda_empty_cache=True,
            )

    def test_tf_scaffold_rejects_cuda_empty_cache(self, mock_file_system):
        from nvflare.app_opt.tf.recipes.scaffold import ScaffoldRecipe

        with pytest.raises(TypeError, match="cuda_empty_cache"):
            ScaffoldRecipe(
                min_clients=2,
                num_rounds=2,
                train_script="train.py",
                cuda_empty_cache=True,
            )

    def test_tf_fedopt_rejects_cuda_empty_cache(self, mock_file_system):
        from nvflare.app_opt.tf.recipes.fedopt import FedOptRecipe

        with pytest.raises(TypeError, match="cuda_empty_cache"):
            FedOptRecipe(
                min_clients=2,
                num_rounds=2,
                train_script="train.py",
                cuda_empty_cache=True,
            )
