# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Integration test for sklearn-linear example using Recipe API."""

import os
import tempfile

import pytest

from nvflare.app_opt.sklearn import SklearnFedAvgRecipe
from nvflare.recipe import SimEnv


class TestSklearnLinearRecipe:
    """Test sklearn linear regression with federated averaging."""

    @pytest.fixture(scope="class")
    def test_data_path(self):
        """Create a small test dataset."""
        import numpy as np
        import pandas as pd

        # Create minimal HIGGS-like test data
        n_samples = 100
        n_features = 28
        data = np.random.randn(n_samples, n_features + 1)
        data[:, 0] = np.random.randint(0, 2, n_samples)  # Binary labels

        temp_dir = tempfile.mkdtemp()
        csv_path = os.path.join(temp_dir, "HIGGS_test.csv")
        pd.DataFrame(data).to_csv(csv_path, index=False, header=False)
        
        yield csv_path
        
        # Cleanup
        os.remove(csv_path)
        os.rmdir(temp_dir)

    def test_sklearn_linear_recipe(self, test_data_path):
        """Test sklearn linear example with Recipe API."""
        n_clients = 2
        num_rounds = 2

        recipe = SklearnFedAvgRecipe(
            name="test_sklearn_linear",
            min_clients=n_clients,
            num_rounds=num_rounds,
            initial_params={
                "n_classes": 2,
                "learning_rate": "constant",
                "eta0": 1e-4,
                "loss": "log_loss",
                "penalty": "l2",
                "fit_intercept": 1,
            },
            train_script=os.path.join(os.path.dirname(__file__), "client.py"),
            train_args=f"--data_path {test_data_path} --train_start 0 --train_end 60 "
                       f"--valid_start 60 --valid_end 100",
        )

        env = SimEnv(num_clients=n_clients, num_threads=n_clients)
        run = recipe.execute(env)

        # Verify execution completed successfully
        assert run.get_job_id() == "test_sklearn_linear"
        assert run.get_status() is None  # SimEnv doesn't track status
        assert run.get_result() is not None
        
        # Check that workspace was created
        workspace = run.get_result()
        assert os.path.exists(workspace)
        print(f"✅ Test passed! Workspace created at: {workspace}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

