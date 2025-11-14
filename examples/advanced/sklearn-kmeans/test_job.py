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

"""Integration test for sklearn-kmeans example using Recipe API."""

import os
import tempfile

import pytest

from nvflare.app_opt.sklearn import KMeansFedAvgRecipe
from nvflare.recipe import SimEnv


class TestSklearnKMeansRecipe:
    """Test sklearn K-Means clustering with federated averaging."""

    @pytest.fixture(scope="class")
    def test_data_path(self):
        """Create a small test dataset."""
        import numpy as np
        import pandas as pd
        from sklearn.datasets import load_iris

        # Load iris dataset
        iris = load_iris()
        data = np.column_stack([iris.data, iris.target])

        temp_dir = tempfile.mkdtemp()
        csv_path = os.path.join(temp_dir, "sklearn_iris_test.csv")
        pd.DataFrame(data).to_csv(csv_path, index=False, header=False)
        
        yield csv_path
        
        # Cleanup
        os.remove(csv_path)
        os.rmdir(temp_dir)

    def test_sklearn_kmeans_recipe(self, test_data_path):
        """Test sklearn K-Means example with Recipe API."""
        n_clients = 2
        num_rounds = 5
        n_clusters = 3

        recipe = KMeansFedAvgRecipe(
            name="test_sklearn_kmeans",
            min_clients=n_clients,
            num_rounds=num_rounds,
            n_clusters=n_clusters,
            train_script=os.path.join(os.path.dirname(__file__), "src", "kmeans_fl.py"),
            train_args=f"--data_path {test_data_path} --train_start 0 --train_end 50 "
                       f"--valid_start 0 --valid_end 150",
        )

        env = SimEnv(num_clients=n_clients, num_threads=n_clients)
        run = recipe.execute(env)

        # Verify execution completed successfully
        assert run.get_job_id() == "test_sklearn_kmeans"
        assert run.get_status() is None  # SimEnv doesn't track status
        assert run.get_result() is not None
        
        # Check that workspace was created
        workspace = run.get_result()
        assert os.path.exists(workspace)
        print(f"✅ Test passed! Workspace created at: {workspace}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

