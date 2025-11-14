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

"""Integration test for sklearn-svm example using Recipe API."""

import os
import tempfile

import pytest

from nvflare.app_opt.sklearn import SVMFedAvgRecipe
from nvflare.recipe import SimEnv


class TestSklearnSVMRecipe:
    """Test sklearn SVM with federated support vector aggregation."""

    @pytest.fixture(scope="class")
    def test_data_path(self):
        """Create a small test dataset."""
        import numpy as np
        import pandas as pd
        from sklearn.datasets import load_breast_cancer

        # Load breast cancer dataset (binary classification)
        cancer = load_breast_cancer()
        # Use subset for faster testing
        data = np.column_stack([cancer.data[:200], cancer.target[:200]])

        temp_dir = tempfile.mkdtemp()
        csv_path = os.path.join(temp_dir, "breast_cancer_test.csv")
        pd.DataFrame(data).to_csv(csv_path, index=False, header=False)
        
        yield csv_path
        
        # Cleanup
        os.remove(csv_path)
        os.rmdir(temp_dir)

    @pytest.mark.parametrize("kernel,backend", [
        ("rbf", "sklearn"),
        ("linear", "sklearn"),
    ])
    def test_sklearn_svm_recipe(self, test_data_path, kernel, backend):
        """Test sklearn SVM example with Recipe API."""
        n_clients = 2

        recipe = SVMFedAvgRecipe(
            name=f"test_sklearn_svm_{kernel}_{backend}",
            min_clients=n_clients,
            kernel=kernel,
            backend=backend,
            train_script=os.path.join(os.path.dirname(__file__), "client.py"),
            train_args=f"--data_path {test_data_path} --kernel {kernel} --backend {backend} "
                       f"--train_start 0 --train_end 100 --valid_start 100 --valid_end 200",
        )

        env = SimEnv(num_clients=n_clients, num_threads=n_clients)
        run = recipe.execute(env)

        # Verify execution completed successfully
        assert run.get_job_id() == f"test_sklearn_svm_{kernel}_{backend}"
        assert run.get_status() is None  # SimEnv doesn't track status
        assert run.get_result() is not None
        
        # Check that workspace was created
        workspace = run.get_result()
        assert os.path.exists(workspace)
        print(f"✅ Test passed for {kernel}/{backend}! Workspace created at: {workspace}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

