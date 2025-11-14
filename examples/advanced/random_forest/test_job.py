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

"""Integration test for random_forest example using Recipe API."""

import json
import os
import tempfile

import pytest

from nvflare.app_opt.xgboost import XGBBaggingRecipe
from nvflare.recipe import SimEnv


class TestRandomForestRecipe:
    """Test XGBoost-based federated Random Forest with bagging aggregation."""

    @pytest.fixture(scope="class")
    def test_data_splits(self):
        """Create test data split files."""
        import numpy as np
        import pandas as pd

        # Create minimal HIGGS-like test data
        n_samples = 200
        n_features = 28
        data = np.random.randn(n_samples, n_features + 1)
        data[:, 0] = np.random.randint(0, 2, n_samples)  # Binary labels

        temp_dir = tempfile.mkdtemp()
        data_path = os.path.join(temp_dir, "HIGGS_test.csv")
        pd.DataFrame(data).to_csv(data_path, index=False, header=False)

        # Create data split files
        split_dir = os.path.join(temp_dir, "splits")
        os.makedirs(split_dir)

        for site_id in [1, 2]:
            split_file = os.path.join(split_dir, f"data_site-{site_id}.json")
            split_data = {
                "data_path": data_path,
                "data_index": {
                    "train": list(range((site_id - 1) * 50, site_id * 50)),
                    "valid": list(range(100, 150)),
                }
            }
            with open(split_file, "w") as f:
                json.dump(split_data, f)
        
        yield temp_dir, data_path, split_dir
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

    def test_random_forest_recipe(self, test_data_splits):
        """Test random forest example with Recipe API."""
        temp_dir, data_path, split_dir = test_data_splits
        n_clients = 2

        # Import here to ensure it's available
        from utils.higgs_data_loader import HIGGSDataLoader

        recipe = XGBBaggingRecipe(
            name="test_random_forest",
            num_rounds=2,
            xgb_params={
                "max_depth": 3,
                "eta": 0.1,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "tree_method": "hist",
                "nthread": 1,
                "num_parallel_tree": 5,
                "subsample": 0.5,
                "colsample_bytree": 0.5,
            },
            xgb_options={
                "early_stopping_rounds": 10,
            },
            num_local_parallel_tree=5,
            local_model_path=os.path.join(temp_dir, "local_model.json"),
            global_model_path=os.path.join(temp_dir, "global_model.json"),
            data_loader_id="dataloader",
        )

        for site_id in range(1, n_clients + 1):
            data_split_file = os.path.join(split_dir, f"data_site-{site_id}.json")
            dataloader = HIGGSDataLoader(data_split_filename=data_split_file)
            recipe.add_to_client(f"site-{site_id}", dataloader, lr_scale=1.0)

        client_names = [f"site-{i}" for i in range(1, n_clients + 1)]
        env = SimEnv(clients=client_names, num_threads=n_clients)
        run = recipe.execute(env)

        # Verify execution completed successfully
        assert run.get_job_id() == "test_random_forest"
        assert run.get_status() is None  # SimEnv doesn't track status
        assert run.get_result() is not None
        
        # Check that workspace was created
        workspace = run.get_result()
        assert os.path.exists(workspace)
        print(f"✅ Test passed! Workspace created at: {workspace}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

