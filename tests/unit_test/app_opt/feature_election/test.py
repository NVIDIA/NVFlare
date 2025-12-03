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

"""
Enhanced Unit Tests for Feature Election
Covers:
1. Initialization & Validation
2. Data Splitting Strategies
3. Job Configuration (FLARE)
4. Simulation (Election, Auto-tuning, Mask Application)
"""

import json
import sys
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from nvflare.app_opt.feature_election import FeatureElection, quick_election

PYIMPETUS_AVAILABLE = find_spec("PyImpetus") is not None


@pytest.fixture
def sample_data():
    """Create a consistent sample dataset for testing."""
    X, y = make_classification(n_samples=200, n_features=20, n_informative=10, n_redundant=5, random_state=42)
    feature_names = [f"feature_{i}" for i in range(20)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df


class TestConfigurationAndValidation:
    """Tests for class initialization, parameter validation, and Job Config generation."""

    def test_initialization_defaults(self):
        """Test default values."""
        fe = FeatureElection()
        assert fe.freedom_degree == 0.5
        assert fe.auto_tune is False
        assert fe.tuning_rounds == 5
        assert fe.fs_method == "lasso"

    def test_initialization_custom(self):
        """Test custom parameters including new auto-tune args."""
        fe = FeatureElection(freedom_degree=0.8, fs_method="random_forest", auto_tune=True, tuning_rounds=10)
        assert fe.freedom_degree == 0.8
        assert fe.auto_tune is True
        assert fe.tuning_rounds == 10

    def test_invalid_parameters(self):
        """Test parameter bounds."""
        with pytest.raises(ValueError, match="freedom_degree"):
            FeatureElection(freedom_degree=1.1)

        with pytest.raises(ValueError, match="aggregation_mode"):
            FeatureElection(aggregation_mode="invalid_mode")

    def test_create_flare_job_structure(self, tmp_path):
        """Test that the generated FL job contains all new fields (auto_tune, phases)."""
        fe = FeatureElection(freedom_degree=0.5, auto_tune=True, tuning_rounds=3)

        output_dir = tmp_path / "jobs"
        paths = fe.create_flare_job(
            job_name="autotune_job", output_dir=str(output_dir), min_clients=2, num_rounds=10  # Total FL rounds
        )

        # 1. Check file existence
        assert Path(paths["server_config"]).exists()
        assert Path(paths["client_config"]).exists()

        # 2. Validate Server Config
        with open(paths["server_config"]) as f:
            server_config = json.load(f)

        workflow_args = server_config["workflows"][0]["args"]

        # Check standard args
        assert workflow_args["freedom_degree"] == 0.5
        assert workflow_args["min_clients"] == 2

        # Check NEW auto-tune args
        assert workflow_args["auto_tune"] is True
        assert workflow_args["tuning_rounds"] == 3
        assert workflow_args["num_rounds"] == 10  # Should be passed to controller for FL phase

        # 3. Validate Client Config
        with open(paths["client_config"]) as f:
            client_config = json.load(f)

        exec_args = client_config["executors"][0]["executor"]["args"]
        assert exec_args["task_name"] == "feature_election"


class TestDataPreparation:
    """Tests for data splitting logic."""

    def test_split_stratified_counts(self, sample_data):
        fe = FeatureElection()
        splits = fe.prepare_data_splits(sample_data, "target", num_clients=3, split_strategy="stratified")

        assert len(splits) == 3
        # Check that we haven't lost data
        total_len = sum(len(x) for x, _ in splits)
        assert total_len == 200

    def test_split_invalid_ratios(self, sample_data):
        fe = FeatureElection()
        with pytest.raises(ValueError):
            fe.prepare_data_splits(sample_data, "target", split_ratios=[0.8, 0.8])  # > 1.0


class TestSimulationLogic:
    """
    Tests the in-memory simulation of the Feature Election process.
    This simulates what happens inside the FLARE Controller/Executor interaction.
    """

    def test_simulate_election_basic(self, sample_data):
        """Test standard one-shot election."""
        fe = FeatureElection(freedom_degree=0.5, fs_method="lasso", auto_tune=False)
        client_data = fe.prepare_data_splits(sample_data, "target", num_clients=2)

        stats = fe.simulate_election(client_data)

        assert fe.global_mask is not None
        assert 0 < np.sum(fe.global_mask) <= 20
        assert stats["freedom_degree"] == 0.5

    def test_simulate_election_with_autotune(self, sample_data):
        """
        Test that simulation runs with auto_tune=True.

        Note: In a pure simulation (without full FL communication overhead),
        we want to ensure the logic flows through the tuning steps.
        """
        # Start with a low freedom degree that likely needs adjustment
        initial_fd = 0.1
        fe = FeatureElection(freedom_degree=initial_fd, fs_method="lasso", auto_tune=True, tuning_rounds=3)

        client_data = fe.prepare_data_splits(sample_data, "target", num_clients=2)

        stats = fe.simulate_election(client_data)

        # The simulation should have updated the freedom_degree in the stats
        # It might be the same if 0.1 was optimal, but the object state should be consistent
        assert fe.global_mask is not None
        assert "freedom_degree" in stats

        # Ensure stats structure contains expected keys
        assert "num_features_selected" in stats
        assert "reduction_ratio" in stats

    def test_boundary_conditions(self, sample_data):
        """Test Intersection (FD=0) and Union (FD=1)."""
        client_data = FeatureElection().prepare_data_splits(sample_data, "target", num_clients=2)

        # Intersection
        fe_int = FeatureElection(freedom_degree=0.0)
        stats_int = fe_int.simulate_election(client_data)
        n_int = stats_int["num_features_selected"]

        # Union
        fe_union = FeatureElection(freedom_degree=1.0)
        stats_union = fe_union.simulate_election(client_data)
        n_union = stats_union["num_features_selected"]

        assert n_int <= n_union
        # Intersection should match intersection_features stat
        assert n_int == stats_int["intersection_features"]

    def test_apply_mask_consistency(self, sample_data):
        """Ensure applying the mask returns the correct dataframe shape."""
        fe = FeatureElection(freedom_degree=0.5)
        client_data = fe.prepare_data_splits(sample_data, "target", num_clients=2)
        fe.simulate_election(client_data)

        num_selected = np.sum(fe.global_mask)

        # Apply to new data
        X_new = sample_data.drop(columns=["target"])
        X_filtered = fe.apply_mask(X_new)

        assert X_filtered.shape[1] == num_selected
        assert X_filtered.shape[0] == 200


class TestQuickElectionHelper:
    """Test the 'one-line' helper function."""

    def test_quick_election_workflow(self, sample_data):
        """Test the end-to-end quick helper."""
        mask, stats = quick_election(
            sample_data, target_col="target", num_clients=2, fs_method="lasso", freedom_degree=0.6
        )

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert stats["num_clients"] == 2


@pytest.mark.skipif(not PYIMPETUS_AVAILABLE, reason="PyImpetus not installed")
class TestAdvancedFeatures:
    """Tests requiring optional dependencies."""

    def test_pyimpetus_method(self, sample_data):
        fe = FeatureElection(fs_method="pyimpetus")
        client_data = fe.prepare_data_splits(sample_data, "target", num_clients=2)
        stats = fe.simulate_election(client_data)

        assert stats["fs_method"] == "pyimpetus"
        assert fe.global_mask is not None


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
