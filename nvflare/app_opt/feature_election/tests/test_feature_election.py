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
Unit tests for Feature Election
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
import json

from nvflare.app_opt.feature_election import FeatureElection, quick_election

# Attempt to import the optional dependency pyimpetus
try:
    import pyimpetus

    PYIMPETUS_AVAILABLE = True
except ImportError:
    PYIMPETUS_AVAILABLE = False


class TestFeatureElection:
    """Test suite for FeatureElection class"""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing"""
        X, y = make_classification(n_samples=200, n_features=20, n_informative=10, n_redundant=5, random_state=42)
        feature_names = [f"feature_{i}" for i in range(20)]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y
        return df

    def test_initialization_valid(self):
        """Test valid initialization"""
        fe = FeatureElection(freedom_degree=0.5, fs_method="lasso")
        assert fe.freedom_degree == 0.5
        assert fe.fs_method == "lasso"
        assert fe.aggregation_mode == "weighted"
        assert fe.global_mask is None

    def test_initialization_invalid_freedom_degree(self):
        """Test invalid freedom degree raises error"""
        with pytest.raises(ValueError, match="freedom_degree must be between 0 and 1"):
            FeatureElection(freedom_degree=1.5)

        with pytest.raises(ValueError, match="freedom_degree must be between 0 and 1"):
            FeatureElection(freedom_degree=-0.1)

    def test_initialization_invalid_aggregation_mode(self):
        """Test invalid aggregation mode raises error"""
        with pytest.raises(ValueError, match="aggregation_mode must be"):
            FeatureElection(aggregation_mode="invalid")

    def test_data_splits_stratified(self, sample_data):
        """Test stratified data splitting"""
        fe = FeatureElection()
        client_data = fe.prepare_data_splits(sample_data, "target", num_clients=3, split_strategy="stratified")

        assert len(client_data) == 3
        total_samples = sum(len(X) for X, _ in client_data)
        assert total_samples == len(sample_data)

        # Check stratification - class ratios should be similar
        original_ratio = sample_data["target"].mean()
        for X, y in client_data:
            client_ratio = y.mean()
            assert abs(client_ratio - original_ratio) < 0.2  # Allow 20% deviation

    def test_data_splits_random(self, sample_data):
        """Test random data splitting"""
        fe = FeatureElection()
        client_data = fe.prepare_data_splits(sample_data, "target", num_clients=4, split_strategy="random")

        assert len(client_data) == 4
        total_samples = sum(len(X) for X, _ in client_data)
        assert total_samples == len(sample_data)

    def test_data_splits_custom_ratios(self, sample_data):
        """Test custom split ratios"""
        fe = FeatureElection()
        ratios = [0.5, 0.3, 0.2]
        client_data = fe.prepare_data_splits(
            sample_data, "target", num_clients=3, split_ratios=ratios, split_strategy="random"
        )

        assert len(client_data) == 3
        # Check approximate ratios (may vary slightly due to rounding)
        for i, (X, _) in enumerate(client_data):
            expected = int(len(sample_data) * ratios[i])
            assert abs(len(X) - expected) <= 5  # Allow small deviation

    def test_data_splits_invalid_ratios(self, sample_data):
        """Test invalid split ratios raise error"""
        fe = FeatureElection()
        with pytest.raises(ValueError, match="Split ratios must sum to 1"):
            fe.prepare_data_splits(sample_data, "target", split_ratios=[0.5, 0.3, 0.3])  # Sums to 1.1

    def test_simulate_election_basic(self, sample_data):
        """Test basic election simulation"""
        fe = FeatureElection(freedom_degree=0.5, fs_method="lasso")
        client_data = fe.prepare_data_splits(sample_data, "target", num_clients=3)

        stats = fe.simulate_election(client_data)

        # Check results
        assert fe.global_mask is not None
        assert len(fe.global_mask) == 20  # Number of features
        assert np.sum(fe.global_mask) > 0  # At least some features selected
        assert np.sum(fe.global_mask) <= 20  # Not more than original features

        # Check stats
        assert stats["num_clients"] == 3
        assert stats["num_features_original"] == 20
        assert stats["num_features_selected"] > 0
        assert 0 <= stats["reduction_ratio"] <= 1
        assert len(stats["client_stats"]) == 3

    def test_simulate_election_auto_tune(self, sample_data):
        """Test election with auto-tuning"""
        fe = FeatureElection(freedom_degree=0.5, fs_method="lasso", auto_tune=True)
        client_data = fe.prepare_data_splits(sample_data, "target", num_clients=3)

        stats = fe.simulate_election(client_data)

        # Freedom degree may have changed
        assert 0 <= fe.freedom_degree <= 1
        assert "freedom_degree" in stats

    @pytest.mark.skipif(not PYIMPETUS_AVAILABLE, reason="PyImpetus dependency not installed.")
    def test_freedom_degree_intersection(self, sample_data):
        """Test freedom_degree=0 gives intersection"""
        fe = FeatureElection(freedom_degree=0.0, fs_method="pyimpetus")
        client_data = fe.prepare_data_splits(sample_data, "target", num_clients=3)

        stats = fe.simulate_election(client_data)

        # With freedom_degree=0, should have intersection
        assert stats["num_features_selected"] == stats["intersection_features"]

    def test_freedom_degree_union(self, sample_data):
        """Test freedom_degree=1 gives union"""
        fe = FeatureElection(freedom_degree=1.0, fs_method="lasso")
        client_data = fe.prepare_data_splits(sample_data, "target", num_clients=3)

        stats = fe.simulate_election(client_data)

        # With freedom_degree=1, should have union
        assert stats["num_features_selected"] == stats["union_features"]

    def test_apply_mask(self, sample_data):
        """Test applying feature mask to new data"""
        fe = FeatureElection(freedom_degree=0.5, fs_method="lasso")
        client_data = fe.prepare_data_splits(sample_data, "target", num_clients=3)
        fe.simulate_election(client_data)

        X = sample_data.drop(columns=["target"])
        X_selected = fe.apply_mask(X)

        assert len(X_selected.columns) == np.sum(fe.global_mask)
        assert all(col in X.columns for col in X_selected.columns)

    def test_apply_mask_no_election(self, sample_data):
        """Test applying mask without running election raises error"""
        fe = FeatureElection()
        X = sample_data.drop(columns=["target"])

        with pytest.raises(ValueError, match="No global mask available"):
            fe.apply_mask(X)

    def test_save_and_load_results(self, sample_data, tmp_path):
        """Test saving and loading results"""
        fe = FeatureElection(freedom_degree=0.5, fs_method="lasso")
        client_data = fe.prepare_data_splits(sample_data, "target", num_clients=3)
        fe.simulate_election(client_data)

        # Save results
        filepath = tmp_path / "results.json"
        fe.save_results(str(filepath))
        assert filepath.exists()

        # Load results
        fe2 = FeatureElection()
        fe2.load_results(str(filepath))

        assert fe2.freedom_degree == fe.freedom_degree
        assert fe2.fs_method == fe.fs_method
        assert np.array_equal(fe2.global_mask, fe.global_mask)
        assert fe2.election_stats == fe.election_stats

    def test_create_flare_job(self, tmp_path):
        """Test FLARE job configuration generation"""
        fe = FeatureElection(freedom_degree=0.5, fs_method="lasso")

        output_dir = tmp_path / "jobs"
        paths = fe.create_flare_job(job_name="test_job", output_dir=str(output_dir), min_clients=2)

        # Check files were created
        assert "job_dir" in paths
        assert "server_config" in paths
        assert "client_config" in paths
        assert "meta" in paths

        # Verify server config
        with open(paths["server_config"]) as f:
            server_config = json.load(f)
        assert server_config["format_version"] == 2
        assert len(server_config["workflows"]) > 0

        # Verify client config
        with open(paths["client_config"]) as f:
            client_config = json.load(f)
        assert client_config["format_version"] == 2
        assert len(client_config["executors"]) > 0


class TestQuickElection:
    """Test suite for quick_election helper function"""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing"""
        X, y = make_classification(n_samples=200, n_features=20, n_informative=10, random_state=42)
        df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(20)])
        df["target"] = y
        return df

    def test_quick_election_basic(self, sample_data):
        """Test basic quick election"""
        mask, stats = quick_election(sample_data, target_col="target", num_clients=3, fs_method="lasso")

        assert isinstance(mask, np.ndarray)
        assert len(mask) == 20
        assert mask.dtype == bool
        assert isinstance(stats, dict)
        assert stats["num_clients"] == 3

    def test_quick_election_auto_tune(self, sample_data):
        """Test quick election with auto-tuning"""
        mask, stats = quick_election(sample_data, target_col="target", num_clients=3, auto_tune=True)

        assert "freedom_degree" in stats
        assert 0 <= stats["freedom_degree"] <= 1


class TestFeatureSelectionMethods:
    """Test different feature selection methods"""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing"""
        X, y = make_classification(n_samples=150, n_features=15, n_informative=8, random_state=42)
        df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(15)])
        df["target"] = y
        return df

    @pytest.mark.parametrize("method", ["lasso", "elastic_net", "random_forest", "mutual_info", "f_classif", "chi2"])
    def test_different_methods(self, sample_data, method):
        """Test that different FS methods work"""
        mask, stats = quick_election(sample_data, target_col="target", num_clients=2, fs_method=method)

        assert len(mask) == 15
        assert np.sum(mask) > 0  # At least some features selected
        assert stats["fs_method"] == method


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_small_dataset(self):
        """Test with very small dataset"""
        X, y = make_classification(n_samples=30, n_features=5, n_informative=3, random_state=42)
        df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(5)])
        df["target"] = y

        mask, stats = quick_election(df, target_col="target", num_clients=2, fs_method="lasso")

        assert len(mask) == 5

    def test_many_clients(self):
        """Test with many clients"""
        X, y = make_classification(n_samples=500, n_features=20, n_informative=10, random_state=42)
        df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(20)])
        df["target"] = y

        mask, stats = quick_election(df, target_col="target", num_clients=10, fs_method="lasso")

        assert stats["num_clients"] == 10

    def test_high_dimensional(self):
        """Test with high-dimensional data"""
        X, y = make_classification(n_samples=200, n_features=100, n_informative=20, random_state=42)
        df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(100)])
        df["target"] = y

        mask, stats = quick_election(df, target_col="target", num_clients=3, fs_method="lasso")

        assert len(mask) == 100
        # Should achieve significant reduction
        assert stats["reduction_ratio"] > 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
