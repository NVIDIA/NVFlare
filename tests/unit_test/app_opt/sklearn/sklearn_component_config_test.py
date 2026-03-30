#!/usr/bin/env python3
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

"""
Test Sklearn recipe component config generation.

Asserts that exported recipes produce correct persistor config (initial_params, model_path).
"""

import json
import os
import tempfile
import unittest


class TestSklearnRecipeComponentConfig(unittest.TestCase):
    """Test Sklearn recipe component config generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = "/workspace/checkpoint.pkl"

        # Create dummy train script
        self.train_script = os.path.join(self.temp_dir, "train.py")
        with open(self.train_script, "w") as f:
            f.write("# Dummy train script\n")

    def tearDown(self):
        """Clean up test artifacts."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _verify_persistor_config(self, config_path, expected_initial_params, expected_model_path):
        """Verify Sklearn persistor component has correct config."""
        with open(config_path, "r") as f:
            config = json.load(f)

        # Find persistor component
        persistor = None
        for comp in config.get("components", []):
            if comp["id"] == "persistor":
                persistor = comp
                break

        self.assertIsNotNone(persistor, "Persistor component not found")
        self.assertIn("joblib", persistor["path"].lower(), f"Expected Joblib persistor, got {persistor['path']}")

        # Verify initial_params (Sklearn uses initial_params in persistor)
        initial_params = persistor["args"].get("initial_params")
        if expected_initial_params:
            self.assertIsNotNone(initial_params, "initial_params should be set")
            self.assertEqual(initial_params, expected_initial_params, "initial_params mismatch")

        # Verify model_path (Sklearn uses model_path, not source_ckpt_file_full_name)
        model_path = persistor["args"].get("model_path")
        self.assertEqual(model_path, expected_model_path, "model_path mismatch")

        print("    ✓ Persistor config verified:")
        print(f"      - initial_params: {initial_params}")
        print(f"      - model_path: {model_path}")

    def test_sklearn_fedavg(self):
        """Test Sklearn FedAvg generates correct config with model_path."""
        print("\n  Testing Sklearn FedAvg...")
        from nvflare.app_opt.sklearn.recipes.fedavg import SklearnFedAvgRecipe

        model_params = {"n_estimators": 10}

        recipe = SklearnFedAvgRecipe(
            name="test-sklearn-fedavg",
            min_clients=2,
            num_rounds=2,
            model_params=model_params,
            model_path=self.model_path,
            train_script=self.train_script,
        )

        job_dir = os.path.join(self.temp_dir, "export")
        recipe.export(job_dir=job_dir)

        server_config = os.path.join(job_dir, "test-sklearn-fedavg", "app/config/config_fed_server.json")
        self._verify_persistor_config(server_config, model_params, self.model_path)

    def test_sklearn_kmeans(self):
        """Test Sklearn KMeans generates correct config with model_path (PR3)."""
        print("\n  Testing Sklearn KMeans (PR3)...")
        from nvflare.app_opt.sklearn.recipes.kmeans import KMeansFedAvgRecipe

        recipe = KMeansFedAvgRecipe(
            name="test-sklearn-kmeans",
            min_clients=2,
            num_rounds=2,
            n_clusters=3,
            model_path=self.model_path,
            train_script=self.train_script,
        )

        job_dir = os.path.join(self.temp_dir, "export_kmeans")
        recipe.export(job_dir=job_dir)

        server_config = os.path.join(job_dir, "test-sklearn-kmeans", "app/config/config_fed_server.json")
        model_params = {"n_clusters": 3}
        self._verify_persistor_config(server_config, model_params, self.model_path)

    def test_sklearn_svm(self):
        """Test Sklearn SVM generates correct config with model_path (PR3)."""
        print("\n  Testing Sklearn SVM (PR3)...")
        from nvflare.app_opt.sklearn.recipes.svm import SVMFedAvgRecipe

        recipe = SVMFedAvgRecipe(
            name="test-sklearn-svm",
            min_clients=2,
            kernel="rbf",
            model_path=self.model_path,
            train_script=self.train_script,
        )

        job_dir = os.path.join(self.temp_dir, "export_svm")
        recipe.export(job_dir=job_dir)

        server_config = os.path.join(job_dir, "test-sklearn-svm", "app/config/config_fed_server.json")
        model_params = {"kernel": "rbf"}
        self._verify_persistor_config(server_config, model_params, self.model_path)
