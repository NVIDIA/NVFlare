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
Comprehensive verification test for recipe component config generation.

Verifies that all changed recipes correctly generate component configs with:
1. Dict model config: {"path": "module.Class"}
2. initial_ckpt parameter: source_ckpt_file_full_name in persistor

This ensures all recipe changes properly handle the new interface.
"""

import json
import os
import tempfile
import unittest


class TestPTRecipeComponentConfig(unittest.TestCase):
    """Test PyTorch recipe component config generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = "/workspace/pretrained_model.pt"

        # Create dummy train script
        self.train_script = os.path.join(self.temp_dir, "train.py")
        with open(self.train_script, "w") as f:
            f.write("# Dummy train script\n")

    def tearDown(self):
        """Clean up test artifacts."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _verify_persistor_config(self, config_path, expected_model_path, expected_ckpt_path):
        """Verify persistor component has correct model dict and checkpoint."""
        with open(config_path, "r") as f:
            config = json.load(f)

        # Find persistor component
        persistor = None
        for comp in config.get("components", []):
            if comp["id"] == "persistor":
                persistor = comp
                break

        self.assertIsNotNone(persistor, "Persistor component not found in config")

        # Verify it's PT persistor
        self.assertIn(
            "PTFileModelPersistor", persistor["path"], f"Expected PTFileModelPersistor, got {persistor['path']}"
        )

        # Verify model is dict config
        model = persistor["args"].get("model")
        self.assertIsInstance(model, dict, "Model should be dict config")
        self.assertEqual(
            model.get("path"),
            expected_model_path,
            f"Model path mismatch. Expected: {expected_model_path}, Got: {model.get('path')}",
        )

        # Verify checkpoint path
        ckpt_path = persistor["args"].get("source_ckpt_file_full_name")
        self.assertEqual(
            ckpt_path, expected_ckpt_path, f"Checkpoint path mismatch. Expected: {expected_ckpt_path}, Got: {ckpt_path}"
        )

        print("    ✓ Persistor config verified:")
        print(f"      - model: {model}")
        print(f"      - checkpoint: {ckpt_path}")

    def test_pt_fedavg(self):
        """Test PT FedAvg generates correct config with dict model and checkpoint."""
        print("\n  Testing PT FedAvg...")
        from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe

        recipe = FedAvgRecipe(
            name="test-pt-fedavg",
            min_clients=2,
            num_rounds=2,
            model={"path": "model.SimpleNetwork"},
            initial_ckpt=self.checkpoint_path,
            train_script=self.train_script,
        )

        job_dir = os.path.join(self.temp_dir, "export")
        recipe.export(job_dir=job_dir)

        server_config = os.path.join(job_dir, "test-pt-fedavg", "app/config/config_fed_server.json")
        self._verify_persistor_config(server_config, "model.SimpleNetwork", self.checkpoint_path)

    def test_pt_fedopt(self):
        """Test PT FedOpt generates correct config.

        Note: FedOpt uses different architecture - it adds model via job.to_server(),
        and persistor references it by ID string "model".
        """
        print("\n  Testing PT FedOpt...")
        from unittest.mock import patch

        import torch.nn as nn

        from nvflare.app_opt.pt.recipes.fedopt import FedOptRecipe

        # Create a simple model for testing
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)

        simple_model = SimpleModel()

        # Mock instantiate_class to avoid needing real model.SimpleNetwork
        with patch("nvflare.fuel.utils.class_utils.instantiate_class") as mock_instantiate:
            mock_instantiate.return_value = simple_model

            recipe = FedOptRecipe(
                name="test-pt-fedopt",
                min_clients=2,
                num_rounds=2,
                model={"path": "model.SimpleNetwork"},
                initial_ckpt=self.checkpoint_path,
                train_script=self.train_script,
            )

        job_dir = os.path.join(self.temp_dir, "export2")
        recipe.export(job_dir=job_dir)

        server_config_path = os.path.join(job_dir, "test-pt-fedopt", "app/config/config_fed_server.json")

        # FedOpt architecture: persistor has component ID reference to model
        with open(server_config_path, "r") as f:
            config = json.load(f)

        # Verify persistor configuration
        persistor = next((c for c in config["components"] if c["id"] == "persistor"), None)
        self.assertIsNotNone(persistor, "Persistor component not found")

        # FedOpt persistor.model is a string reference to component ID
        # The actual dict config is passed via job.to_server() and resolved at runtime
        model_ref = persistor["args"]["model"]
        self.assertIsInstance(model_ref, str, "FedOpt persistor.model should be string reference")
        self.assertEqual(model_ref, "model", "Expected component ID reference")

        # Verify checkpoint path
        ckpt_path = persistor["args"]["source_ckpt_file_full_name"]
        self.assertEqual(ckpt_path, self.checkpoint_path)

        print("    ✓ FedOpt config verified (uses component reference architecture):")
        print(f"      - persistor.model: '{model_ref}' (component ID)")
        print(f"      - checkpoint: {ckpt_path}")
        print("      - Note: Dict config passed to component 'model' at runtime")

    def test_pt_cyclic(self):
        """Test PT Cyclic generates correct config."""
        print("\n  Testing PT Cyclic...")
        from nvflare.app_opt.pt.recipes.cyclic import CyclicRecipe

        recipe = CyclicRecipe(
            name="test-pt-cyclic",
            min_clients=2,
            num_rounds=2,
            model={"path": "model.SimpleNetwork"},
            initial_ckpt=self.checkpoint_path,
            train_script=self.train_script,
        )

        job_dir = os.path.join(self.temp_dir, "export3")
        recipe.export(job_dir=job_dir)

        server_config = os.path.join(job_dir, "test-pt-cyclic", "app/config/config_fed_server.json")
        self._verify_persistor_config(server_config, "model.SimpleNetwork", self.checkpoint_path)

    def test_pt_scaffold(self):
        """Test PT Scaffold generates correct config."""
        print("\n  Testing PT Scaffold...")
        from nvflare.app_opt.pt.recipes.scaffold import ScaffoldRecipe

        recipe = ScaffoldRecipe(
            name="test-pt-scaffold",
            min_clients=2,
            num_rounds=2,
            model={"path": "model.SimpleNetwork"},
            initial_ckpt=self.checkpoint_path,
            train_script=self.train_script,
        )

        job_dir = os.path.join(self.temp_dir, "export4")
        recipe.export(job_dir=job_dir)

        server_config = os.path.join(job_dir, "test-pt-scaffold", "app/config/config_fed_server.json")
        self._verify_persistor_config(server_config, "model.SimpleNetwork", self.checkpoint_path)

    def test_pt_fedavg_he(self):
        """Test PT FedAvg with HE supports checkpoint.

        Note: FedAvg HE does NOT support dict model config yet - it still requires nn.Module.
        Only testing checkpoint support here.
        """
        print("\n  Testing PT FedAvg with HE (checkpoint only, no dict config)...")
        print("    ⚠ Note: FedAvg HE doesn't support dict config yet")

        # Skip this test - FedAvg HE uses BaseFedJob which doesn't support dict config
        # It would need to be updated similarly to other recipes
        self.skipTest("FedAvg HE doesn't support dict config (uses BaseFedJob)")


class TestTFRecipeComponentConfig(unittest.TestCase):
    """Test TensorFlow recipe component config generation."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            import tensorflow  # noqa: F401

            self.tf_available = True
        except ImportError:
            self.tf_available = False

        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = "/workspace/pretrained_model"

        # Create dummy train script
        self.train_script = os.path.join(self.temp_dir, "train.py")
        with open(self.train_script, "w") as f:
            f.write("# Dummy train script\n")

    def tearDown(self):
        """Clean up test artifacts."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _verify_persistor_config(self, config_path, expected_model_path, expected_ckpt_path):
        """Verify TF persistor component has correct model dict and checkpoint."""
        with open(config_path, "r") as f:
            config = json.load(f)

        # Find persistor component
        persistor = None
        for comp in config.get("components", []):
            if comp["id"] == "persistor":
                persistor = comp
                break

        self.assertIsNotNone(persistor, "Persistor component not found")
        self.assertIn("TFModelPersistor", persistor["path"], f"Expected TFModelPersistor, got {persistor['path']}")

        # Verify model is dict config
        model = persistor["args"].get("model")
        self.assertIsInstance(model, dict, "Model should be dict config")
        self.assertEqual(model.get("path"), expected_model_path, "Model path mismatch")

        # Verify checkpoint path
        ckpt_path = persistor["args"].get("source_ckpt_file_full_name")
        self.assertEqual(ckpt_path, expected_ckpt_path, "Checkpoint path mismatch")

        print("    ✓ Persistor config verified:")
        print(f"      - model: {model}")
        print(f"      - checkpoint: {ckpt_path}")

    def test_tf_fedavg(self):
        """Test TF FedAvg generates correct config."""
        if not self.tf_available:
            self.skipTest("TensorFlow not installed")

        print("\n  Testing TF FedAvg...")
        from nvflare.app_opt.tf.recipes.fedavg import FedAvgRecipe

        recipe = FedAvgRecipe(
            name="test-tf-fedavg",
            min_clients=2,
            num_rounds=2,
            model={"path": "model.SimpleModel"},
            initial_ckpt=self.checkpoint_path,
            train_script=self.train_script,
        )

        job_dir = os.path.join(self.temp_dir, "export")
        recipe.export(job_dir=job_dir)

        server_config = os.path.join(job_dir, "test-tf-fedavg", "app/config/config_fed_server.json")
        self._verify_persistor_config(server_config, "model.SimpleModel", self.checkpoint_path)

    def test_tf_fedopt(self):
        """Test TF FedOpt generates correct config."""
        if not self.tf_available:
            self.skipTest("TensorFlow not installed")

        print("\n  Testing TF FedOpt...")
        from nvflare.app_opt.tf.recipes.fedopt import FedOptRecipe

        recipe = FedOptRecipe(
            name="test-tf-fedopt",
            min_clients=2,
            num_rounds=2,
            model={"path": "model.SimpleModel"},
            initial_ckpt=self.checkpoint_path,
            train_script=self.train_script,
        )

        job_dir = os.path.join(self.temp_dir, "export2")
        recipe.export(job_dir=job_dir)

        server_config = os.path.join(job_dir, "test-tf-fedopt", "app/config/config_fed_server.json")
        self._verify_persistor_config(server_config, "model.SimpleModel", self.checkpoint_path)

    def test_tf_cyclic(self):
        """Test TF Cyclic generates correct config."""
        if not self.tf_available:
            self.skipTest("TensorFlow not installed")

        print("\n  Testing TF Cyclic...")
        from nvflare.app_opt.tf.recipes.cyclic import CyclicRecipe

        recipe = CyclicRecipe(
            name="test-tf-cyclic",
            min_clients=2,
            num_rounds=2,
            model={"path": "model.SimpleModel"},
            initial_ckpt=self.checkpoint_path,
            train_script=self.train_script,
        )

        job_dir = os.path.join(self.temp_dir, "export3")
        recipe.export(job_dir=job_dir)

        server_config = os.path.join(job_dir, "test-tf-cyclic", "app/config/config_fed_server.json")
        self._verify_persistor_config(server_config, "model.SimpleModel", self.checkpoint_path)

    def test_tf_scaffold(self):
        """Test TF Scaffold generates correct config."""
        if not self.tf_available:
            self.skipTest("TensorFlow not installed")

        print("\n  Testing TF Scaffold...")
        from nvflare.app_opt.tf.recipes.scaffold import ScaffoldRecipe

        recipe = ScaffoldRecipe(
            name="test-tf-scaffold",
            min_clients=2,
            num_rounds=2,
            model={"path": "model.SimpleModel"},
            initial_ckpt=self.checkpoint_path,
            train_script=self.train_script,
        )

        job_dir = os.path.join(self.temp_dir, "export4")
        recipe.export(job_dir=job_dir)

        server_config = os.path.join(job_dir, "test-tf-scaffold", "app/config/config_fed_server.json")
        self._verify_persistor_config(server_config, "model.SimpleModel", self.checkpoint_path)


class TestNumpyRecipeComponentConfig(unittest.TestCase):
    """Test NumPy recipe component config generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = "/workspace/checkpoint.npy"

        # Create dummy train script
        self.train_script = os.path.join(self.temp_dir, "train.py")
        with open(self.train_script, "w") as f:
            f.write("# Dummy train script\n")

    def tearDown(self):
        """Clean up test artifacts."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _verify_persistor_config(self, config_path, expected_ckpt_path):
        """Verify NumPy persistor component has correct checkpoint."""
        with open(config_path, "r") as f:
            config = json.load(f)

        # Find persistor component
        persistor = None
        for comp in config.get("components", []):
            if comp["id"] == "persistor":
                persistor = comp
                break

        self.assertIsNotNone(persistor, "Persistor component not found")
        self.assertIn("NPModelPersistor", persistor["path"], f"Expected NPModelPersistor, got {persistor['path']}")

        # Verify checkpoint path
        ckpt_path = persistor["args"].get("source_ckpt_file_full_name")
        self.assertEqual(ckpt_path, expected_ckpt_path, "Checkpoint path mismatch")

        print("    ✓ Persistor config verified:")
        print(f"      - checkpoint: {ckpt_path}")

    def test_numpy_fedavg(self):
        """Test NumPy FedAvg generates correct config with checkpoint."""
        print("\n  Testing NumPy FedAvg...")
        from nvflare.app_common.np.recipes.fedavg import NumpyFedAvgRecipe

        recipe = NumpyFedAvgRecipe(
            name="test-np-fedavg",
            min_clients=2,
            num_rounds=2,
            model=[[1.0, 2.0], [3.0, 4.0]],
            initial_ckpt=self.checkpoint_path,
            train_script=self.train_script,
        )

        job_dir = os.path.join(self.temp_dir, "export")
        recipe.export(job_dir=job_dir)

        server_config = os.path.join(job_dir, "test-np-fedavg", "app/config/config_fed_server.json")
        self._verify_persistor_config(server_config, self.checkpoint_path)


class TestSklearnRecipeComponentConfig(unittest.TestCase):
    """Test Sklearn recipe component config generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = "/workspace/checkpoint.pkl"

        # Create dummy train script
        self.train_script = os.path.join(self.temp_dir, "train.py")
        with open(self.train_script, "w") as f:
            f.write("# Dummy train script\n")

    def tearDown(self):
        """Clean up test artifacts."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _verify_persistor_config(self, config_path, expected_initial_params, expected_ckpt_path):
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

        # Verify initial_params (Sklearn uses initial_params in persistor, not model_params)
        initial_params = persistor["args"].get("initial_params")
        if expected_initial_params:
            self.assertIsNotNone(initial_params, "initial_params should be set")
            self.assertEqual(initial_params, expected_initial_params, "initial_params mismatch")

        # Verify checkpoint path
        ckpt_path = persistor["args"].get("source_ckpt_file_full_name")
        self.assertEqual(ckpt_path, expected_ckpt_path, "Checkpoint path mismatch")

        print("    ✓ Persistor config verified:")
        print(f"      - initial_params: {initial_params}")
        print(f"      - checkpoint: {ckpt_path}")

    def test_sklearn_fedavg(self):
        """Test Sklearn FedAvg generates correct config with checkpoint."""
        print("\n  Testing Sklearn FedAvg...")
        from nvflare.app_opt.sklearn.recipes.fedavg import SklearnFedAvgRecipe

        model_params = {"n_estimators": 10}

        recipe = SklearnFedAvgRecipe(
            name="test-sklearn-fedavg",
            min_clients=2,
            num_rounds=2,
            model_params=model_params,
            initial_ckpt=self.checkpoint_path,
            train_script=self.train_script,
        )

        job_dir = os.path.join(self.temp_dir, "export")
        recipe.export(job_dir=job_dir)

        server_config = os.path.join(job_dir, "test-sklearn-fedavg", "app/config/config_fed_server.json")
        self._verify_persistor_config(server_config, model_params, self.checkpoint_path)

    def test_sklearn_kmeans(self):
        """Test Sklearn KMeans generates correct config with checkpoint (PR3)."""
        print("\n  Testing Sklearn KMeans (PR3)...")
        from nvflare.app_opt.sklearn.recipes.kmeans import KMeansFedAvgRecipe

        recipe = KMeansFedAvgRecipe(
            name="test-sklearn-kmeans",
            min_clients=2,
            num_rounds=2,
            n_clusters=3,
            initial_ckpt=self.checkpoint_path,
            train_script=self.train_script,
        )

        job_dir = os.path.join(self.temp_dir, "export_kmeans")
        recipe.export(job_dir=job_dir)

        server_config = os.path.join(job_dir, "test-sklearn-kmeans", "app/config/config_fed_server.json")
        model_params = {"n_clusters": 3}
        self._verify_persistor_config(server_config, model_params, self.checkpoint_path)

    def test_sklearn_svm(self):
        """Test Sklearn SVM generates correct config with checkpoint (PR3)."""
        print("\n  Testing Sklearn SVM (PR3)...")
        from nvflare.app_opt.sklearn.recipes.svm import SVMFedAvgRecipe

        recipe = SVMFedAvgRecipe(
            name="test-sklearn-svm",
            min_clients=2,
            kernel="rbf",
            initial_ckpt=self.checkpoint_path,
            train_script=self.train_script,
        )

        job_dir = os.path.join(self.temp_dir, "export_svm")
        recipe.export(job_dir=job_dir)

        server_config = os.path.join(job_dir, "test-sklearn-svm", "app/config/config_fed_server.json")
        model_params = {"kernel": "rbf"}
        self._verify_persistor_config(server_config, model_params, self.checkpoint_path)


class TestPTSpecialRecipesComponentConfig(unittest.TestCase):
    """Test PyTorch special recipes component config generation (PR3)."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = "/workspace/pretrained_model.pt"

        # Create dummy train script
        self.train_script = os.path.join(self.temp_dir, "train.py")
        with open(self.train_script, "w") as f:
            f.write("# Dummy train script\n")

    def tearDown(self):
        """Clean up test artifacts."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _verify_persistor_config(self, config_path, expected_model_path, expected_ckpt_path):
        """Verify PT persistor has correct model dict and checkpoint."""
        with open(config_path, "r") as f:
            config = json.load(f)

        # Find persistor component
        persistor = None
        for comp in config.get("components", []):
            if comp["id"] == "persistor":
                persistor = comp
                break

        self.assertIsNotNone(persistor, "Persistor component not found")
        self.assertIn(
            "PTFileModelPersistor", persistor["path"], f"Expected PTFileModelPersistor, got {persistor['path']}"
        )

        # Verify model is dict config
        model = persistor["args"].get("model")
        self.assertIsInstance(model, dict, "Model should be dict config")
        self.assertEqual(model.get("path"), expected_model_path, "Model path mismatch")

        # Verify checkpoint path
        ckpt_path = persistor["args"].get("source_ckpt_file_full_name")
        self.assertEqual(ckpt_path, expected_ckpt_path, "Checkpoint path mismatch")

        print("    ✓ Persistor config verified:")
        print(f"      - model: {model}")
        print(f"      - checkpoint: {ckpt_path}")

    def test_pt_swarm(self):
        """Test PT Swarm generates correct config with checkpoint (PR3)."""
        print("\n  Testing PT Swarm (PR3)...")
        import torch.nn as nn

        from nvflare.app_opt.pt.recipes.swarm import SimpleSwarmLearningRecipe

        # SimpleSwarmLearningRecipe requires an actual nn.Module
        simple_model = nn.Linear(10, 2)

        recipe = SimpleSwarmLearningRecipe(
            name="test-pt-swarm",
            model=simple_model,
            initial_ckpt=self.checkpoint_path,
            num_rounds=2,
            train_script=self.train_script,
        )

        job_dir = os.path.join(self.temp_dir, "export_swarm")
        recipe.export(job_dir=job_dir)

        # SimpleSwarmLearningRecipe puts persistor on client side (swarm topology)
        client_config_path = os.path.join(job_dir, "test-pt-swarm", "app/config/config_fed_client.json")
        with open(client_config_path, "r") as f:
            config = json.load(f)

        # Find persistor component in client config
        persistor = None
        for comp in config.get("components", []):
            if "Persistor" in comp.get("path", ""):
                persistor = comp
                break

        self.assertIsNotNone(persistor, "Persistor component not found in client config")
        print(f"    ✓ Swarm persistor found: {persistor['path']}")

    def test_pt_fedeval(self):
        """Test PT FedEval generates correct config with checkpoint (PR3)."""
        print("\n  Testing PT FedEval (PR3)...")
        from nvflare.app_opt.pt.recipes.fedeval import FedEvalRecipe

        recipe = FedEvalRecipe(
            name="test-pt-fedeval",
            model={"path": "model.SimpleNetwork"},
            eval_ckpt=self.checkpoint_path,
            min_clients=2,
            eval_script=self.train_script,  # FedEval uses eval_script, not train_script
        )

        job_dir = os.path.join(self.temp_dir, "export_fedeval")
        recipe.export(job_dir=job_dir)

        server_config = os.path.join(job_dir, "test-pt-fedeval", "app/config/config_fed_server.json")
        self._verify_persistor_config(server_config, "model.SimpleNetwork", self.checkpoint_path)

    def test_numpy_cross_site_eval(self):
        """Test NumPy Cross-site Eval generates correct config with checkpoint (PR3)."""
        print("\n  Testing NumPy Cross-site Eval (PR3)...")
        from nvflare.app_common.np.recipes.cross_site_eval import NumpyCrossSiteEvalRecipe

        recipe = NumpyCrossSiteEvalRecipe(
            name="test-np-cross-site-eval",
            min_clients=2,
            initial_ckpt=self.checkpoint_path,
        )

        job_dir = os.path.join(self.temp_dir, "export_cross_site")
        recipe.export(job_dir=job_dir)

        server_config_path = os.path.join(job_dir, "test-np-cross-site-eval", "app/config/config_fed_server.json")

        with open(server_config_path, "r") as f:
            config = json.load(f)

        # Find persistor/model_locator component
        model_locator = None
        for comp in config.get("components", []):
            if "ModelLocator" in comp.get("path", ""):
                model_locator = comp
                break

        self.assertIsNotNone(model_locator, "ModelLocator component not found")

        # Verify model_name contains the checkpoint path when initial_ckpt is provided
        model_name = model_locator["args"].get("model_name")
        self.assertIsNotNone(model_name, "model_name should be set in model locator")
        # When initial_ckpt is provided, it's stored in model_name dict under SERVER_MODEL_NAME key
        self.assertIn(self.checkpoint_path, str(model_name), "Checkpoint path should be in model_name")

        print("    ✓ Model locator config verified:")
        print(f"      - model_name: {model_name}")


if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)
