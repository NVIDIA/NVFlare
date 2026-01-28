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

"""Unit tests for lazy model instantiation feature in PTModel and PTFileModelPersistor."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch.nn as nn

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.app_opt.pt.job_config.model import PTModel


class SimpleTestModel(nn.Module):
    """Simple test model for unit tests."""

    def __init__(self, input_dim=10, output_dim=5):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class TestPTModelLazyInstantiation(unittest.TestCase):
    """Test cases for PTModel with lazy instantiation support."""

    def test_ptmodel_with_nn_module(self):
        """Test PTModel with nn.Module instance (existing functionality)."""
        model = SimpleTestModel(input_dim=10, output_dim=5)
        pt_model = PTModel(model=model)

        self.assertIsInstance(pt_model.model, nn.Module)
        self.assertEqual(pt_model.model, model)

    def test_ptmodel_with_dict_config(self):
        """Test PTModel with dict config for lazy instantiation."""
        model_config = {
            "path": "tests.unit_test.app_opt.pt.test_lazy_model_instantiation.SimpleTestModel",
            "args": {"input_dim": 20, "output_dim": 10},
        }

        pt_model = PTModel(model=model_config)

        self.assertIsInstance(pt_model.model, dict)
        self.assertEqual(pt_model.model["path"], model_config["path"])
        self.assertEqual(pt_model.model["args"], model_config["args"])

    def test_ptmodel_with_dict_config_no_args(self):
        """Test PTModel with dict config without args."""
        model_config = {
            "path": "tests.unit_test.app_opt.pt.test_lazy_model_instantiation.SimpleTestModel",
        }

        pt_model = PTModel(model=model_config)

        self.assertIsInstance(pt_model.model, dict)
        self.assertEqual(pt_model.model["path"], model_config["path"])
        self.assertNotIn("args", pt_model.model)


class TestPTFileModelPersistorLazyInstantiation(unittest.TestCase):
    """Test cases for PTFileModelPersistor with lazy instantiation support."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.fl_ctx = self._create_mock_fl_context()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_mock_fl_context(self):
        """Create a mock FLContext for testing."""
        from nvflare.apis.fl_constant import FLContextKey

        fl_ctx = FLContext()
        fl_ctx.set_prop(FLContextKey.APP_ROOT, self.temp_dir)

        # Create mock run args
        mock_args = MagicMock()
        mock_args.env = "environment.json"
        fl_ctx.set_prop(FLContextKey.ARGS, mock_args)

        return fl_ctx

    def test_persistor_with_nn_module(self):
        """Test PTFileModelPersistor with nn.Module instance."""
        model = SimpleTestModel(input_dim=10, output_dim=5)
        persistor = PTFileModelPersistor(model=model)

        # Initialize persistor
        persistor.handle_event(EventType.START_RUN, self.fl_ctx)

        # Model should remain as nn.Module
        self.assertIsInstance(persistor.model, nn.Module)
        self.assertEqual(type(persistor.model).__name__, "SimpleTestModel")

    def test_persistor_with_dict_config(self):
        """Test PTFileModelPersistor with dict config for lazy instantiation."""
        model_config = {
            "path": "tests.unit_test.app_opt.pt.test_lazy_model_instantiation.SimpleTestModel",
            "args": {"input_dim": 20, "output_dim": 10},
        }

        persistor = PTFileModelPersistor(model=model_config)

        # Before initialization, model should be dict
        self.assertIsInstance(persistor.model, dict)

        # Initialize persistor - should instantiate model from config
        persistor.handle_event(EventType.START_RUN, self.fl_ctx)

        # After initialization, model should be instantiated
        self.assertIsInstance(persistor.model, nn.Module)
        self.assertEqual(type(persistor.model).__name__, "SimpleTestModel")
        self.assertEqual(persistor.model.linear.in_features, 20)
        self.assertEqual(persistor.model.linear.out_features, 10)

    def test_persistor_with_dict_config_no_args(self):
        """Test PTFileModelPersistor with dict config without args."""
        model_config = {
            "path": "tests.unit_test.app_opt.pt.test_lazy_model_instantiation.SimpleTestModel",
        }

        persistor = PTFileModelPersistor(model=model_config)

        # Initialize persistor
        persistor.handle_event(EventType.START_RUN, self.fl_ctx)

        # Model should be instantiated with default args
        self.assertIsInstance(persistor.model, nn.Module)
        self.assertEqual(type(persistor.model).__name__, "SimpleTestModel")
        self.assertEqual(persistor.model.linear.in_features, 10)  # default
        self.assertEqual(persistor.model.linear.out_features, 5)  # default

    def test_persistor_with_invalid_dict_config_missing_path(self):
        """Test PTFileModelPersistor with invalid dict config (missing 'path')."""
        invalid_config = {"args": {"input_dim": 10}}

        persistor = PTFileModelPersistor(model=invalid_config)

        # Initialize should fail with system_panic
        with patch.object(persistor, "system_panic") as mock_panic:
            persistor.handle_event(EventType.START_RUN, self.fl_ctx)
            mock_panic.assert_called_once()
            call_args = mock_panic.call_args
            self.assertIn("must contain 'path' key", call_args[1]["reason"])

    def test_persistor_with_invalid_class_path(self):
        """Test PTFileModelPersistor with invalid class path."""
        model_config = {
            "path": "nonexistent.module.InvalidModel",
            "args": {},
        }

        persistor = PTFileModelPersistor(model=model_config)

        # Initialize should fail with system_panic
        with patch.object(persistor, "system_panic") as mock_panic:
            persistor.handle_event(EventType.START_RUN, self.fl_ctx)
            mock_panic.assert_called_once()
            call_args = mock_panic.call_args
            self.assertIn("Cannot instantiate model from config", call_args[1]["reason"])

    def test_persistor_with_non_nn_module_class(self):
        """Test PTFileModelPersistor when instantiated class is not nn.Module."""
        model_config = {
            "path": "builtins.dict",  # dict is not an nn.Module
            "args": {},
        }

        persistor = PTFileModelPersistor(model=model_config)

        # Initialize should fail with system_panic
        with patch.object(persistor, "system_panic") as mock_panic:
            persistor.handle_event(EventType.START_RUN, self.fl_ctx)
            mock_panic.assert_called_once()
            call_args = mock_panic.call_args
            self.assertIn("must be torch.nn.Module", call_args[1]["reason"])

    def test_persistor_string_component_id_still_works(self):
        """Test that existing string component ID functionality still works."""
        model = SimpleTestModel()

        # Mock FL context with engine
        fl_ctx = self._create_mock_fl_context()

        # Mock engine that returns the model
        mock_engine = MagicMock()
        mock_engine.get_component.return_value = model

        # Mock get_engine to return our mock engine
        with patch.object(fl_ctx, "get_engine", return_value=mock_engine):
            persistor = PTFileModelPersistor(model="model_component_id")

            # Before initialization, model should be string
            self.assertEqual(persistor.model, "model_component_id")

            # Initialize persistor
            persistor.handle_event(EventType.START_RUN, fl_ctx)

            # Model should be resolved from engine
            mock_engine.get_component.assert_called_once_with("model_component_id")


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with existing code."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_nn_module_still_works(self):
        """Test that existing code using nn.Module still works."""
        model = SimpleTestModel(input_dim=12, output_dim=6)

        # Create PTModel with nn.Module
        pt_model = PTModel(model=model)
        self.assertIsInstance(pt_model.model, nn.Module)

        # Create persistor
        persistor = PTFileModelPersistor(model=pt_model.model)
        self.assertIsInstance(persistor.model, nn.Module)

        # Initialize persistor
        from nvflare.apis.fl_constant import FLContextKey

        fl_ctx = FLContext()
        fl_ctx.set_prop(FLContextKey.APP_ROOT, self.temp_dir)

        mock_args = MagicMock()
        mock_args.env = "environment.json"
        fl_ctx.set_prop(FLContextKey.ARGS, mock_args)

        persistor.handle_event(EventType.START_RUN, fl_ctx)

        # Model should still be the same nn.Module
        self.assertIsInstance(persistor.model, nn.Module)
        self.assertEqual(persistor.model.linear.in_features, 12)
        self.assertEqual(persistor.model.linear.out_features, 6)


if __name__ == "__main__":
    unittest.main()
