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

"""Unit tests for model_config_handler module."""

import pytest

from nvflare.fuel.utils.constants import FrameworkType
from nvflare.recipe.model_config_handler import ModelConfigHandler


class TestModelConfigHandlerInit:
    """Tests for ModelConfigHandler initialization."""

    def test_empty_handler(self):
        """Empty handler should be valid."""
        handler = ModelConfigHandler()
        assert not handler.has_model
        assert not handler.has_checkpoint
        assert handler.framework is None

    def test_handler_with_model_instance(self):
        """Handler with model instance should work."""
        try:
            import torch.nn as nn

            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(10, 2)

            model = Net()
            handler = ModelConfigHandler(model=model)

            assert handler.has_model
            assert handler.model_instance is model
            assert handler.framework == FrameworkType.PYTORCH
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_handler_with_dict_config(self):
        """Handler with dict config should parse correctly."""
        model_dict = {"path": "my_module.Net", "args": {"size": 10}}
        handler = ModelConfigHandler(model=model_dict, framework=FrameworkType.PYTORCH)

        assert handler.has_model
        assert handler.model_instance is None
        assert handler.framework == FrameworkType.PYTORCH

    def test_handler_with_checkpoint(self):
        """Handler with checkpoint should store path."""
        handler = ModelConfigHandler(
            initial_ckpt="/data/model.h5",
            framework=FrameworkType.TENSORFLOW,
        )

        assert handler.has_checkpoint
        assert handler.config.initial_ckpt == "/data/model.h5"

    def test_handler_validates_checkpoint_type(self):
        """Handler should validate checkpoint type."""
        with pytest.raises(TypeError, match="must be str"):
            ModelConfigHandler(initial_ckpt=123, framework=FrameworkType.PYTORCH)

    def test_handler_validates_absolute_path(self):
        """Handler should validate absolute path."""
        with pytest.raises(ValueError, match="must be an absolute path"):
            ModelConfigHandler(
                model={"path": "test.Net", "args": {}},
                initial_ckpt="relative/path.pt",
                framework=FrameworkType.PYTORCH,
            )

    def test_pytorch_ckpt_without_model_raises(self):
        """PyTorch checkpoint without model should raise."""
        with pytest.raises(ValueError, match="requires initial_model"):
            ModelConfigHandler(
                initial_ckpt="/data/model.pt",
                framework=FrameworkType.PYTORCH,
            )

    def test_tensorflow_ckpt_without_model_allowed(self):
        """TensorFlow checkpoint without model should be allowed."""
        handler = ModelConfigHandler(
            initial_ckpt="/data/model.h5",
            framework=FrameworkType.TENSORFLOW,
        )
        assert handler.has_checkpoint
        assert not handler.has_model


class TestModelConfigHandlerCreatePersistor:
    """Tests for create_persistor method."""

    def test_no_model_no_checkpoint_returns_none(self):
        """No model or checkpoint should return None."""
        handler = ModelConfigHandler()
        persistor = handler.create_persistor()
        assert persistor is None

    def test_custom_persistor_takes_precedence(self):
        """Custom persistor should be returned as-is."""
        from unittest.mock import MagicMock

        custom = MagicMock()
        handler = ModelConfigHandler(framework=FrameworkType.PYTORCH)
        result = handler.create_persistor(custom_persistor=custom)
        assert result is custom

    def test_pytorch_persistor_created(self):
        """PyTorch persistor should be created for PT framework."""
        try:
            import torch.nn as nn

            from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(10, 2)

            model = Net()
            handler = ModelConfigHandler(model=model)
            persistor = handler.create_persistor()

            assert isinstance(persistor, PTFileModelPersistor)
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_tensorflow_persistor_created(self):
        """TensorFlow persistor should be created for TF framework."""
        try:
            import tensorflow as tf

            from nvflare.app_opt.tf.model_persistor import TFModelPersistor

            model = tf.keras.Sequential([tf.keras.layers.Dense(2)])
            handler = ModelConfigHandler(model=model)
            persistor = handler.create_persistor()

            assert isinstance(persistor, TFModelPersistor)
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_numpy_persistor_created(self):
        """NumPy persistor should be created for NP framework."""
        try:
            import numpy as np

            from nvflare.app_common.np.np_model_persistor import NPModelPersistor

            model = np.array([[1, 2], [3, 4]])
            handler = ModelConfigHandler(model=model)
            persistor = handler.create_persistor()

            assert isinstance(persistor, NPModelPersistor)
        except ImportError:
            pytest.skip("NumPy not installed")

    def test_persistor_with_checkpoint(self):
        """Persistor should include checkpoint path."""
        try:
            import torch.nn as nn

            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(10, 2)

            model = Net()
            handler = ModelConfigHandler(
                model=model,
                initial_ckpt="/data/pretrained.pt",
            )
            persistor = handler.create_persistor()

            assert persistor.source_ckpt_file_full_name == "/data/pretrained.pt"
        except ImportError:
            pytest.skip("PyTorch not installed")


class TestModelConfigHandlerToConfigDict:
    """Tests for to_config_dict method."""

    def test_empty_handler_returns_empty(self):
        """Empty handler should return empty dict."""
        handler = ModelConfigHandler()
        result = handler.to_config_dict()
        assert result == {}

    def test_full_config_exported(self):
        """Full config should be exported."""
        model_dict = {"path": "my_module.Net", "args": {"size": 10}}
        handler = ModelConfigHandler(
            model=model_dict,
            initial_ckpt="/data/model.pt",
            framework=FrameworkType.PYTORCH,
        )

        result = handler.to_config_dict()

        assert result["model_class_path"] == "my_module.Net"
        assert result["model_args"] == {"size": 10}
        assert result["initial_ckpt"] == "/data/model.pt"
        assert "framework" in result
