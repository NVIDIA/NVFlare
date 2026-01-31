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

"""Unit tests for model_config module."""

import pytest

from nvflare.fuel.utils.constants import FrameworkType
from nvflare.recipe.model_config import (
    KNOWN_EXTENSIONS,
    ModelConfig,
    detect_framework_from_model,
    validate_checkpoint_path,
)


class TestValidateCheckpointPath:
    """Tests for validate_checkpoint_path function."""

    def test_none_path_passes(self):
        """None path should pass without error."""
        validate_checkpoint_path(None, FrameworkType.PYTORCH, has_model=True)

    def test_invalid_type_raises_type_error(self):
        """Non-string path should raise TypeError."""
        with pytest.raises(TypeError, match="initial_ckpt must be str"):
            validate_checkpoint_path(123, FrameworkType.PYTORCH, has_model=True)

        with pytest.raises(TypeError, match="initial_ckpt must be str"):
            validate_checkpoint_path(["path"], FrameworkType.PYTORCH, has_model=True)

    def test_relative_path_raises_value_error(self):
        """Relative path should raise ValueError."""
        with pytest.raises(ValueError, match="must be an absolute path"):
            validate_checkpoint_path("models/model.pt", FrameworkType.PYTORCH, has_model=True)

        with pytest.raises(ValueError, match="must be an absolute path"):
            validate_checkpoint_path("./model.pt", FrameworkType.PYTORCH, has_model=True)

    def test_absolute_path_passes(self):
        """Absolute path should pass."""
        validate_checkpoint_path("/data/models/model.pt", FrameworkType.PYTORCH, has_model=True)
        validate_checkpoint_path("/opt/models/model.h5", FrameworkType.TENSORFLOW, has_model=True)

    def test_pytorch_without_model_raises_error(self):
        """PyTorch checkpoint without model should raise ValueError."""
        with pytest.raises(ValueError, match="PyTorch initial_ckpt requires initial_model"):
            validate_checkpoint_path("/data/model.pt", FrameworkType.PYTORCH, has_model=False)

    def test_pytorch_with_model_passes(self):
        """PyTorch checkpoint with model should pass."""
        validate_checkpoint_path("/data/model.pt", FrameworkType.PYTORCH, has_model=True)

    def test_tensorflow_without_model_passes(self):
        """TensorFlow checkpoint without model should pass (can load full model)."""
        validate_checkpoint_path("/data/model.h5", FrameworkType.TENSORFLOW, has_model=False)

    def test_numpy_without_model_passes(self):
        """NumPy checkpoint without model should pass."""
        validate_checkpoint_path("/data/model.npy", FrameworkType.NUMPY, has_model=False)

    def test_unknown_extension_warns_but_passes(self, caplog):
        """Unknown extension should log warning but not raise error."""
        import logging

        with caplog.at_level(logging.WARNING):
            validate_checkpoint_path("/data/model.xyz", FrameworkType.PYTORCH, has_model=True)

        assert "unrecognized extension" in caplog.text.lower()

    def test_no_extension_logs_debug(self, caplog):
        """Path without extension should log debug message."""
        import logging

        with caplog.at_level(logging.DEBUG):
            validate_checkpoint_path("/data/saved_model", FrameworkType.TENSORFLOW, has_model=False)

        assert "directory-based model" in caplog.text.lower()

    def test_known_extensions_pass_silently(self, caplog):
        """Known extensions should not log warnings."""
        import logging

        with caplog.at_level(logging.WARNING):
            validate_checkpoint_path("/data/model.pt", FrameworkType.PYTORCH, has_model=True)
            validate_checkpoint_path("/data/model.pth", FrameworkType.PYTORCH, has_model=True)
            validate_checkpoint_path("/data/model.h5", FrameworkType.TENSORFLOW, has_model=False)

        assert "unrecognized" not in caplog.text.lower()


class TestDetectFrameworkFromModel:
    """Tests for detect_framework_from_model function."""

    def test_none_returns_none(self):
        """None model should return None."""
        assert detect_framework_from_model(None) is None

    def test_pytorch_model_detected(self):
        """PyTorch nn.Module should be detected."""
        try:
            import torch.nn as nn

            class SimpleNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(10, 2)

            model = SimpleNet()
            assert detect_framework_from_model(model) == FrameworkType.PYTORCH
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_tensorflow_model_detected(self):
        """TensorFlow keras.Model should be detected."""
        try:
            import tensorflow as tf

            model = tf.keras.Sequential([tf.keras.layers.Dense(2)])
            assert detect_framework_from_model(model) == FrameworkType.TENSORFLOW
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_numpy_array_detected(self):
        """NumPy array should be detected."""
        try:
            import numpy as np

            model = np.array([1, 2, 3])
            assert detect_framework_from_model(model) == FrameworkType.NUMPY
        except ImportError:
            pytest.skip("NumPy not installed")

    def test_list_not_detected_as_framework(self):
        """Python list should NOT auto-detect as any framework (ambiguous)."""
        model = [1, 2, 3]
        # Plain list is ambiguous - could be used with any framework
        # User should explicitly specify framework when using plain lists
        assert detect_framework_from_model(model) is None

    def test_unknown_type_returns_none(self):
        """Unknown type should return None."""
        assert detect_framework_from_model("string") is None
        assert detect_framework_from_model({"dict": "value"}) is None


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_empty_config(self):
        """Empty config should be valid."""
        config = ModelConfig()
        assert config.model is None
        assert config.initial_ckpt is None
        assert not config.has_model
        assert not config.has_checkpoint

    def test_model_instance_config(self):
        """Config with model instance should work."""
        try:
            import torch.nn as nn

            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(10, 2)

            model = Net()
            config = ModelConfig(model=model)

            assert config.has_model
            assert config.model_instance is model
            assert not config.is_dict_config
            assert config.framework == FrameworkType.PYTORCH
            assert "Net" in config.model_class_path
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_dict_config(self):
        """Config with dict should parse correctly."""
        model_dict = {"path": "my_module.models.Net", "args": {"num_classes": 10}}
        config = ModelConfig(model=model_dict, framework=FrameworkType.PYTORCH)

        assert config.has_model
        assert config.is_dict_config
        assert config.model_instance is None
        assert config.model_class_path == "my_module.models.Net"
        assert config.model_args == {"num_classes": 10}

    def test_dict_config_missing_path_raises(self):
        """Dict config without 'path' key should raise."""
        with pytest.raises(ValueError, match="must have 'path' key"):
            ModelConfig(model={"args": {}}, framework=FrameworkType.PYTORCH)

    def test_checkpoint_without_framework_raises(self):
        """Checkpoint without framework should raise when no model instance."""
        with pytest.raises(ValueError, match="framework must be specified"):
            ModelConfig(initial_ckpt="/data/model.pt")

    def test_checkpoint_with_framework(self):
        """Checkpoint with framework should work."""
        config = ModelConfig(
            initial_ckpt="/data/model.h5",
            framework=FrameworkType.TENSORFLOW,
        )
        assert config.has_checkpoint
        assert config.initial_ckpt == "/data/model.h5"

    def test_to_config_dict(self):
        """to_config_dict should export configuration."""
        model_dict = {"path": "my_module.Net", "args": {"size": 10}}
        config = ModelConfig(
            model=model_dict,
            initial_ckpt="/data/model.pt",
            framework=FrameworkType.PYTORCH,
        )

        result = config.to_config_dict()
        assert result["model_class_path"] == "my_module.Net"
        assert result["model_args"] == {"size": 10}
        assert result["initial_ckpt"] == "/data/model.pt"


class TestKnownExtensions:
    """Tests for KNOWN_EXTENSIONS constant."""

    def test_pytorch_extensions(self):
        """PyTorch should have common extensions."""
        pt_exts = KNOWN_EXTENSIONS[FrameworkType.PYTORCH]
        assert ".pt" in pt_exts
        assert ".pth" in pt_exts
        assert ".ckpt" in pt_exts
        assert ".safetensors" in pt_exts

    def test_tensorflow_extensions(self):
        """TensorFlow should have common extensions."""
        tf_exts = KNOWN_EXTENSIONS[FrameworkType.TENSORFLOW]
        assert ".h5" in tf_exts
        assert ".keras" in tf_exts
        assert ".pb" in tf_exts

    def test_numpy_extensions(self):
        """NumPy should have .npy and .npz."""
        np_exts = KNOWN_EXTENSIONS[FrameworkType.NUMPY]
        assert ".npy" in np_exts
        assert ".npz" in np_exts
