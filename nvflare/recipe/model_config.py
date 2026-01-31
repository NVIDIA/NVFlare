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

"""ModelConfig dataclass and validation utilities for recipe model configuration."""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

from nvflare.fuel.utils.constants import FrameworkType

logger = logging.getLogger(__name__)

# Known file extensions by framework (for guidance, not strict enforcement)
KNOWN_EXTENSIONS = {
    FrameworkType.PYTORCH: (".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".onnx", ".mar"),
    FrameworkType.TENSORFLOW: (".h5", ".keras", ".pb", ".ckpt", ".tflite", ".weights.h5", ".onnx"),
    FrameworkType.NUMPY: (".npy", ".npz"),
    FrameworkType.RAW: (".npy", ".npz", ".pkl", ".joblib", ".json", ".onnx", ".safetensors"),
}

# Frameworks that support checkpoint-only mode (without model class)
CHECKPOINT_ONLY_FRAMEWORKS = {
    FrameworkType.TENSORFLOW,  # Keras .h5/SavedModel contains full model
    FrameworkType.NUMPY,  # .npy is just data
    FrameworkType.RAW,  # Depends on format
}


def validate_checkpoint_path(
    path: Optional[str],
    framework: FrameworkType,
    has_model: bool = False,
) -> None:
    """Validate checkpoint path format without checking file existence.

    Args:
        path: The checkpoint path to validate.
        framework: The ML framework type.
        has_model: Whether a model class/dict is also provided.

    Raises:
        TypeError: If path is not a string.
        ValueError: If path is not absolute, or if PyTorch ckpt provided without model.
    """
    if path is None:
        return

    # Type check - must be string
    if not isinstance(path, str):
        raise TypeError(f"initial_ckpt must be str, got {type(path).__name__}")

    # Must be absolute path
    if not os.path.isabs(path):
        raise ValueError(f"initial_ckpt must be an absolute path, got: {path}")

    # PyTorch requires model class for architecture (state_dict only contains weights)
    if framework == FrameworkType.PYTORCH and not has_model:
        raise ValueError(
            "PyTorch initial_ckpt requires initial_model (class instance or dict) "
            "because .pt/.pth files only contain state_dict, not model architecture. "
            "Provide initial_model=YourModel() or initial_model={'path': '...', 'args': {...}}"
        )

    # Check file extension - warn on unknown but don't error
    ext = os.path.splitext(path)[1].lower()

    # No extension - might be directory-based model (SavedModel, HF model dir)
    if not ext:
        logger.debug(f"initial_ckpt '{path}' has no extension - assuming directory-based model")
        return

    # Check against known extensions
    known = KNOWN_EXTENSIONS.get(framework, ())
    if known and ext not in known:
        logger.warning(
            f"initial_ckpt has unrecognized extension '{ext}' for framework {framework.name}. "
            f"Common extensions: {known}. Proceeding - ensure your persistor can handle this format."
        )


def detect_framework_from_model(model: Any) -> Optional[FrameworkType]:
    """Detect framework type from model object.

    Detection order: PyTorch -> NumPy -> TensorFlow

    Args:
        model: The model object to inspect.

    Returns:
        FrameworkType if detected, None otherwise.
    """
    if model is None:
        return None

    # Check for PyTorch first
    try:
        import torch.nn as nn

        if isinstance(model, nn.Module):
            return FrameworkType.PYTORCH
    except ImportError:
        pass

    # Check for NumPy second
    try:
        import numpy as np

        if isinstance(model, (np.ndarray, list)):
            return FrameworkType.NUMPY
    except ImportError:
        pass

    # Check for TensorFlow/Keras last
    try:
        import tensorflow as tf

        if isinstance(model, tf.keras.Model):
            return FrameworkType.TENSORFLOW
    except ImportError:
        pass

    return None


@dataclass
class ModelConfig:
    """Configuration for model input in recipes.

    Supports three input modes:
    1. Model class instance (e.g., Net()) - backward compatible
    2. Dict config ({"path": "module.Class", "args": {...}}) - new
    3. Checkpoint path (absolute path to weights file) - new

    Attributes:
        model: Model class instance, dict config, or None.
        initial_ckpt: Absolute path to checkpoint file (may not exist locally).
        framework: The ML framework type. Auto-detected if not provided.
        model_class_path: Fully qualified class path (extracted from dict or derived).
        model_args: Constructor arguments for the model class.
    """

    model: Union[Any, Dict, None] = None
    initial_ckpt: Optional[str] = None
    framework: Optional[FrameworkType] = None
    model_class_path: Optional[str] = field(default=None, init=False)
    model_args: Optional[Dict] = field(default=None, init=False)

    def __post_init__(self):
        """Parse model input and validate configuration."""
        self._parse_model_input()
        self._auto_detect_framework()
        self._validate()

    def _parse_model_input(self):
        """Parse model input to extract class path and args if dict."""
        if isinstance(self.model, dict):
            # Dict config mode: {"path": "...", "args": {...}}
            self.model_class_path = self.model.get("path")
            self.model_args = self.model.get("args", {})

            if not self.model_class_path:
                raise ValueError(
                    "Dict model config must have 'path' key with fully qualified class path. "
                    "Example: {'path': 'my_module.models.Net', 'args': {'num_classes': 10}}"
                )
        elif self.model is not None:
            # Model instance mode: extract class path from object
            cls = type(self.model)
            self.model_class_path = f"{cls.__module__}.{cls.__qualname__}"
            self.model_args = None  # Args not available from instance

    def _auto_detect_framework(self):
        """Auto-detect framework from model if not explicitly set."""
        if self.framework is not None:
            return

        if self.model is not None and not isinstance(self.model, dict):
            detected = detect_framework_from_model(self.model)
            if detected:
                self.framework = detected
                logger.debug(f"Auto-detected framework: {self.framework.name}")

    def _validate(self):
        """Validate the configuration."""
        has_model = self.model is not None

        # Validate checkpoint path
        if self.initial_ckpt is not None:
            if self.framework is None:
                raise ValueError(
                    "framework must be specified when using initial_ckpt without a model instance. "
                    "Example: framework=FrameworkType.PYTORCH"
                )
            validate_checkpoint_path(self.initial_ckpt, self.framework, has_model)

    @property
    def has_model(self) -> bool:
        """Check if a model (instance or dict) is provided."""
        return self.model is not None

    @property
    def has_checkpoint(self) -> bool:
        """Check if a checkpoint path is provided."""
        return self.initial_ckpt is not None

    @property
    def is_dict_config(self) -> bool:
        """Check if model is provided as dict config."""
        return isinstance(self.model, dict)

    @property
    def model_instance(self) -> Optional[Any]:
        """Get the model instance if available (None for dict config)."""
        if self.is_dict_config:
            return None
        return self.model

    def to_config_dict(self) -> Dict[str, Any]:
        """Export model configuration as dict for JSON serialization.

        Returns:
            Dict containing model class path, args, and checkpoint info.
        """
        config = {}

        if self.model_class_path:
            config["model_class_path"] = self.model_class_path

        if self.model_args:
            config["model_args"] = self.model_args

        if self.initial_ckpt:
            config["initial_ckpt"] = self.initial_ckpt

        if self.framework:
            config["framework"] = self.framework.value

        return config
