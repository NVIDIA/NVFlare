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

"""ModelConfigHandler for composition-based model configuration in recipes.

This module provides a standalone handler class that recipes use via composition
(not inheritance) to handle model configuration, persistor creation, and job setup.
"""

import logging
from typing import Any, Dict, Optional, Union

from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.fuel.utils.constants import FrameworkType
from nvflare.recipe.model_config import ModelConfig

logger = logging.getLogger(__name__)


class ModelConfigHandler:
    """Handler for model configuration in recipes (composition pattern).

    This class is instantiated by recipes to handle model input parsing,
    persistor creation, and model component setup. It is NOT a base class -
    recipes create their own handler instance.

    Usage in a recipe:
        ```python
        class MyRecipe(Recipe):
            def __init__(self, initial_model, initial_ckpt, framework, ...):
                # Create handler instance (composition)
                self._model_handler = ModelConfigHandler(
                    model=initial_model,
                    initial_ckpt=initial_ckpt,
                    framework=framework
                )

                # Use handler to setup model components
                persistor_id = self._model_handler.setup_model_components(job)
        ```

    Attributes:
        config: The parsed ModelConfig object.
    """

    def __init__(
        self,
        model: Union[Any, Dict, None] = None,
        initial_ckpt: Optional[str] = None,
        framework: Optional[FrameworkType] = None,
    ):
        """Initialize the handler with model configuration.

        Args:
            model: Model class instance, dict config, or None.
                - Instance: e.g., Net() - existing behavior
                - Dict: {"path": "module.Class", "args": {...}} - new
                - None: no initial model
            initial_ckpt: Absolute path to checkpoint file.
                May not exist on local machine (server-side path).
            framework: The ML framework type. Required if model is dict or None.
                Auto-detected from model instance if not provided.

        Raises:
            TypeError: If initial_ckpt is not a string.
            ValueError: If validation fails (e.g., relative path, missing framework).
        """
        self.config = ModelConfig(
            model=model,
            initial_ckpt=initial_ckpt,
            framework=framework,
        )

    @property
    def framework(self) -> Optional[FrameworkType]:
        """Get the framework type."""
        return self.config.framework

    @property
    def has_model(self) -> bool:
        """Check if a model is configured."""
        return self.config.has_model

    @property
    def has_checkpoint(self) -> bool:
        """Check if a checkpoint is configured."""
        return self.config.has_checkpoint

    @property
    def model_instance(self) -> Optional[Any]:
        """Get the model instance if available."""
        return self.config.model_instance

    def create_persistor(
        self,
        custom_persistor: Optional[ModelPersistor] = None,
        **kwargs,
    ) -> Optional[ModelPersistor]:
        """Create an appropriate persistor for the configured model.

        Args:
            custom_persistor: User-provided persistor (takes precedence).
            **kwargs: Additional arguments passed to persistor constructor.

        Returns:
            ModelPersistor instance, or None if no model/checkpoint configured.
        """
        # User-provided persistor takes precedence
        if custom_persistor is not None:
            return custom_persistor

        # No model or checkpoint - no persistor needed
        if not self.has_model and not self.has_checkpoint:
            return None

        framework = self.config.framework
        if framework is None:
            logger.warning("No framework specified, cannot create persistor automatically")
            return None

        # Create framework-specific persistor
        if framework == FrameworkType.PYTORCH:
            return self._create_pt_persistor(**kwargs)
        elif framework == FrameworkType.TENSORFLOW:
            return self._create_tf_persistor(**kwargs)
        elif framework == FrameworkType.NUMPY:
            return self._create_np_persistor(**kwargs)
        elif framework == FrameworkType.RAW:
            return self._create_raw_persistor(**kwargs)
        else:
            logger.warning(f"Unknown framework {framework}, cannot create persistor automatically")
            return None

    def _build_persistor_kwargs(self, base_kwargs: Dict, user_kwargs: Dict) -> Dict:
        """Build final persistor kwargs with checkpoint and user overrides.

        Args:
            base_kwargs: Framework-specific base arguments.
            user_kwargs: User-provided kwargs to merge.

        Returns:
            Merged kwargs dict.
        """
        # Add checkpoint path if available
        if self.config.initial_ckpt:
            base_kwargs["source_ckpt_file_full_name"] = self.config.initial_ckpt

        # User kwargs override base
        base_kwargs.update(user_kwargs)
        return base_kwargs

    def _create_pt_persistor(self, **kwargs) -> ModelPersistor:
        """Create PyTorch persistor."""
        from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

        persistor_kwargs = {}

        # Add model if available (for architecture)
        if self.config.model_instance is not None:
            persistor_kwargs["model"] = self.config.model_instance
        elif self.config.is_dict_config:
            # Dict config - pass class path as string (resolved at runtime)
            persistor_kwargs["model"] = self.config.model_class_path

        return PTFileModelPersistor(**self._build_persistor_kwargs(persistor_kwargs, kwargs))

    def _create_tf_persistor(self, **kwargs) -> ModelPersistor:
        """Create TensorFlow persistor."""
        from nvflare.app_opt.tf.model_persistor import TFModelPersistor

        persistor_kwargs = {}

        # Add model if available
        if self.config.model_instance is not None:
            persistor_kwargs["model"] = self.config.model_instance

        return TFModelPersistor(**self._build_persistor_kwargs(persistor_kwargs, kwargs))

    def _create_np_persistor(self, **kwargs) -> ModelPersistor:
        """Create NumPy persistor."""
        from nvflare.app_common.np.np_model_persistor import NPModelPersistor

        persistor_kwargs = {}

        # Add initial model if available (as list for JSON serialization)
        if self.config.model_instance is not None:
            model = self.config.model_instance
            # Convert numpy array to list for JSON serialization
            if hasattr(model, "tolist"):
                persistor_kwargs["initial_model"] = model.tolist()
            elif isinstance(model, list):
                persistor_kwargs["initial_model"] = model

        return NPModelPersistor(**self._build_persistor_kwargs(persistor_kwargs, kwargs))

    def _create_raw_persistor(self, **kwargs) -> ModelPersistor:
        """Create RAW/sklearn persistor."""
        from nvflare.app_opt.sklearn.joblib_model_param_persistor import JoblibModelParamPersistor

        persistor_kwargs = {}

        # Add initial params if available
        if self.config.model_instance is not None:
            persistor_kwargs["initial_params"] = self.config.model_instance
        elif self.config.model_args:
            persistor_kwargs["initial_params"] = self.config.model_args

        return JoblibModelParamPersistor(**self._build_persistor_kwargs(persistor_kwargs, kwargs))

    def setup_model_components(
        self,
        job,
        custom_persistor: Optional[ModelPersistor] = None,
        persistor_id: str = "persistor",
        **kwargs,
    ) -> str:
        """Setup model and persistor components in a job.

        This method creates the appropriate persistor and adds it to the job.
        It handles the full setup flow for model components.

        Args:
            job: The FedJob or BaseFedJob instance.
            custom_persistor: User-provided persistor (takes precedence).
            persistor_id: ID to use for the persistor component.
            **kwargs: Additional arguments passed to persistor constructor.

        Returns:
            str: The persistor_id if persistor was added, empty string otherwise.
        """
        persistor = self.create_persistor(custom_persistor=custom_persistor, **kwargs)

        if persistor is None:
            return ""

        # Add persistor to job
        result_id = job.to_server(persistor, id=persistor_id)

        # Store in job.comp_ids if available
        if hasattr(job, "comp_ids"):
            job.comp_ids["persistor_id"] = result_id

        return result_id

    def get_initial_model_params(self) -> Optional[Dict]:
        """Get initial model parameters as dict (for controllers that accept it).

        This is used by controllers like FedAvg that can accept initial_model
        as a dict of parameters directly (alternative to using persistor).

        Returns:
            Dict of model parameters, or None if not available.
        """
        if not self.has_model:
            return None

        model = self.config.model_instance
        if model is None:
            # Dict config - can't get params without instantiation
            return None

        # Try to get state_dict for PyTorch models
        if hasattr(model, "state_dict"):
            return model.state_dict()

        # Try to get weights for Keras models
        if hasattr(model, "get_weights"):
            weights = model.get_weights()
            return {"weights": weights}

        # For numpy arrays, return directly
        if hasattr(model, "tolist"):
            return {"weights": model}

        # For dicts, return directly
        if isinstance(model, dict):
            return model

        return None

    def to_config_dict(self) -> Dict[str, Any]:
        """Export configuration as dict for JSON serialization.

        Returns:
            Dict containing the model configuration.
        """
        return self.config.to_config_dict()
