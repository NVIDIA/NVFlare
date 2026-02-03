# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any, Dict, Optional, Union

import tensorflow as tf

from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_opt.tf.model_persistor import TFModelPersistor
from nvflare.job_config.api import validate_object_for_job


class TFModel:
    def __init__(
        self,
        model: Union[tf.keras.Model, Dict[str, Any], None] = None,
        persistor: Optional[ModelPersistor] = None,
        initial_ckpt: Optional[str] = None,
    ):
        """TensorFlow model wrapper.

        Supports three input modes:
        1. tf.keras.Model instance - existing behavior
        2. Dict config {"path": "module.Class", "args": {...}} - new
        3. Checkpoint path only (TF can load full model from .h5/SavedModel) - new

        Args:
            model: Model input, can be:
                - tf.keras.Model: Model instance (existing behavior)
                - dict: {"path": "fully.qualified.Class", "args": {...}}
                - None: Use with initial_ckpt for checkpoint-only loading
            persistor (Optional[ModelPersistor]): Custom persistor.
                If provided, takes precedence over automatic creation.
            initial_ckpt (str, optional): Absolute path to checkpoint file or SavedModel dir.
                May not exist locally (server-side path).
                TensorFlow can load full model from .h5 or SavedModel directory.
        """
        self.model = model
        self.initial_ckpt = initial_ckpt

        if persistor:
            validate_object_for_job("persistor", persistor, ModelPersistor)
        self.persistor = persistor

        # Extract model class path if dict config
        self.model_class_path = None
        self.model_args = None
        if isinstance(model, dict):
            self.model_class_path = model.get("path")
            self.model_args = model.get("args", {})
            if not self.model_class_path:
                raise ValueError("Dict model config must have 'path' key with fully qualified class path.")

    def add_to_fed_job(self, job, ctx):
        """This method is used by Job API.

        Args:
            job: the Job object to add to
            ctx: Job Context

        Returns:
            str: persistor_id of component added
        """
        if self.persistor:
            persistor = self.persistor
        elif isinstance(self.model, tf.keras.Model):
            # Model instance provided
            persistor = self._create_persistor_for_model()
        elif isinstance(self.model, dict):
            # Dict config provided
            persistor = self._create_persistor_for_dict_config()
        elif self.initial_ckpt:
            # Checkpoint only (TF can load full model from file)
            persistor = self._create_persistor_for_checkpoint_only()
        else:
            raise ValueError(
                f"Unsupported model configuration. Provide tf.keras.Model, dict config, "
                f"or initial_ckpt path. Got model={type(self.model)}."
            )

        persistor_id = job.add_component(comp_id="persistor", obj=persistor, ctx=ctx)
        return persistor_id

    def _create_persistor_for_model(self) -> ModelPersistor:
        """Create persistor for tf.keras.Model."""
        persistor_kwargs = {"model": self.model}

        if self.initial_ckpt:
            persistor_kwargs["source_ckpt_file_full_name"] = self.initial_ckpt

        return TFModelPersistor(**persistor_kwargs)

    def _create_persistor_for_dict_config(self) -> ModelPersistor:
        """Create persistor for dict config model."""
        # For dict config, pass full dict with path and args (resolved at runtime via instantiate_class)
        persistor_kwargs = {"model": self.model}  # Pass full dict {"path": "...", "args": {...}}

        if self.initial_ckpt:
            persistor_kwargs["source_ckpt_file_full_name"] = self.initial_ckpt

        return TFModelPersistor(**persistor_kwargs)

    def _create_persistor_for_checkpoint_only(self) -> ModelPersistor:
        """Create persistor for checkpoint-only mode (TF can load full model)."""
        return TFModelPersistor(
            model=None,
            source_ckpt_file_full_name=self.initial_ckpt,
        )
