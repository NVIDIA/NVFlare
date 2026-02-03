# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import os
from typing import Any, Dict, Optional, Union

import tensorflow as tf

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey, make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.model_desc import ModelDescriptor
from nvflare.app_opt.tf.utils import flat_layer_weights_dict, unflat_layer_weights_dict


class TFModelPersistor(ModelPersistor):
    def __init__(
        self,
        model: Optional[Union[tf.keras.Model, Dict[str, Any]]] = None,
        save_name: str = "tf_model.weights.h5",
        filter_id: Optional[str] = None,
        source_ckpt_file_full_name: Optional[str] = None,
    ):
        """Persist TensorFlow/Keras model to/from file system.

        This persistor supports loading from a source checkpoint file, which is useful
        for pre-trained models. The checkpoint path may be a server-side path that
        doesn't exist on the job submission machine.

        Args:
            model: Model input. Can be one of:
                - tf.keras.Model: Direct model instance
                - dict: {"path": "fully.qualified.Class", "args": {...}} for dynamic instantiation
                - None: If source_ckpt_file_full_name points to a full model file
            save_name: Filename for saving model weights. Defaults to "tf_model.weights.h5".
            filter_id: Optional filter component ID for model serialization.
            source_ckpt_file_full_name: Full path to source checkpoint file.
                This path may not exist locally (server-side path).
        """
        super().__init__(
            filter_id=filter_id,
        )
        self.save_name = save_name
        self.model = model
        self.source_ckpt_file_full_name = source_ckpt_file_full_name
        # Note: We don't validate existence here because the checkpoint path may be
        # a server-side path that doesn't exist on the job submission machine.

    def _initialize(self, fl_ctx: FLContext):
        workspace = fl_ctx.get_engine().get_workspace()
        app_root = workspace.get_app_dir(fl_ctx.get_job_id())
        self._model_save_path = os.path.join(app_root, self.save_name)
        self.log_dir = app_root

    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        """Initializes and loads the Model.

        Args:
            fl_ctx: FLContext

        Returns:
            ModelLearnable object
        """
        # Handle dict config: {"path": "module.Class", "args": {...}}
        if isinstance(self.model, dict):
            from nvflare.fuel.utils.class_utils import instantiate_class

            class_path = self.model.get("path")
            class_args = self.model.get("args", {})
            if not class_path:
                self.system_panic(reason="Dict model config must have 'path' key with class path", fl_ctx=fl_ctx)
                return None
            try:
                self.model = instantiate_class(class_path, class_args)
            except Exception as e:
                self.system_panic(
                    reason=f"Failed to instantiate model class '{class_path}': {e}",
                    fl_ctx=fl_ctx,
                )
                return None
            if not isinstance(self.model, tf.keras.Model):
                self.system_panic(
                    reason=f"Expected model class '{class_path}' to be tf.keras.Model but got {type(self.model)}",
                    fl_ctx=fl_ctx,
                )
                return None

        # Priority: source_ckpt_file_full_name > previously saved model > model weights
        if self.source_ckpt_file_full_name:
            # If user explicitly specified a checkpoint, it MUST exist (fail fast to catch config errors)
            if not os.path.exists(self.source_ckpt_file_full_name):
                self.system_panic(
                    reason=f"Source checkpoint not found: {self.source_ckpt_file_full_name}. "
                    "Check that the checkpoint exists at runtime.",
                    fl_ctx=fl_ctx,
                )
                return None

            self.logger.info(f"Loading model from source checkpoint: {self.source_ckpt_file_full_name}")
            # Check if it's a full model file or just weights
            if self.source_ckpt_file_full_name.endswith((".h5", ".keras")):
                try:
                    # Try loading as full model first
                    self.model = tf.keras.models.load_model(self.source_ckpt_file_full_name)
                except Exception as e:
                    # Fall back to loading weights only if file format suggests weights-only
                    self.logger.info(f"Could not load as full model ({e}), attempting weights-only load")
                    if self.model is not None:
                        self.model.load_weights(self.source_ckpt_file_full_name)
                    else:
                        self.system_panic(
                            reason=f"Cannot load weights from {self.source_ckpt_file_full_name} without a model. "
                            "Provide a model instance or use a full model file.",
                            fl_ctx=fl_ctx,
                        )
                        return None
            elif os.path.isdir(self.source_ckpt_file_full_name):
                # SavedModel directory
                self.model = tf.keras.models.load_model(self.source_ckpt_file_full_name)
            else:
                # Assume weights file
                if self.model is not None:
                    self.model.load_weights(self.source_ckpt_file_full_name)
                else:
                    self.system_panic(
                        reason=f"Cannot load weights from {self.source_ckpt_file_full_name} without a model.",
                        fl_ctx=fl_ctx,
                    )
                    return None
        elif os.path.exists(self._model_save_path):
            self.logger.info("Loading server model and weights")
            if self.model is not None:
                self.model.load_weights(self._model_save_path)
            else:
                self.system_panic(
                    reason=f"Cannot load weights from {self._model_save_path} without a model. Provide a model instance.",
                    fl_ctx=fl_ctx,
                )
                return None

        # Ensure model exists before proceeding
        if self.model is None:
            self.system_panic(
                reason="No model available. Provide either a model instance, source_ckpt_file_full_name with a full model, "
                "or a previously saved model.",
                fl_ctx=fl_ctx,
            )
            return None

        # build model if not built yet
        if not self.model.built:
            if hasattr(self.model, "_input_shape"):
                self.model.build(input_shape=self.model._input_shape)
            else:
                self.system_panic(
                    reason="To use delayed model build, you need to set model._input_shape", fl_ctx=fl_ctx
                )
                return None

        # get flat model parameters
        layer_weights_dict = {layer.name: layer.get_weights() for layer in self.model.layers}
        result = flat_layer_weights_dict(layer_weights_dict)

        model_learnable = make_model_learnable(result, dict())
        return model_learnable

    def handle_event(self, event: str, fl_ctx: FLContext):
        if event == EventType.START_RUN:
            self._initialize(fl_ctx)

    def save_model(self, model_learnable: ModelLearnable, fl_ctx: FLContext):
        """Saves model.

        Args:
            model_learnable: ModelLearnable object
            fl_ctx: FLContext
        """
        result = unflat_layer_weights_dict(model_learnable[ModelLearnableKey.WEIGHTS])
        for k in result:
            layer = self.model.get_layer(name=k)
            layer.set_weights(result[k])
        self.model.save_weights(self._model_save_path)

    def get_model(self, model_file: str, fl_ctx: FLContext) -> ModelLearnable:
        """Get a specific model by file name for cross-site evaluation.

        Args:
            model_file: Name/path of the model file to load
            fl_ctx: FLContext

        Returns:
            ModelLearnable object or None if model not found
        """
        inventory = self.get_model_inventory(fl_ctx)
        if not inventory:
            return None

        desc = inventory.get(model_file)
        if not desc:
            return None

        location = desc.location
        return self._get_model_from_location(location, fl_ctx)

    def _get_model_from_location(self, location: str, fl_ctx: FLContext) -> ModelLearnable:
        """Load model from a specific file location.

        Args:
            location: Full path to model file
            fl_ctx: FLContext

        Returns:
            ModelLearnable object or None if loading fails
        """
        try:
            if os.path.exists(location):
                self.logger.info(f"Loading TensorFlow model from {location}")
                self.model.load_weights(location)

                # build model if not built yet
                if not self.model.built:
                    if hasattr(self.model, "_input_shape"):
                        self.model.build(input_shape=self.model._input_shape)
                    else:
                        raise AttributeError("To use delayed model build, you need to set model._input_shape")

                # get flat model parameters
                layer_weights_dict = {layer.name: layer.get_weights() for layer in self.model.layers}
                result = flat_layer_weights_dict(layer_weights_dict)

                model_learnable = make_model_learnable(result, dict())
                return model_learnable
            else:
                self.logger.error(f"Model file not found: {location}")
                return None
        except Exception as e:
            self.log_exception(fl_ctx, f"Error loading TensorFlow model from {location}: {e}")
            return None

    def get_model_inventory(self, fl_ctx: FLContext) -> Dict[str, ModelDescriptor]:
        """Get inventory of available models for cross-site evaluation.

        Args:
            fl_ctx: FLContext

        Returns:
            Dictionary mapping model names to ModelDescriptor objects
        """
        model_inventory = {}

        # Check for the main saved model
        if hasattr(self, "_model_save_path") and os.path.exists(self._model_save_path):
            _, tail = os.path.split(self.save_name)
            model_inventory[tail] = ModelDescriptor(
                name=self.save_name,
                location=self._model_save_path,
                model_format="TensorFlow",
                props={},
            )

        return model_inventory
