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

import base64
import importlib
from typing import Any, List

import numpy as np
import torch

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants
from nvflare.edge.constants import MsgKey
from nvflare.edge.model_protocol import ModelBufferType, ModelEncoding, ModelExchangeFormat, ModelNativeFormat
from nvflare.edge.models.model import export_model


class ExecutorchShareableGenerator(ShareableGenerator):
    def __init__(self, base_model_path: str, executorch_model_path: str, input_shape: List, output_shape: List):
        super().__init__()
        self.base_model_path = base_model_path
        self.executorch_model_path = executorch_model_path
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = None

    def _load_model(self, model_path: str, fl_ctx: FLContext) -> Any:
        """Load model from model path.

        Args:
            model_path (str): model path in format "module.submodule.ClassName"
            fl_ctx (FLContext): FL context

        Returns:
            model class instance
        """
        try:
            module_path, class_name = model_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            return model_class
        except Exception as e:
            self.system_panic(
                reason=f"Failed to load model class from '{model_path}': {str(e)}",
                fl_ctx=fl_ctx,
            )
            return None

    def _export_current_model(self) -> bytes:
        """Export current model in ExecutorTorch format."""
        input_tensor = torch.randn(self.input_shape)
        label_tensor = torch.ones(self.output_shape, dtype=torch.int64)
        model_buffer = export_model(self.model, input_tensor, label_tensor).buffer
        base64_encoded = base64.b64encode(model_buffer).decode("utf-8")
        return base64_encoded

    def handle_event(self, event: str, fl_ctx: FLContext):
        if event == EventType.START_RUN:
            base_model = self._load_model(self.base_model_path, fl_ctx)
            executorch_model = self._load_model(self.executorch_model_path, fl_ctx)
            base_model_inst = base_model()
            self.model = executorch_model(base_model_inst)
            # Verify self.model is a torch model
            if not isinstance(self.model, torch.nn.Module):
                self.system_panic(reason="Model is not a torch model", fl_ctx=fl_ctx)

    def learnable_to_shareable(self, model_learnable: ModelLearnable, fl_ctx: FLContext) -> Shareable:
        """Convert ModelLearnable to Shareable.

        Args:
            model_learnable (ModelLearnable): model to be converted
            fl_ctx (FLContext): FL context

        Returns:
            Shareable: a shareable containing a DXO object.
        """
        # Compose shareable
        task_data = Shareable()
        # Update model weights using global model weights
        model_weights = model_learnable[ModelLearnableKey.WEIGHTS]
        # Add 'net' to model_weight keys and convert numpy to tensor
        # so that it can be loaded by model.load_state_dict
        model_weights = {"net." + k: torch.from_numpy(v) for k, v in model_weights.items()}
        self.model.load_state_dict(model_weights)
        # Convert to buffer
        model_buffer = self._export_current_model()
        task_data[MsgKey.PAYLOAD] = {
            ModelExchangeFormat.MODEL_BUFFER: model_buffer,
            ModelExchangeFormat.MODEL_BUFFER_TYPE: ModelBufferType.EXECUTORCH,
            ModelExchangeFormat.MODEL_BUFFER_NATIVE_FORMAT: ModelNativeFormat.BINARY,
            ModelExchangeFormat.MODEL_BUFFER_ENCODING: ModelEncoding.BASE64,
        }
        return task_data

    def shareable_to_learnable(self, shareable: Shareable, fl_ctx: FLContext) -> ModelLearnable:
        """Convert Shareable to ModelLearnable.

        Supporting TYPE == TYPE_WEIGHT_DIFF or TYPE_WEIGHTS

        Args:
            shareable (Shareable): Shareable that contains a DXO object
            fl_ctx (FLContext): FL context

        Returns:
            A ModelLearnable object

        Raises:
            TypeError: if shareable is not of type shareable
            ValueError: if data_kind is not `DataKind.WEIGHTS` and is not `DataKind.WEIGHT_DIFF`
        """
        if not isinstance(shareable, Shareable):
            raise TypeError("shareable must be Shareable, but got {}.".format(type(shareable)))

        base_model = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
        weight_to_add = shareable.get(MsgKey.RESULT)
        divide_factor = shareable.get(MsgKey.NUM_DEVICES)

        # apply updates - only diff mode from device
        if not base_model:
            self.system_panic(reason="No global base model found for processing WEIGHT_DIFF!", fl_ctx=fl_ctx)
            return base_model
        weights = base_model[ModelLearnableKey.WEIGHTS]
        # apply updates
        for k, v in weight_to_add.items():
            # weights_to_add in executorch json format, convert to numpy array
            # and divide by number of devices
            weight_to_add = np.array(v["data"]).reshape(v["sizes"]) / divide_factor
            weights[k] = weights[k] + weight_to_add

        return base_model
