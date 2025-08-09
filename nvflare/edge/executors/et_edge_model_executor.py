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
import time
from typing import Optional

import torch

from nvflare.apis.dxo import DataKind, from_dict
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey
from nvflare.edge.constants import CookieKey, MsgKey
from nvflare.edge.executors.edge_model_executor import EdgeModelExecutor, ModelUpdate
from nvflare.edge.executors.hug import TaskInfo
from nvflare.edge.model_protocol import ModelBufferType, ModelEncoding, ModelExchangeFormat, ModelNativeFormat
from nvflare.edge.models.model import DeviceModel, export_model_to_bytes
from nvflare.edge.mud import BaseState
from nvflare.edge.web.models.result_report import ResultReport


class ETEdgeModelExecutor(EdgeModelExecutor):
    def __init__(
        self,
        et_model: DeviceModel,
        input_shape,
        output_shape,
        aggr_factory_id: str,
        max_model_versions: int,
        update_timeout=60,
    ):
        """Initializes an edge model executor for on-device training using ExecuTorch.

        This constructor sets up the executor with a training-ready PyTorch model
        (wrapped to include loss computation), along with model input/output shapes
        and versioning/update control parameters.

        Args:
            et_model (DeviceModel): A PyTorch model wrapped for ExecuTorch export.
                See `nvflare/edge/models/model.py` for wrapping examples.
            input_shape (tuple): Shape of the input tensor (e.g., (1, 3, 224, 224)).
            output_shape (tuple): Shape of the label/output tensor (e.g., (1,) for class index).
            aggr_factory_id (str): Identifier used for selecting the model aggregation strategy.
            max_model_versions (int): Maximum number of model versions to retain or track.
            update_timeout (int, optional): Timeout in seconds for applying model updates. Defaults to 60.
        """
        EdgeModelExecutor.__init__(self, aggr_factory_id, max_model_versions, update_timeout)
        self.et_model = et_model
        self.input_shape = input_shape
        self.output_shape = output_shape

    def _export_model_weights_to_pte_b64str(self, model_weights) -> str:
        model_weights = {"net." + k: torch.tensor(v) for k, v in model_weights.items()}
        self.et_model.load_state_dict(model_weights)
        # Convert to buffer
        model_buffer = export_model_to_bytes(self.et_model, self.input_shape, self.output_shape)
        model_str = base64.b64encode(model_buffer).decode("utf-8")
        return model_str

    def _convert_task(self, task_state: BaseState, current_task: TaskInfo, fl_ctx: FLContext) -> dict:
        """Convert task_data to a plain dict"""
        self.log_info(fl_ctx, f"ETEdgeModelExecutor Converting task for task: {current_task.id}")

        # Add model version to the payload to track the version of the model being processed.
        model_dxo = task_state.model
        model_dxo.set_meta_prop(MsgKey.MODEL_VERSION, task_state.model_version)
        model_dict = model_dxo.to_dict()
        self.log_info(fl_ctx, f"ETEdgeModelExecutor model_dict data keys are: {model_dict['data'].keys()}")
        model_dict["data"] = self._export_model_weights_to_pte_b64str(model_dict["data"])
        model_dict["meta"].update(
            {
                ModelExchangeFormat.MODEL_BUFFER_TYPE: ModelBufferType.EXECUTORCH,
                ModelExchangeFormat.MODEL_BUFFER_NATIVE_FORMAT: ModelNativeFormat.BINARY,
                ModelExchangeFormat.MODEL_BUFFER_ENCODING: ModelEncoding.BASE64,
            }
        )
        model_dict["kind"] = DataKind.APP_DEFINED
        self.log_info(fl_ctx, f"ETEdgeModelExecutor model_dict keys are: {model_dict.keys()}")
        return model_dict

    def _convert_to_tensor_dxo(self, result_dict: dict, fl_ctx: FLContext):
        """Convert the result_dict to a tensor DXO"""
        d = {}
        d["meta"] = result_dict["meta"]
        d["kind"] = DataKind.WEIGHT_DIFF
        tensor_dict = {}
        for key, value in result_dict["data"].items():
            tensor = torch.Tensor(value["data"]).reshape(value["sizes"]).cpu().numpy()
            tensor_dict[key] = tensor

        d["data"] = {"dict": tensor_dict}
        return d

    def _convert_device_result_to_model_update(
        self, result_report: ResultReport, current_task: TaskInfo, fl_ctx: FLContext
    ) -> Optional[ModelUpdate]:
        self.log_info(fl_ctx, f"ETEdgeModelExecutor Converting result for task: {current_task.id}")

        device_id = result_report.get_device_id()
        cookie = result_report.cookie
        if not cookie:
            self.log_error(fl_ctx, f"missing cookie in result report from device {device_id}")
            raise ValueError("missing cookie")

        model_version = cookie.get(CookieKey.MODEL_VERSION)
        if not model_version:
            self.log_error(
                fl_ctx, f"missing '{CookieKey.MODEL_VERSION}' cookie in result report from device {device_id}"
            )
            raise ValueError(f"missing '{CookieKey.MODEL_VERSION}' cookie")

        result_dict = result_report.result

        # Convert the result_dict json to a tensor DXO dict
        self.log_info(fl_ctx, "ETEdgeModelExecutor converting result_dict to tensor DXO")
        result_dict = self._convert_to_tensor_dxo(result_dict, fl_ctx)

        if not isinstance(result_dict, dict) or "data" not in result_dict or "dict" not in result_dict["data"]:
            self.log_error(fl_ctx, f"result_report.result is not a valid structure: {result_report.result}")
            raise ValueError("result_report.result is not a valid structure")

        result_dict["data"]["dict"] = {k.removeprefix("net."): v for k, v in result_dict["data"]["dict"].items()}
        self.log_info(fl_ctx, f"ETEdgeModelExecutor result_dict data keys are: {result_dict['data'].keys()}")

        try:
            dxo = from_dict(result_dict)
        except Exception as e:
            self.log_error(fl_ctx, f"Failed to convert result_report.result to DXO: {e}")
            raise ValueError("Failed to convert result_report.result to DXO") from e

        dxo.set_meta_prop(ReservedHeaderKey.TASK_ID, current_task.id)

        return ModelUpdate(
            model_version=model_version,
            update=dxo.to_shareable(),
            devices={result_report.get_device_id(): time.time()},
        )
