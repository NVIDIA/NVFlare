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

import re
from typing import Union

import numpy as np
import torch
from bitsandbytes.functional import QuantState, dequantize_4bit, dequantize_blockwise

from nvflare.apis.dxo import DXO, DataKind, MetaKey
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_opt.pt.quantization.constant import QUANTIZATION_TYPE

from .ada_quant import AdaQuantizer


class ModelDequantizer(DXOFilter):
    def __init__(self):
        """Filter to dequantize Shareable object to recover from quantization

        Args:
            None

        """

        # support weight and weight_diff data kinds
        data_kinds = [DataKind.WEIGHTS, DataKind.WEIGHT_DIFF]
        super().__init__(supported_data_kinds=data_kinds, data_kinds_to_filter=data_kinds)
        self.logger.info("Using model dequantizer.")

    def dequantization(
        self,
        params: dict,
        quant_state: dict,
        quantization_type: str,
        source_datatype: dict,
        fl_ctx: FLContext,
    ):
        n_params = len(params.keys())
        self.log_info(fl_ctx, f"Running dequantization on {n_params} variables")
        n_bytes_before = 0
        n_bytes_after = 0
        n_bytes_meta = 0
        n_quant_params = 0
        for param_name in params:
            source_data_type = source_datatype[param_name]

            if quantization_type != "adaquant":
                # get the bits information
                source_date_bits = int(re.findall(r"\d+", source_data_type)[0])
                quantization_bits = int(re.findall(r"\d+", quantization_type)[0])

                # only dequantize if the quantization type is lower than the source data type
                if quantization_bits >= source_date_bits:
                    self.log_info(
                        fl_ctx,
                        f"Skipping dequantization for {param_name}, quantization bit {quantization_type} >= source data bit {source_data_type}",
                    )
                    continue
            values = params[param_name]
            n_bytes_before += values.nbytes
            if param_name not in quant_state:
                n_bytes_after += values.nbytes
                continue
            for item in quant_state[param_name].values():
                if isinstance(item, (np.ndarray, torch.Tensor)):
                    n_bytes_meta += item.nbytes

            if isinstance(values, np.ndarray):
                # if numpy, convert to torch
                source_data_format = "numpy"
            elif isinstance(values, torch.Tensor):
                source_data_format = "torch"
            else:
                raise ValueError(f"Invalid source data type: {type(values)}, valid: numpy or torch")

            n_quant_params += 1
            if quantization_type == "float16":
                # direct assign and convert back to higher precision
                params[param_name] = values
            elif quantization_type in ["blockwise8", "float4", "normfloat4"]:
                # use bitsandbytes to dequantize the values
                # need GPU for general support
                quantized = self.to_torch_tensor(values).cuda()
                # create QuantState object
                quantized_state_dict = quant_state[param_name]
                for k, v in quantized_state_dict.items():
                    if isinstance(v, (np.ndarray, torch.Tensor)):
                        quantized_state_dict[k] = self.to_torch_tensor(v).cuda()
                quantized_state = QuantState.from_dict(quantized_state_dict, device=torch.device("cuda"))

                if quantization_type == "blockwise8":
                    dequantized = dequantize_blockwise(quantized, quant_state=quantized_state)
                else:
                    dequantized = dequantize_4bit(quantized, quant_state=quantized_state)
                params[param_name] = self.to_source_data(dequantized, source_data_format)

            elif quantization_type == "ada_quant":
                values_tensor = self.to_torch_tensor(values)
                if param_name not in quant_state:
                    dequantized = values_tensor
                else:
                    dequantized = AdaQuantizer().dequantized(values_tensor, quant_state[param_name])
                params[param_name] = self.to_source_data(dequantized, source_data_format)
            else:
                raise ValueError(f"Invalid quantization type: {quantization_type}")

            # assign back
            if source_data_format == "numpy":
                # convert back to original data type
                if source_data_type == "float32":
                    params[param_name] = params[param_name].astype(np.float32)
                elif source_data_type == "float16":
                    params[param_name] = params[param_name].astype(np.float16)
            elif source_data_format == "torch":
                # convert back to original data type
                if source_data_type == "float32":
                    params[param_name] = params[param_name].float()
                elif source_data_type == "float16":
                    params[param_name] = params[param_name].half()
                elif source_data_type == "bfloat16":
                    params[param_name] = params[param_name].bfloat16()

            n_bytes_after += params[param_name].nbytes

        self.log_info(
            fl_ctx,
            f"Dequantized {n_quant_params}/{n_params} params."
            f" Before dequantization: {n_bytes_before / (1024**2):.2f} MB with meta: {n_bytes_meta / (1024**2):.2f} MB."
            f" After dequantization: {n_bytes_after / (1024**2):.2f} MB.",
        )
        return params

    def to_torch_tensor(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(data, np.ndarray):
            return torch.as_tensor(data)
        return data

    def to_source_data(self, tensor: torch.Tensor, source_data_format: str) -> Union[np.ndarray, torch.Tensor]:
        if source_data_format == "numpy":
            return tensor.numpy()
        return tensor.cpu()

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> Union[None, DXO]:
        """Filter process apply to the Shareable object.

        Args:
            dxo: data to be processed
            shareable: that the dxo belongs to
            fl_ctx: FLContext

        Returns: DXO object with dequantized weights

        """

        self.log_info(fl_ctx, "Running dequantization...")

        # check config
        quantization_type = dxo.get_meta_prop(key=MetaKey.PROCESSED_ALGORITHM, default=None)
        if quantization_type.upper() not in QUANTIZATION_TYPE:
            raise ValueError(f"Invalid quantization type: {quantization_type}, valid: {QUANTIZATION_TYPE}")
        source_datatype = dxo.get_meta_prop(key="source_datatype", default=None)
        dequantized_params = self.dequantization(
            params=dxo.data,
            quant_state=dxo.meta["quant_state"],
            quantization_type=quantization_type,
            source_datatype=source_datatype,
            fl_ctx=fl_ctx,
        )
        # Compose new DXO with dequantized data
        dxo.data = dequantized_params
        dxo.remove_meta_props(
            [
                MetaKey.PROCESSED_ALGORITHM,
                "quant_state",
                "source_datatype",
                "quantized_flag",
            ]
        )
        dxo.update_shareable(shareable)
        self.log_info(fl_ctx, "Dequantized back to original precision")

        return dxo
