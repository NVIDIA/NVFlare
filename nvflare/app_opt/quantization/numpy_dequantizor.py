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

from typing import Union

import numpy as np
import torch
from bitsandbytes.functional import QuantState, dequantize_4bit, dequantize_blockwise

from nvflare.apis.dxo import DXO, DataKind, MetaKey
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_opt.quantization.constant import DATA_TYPE, QUANTIZATION_TYPE


class NumpyModelDequantizor(DXOFilter):
    def __init__(self, source_data_type="float32"):
        """Filter to dequantize Shareable object to recover from quantization

        Args:
            source_data_type: original data type of the model

        """

        # support weight and weight_diff data kinds
        data_kinds = [DataKind.WEIGHTS, DataKind.WEIGHT_DIFF]
        super().__init__(supported_data_kinds=data_kinds, data_kinds_to_filter=data_kinds)

        # assign data type and check if it is valid
        self.logger.info("Using model dequantizator.")
        if source_data_type.upper() not in DATA_TYPE:
            raise ValueError(f"Invalid source data type: {source_data_type}, valid: {DATA_TYPE}")
        else:
            self.source_data_type = source_data_type

    def dequantization(self, params: dict, quant_state: dict, quant_type: str, fl_ctx: FLContext):
        n_params = len(params.keys())
        self.log_info(fl_ctx, f"Running dequantization on {n_params} variables")
        n_bytes_before = 0
        n_bytes_after = 0
        n_bytes_meta = 0
        n_quant_params = 0
        for i, param_name in enumerate(params.keys()):
            if self.source_data_type == "float32":
                values = params[param_name]
                n_bytes_before += values.nbytes
                for item in quant_state[param_name].values():
                    if isinstance(item, np.ndarray):
                        n_bytes_meta += item.nbytes
                if self.source_data_type != quant_type:
                    # if the source data type is not the same as the quantization type, convert it
                    n_quant_params += 1
                    if quant_type == "float16":
                        # direct convert
                        values = values.astype(np.float32)
                        params[param_name] = values
                    elif quant_type in ["blockwise8", "float4", "normfloat4"]:
                        # use bitsandbytes to dequantize the values
                        # extract quantization state
                        if quant_type == "blockwise8":
                            quantized = torch.as_tensor(values)
                            absmax = torch.as_tensor(quant_state[param_name]["absmax"])
                            code = torch.as_tensor(quant_state[param_name]["code"])
                            # de-quanitze
                            dequantized = dequantize_blockwise(quantized, absmax=absmax, code=code)
                            params[param_name] = dequantized.numpy()
                        else:
                            # first convert numpy array to tensor, need to use GPU
                            quantized = torch.as_tensor(values).cuda()
                            # create QuantState object
                            quantize_state = QuantState(
                                quant_type=quant_state[param_name]["quant_type"],
                                absmax=torch.as_tensor(quant_state[param_name]["absmax"]).cuda(),
                                blocksize=quant_state[param_name]["blocksize"],
                                code=torch.as_tensor(quant_state[param_name]["quant_map"]).cuda(),
                                dtype=getattr(torch, quant_state[param_name]["dtype"]),
                                shape=torch.Size(quant_state[param_name]["shape"]),
                            )
                            # de-quanitze
                            if quant_type == "float4":
                                dequantized = dequantize_4bit(quantized, quantize_state, quant_type="fp4")
                            else:
                                dequantized = dequantize_4bit(quantized, quantize_state, quant_type="nf4")
                            params[param_name] = dequantized.cpu().numpy()
                n_bytes_after += params[param_name].nbytes

        self.log_info(
            fl_ctx,
            f"Dequantized {n_quant_params}/{n_params} params."
            f" Before dequantization: {n_bytes_before / (1024 ** 2):.2f} MB with meta: {n_bytes_meta / (1024 ** 2):.2f} MB."
            f" After dequantization: {n_bytes_after / (1024 ** 2):.2f} MB.",
        )
        return params

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

        dequantized_params = self.dequantization(
            params=dxo.data, quant_state=dxo.meta["quant_state"], quant_type=quantization_type, fl_ctx=fl_ctx
        )
        # Compose new DXO with dequantized data
        dxo.data = dequantized_params
        dxo.remove_meta_props(MetaKey.PROCESSED_ALGORITHM)
        dxo.remove_meta_props("quant_state")
        dxo.update_shareable(shareable)
        self.log_info(fl_ctx, f"Dequantized back to {self.source_data_type}")

        return dxo
