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
from bitsandbytes.functional import quantize_blockwise

from nvflare.apis.dxo import DXO, DataKind, MetaKey
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_opt.quantization.constant import DATA_TYPE, QUANTIZATION_TYPE


class ModelQuantizor(DXOFilter):
    def __init__(
        self,
        quantization_type="float16",
    ):
        """Filter to quantize Shareable object to reduce communication burden.

        Args:
            quantization_type: method used for quantization

        """

        # support weight and weight_diff data kinds
        data_kinds = [DataKind.WEIGHTS, DataKind.WEIGHT_DIFF]
        super().__init__(supported_data_kinds=data_kinds, data_kinds_to_filter=data_kinds)

        # assign quantization type and check if it is valid
        self.logger.info("Using model quantizator.")
        if quantization_type.upper() not in QUANTIZATION_TYPE:
            raise ValueError(f"Invalid quantization type: {quantization_type}")
        else:
            self.quantization_type = quantization_type

        # quantization constants
        self.FP16_MIN = np.finfo(np.float16).min
        self.FP16_MAX = np.finfo(np.float16).max

    def quantization(self, params: dict, fl_ctx: FLContext):
        n_params = len(params.keys())
        self.log_info(fl_ctx, f"Running quantization on {n_params} variables")
        n_bytes_before = 0
        n_bytes_after = 0
        n_bytes_meta = 0
        quant_state = {"absmax": {}, "codebook": {}}
        for i, param_name in enumerate(params.keys()):
            values = params[param_name]
            # check the data type of the values and if it is valid
            source_data_type = values.dtype.name
            if source_data_type.upper() not in DATA_TYPE:
                raise ValueError(f"Invalid source data type: {source_data_type}")
            # add the number of bytes of the values
            n_bytes_before += values.nbytes
            if source_data_type == "float32":
                if self.quantization_type == "float16":
                    # first clamp the values to the range of float16
                    values = np.clip(values, self.FP16_MIN, self.FP16_MAX)
                    # then convert to float16
                    values = values.astype(np.float16)
                    n_bytes_after += values.nbytes
                    params[param_name] = values
                elif self.quantization_type == "blockwise8":
                    # use bitsandbytes to quantize the values
                    # input is a tensor, output is a tuple of (quantized tensor, (absmax, codebook))
                    # first convert numpy array to tensor
                    values_tensor = torch.as_tensor(values)
                    # quantize the tensor
                    quantized, quantized_state = quantize_blockwise(values_tensor, blocksize=4096, nested=False)
                    absmax = quantized_state.absmax
                    codebook = quantized_state.code
                    values = quantized.numpy()
                    # add the number of bytes
                    n_bytes_after += values.nbytes
                    params[param_name] = values
                    # also add the meta information
                    quant_state["absmax"][param_name] = absmax.numpy()
                    n_bytes_meta += absmax.nbytes
                    quant_state["codebook"][param_name] = codebook.numpy()
                    n_bytes_meta += codebook.nbytes

        self.log_info(
            fl_ctx,
            f"Quantized all {n_params} params."
            f" Before quantization: {n_bytes_before} bytes."
            f" After quantization: {n_bytes_after} bytes with meta: {n_bytes_meta} bytes.",
        )
        return params, quant_state

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> Union[None, DXO]:
        """Filter process apply to the Shareable object.

        Args:
            dxo: data to be processed
            shareable: that the dxo belongs to
            fl_ctx: FLContext

        Returns: DXO object with quantized weights

        """

        self.log_info(fl_ctx, "Running quantization...")
        quantized_params, quant_state = self.quantization(params=dxo.data, fl_ctx=fl_ctx)
        # Compose new DXO with quantized data
        # Add quant_state to the new DXO meta
        new_dxo = DXO(data_kind=dxo.data_kind, data=quantized_params, meta=dxo.meta)
        new_dxo.set_meta_prop(key=MetaKey.PROCESSED_ALGORITHM, value=self.quantization_type)
        new_dxo.set_meta_prop(key="quant_state", value=quant_state)
        self.log_info(fl_ctx, f"Quantized to {self.quantization_type}")

        return new_dxo
