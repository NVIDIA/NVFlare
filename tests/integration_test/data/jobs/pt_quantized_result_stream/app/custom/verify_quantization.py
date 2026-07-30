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

import numpy as np
import torch

from nvflare.apis.dxo import DXO, DataKind, MetaKey
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable

_DATA_KINDS = [DataKind.WEIGHTS, DataKind.WEIGHT_DIFF]
_QUANTIZATION_META_KEYS = [
    MetaKey.PROCESSED_ALGORITHM,
    "quant_state",
    "source_datatype",
    "quantized_flag",
]


class VerifyTaskParams(DXOFilter):
    def __init__(self, expected_format: str):
        super().__init__(supported_data_kinds=_DATA_KINDS, data_kinds_to_filter=_DATA_KINDS)
        if expected_format not in ("numpy", "pytorch"):
            raise ValueError(f"expected_format must be 'numpy' or 'pytorch', got {expected_format!r}")
        self.expected_format = expected_format

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> DXO:
        expected_type = np.ndarray if self.expected_format == "numpy" else torch.Tensor
        expected_dtype = np.dtype("float32") if self.expected_format == "numpy" else torch.float32

        for name, value in dxo.data.items():
            if not isinstance(value, expected_type):
                raise TypeError(f"server task parameter {name!r} is {type(value)}, expected {self.expected_format}")
            if value.dtype != expected_dtype:
                raise TypeError(f"server task parameter {name!r} has dtype {value.dtype}, expected {expected_dtype}")

        self.log_info(fl_ctx, f"VERIFIED_SERVER_TASK_FORMAT: {len(dxo.data)} {self.expected_format} parameters")
        return dxo


class VerifyQuantizedResult(DXOFilter):
    def __init__(self):
        super().__init__(supported_data_kinds=_DATA_KINDS, data_kinds_to_filter=_DATA_KINDS)

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> DXO:
        algorithm = dxo.get_meta_prop(MetaKey.PROCESSED_ALGORITHM)
        if algorithm != "float16":
            raise ValueError(f"expected float16 quantization metadata but got {algorithm!r}")
        if dxo.get_meta_prop("quantized_flag") is not True:
            raise ValueError("quantized_flag is not set")

        quant_state = dxo.get_meta_prop("quant_state")
        source_datatype = dxo.get_meta_prop("source_datatype")
        if not isinstance(quant_state, dict) or set(quant_state) != set(dxo.data):
            raise ValueError("quant_state does not describe every model parameter")
        if not isinstance(source_datatype, dict) or set(source_datatype) != set(dxo.data):
            raise ValueError("source_datatype does not describe every model parameter")

        for name, value in dxo.data.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"quantized parameter {name!r} is {type(value)}, not torch.Tensor")
            if value.dtype != torch.float16:
                raise TypeError(f"quantized parameter {name!r} has dtype {value.dtype}, expected torch.float16")
            if source_datatype[name] != "float32":
                raise ValueError(
                    f"parameter {name!r} records source dtype {source_datatype[name]!r}, expected 'float32'"
                )

        self.log_info(fl_ctx, f"VERIFIED_QUANTIZED_RESULT: {len(dxo.data)} float16 tensors")
        return dxo


class VerifyDequantizedResult(DXOFilter):
    def __init__(self):
        super().__init__(supported_data_kinds=_DATA_KINDS, data_kinds_to_filter=_DATA_KINDS)

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> DXO:
        unexpected_meta = [key for key in _QUANTIZATION_META_KEYS if dxo.get_meta_prop(key) is not None]
        if unexpected_meta:
            raise ValueError(f"quantization metadata remains after dequantization: {unexpected_meta}")

        for name, value in dxo.data.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"dequantized parameter {name!r} is {type(value)}, not torch.Tensor")
            if value.dtype != torch.float32:
                raise TypeError(f"dequantized parameter {name!r} has dtype {value.dtype}, expected torch.float32")

        self.log_info(fl_ctx, f"VERIFIED_DEQUANTIZED_RESULT: {len(dxo.data)} float32 tensors")
        return dxo
