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

from typing import List, Optional
import numpy as np

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class Int4QuantFilter(DXOFilter):
    """Filter that applies block-wise INT4 symmetric quantization to model weights or weight diffs.

    Each tensor is divided into contiguous blocks of `block_size` elements. A dynamic scale factor
    is computed per block, and values are linearly quantized to 4-bit integers in [-8, 7].
    The quantized representation provides an ~8x reduction in effective communication precision
    relative to float32.

    Args:
        block_size (int): Number of elements per quantization block. Defaults to 64. Must be positive and even.
        min_elements (int): Minimum number of elements in a tensor to qualify for quantization. Defaults to 128.
        data_kinds_to_filter (list of DataKind, optional): Filter targets. Defaults to [DataKind.WEIGHTS, DataKind.WEIGHT_DIFF].
    """

    def __init__(
        self,
        block_size: int = 64,
        min_elements: int = 128,
        data_kinds_to_filter: Optional[List[DataKind]] = None,
    ):
        if block_size <= 0 or block_size % 2 != 0:
            raise ValueError(f"block_size must be a positive even integer, got {block_size}")
        if min_elements < 0:
            raise ValueError(f"min_elements must be non-negative, got {min_elements}")

        supported = [DataKind.WEIGHTS, DataKind.WEIGHT_DIFF]
        super().__init__(supported_data_kinds=supported, data_kinds_to_filter=data_kinds_to_filter)
        self.block_size = block_size
        self.min_elements = min_elements

    def _quantize_array(self, arr: np.ndarray) -> np.ndarray:
        if not np.issubdtype(arr.dtype, np.floating) or arr.size < self.min_elements:
            return arr

        orig_shape = arr.shape
        flat = arr.flatten().astype(np.float32)
        n = flat.size

        pad_len = (self.block_size - (n % self.block_size)) % self.block_size
        padded = np.pad(flat, (0, pad_len), mode="constant") if pad_len > 0 else flat

        blocks = padded.reshape(-1, self.block_size)
        max_abs = np.max(np.abs(blocks), axis=1, keepdims=True)
        scale = np.clip(max_abs / 7.0, 1e-8, None)

        quantized = np.clip(np.round(blocks / scale), -8, 7)
        recon_blocks = quantized * scale
        recon = recon_blocks.flatten()[:n].reshape(orig_shape).astype(arr.dtype)

        return recon

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> DXO:
        if not isinstance(dxo.data, dict):
            return dxo

        for name, value in dxo.data.items():
            if isinstance(value, np.ndarray):
                dxo.data[name] = self._quantize_array(value)

        dxo.set_meta_prop("int4_quantized", True)
        return dxo
