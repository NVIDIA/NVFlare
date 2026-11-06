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

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class Int4QuantFilter(DXOFilter):
    """Filter that applies block-wise INT4 uniform quantization to model weights or weight diffs.

    Each tensor is divided into contiguous blocks of `block_size` elements. A dynamic scale factor
    is computed per block, and values are linearly quantized to 4-bit integers in [-8, 7].
    In quantize mode (default), pairs of 4-bit nibbles are packed into uint8 bytes, providing an ~8x
    reduction in wire payload size relative to float32. In dequantize mode, packed bytes are reconstructed
    back into full floating-point arrays using per-block scales.

    Args:
        direction (str): Direction of operation: "quantize", "dequantize", or "simulate". Defaults to "quantize".
        block_size (int): Number of elements per quantization block. Defaults to 64. Must be a positive even integer.
        min_elements (int): Minimum number of elements in a tensor to qualify for quantization. Defaults to 128.
        data_kinds_to_filter (list of DataKind, optional): Filter targets. Defaults to [DataKind.WEIGHTS, DataKind.WEIGHT_DIFF].
    """

    QUANTIZE = "quantize"
    DEQUANTIZE = "dequantize"
    SIMULATE = "simulate"

    def __init__(
        self,
        direction: str = QUANTIZE,
        block_size: int = 64,
        min_elements: int = 128,
        data_kinds_to_filter: Optional[List[DataKind]] = None,
    ):
        if direction not in (self.QUANTIZE, self.DEQUANTIZE, self.SIMULATE):
            raise ValueError(
                f"direction must be one of {(self.QUANTIZE, self.DEQUANTIZE, self.SIMULATE)}, got {direction}"
            )
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0 or block_size % 2 != 0:
            raise ValueError(f"block_size must be a positive even integer, got {block_size}")
        if isinstance(min_elements, bool) or not isinstance(min_elements, int) or min_elements < 0:
            raise ValueError(f"min_elements must be a non-negative integer, got {min_elements}")

        supported = [DataKind.WEIGHTS, DataKind.WEIGHT_DIFF]
        super().__init__(supported_data_kinds=supported, data_kinds_to_filter=data_kinds_to_filter)
        self.direction = direction
        self.block_size = block_size
        self.min_elements = min_elements

    def _pack_array(self, arr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        orig_shape = arr.shape
        flat = arr.flatten().astype(np.float32)
        n = flat.size

        pad_len = (self.block_size - (n % self.block_size)) % self.block_size
        padded = np.pad(flat, (0, pad_len), mode="constant") if pad_len > 0 else flat

        blocks = padded.reshape(-1, self.block_size)
        max_abs = np.max(np.abs(blocks), axis=1, keepdims=True)
        zero_mask = max_abs == 0.0
        scale = max_abs / 7.0
        safe_scale = np.where(zero_mask, 1.0, scale)

        q_signed = np.where(zero_mask, 0, np.clip(np.round(blocks / safe_scale), -8, 7)).astype(np.int8)
        q_unsigned = (q_signed + 8).astype(np.uint8)

        flat_q = q_unsigned.flatten()
        q0 = flat_q[0::2]
        q1 = flat_q[1::2]
        packed = ((q0 << 4) | (q1 & 0x0F)).astype(np.uint8)

        meta = {
            "shape": orig_shape,
            "pad_len": pad_len,
            "orig_dtype": str(arr.dtype),
            "scales": scale.flatten().astype(np.float32),
        }
        return packed, meta

    def _unpack_array(self, packed: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
        scales = meta["scales"]
        pad_len = meta["pad_len"]
        orig_shape = meta["shape"]
        orig_dtype = np.dtype(meta["orig_dtype"])

        uq0 = (packed >> 4) & 0x0F
        uq1 = packed & 0x0F
        unpacked = np.empty(packed.size * 2, dtype=np.uint8)
        unpacked[0::2] = uq0
        unpacked[1::2] = uq1

        unp_signed = unpacked.astype(np.int8) - 8
        rec_blocks = unp_signed.reshape(-1, self.block_size) * scales.reshape(-1, 1)
        rec_flat = rec_blocks.flatten()
        if pad_len > 0:
            rec_flat = rec_flat[:-pad_len]
        return rec_flat.reshape(orig_shape).astype(orig_dtype)

    def _simulate_array(self, arr: np.ndarray) -> np.ndarray:
        orig_shape = arr.shape
        flat = arr.flatten().astype(np.float32)
        n = flat.size

        pad_len = (self.block_size - (n % self.block_size)) % self.block_size
        padded = np.pad(flat, (0, pad_len), mode="constant") if pad_len > 0 else flat

        blocks = padded.reshape(-1, self.block_size)
        max_abs = np.max(np.abs(blocks), axis=1, keepdims=True)
        zero_mask = max_abs == 0.0
        scale = max_abs / 7.0
        safe_scale = np.where(zero_mask, 1.0, scale)

        quantized = np.where(zero_mask, 0.0, np.clip(np.round(blocks / safe_scale), -8, 7))
        recon_blocks = quantized * scale
        rec_flat = recon_blocks.flatten()
        if pad_len > 0:
            rec_flat = rec_flat[:-pad_len]
        return rec_flat.reshape(orig_shape).astype(arr.dtype)

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> DXO:
        if not isinstance(dxo.data, dict):
            return dxo

        if self.direction == self.QUANTIZE:
            int4_meta = {}
            for name, value in list(dxo.data.items()):
                if (
                    isinstance(value, np.ndarray)
                    and np.issubdtype(value.dtype, np.floating)
                    and value.size >= self.min_elements
                ):
                    packed, meta = self._pack_array(value)
                    dxo.data[name] = packed
                    int4_meta[name] = meta

            if int4_meta:
                dxo.set_meta_prop("int4_meta", int4_meta)
                dxo.set_meta_prop("int4_quantized", True)

        elif self.direction == self.DEQUANTIZE:
            int4_meta = dxo.get_meta_prop("int4_meta", {})
            if int4_meta:
                for name, meta in int4_meta.items():
                    if name in dxo.data and isinstance(dxo.data[name], np.ndarray):
                        dxo.data[name] = self._unpack_array(dxo.data[name], meta)
                dxo.remove_meta_props(["int4_meta", "int4_quantized"])

        elif self.direction == self.SIMULATE:
            for name, value in dxo.data.items():
                if (
                    isinstance(value, np.ndarray)
                    and np.issubdtype(value.dtype, np.floating)
                    and value.size >= self.min_elements
                ):
                    dxo.data[name] = self._simulate_array(value)
            dxo.set_meta_prop("int4_quantized", True)

        return dxo


class Int4DequantFilter(Int4QuantFilter):
    """Convenience filter for dequantizing INT4-quantized DXO payloads back to floating-point arrays."""

    def __init__(
        self,
        block_size: int = 64,
        min_elements: int = 128,
        data_kinds_to_filter: Optional[List[DataKind]] = None,
    ):
        super().__init__(
            direction=self.DEQUANTIZE,
            block_size=block_size,
            min_elements=min_elements,
            data_kinds_to_filter=data_kinds_to_filter,
        )
