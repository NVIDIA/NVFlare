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

from typing import Any, Dict, Tuple

import numpy as np


class Int4BlockQuantizer:
    """CPU/NumPy block-wise symmetric INT4 quantizer with nibble packing.

    Splits flattened floating-point arrays into contiguous blocks of `block_size`
    elements, computes a per-block symmetric scale (`max_abs / 7.0`), maps values
    to signed 4-bit integers in `[-8, 7]`, and packs two 4-bit nibbles into each
    `uint8` byte.

    Args:
        block_size: Number of elements per quantization block. Must be a positive
            even integer. Defaults to 64.
    """

    def __init__(self, block_size: int = 64) -> None:
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0 or block_size % 2 != 0:
            raise ValueError(f"block_size must be a positive even integer, got {block_size}")
        self.block_size = block_size

    def quantize(self, values: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not isinstance(values, np.ndarray):
            raise ValueError(f"Expected numpy.ndarray, got {type(values)}")

        if not np.issubdtype(values.dtype, np.floating) or values.dtype.itemsize > 8:
            raise ValueError(
                f"Unsupported source dtype '{values.dtype}'. Only float16, float32, and float64 are supported."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError("Input array contains non-finite values (NaN or Inf)")

        work_dtype = np.float64 if values.dtype.itemsize == 8 else np.float32
        orig_shape = values.shape

        if values.size == 0:
            return np.empty(0, dtype=np.uint8), {
                "scales": np.empty(0, dtype=work_dtype),
                "shape": orig_shape,
                "pad_len": 0,
                "block_size": self.block_size,
                "orig_dtype": str(values.dtype),
            }

        flat = values.reshape(-1).astype(work_dtype, copy=False)
        n = flat.size

        pad_len = (self.block_size - (n % self.block_size)) % self.block_size
        padded = np.pad(flat, (0, pad_len), mode="constant") if pad_len > 0 else flat

        blocks = padded.reshape(-1, self.block_size)
        max_abs = np.max(np.abs(blocks), axis=1, keepdims=True)
        scale = max_abs / 7.0
        # Guard against 1-ULP round-up overflow when reconstructing 7 * scale near finfo.max
        with np.errstate(over="ignore"):
            overflow_mask = np.isinf(scale * 7.0)
        if np.any(overflow_mask):
            scale = np.where(overflow_mask, np.nextafter(scale, 0.0), scale)

        zero_mask = (max_abs == 0.0) | (scale == 0.0)
        safe_max = np.where(zero_mask, 1.0, max_abs)

        # Normalize by max_abs before multiplying by 7.0 to avoid overflow/underflow on extreme float64 ranges
        normalized = np.where(zero_mask, 0.0, (blocks / safe_max) * 7.0)
        q_signed = np.clip(np.round(normalized), -8, 7).astype(np.int8)
        q_unsigned = (q_signed + 8).astype(np.uint8)

        flat_q = q_unsigned.reshape(-1)
        q0 = flat_q[0::2]
        q1 = flat_q[1::2]
        packed = ((q0 << 4) | (q1 & 0x0F)).astype(np.uint8)

        quant_state = {
            "scales": scale.reshape(-1).astype(work_dtype, copy=False),
            "shape": orig_shape,
            "pad_len": int(pad_len),
            "block_size": self.block_size,
            "orig_dtype": str(values.dtype),
        }
        return packed, quant_state

    def dequantize(self, packed: np.ndarray, quant_state: Dict[str, Any]) -> np.ndarray:
        if not isinstance(packed, np.ndarray):
            raise ValueError(f"Expected numpy.ndarray for packed data, got {type(packed)}")
        if packed.dtype != np.uint8:
            raise ValueError(f"Expected uint8 packed array, got {packed.dtype}")
        if not isinstance(quant_state, dict):
            raise ValueError("quant_state must be a dictionary")

        block_size = quant_state.get("block_size", self.block_size)
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0 or block_size % 2 != 0:
            raise ValueError(f"block_size must be a positive even integer, got {block_size}")

        raw_dtype = quant_state.get("orig_dtype", "float32")
        try:
            target_dtype = np.dtype(raw_dtype)
        except Exception as e:
            raise ValueError(f"Invalid orig_dtype '{raw_dtype}': {e}") from e

        if not np.issubdtype(target_dtype, np.floating) or target_dtype.itemsize > 8:
            raise ValueError(
                f"Unsupported or untrusted target dtype '{raw_dtype}'. "
                "Only standard floating-point types (float16, float32, float64) are permitted."
            )

        pad_len = quant_state.get("pad_len")
        if (
            isinstance(pad_len, bool)
            or not isinstance(pad_len, (int, np.integer))
            or pad_len < 0
            or pad_len >= block_size
        ):
            raise ValueError(f"Invalid pad_len {pad_len} for block_size {block_size}")
        pad_len = int(pad_len)

        raw_shape = quant_state.get("shape")
        if not isinstance(raw_shape, (tuple, list)):
            raise ValueError(f"Invalid shape metadata: {raw_shape}")
        orig_shape = tuple(raw_shape)
        if any(isinstance(dim, bool) or not isinstance(dim, (int, np.integer)) or dim < 0 for dim in orig_shape):
            raise ValueError(f"Invalid negative or non-integer shape {orig_shape}")

        total_elements = int(np.prod(orig_shape, dtype=np.int64))
        padded_elements = total_elements + pad_len
        if padded_elements % block_size != 0:
            raise ValueError(f"Padded element count {padded_elements} is not divisible by block_size {block_size}")

        if padded_elements != packed.size * 2:
            raise ValueError(
                f"Packed buffer size {packed.size} does not match reconstructed size for shape {orig_shape} with pad {pad_len}"
            )

        num_blocks = padded_elements // block_size
        scales = quant_state.get("scales")
        if not isinstance(scales, np.ndarray):
            raise ValueError(f"Expected numpy.ndarray for scales, got {type(scales)}")
        if not np.issubdtype(scales.dtype, np.floating) or scales.dtype.itemsize > 8:
            raise ValueError(f"Unsupported scales dtype '{scales.dtype}'")
        if scales.ndim != 1 or scales.size != num_blocks:
            raise ValueError(
                f"Scale count mismatch: expected 1D scales array of length {num_blocks}, got shape {scales.shape}"
            )
        if not np.all(np.isfinite(scales)):
            raise ValueError("scales contains non-finite values (NaN or Inf)")

        if total_elements == 0:
            return np.empty(orig_shape, dtype=target_dtype)

        flat_packed = packed.reshape(-1)
        uq0 = (flat_packed >> 4) & 0x0F
        uq1 = flat_packed & 0x0F
        unpacked = np.empty(flat_packed.size * 2, dtype=np.uint8)
        unpacked[0::2] = uq0
        unpacked[1::2] = uq1

        unp_signed = unpacked.astype(np.int8) - 8
        with np.errstate(over="ignore"):
            rec_blocks = unp_signed.reshape(num_blocks, block_size) * scales.reshape(num_blocks, 1)
        if not np.all(np.isfinite(rec_blocks)):
            raise ValueError("Dequantized values overflowed floating-point range")
        rec_flat = rec_blocks.reshape(-1)
        if pad_len > 0:
            rec_flat = rec_flat[:-pad_len]
        with np.errstate(over="ignore"):
            result = rec_flat.reshape(orig_shape).astype(target_dtype, copy=False)
        if not np.all(np.isfinite(result)):
            raise ValueError(f"Dequantized values overflowed target dtype '{target_dtype}'")
        return result
