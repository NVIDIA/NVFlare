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
import pytest

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.filters.int4_quant_filter import Int4DequantFilter, Int4QuantFilter


class TestInt4QuantFilter:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            Int4QuantFilter(block_size=63)
        with pytest.raises(ValueError):
            Int4QuantFilter(block_size=0)
        with pytest.raises(ValueError):
            Int4QuantFilter(block_size=-2)
        with pytest.raises(ValueError):
            Int4QuantFilter(block_size=64.0)
        with pytest.raises(ValueError):
            Int4QuantFilter(block_size=True)
        with pytest.raises(ValueError):
            Int4QuantFilter(min_elements=-1)
        with pytest.raises(ValueError):
            Int4QuantFilter(min_elements=128.5)
        with pytest.raises(ValueError):
            Int4QuantFilter(min_elements=False)
        with pytest.raises(ValueError):
            Int4QuantFilter(direction="invalid_mode")

    def test_wire_packing_and_dequantize_roundtrip(self):
        quant_filter = Int4QuantFilter(direction=Int4QuantFilter.QUANTIZE, block_size=64, min_elements=64)
        dequant_filter = Int4DequantFilter(block_size=64, min_elements=64)

        rng = np.random.default_rng(42)
        orig_arr = rng.standard_normal((10, 20)).astype(np.float32)  # 200 elements, 800 bytes

        dxo = DXO(data_kind=DataKind.WEIGHTS, data={"w1": orig_arr.copy()})
        shareable = Shareable()
        fl_ctx = FLContext()

        # Client outbound quantization
        quant_dxo = quant_filter.process_dxo(dxo, shareable, fl_ctx)
        assert quant_dxo.get_meta_prop("int4_quantized") is True
        packed = quant_dxo.data["w1"]
        assert packed.dtype == np.uint8
        # Padded elements: 256 -> packed bytes: 128 (8x reduction relative to float32 = 1024 bytes)
        assert packed.nbytes == 128

        # Server inbound dequantization
        restored_dxo = dequant_filter.process_dxo(quant_dxo, shareable, fl_ctx)
        restored = restored_dxo.data["w1"]
        assert restored.shape == orig_arr.shape
        assert restored.dtype == orig_arr.dtype

        mse = np.mean((orig_arr - restored) ** 2)
        assert mse < 0.05

    def test_small_updates_retained(self):
        quant_filter = Int4QuantFilter(direction=Int4QuantFilter.QUANTIZE, block_size=64, min_elements=64)
        dequant_filter = Int4DequantFilter(block_size=64, min_elements=64)

        # Small update array representable in float32 (e.g. 1e-9)
        small_val = 1e-9
        arr = np.full((8, 16), small_val, dtype=np.float32)

        dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data={"diff": arr.copy()})
        shareable = Shareable()
        fl_ctx = FLContext()

        quant_dxo = quant_filter.process_dxo(dxo, shareable, fl_ctx)
        restored_dxo = dequant_filter.process_dxo(quant_dxo, shareable, fl_ctx)

        restored = restored_dxo.data["diff"]
        assert np.allclose(restored, small_val, atol=1e-12)

    def test_zero_blocks_handled(self):
        quant_filter = Int4QuantFilter(direction=Int4QuantFilter.QUANTIZE, block_size=64, min_elements=64)
        dequant_filter = Int4DequantFilter(block_size=64, min_elements=64)

        zeros = np.zeros((8, 16), dtype=np.float32)
        dxo = DXO(data_kind=DataKind.WEIGHTS, data={"w": zeros.copy()})
        shareable = Shareable()
        fl_ctx = FLContext()

        quant_dxo = quant_filter.process_dxo(dxo, shareable, fl_ctx)
        restored_dxo = dequant_filter.process_dxo(quant_dxo, shareable, fl_ctx)

        restored = restored_dxo.data["w"]
        assert np.array_equal(restored, zeros)

    def test_simulate_mode(self):
        sim_filter = Int4QuantFilter(direction=Int4QuantFilter.SIMULATE, block_size=64, min_elements=64)
        rng = np.random.default_rng(0)
        orig_arr = rng.standard_normal((8, 16)).astype(np.float32)

        dxo = DXO(data_kind=DataKind.WEIGHTS, data={"w": orig_arr.copy()})
        shareable = Shareable()
        fl_ctx = FLContext()

        res_dxo = sim_filter.process_dxo(dxo, shareable, fl_ctx)
        assert res_dxo.get_meta_prop("int4_quantized") is True
        res = res_dxo.data["w"]
        assert res.dtype == np.float32
        assert res.shape == orig_arr.shape

        mse = np.mean((orig_arr - res) ** 2)
        assert mse < 0.05

    def test_min_elements_cutoff(self):
        filter_inst = Int4QuantFilter(block_size=64, min_elements=128)
        small_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        dxo = DXO(data_kind=DataKind.WEIGHTS, data={"small": small_arr.copy()})
        res_dxo = filter_inst.process_dxo(dxo, Shareable(), FLContext())

        assert np.array_equal(res_dxo.data["small"], small_arr)
        assert res_dxo.data["small"].dtype == np.float32

    def test_non_floating_types_untouched(self):
        filter_inst = Int4QuantFilter(block_size=64, min_elements=10)
        int_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64)

        dxo = DXO(data_kind=DataKind.WEIGHTS, data={"steps": int_arr.copy()})
        res_dxo = filter_inst.process_dxo(dxo, Shareable(), FLContext())

        assert np.array_equal(res_dxo.data["steps"], int_arr)
        assert res_dxo.data["steps"].dtype == np.int64
