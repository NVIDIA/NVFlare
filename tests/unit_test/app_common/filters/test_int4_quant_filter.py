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
from nvflare.app_common.filters.int4_quant_filter import Int4QuantFilter


class TestInt4QuantFilter:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            Int4QuantFilter(block_size=63)
        with pytest.raises(ValueError):
            Int4QuantFilter(block_size=0)
        with pytest.raises(ValueError):
            Int4QuantFilter(min_elements=-1)

    def test_quantize_roundtrip_fidelity(self):
        filter_inst = Int4QuantFilter(block_size=64, min_elements=64)
        rng = np.random.default_rng(42)
        arr = rng.standard_normal((10, 20)).astype(np.float32)

        quantized = filter_inst._quantize_array(arr)
        assert quantized.shape == arr.shape
        assert quantized.dtype == arr.dtype

        mse = np.mean((arr - quantized) ** 2)
        assert mse < 0.05

    def test_min_elements_cutoff(self):
        filter_inst = Int4QuantFilter(block_size=64, min_elements=128)
        small_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        res = filter_inst._quantize_array(small_arr)
        assert np.array_equal(res, small_arr)

    def test_non_floating_types_untouched(self):
        filter_inst = Int4QuantFilter(block_size=64, min_elements=10)
        int_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64)

        res = filter_inst._quantize_array(int_arr)
        assert np.array_equal(res, int_arr)
        assert res.dtype == np.int64

    def test_process_dxo_pipeline(self):
        filter_inst = Int4QuantFilter(block_size=64, min_elements=64)
        rng = np.random.default_rng(0)

        data = {
            "w1": rng.standard_normal((8, 16)).astype(np.float32),
            "step": np.array([100], dtype=np.int32),
        }
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=data)
        shareable = Shareable()
        fl_ctx = FLContext()

        res_dxo = filter_inst.process_dxo(dxo, shareable, fl_ctx)

        assert res_dxo.get_meta_prop("int4_quantized") is True
        assert np.array_equal(res_dxo.data["step"], np.array([100], dtype=np.int32))
        assert res_dxo.data["w1"].shape == (8, 16)
