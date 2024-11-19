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

import numpy as np
import pytest

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.quantization.numpy_dequantizor import NumpyModelDequantizor
from nvflare.app_opt.quantization.numpy_quantizor import NumpyModelQuantizor

TEST_CASES = [
    (
        {"a": np.array([1.0, 2.0, 3.0, 70000.0], dtype="float32")},
        "float16",
        {"a": np.array([1.0, 2.0, 3.0, 65504.0], dtype="float32")},
    ),
    (
        {"a": np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")},
        "blockwise8",
        {"a": np.array([0.99062496, 2.003125, 3.015625, 4.0], dtype="float32")},
    ),
]


class TestQuantization:
    @pytest.mark.parametrize("input_data, quantization_type, expected_data", TEST_CASES)
    def test_quantization(self, input_data, quantization_type, expected_data):
        dxo = DXO(
            data_kind=DataKind.WEIGHTS,
            data=input_data,
        )
        fl_ctx = FLContext()
        f_quant = NumpyModelQuantizor(quantization_type=quantization_type)
        quant_dxo = f_quant.process_dxo(dxo, dxo.to_shareable(), fl_ctx)
        f_dequant = NumpyModelDequantizor(source_data_type="float32")
        dequant_dxo = f_dequant.process_dxo(quant_dxo, dxo.to_shareable(), fl_ctx)
        dequant_data = dequant_dxo.data
        for key in dequant_data.keys():
            dequant_array = dequant_data[key]
            expected_array = expected_data[key]
            assert np.allclose(dequant_array, expected_array)
