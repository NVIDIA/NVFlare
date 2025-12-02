# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import unittest

import numpy as np
import torch

from nvflare.apis.dxo import DXO, DataKind, MetaKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_opt.pt.quantization.quantizer import ModelQuantizer


class TestModelQuantizer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.fl_ctx = FLContext()

    def test_initialization_default(self):
        """Test ModelQuantizer initialization with default parameters."""
        quantizer = ModelQuantizer()
        self.assertEqual(quantizer.quantization_type, "float16")

    def test_initialization_custom(self):
        """Test ModelQuantizer initialization with custom parameters."""
        quantizer = ModelQuantizer(quantization_type="adaquant")
        self.assertEqual(quantizer.quantization_type, "adaquant")

    def test_initialization_invalid_type(self):
        """Test ModelQuantizer initialization with invalid quantization type."""
        with self.assertRaises(ValueError):
            ModelQuantizer(quantization_type="invalid_type")

    def test_initialization_case_insensitive(self):
        """Test ModelQuantizer initialization is case insensitive."""
        quantizer = ModelQuantizer(quantization_type="FLOAT16")
        self.assertEqual(quantizer.quantization_type, "float16")

    def test_quantization_float16_numpy(self):
        """Test float16 quantization with numpy arrays."""
        quantizer = ModelQuantizer(quantization_type="float16")
        params = {
            "layer1.weight": np.random.randn(10, 10).astype(np.float32),
            "layer2.bias": np.random.randn(10).astype(np.float32),
        }

        quantized_params, quant_state, source_datatype = quantizer.quantization(params, self.fl_ctx)

        self.assertIn("layer1.weight", quantized_params)
        self.assertIn("layer2.bias", quantized_params)
        self.assertEqual(quantized_params["layer1.weight"].dtype, np.float16)
        self.assertEqual(quantized_params["layer2.bias"].dtype, np.float16)
        self.assertEqual(source_datatype["layer1.weight"], "float32")

    def test_quantization_float16_torch(self):
        """Test float16 quantization with torch tensors."""
        quantizer = ModelQuantizer(quantization_type="float16")
        params = {
            "layer1.weight": torch.randn(10, 10, dtype=torch.float32),
            "layer2.bias": torch.randn(10, dtype=torch.float32),
        }

        quantized_params, quant_state, source_datatype = quantizer.quantization(params, self.fl_ctx)

        self.assertIn("layer1.weight", quantized_params)
        self.assertIn("layer2.bias", quantized_params)
        self.assertEqual(quantized_params["layer1.weight"].dtype, torch.float16)
        self.assertEqual(quantized_params["layer2.bias"].dtype, torch.float16)
        self.assertEqual(source_datatype["layer1.weight"], "float32")

    def test_quantization_float16_clamping(self):
        """Test float16 quantization properly clamps extreme values."""
        quantizer = ModelQuantizer(quantization_type="float16")
        # Create values that exceed float16 range
        large_value = 1e10
        params = {
            "layer1.weight": np.array([large_value, -large_value]).astype(np.float32),
        }

        quantized_params, quant_state, source_datatype = quantizer.quantization(params, self.fl_ctx)

        # Values should be clamped to float16 range
        self.assertTrue(np.all(quantized_params["layer1.weight"] <= np.finfo(np.float16).max))
        self.assertTrue(np.all(quantized_params["layer1.weight"] >= np.finfo(np.float16).min))

    def test_quantization_adaquant_numpy(self):
        """Test adaquant quantization with numpy arrays (converted to torch for AdaQuant)."""
        quantizer = ModelQuantizer(quantization_type="adaquant")
        # AdaQuant works with tensors only, so convert numpy to torch
        params = {
            "layer1.weight": torch.from_numpy(np.random.randn(10, 10).astype(np.float32)),
            "layer2.bias": torch.from_numpy(np.random.randn(10).astype(np.float32)),
        }

        quantized_params, quant_state, source_datatype = quantizer.quantization(params, self.fl_ctx)

        self.assertIn("layer1.weight", quantized_params)
        self.assertIn("layer2.bias", quantized_params)
        # AdaQuant may or may not have quant_state depending on tensor
        self.assertEqual(source_datatype["layer1.weight"], "float32")

    def test_quantization_adaquant_torch(self):
        """Test adaquant quantization with torch tensors."""
        quantizer = ModelQuantizer(quantization_type="adaquant")
        params = {
            "layer1.weight": torch.randn(10, 10, dtype=torch.float32),
            "layer2.bias": torch.randn(10, dtype=torch.float32),
        }

        quantized_params, quant_state, source_datatype = quantizer.quantization(params, self.fl_ctx)

        self.assertIn("layer1.weight", quantized_params)
        self.assertIn("layer2.bias", quantized_params)
        self.assertEqual(source_datatype["layer1.weight"], "float32")

    def test_quantization_skip_lower_precision(self):
        """Test quantization skips when target precision >= source precision."""
        quantizer = ModelQuantizer(quantization_type="float16")
        params = {
            "layer1.weight": np.random.randn(10, 10).astype(np.float16),
        }

        quantized_params, quant_state, source_datatype = quantizer.quantization(params, self.fl_ctx)

        # Should skip quantization since source is already float16
        self.assertEqual(quantized_params["layer1.weight"].dtype, np.float16)

    def test_quantization_invalid_data_type(self):
        """Test quantization with invalid data type."""
        quantizer = ModelQuantizer(quantization_type="float16")
        params = {
            "layer1.weight": [1, 2, 3],  # List, not numpy or torch
        }

        with self.assertRaises(ValueError):
            quantizer.quantization(params, self.fl_ctx)

    def test_quantization_invalid_source_dtype(self):
        """Test quantization with unsupported source dtype."""
        quantizer = ModelQuantizer(quantization_type="float16")
        params = {
            "layer1.weight": np.random.randn(10, 10).astype(np.float64),
        }

        with self.assertRaises(ValueError):
            quantizer.quantization(params, self.fl_ctx)

    def test_process_dxo_first_time(self):
        """Test process_dxo on first quantization."""
        quantizer = ModelQuantizer(quantization_type="float16")

        params = {
            "layer1.weight": torch.randn(10, 10, dtype=torch.float32),
        }
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=params)
        shareable = Shareable()

        result_dxo = quantizer.process_dxo(dxo, shareable, self.fl_ctx)

        self.assertIsNotNone(result_dxo)
        self.assertEqual(result_dxo.get_meta_prop(MetaKey.PROCESSED_ALGORITHM), "float16")
        self.assertTrue(result_dxo.get_meta_prop("quantized_flag"))
        self.assertIsNotNone(result_dxo.get_meta_prop("quant_state"))
        self.assertIsNotNone(result_dxo.get_meta_prop("source_datatype"))

    def test_process_dxo_already_quantized(self):
        """Test process_dxo skips already quantized data."""
        quantizer = ModelQuantizer(quantization_type="float16")

        params = {
            "layer1.weight": torch.randn(10, 10, dtype=torch.float16),
        }
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=params)
        dxo.set_meta_prop("quantized_flag", True)
        shareable = Shareable()

        result_dxo = quantizer.process_dxo(dxo, shareable, self.fl_ctx)

        self.assertIsNotNone(result_dxo)
        # Should skip quantization
        self.assertTrue(result_dxo.get_meta_prop("quantized_flag"))

    def test_process_dxo_weight_diff(self):
        """Test process_dxo with WEIGHT_DIFF data kind."""
        quantizer = ModelQuantizer(quantization_type="float16")

        params = {
            "layer1.weight": torch.randn(10, 10, dtype=torch.float32),
        }
        dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=params)
        shareable = Shareable()

        result_dxo = quantizer.process_dxo(dxo, shareable, self.fl_ctx)

        self.assertIsNotNone(result_dxo)
        self.assertEqual(result_dxo.get_meta_prop(MetaKey.PROCESSED_ALGORITHM), "float16")

    def test_quantization_multiple_params(self):
        """Test quantization with multiple parameters."""
        quantizer = ModelQuantizer(quantization_type="float16")
        params = {
            "layer1.weight": torch.randn(10, 10, dtype=torch.float32),
            "layer1.bias": torch.randn(10, dtype=torch.float32),
            "layer2.weight": torch.randn(5, 10, dtype=torch.float32),
            "layer2.bias": torch.randn(5, dtype=torch.float32),
        }

        quantized_params, quant_state, source_datatype = quantizer.quantization(params, self.fl_ctx)

        self.assertEqual(len(quantized_params), 4)
        for key in params.keys():
            self.assertIn(key, quantized_params)
            self.assertEqual(quantized_params[key].dtype, torch.float16)
            self.assertEqual(source_datatype[key], "float32")

    def test_quantization_mixed_shapes(self):
        """Test quantization with various tensor shapes."""
        quantizer = ModelQuantizer(quantization_type="float16")
        params = {
            "scalar": torch.tensor([1.0], dtype=torch.float32),
            "vector": torch.randn(100, dtype=torch.float32),
            "matrix": torch.randn(10, 20, dtype=torch.float32),
            "tensor_3d": torch.randn(5, 10, 15, dtype=torch.float32),
            "tensor_4d": torch.randn(2, 3, 4, 5, dtype=torch.float32),
        }

        quantized_params, quant_state, source_datatype = quantizer.quantization(params, self.fl_ctx)

        for key, original_tensor in params.items():
            self.assertIn(key, quantized_params)
            self.assertEqual(quantized_params[key].shape, original_tensor.shape)
            self.assertEqual(quantized_params[key].dtype, torch.float16)

    def test_quantization_preserves_numpy_format(self):
        """Test that quantization preserves numpy format when input is numpy."""
        quantizer = ModelQuantizer(quantization_type="float16")
        params = {
            "layer1.weight": np.random.randn(10, 10).astype(np.float32),
        }

        quantized_params, quant_state, source_datatype = quantizer.quantization(params, self.fl_ctx)

        self.assertIsInstance(quantized_params["layer1.weight"], np.ndarray)

    def test_quantization_preserves_torch_format(self):
        """Test that quantization preserves torch format when input is torch."""
        quantizer = ModelQuantizer(quantization_type="float16")
        params = {
            "layer1.weight": torch.randn(10, 10, dtype=torch.float32),
        }

        quantized_params, quant_state, source_datatype = quantizer.quantization(params, self.fl_ctx)

        self.assertIsInstance(quantized_params["layer1.weight"], torch.Tensor)

    def test_quantization_bfloat16_torch(self):
        """Test quantization with bfloat16 source data."""
        quantizer = ModelQuantizer(quantization_type="adaquant")
        params = {
            "layer1.weight": torch.randn(10, 10, dtype=torch.bfloat16),
        }

        quantized_params, quant_state, source_datatype = quantizer.quantization(params, self.fl_ctx)

        self.assertIn("layer1.weight", quantized_params)
        self.assertEqual(source_datatype["layer1.weight"], "bfloat16")

    def test_quantization_empty_params(self):
        """Test quantization with empty parameters."""
        quantizer = ModelQuantizer(quantization_type="float16")
        params = {}

        quantized_params, quant_state, source_datatype = quantizer.quantization(params, self.fl_ctx)

        self.assertEqual(len(quantized_params), 0)
        self.assertEqual(len(quant_state), 0)
        self.assertEqual(len(source_datatype), 0)

    def test_quantization_large_tensor(self):
        """Test quantization with large tensors."""
        quantizer = ModelQuantizer(quantization_type="float16")
        # Use float32 to ensure quantization to float16 reduces size
        original_tensor = torch.randn(1000, 1000, dtype=torch.float32)
        original_size = original_tensor.nbytes

        params = {
            "large_layer": original_tensor,
        }

        quantized_params, quant_state, source_datatype = quantizer.quantization(params, self.fl_ctx)

        self.assertIn("large_layer", quantized_params)
        self.assertEqual(quantized_params["large_layer"].shape, (1000, 1000))
        self.assertEqual(quantized_params["large_layer"].dtype, torch.float16)
        # Verify size reduction (float32 -> float16 should be 50% reduction)
        quantized_size = quantized_params["large_layer"].nbytes
        self.assertLess(quantized_size, original_size)
        self.assertEqual(quantized_size, original_size // 2)


if __name__ == "__main__":
    unittest.main()
