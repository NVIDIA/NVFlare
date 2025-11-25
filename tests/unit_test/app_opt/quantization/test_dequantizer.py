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
from nvflare.app_opt.pt.quantization.dequantizer import ModelDequantizer
from nvflare.app_opt.pt.quantization.quantizer import ModelQuantizer


class TestModelDequantizer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.fl_ctx = FLContext()
        self.dequantizer = ModelDequantizer()

    def test_initialization(self):
        """Test ModelDequantizer initialization."""
        dequantizer = ModelDequantizer()
        self.assertIsNotNone(dequantizer)

    def test_dequantization_float16_numpy(self):
        """Test float16 dequantization with numpy arrays."""
        # First quantize
        quantizer = ModelQuantizer(quantization_type="float16")
        original_params = {
            "layer1.weight": np.random.randn(10, 10).astype(np.float32),
            "layer2.bias": np.random.randn(10).astype(np.float32),
        }
        quantized_params, quant_state, source_datatype = quantizer.quantization(original_params.copy(), self.fl_ctx)

        # Then dequantize
        dequantized_params = self.dequantizer.dequantization(
            quantized_params, quant_state, "float16", source_datatype, self.fl_ctx
        )

        self.assertIn("layer1.weight", dequantized_params)
        self.assertIn("layer2.bias", dequantized_params)
        self.assertEqual(dequantized_params["layer1.weight"].dtype, np.float32)
        self.assertEqual(dequantized_params["layer2.bias"].dtype, np.float32)
        # Check approximate equality (some precision loss expected)
        np.testing.assert_allclose(
            dequantized_params["layer1.weight"], original_params["layer1.weight"], rtol=1e-2, atol=1e-2
        )

    def test_dequantization_float16_torch(self):
        """Test float16 dequantization with torch tensors."""
        # First quantize
        quantizer = ModelQuantizer(quantization_type="float16")
        original_params = {
            "layer1.weight": torch.randn(10, 10, dtype=torch.float32),
            "layer2.bias": torch.randn(10, dtype=torch.float32),
        }
        quantized_params, quant_state, source_datatype = quantizer.quantization(original_params.copy(), self.fl_ctx)

        # Then dequantize
        dequantized_params = self.dequantizer.dequantization(
            quantized_params, quant_state, "float16", source_datatype, self.fl_ctx
        )

        self.assertIn("layer1.weight", dequantized_params)
        self.assertIn("layer2.bias", dequantized_params)
        self.assertEqual(dequantized_params["layer1.weight"].dtype, torch.float32)
        self.assertEqual(dequantized_params["layer2.bias"].dtype, torch.float32)

    def test_dequantization_adaquant_torch(self):
        """Test adaquant dequantization with torch tensors (AdaQuant requires tensors)."""
        quantizer = ModelQuantizer(quantization_type="adaquant")
        # AdaQuant works with tensors only
        original_params = {
            "layer1.weight": torch.randn(10, 10, dtype=torch.float32),
            "layer2.bias": torch.randn(10, dtype=torch.float32),
        }

        quantized_params, quant_state, source_datatype = quantizer.quantization(original_params.copy(), self.fl_ctx)

        dequantized_params = self.dequantizer.dequantization(
            quantized_params, quant_state, "adaquant", source_datatype, self.fl_ctx
        )

        self.assertIn("layer1.weight", dequantized_params)
        self.assertIn("layer2.bias", dequantized_params)
        self.assertEqual(dequantized_params["layer1.weight"].dtype, torch.float32)
        # Check approximate equality
        self.assertTrue(
            torch.allclose(dequantized_params["layer1.weight"], original_params["layer1.weight"], rtol=0.1, atol=0.1)
        )

    def test_dequantization_skip_lower_precision(self):
        """Test dequantization skips when target precision >= source precision."""
        quantizer = ModelQuantizer(quantization_type="float16")
        original_params = {
            "layer1.weight": np.random.randn(10, 10).astype(np.float16),
        }

        quantized_params, quant_state, source_datatype = quantizer.quantization(original_params.copy(), self.fl_ctx)

        dequantized_params = self.dequantizer.dequantization(
            quantized_params, quant_state, "float16", source_datatype, self.fl_ctx
        )

        # Should have original param (skipped quantization/dequantization)
        self.assertIn("layer1.weight", dequantized_params)

    def test_dequantization_invalid_data_type(self):
        """Test dequantization with invalid data type."""
        params = {
            "layer1.weight": [1, 2, 3],  # List, not numpy or torch
        }
        quant_state = {"layer1.weight": {}}
        source_datatype = {"layer1.weight": "float32"}

        # The code tries to access .nbytes before type checking, so AttributeError is raised
        with self.assertRaises(AttributeError):
            self.dequantizer.dequantization(params, quant_state, "float16", source_datatype, self.fl_ctx)

    def test_dequantization_invalid_quantization_type(self):
        """Test dequantization with invalid quantization type (without digits)."""
        params = {
            "layer1.weight": np.random.randn(10, 10).astype(np.float16),
        }
        quant_state = {"layer1.weight": {}}
        source_datatype = {"layer1.weight": "float32"}

        # The code tries to extract digits from quantization_type, raising IndexError if none exist
        with self.assertRaises(IndexError):
            self.dequantizer.dequantization(params, quant_state, "invalid_type", source_datatype, self.fl_ctx)

    def test_process_dxo_with_quantized_data(self):
        """Test process_dxo with properly quantized DXO."""
        # First quantize
        quantizer = ModelQuantizer(quantization_type="float16")
        original_params = {
            "layer1.weight": torch.randn(10, 10, dtype=torch.float32),
        }
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=original_params)
        shareable = Shareable()

        quantized_dxo = quantizer.process_dxo(dxo, shareable, self.fl_ctx)

        # Then dequantize
        dequantized_dxo = self.dequantizer.process_dxo(quantized_dxo, shareable, self.fl_ctx)

        self.assertIsNotNone(dequantized_dxo)
        self.assertIn("layer1.weight", dequantized_dxo.data)
        self.assertEqual(dequantized_dxo.data["layer1.weight"].dtype, torch.float32)
        # Meta properties should be removed after dequantization
        self.assertIsNone(dequantized_dxo.get_meta_prop(MetaKey.PROCESSED_ALGORITHM))
        self.assertIsNone(dequantized_dxo.get_meta_prop("quant_state"))
        self.assertIsNone(dequantized_dxo.get_meta_prop("source_datatype"))
        self.assertIsNone(dequantized_dxo.get_meta_prop("quantized_flag"))

    def test_process_dxo_invalid_quantization_type(self):
        """Test process_dxo with invalid quantization type in meta."""
        params = {
            "layer1.weight": torch.randn(10, 10, dtype=torch.float16),
        }
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=params)
        dxo.set_meta_prop(MetaKey.PROCESSED_ALGORITHM, "invalid_type")
        dxo.set_meta_prop("quant_state", {})
        dxo.set_meta_prop("source_datatype", {})
        shareable = Shareable()

        with self.assertRaises(ValueError):
            self.dequantizer.process_dxo(dxo, shareable, self.fl_ctx)

    def test_roundtrip_float16_numpy(self):
        """Test full quantization-dequantization roundtrip with float16 and numpy."""
        quantizer = ModelQuantizer(quantization_type="float16")

        original_params = {
            "layer1.weight": np.random.randn(10, 10).astype(np.float32),
            "layer2.bias": np.random.randn(10).astype(np.float32),
        }

        # Quantize
        quantized_params, quant_state, source_datatype = quantizer.quantization(original_params.copy(), self.fl_ctx)

        # Dequantize
        dequantized_params = self.dequantizer.dequantization(
            quantized_params, quant_state, "float16", source_datatype, self.fl_ctx
        )

        # Verify shapes match
        for key in original_params.keys():
            self.assertEqual(dequantized_params[key].shape, original_params[key].shape)
            self.assertEqual(dequantized_params[key].dtype, original_params[key].dtype)
            # Verify approximate equality
            np.testing.assert_allclose(dequantized_params[key], original_params[key], rtol=1e-2, atol=1e-2)

    def test_roundtrip_float16_torch(self):
        """Test full quantization-dequantization roundtrip with float16 and torch."""
        quantizer = ModelQuantizer(quantization_type="float16")

        original_params = {
            "layer1.weight": torch.randn(10, 10, dtype=torch.float32),
            "layer2.bias": torch.randn(10, dtype=torch.float32),
        }

        # Quantize
        quantized_params, quant_state, source_datatype = quantizer.quantization(original_params.copy(), self.fl_ctx)

        # Dequantize
        dequantized_params = self.dequantizer.dequantization(
            quantized_params, quant_state, "float16", source_datatype, self.fl_ctx
        )

        # Verify shapes match
        for key in original_params.keys():
            self.assertEqual(dequantized_params[key].shape, original_params[key].shape)
            self.assertEqual(dequantized_params[key].dtype, original_params[key].dtype)
            # Verify approximate equality
            self.assertTrue(torch.allclose(dequantized_params[key], original_params[key], rtol=1e-2, atol=1e-2))

    def test_roundtrip_adaquant_torch(self):
        """Test full quantization-dequantization roundtrip with adaquant and torch."""
        quantizer = ModelQuantizer(quantization_type="adaquant")

        # AdaQuant works with tensors only
        original_params = {
            "layer1.weight": torch.randn(10, 10, dtype=torch.float32),
            "layer2.bias": torch.randn(10, dtype=torch.float32),
        }

        # Quantize
        quantized_params, quant_state, source_datatype = quantizer.quantization(original_params.copy(), self.fl_ctx)

        # Dequantize
        dequantized_params = self.dequantizer.dequantization(
            quantized_params, quant_state, "adaquant", source_datatype, self.fl_ctx
        )

        # Verify shapes match
        for key in original_params.keys():
            self.assertEqual(dequantized_params[key].shape, original_params[key].shape)
            self.assertEqual(dequantized_params[key].dtype, original_params[key].dtype)

    def test_dequantization_bfloat16_source(self):
        """Test dequantization back to bfloat16."""
        quantizer = ModelQuantizer(quantization_type="adaquant")
        original_params = {
            "layer1.weight": torch.randn(10, 10, dtype=torch.bfloat16),
        }

        quantized_params, quant_state, source_datatype = quantizer.quantization(original_params.copy(), self.fl_ctx)

        dequantized_params = self.dequantizer.dequantization(
            quantized_params, quant_state, "adaquant", source_datatype, self.fl_ctx
        )

        self.assertEqual(dequantized_params["layer1.weight"].dtype, torch.bfloat16)

    def test_dequantization_multiple_params(self):
        """Test dequantization with multiple parameters."""
        quantizer = ModelQuantizer(quantization_type="float16")
        original_params = {
            "layer1.weight": torch.randn(10, 10, dtype=torch.float32),
            "layer1.bias": torch.randn(10, dtype=torch.float32),
            "layer2.weight": torch.randn(5, 10, dtype=torch.float32),
            "layer2.bias": torch.randn(5, dtype=torch.float32),
        }

        quantized_params, quant_state, source_datatype = quantizer.quantization(original_params.copy(), self.fl_ctx)

        dequantized_params = self.dequantizer.dequantization(
            quantized_params, quant_state, "float16", source_datatype, self.fl_ctx
        )

        self.assertEqual(len(dequantized_params), 4)
        for key in original_params.keys():
            self.assertIn(key, dequantized_params)
            self.assertEqual(dequantized_params[key].dtype, torch.float32)

    def test_dequantization_empty_params(self):
        """Test dequantization with empty parameters."""
        params = {}
        quant_state = {}
        source_datatype = {}

        dequantized_params = self.dequantizer.dequantization(
            params, quant_state, "float16", source_datatype, self.fl_ctx
        )

        self.assertEqual(len(dequantized_params), 0)

    def test_dequantization_preserves_numpy_format(self):
        """Test that dequantization preserves numpy format when input is numpy."""
        quantizer = ModelQuantizer(quantization_type="float16")
        original_params = {
            "layer1.weight": np.random.randn(10, 10).astype(np.float32),
        }

        quantized_params, quant_state, source_datatype = quantizer.quantization(original_params.copy(), self.fl_ctx)

        dequantized_params = self.dequantizer.dequantization(
            quantized_params, quant_state, "float16", source_datatype, self.fl_ctx
        )

        self.assertIsInstance(dequantized_params["layer1.weight"], np.ndarray)

    def test_dequantization_preserves_torch_format(self):
        """Test that dequantization preserves torch format when input is torch."""
        quantizer = ModelQuantizer(quantization_type="float16")
        original_params = {
            "layer1.weight": torch.randn(10, 10, dtype=torch.float32),
        }

        quantized_params, quant_state, source_datatype = quantizer.quantization(original_params.copy(), self.fl_ctx)

        dequantized_params = self.dequantizer.dequantization(
            quantized_params, quant_state, "float16", source_datatype, self.fl_ctx
        )

        self.assertIsInstance(dequantized_params["layer1.weight"], torch.Tensor)


if __name__ == "__main__":
    unittest.main()
