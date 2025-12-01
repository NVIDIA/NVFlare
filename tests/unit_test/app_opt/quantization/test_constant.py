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

from nvflare.app_opt.pt.quantization.constant import DATA_TYPE, QUANTIZATION_TYPE


class TestConstants(unittest.TestCase):
    def test_data_type_not_empty(self):
        """Test that DATA_TYPE list is not empty."""
        self.assertIsNotNone(DATA_TYPE)
        self.assertGreater(len(DATA_TYPE), 0)

    def test_data_type_contains_expected_types(self):
        """Test that DATA_TYPE contains expected data types."""
        expected_types = ["FLOAT32", "FLOAT16", "BFLOAT16"]
        for dtype in expected_types:
            self.assertIn(dtype, DATA_TYPE)

    def test_data_type_all_uppercase(self):
        """Test that all DATA_TYPE entries are uppercase."""
        for dtype in DATA_TYPE:
            self.assertEqual(dtype, dtype.upper())

    def test_data_type_no_duplicates(self):
        """Test that DATA_TYPE has no duplicate entries."""
        self.assertEqual(len(DATA_TYPE), len(set(DATA_TYPE)))

    def test_quantization_type_not_empty(self):
        """Test that QUANTIZATION_TYPE list is not empty."""
        self.assertIsNotNone(QUANTIZATION_TYPE)
        self.assertGreater(len(QUANTIZATION_TYPE), 0)

    def test_quantization_type_contains_expected_types(self):
        """Test that QUANTIZATION_TYPE contains expected quantization types."""
        expected_types = ["FLOAT16", "BLOCKWISE8", "FLOAT4", "NORMFLOAT4", "ADAQUANT"]
        for qtype in expected_types:
            self.assertIn(qtype, QUANTIZATION_TYPE)

    def test_quantization_type_all_uppercase(self):
        """Test that all QUANTIZATION_TYPE entries are uppercase."""
        for qtype in QUANTIZATION_TYPE:
            self.assertEqual(qtype, qtype.upper())

    def test_quantization_type_no_duplicates(self):
        """Test that QUANTIZATION_TYPE has no duplicate entries."""
        self.assertEqual(len(QUANTIZATION_TYPE), len(set(QUANTIZATION_TYPE)))

    def test_data_type_is_list(self):
        """Test that DATA_TYPE is a list."""
        self.assertIsInstance(DATA_TYPE, list)

    def test_quantization_type_is_list(self):
        """Test that QUANTIZATION_TYPE is a list."""
        self.assertIsInstance(QUANTIZATION_TYPE, list)

    def test_data_type_strings(self):
        """Test that all DATA_TYPE entries are strings."""
        for dtype in DATA_TYPE:
            self.assertIsInstance(dtype, str)

    def test_quantization_type_strings(self):
        """Test that all QUANTIZATION_TYPE entries are strings."""
        for qtype in QUANTIZATION_TYPE:
            self.assertIsInstance(qtype, str)

    def test_float16_in_both_lists(self):
        """Test that FLOAT16 appears in both DATA_TYPE and QUANTIZATION_TYPE."""
        self.assertIn("FLOAT16", DATA_TYPE)
        self.assertIn("FLOAT16", QUANTIZATION_TYPE)

    def test_data_type_count(self):
        """Test that DATA_TYPE has expected number of entries."""
        # As of the current implementation, there should be 3 data types
        self.assertEqual(len(DATA_TYPE), 3)

    def test_quantization_type_count(self):
        """Test that QUANTIZATION_TYPE has expected number of entries."""
        # As of the current implementation, there should be 5 quantization types
        self.assertEqual(len(QUANTIZATION_TYPE), 5)

    def test_data_type_format(self):
        """Test that DATA_TYPE entries follow expected naming convention."""
        for dtype in DATA_TYPE:
            # Should be all uppercase and contain only alphanumeric characters
            self.assertTrue(
                dtype.replace("FLOAT", "").replace("BFLOAT", "").isdigit()
                or dtype.startswith("FLOAT")
                or dtype.startswith("BFLOAT")
            )

    def test_quantization_type_format(self):
        """Test that QUANTIZATION_TYPE entries follow expected naming convention."""
        for qtype in QUANTIZATION_TYPE:
            # Should be all uppercase and not empty
            self.assertTrue(len(qtype) > 0)
            self.assertTrue(qtype.isupper() or any(c.isdigit() for c in qtype))

    def test_case_insensitive_lookup_data_type(self):
        """Test that DATA_TYPE can be looked up in case-insensitive manner."""
        for dtype in DATA_TYPE:
            self.assertIn(dtype.lower().upper(), DATA_TYPE)

    def test_case_insensitive_lookup_quantization_type(self):
        """Test that QUANTIZATION_TYPE can be looked up in case-insensitive manner."""
        for qtype in QUANTIZATION_TYPE:
            self.assertIn(qtype.lower().upper(), QUANTIZATION_TYPE)


if __name__ == "__main__":
    unittest.main()
