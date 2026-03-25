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

import pathlib
import sys
import tempfile
import unittest

EXAMPLE_DIR = pathlib.Path(__file__).resolve().parents[5] / "examples" / "advanced" / "medgemma"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from download_data import resolve_extraction_path  # noqa: E402


class MedGemmaDownloadDataTest(unittest.TestCase):
    def test_resolve_extraction_path_allows_descendants(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            expected_path = str(pathlib.Path(temp_dir).resolve() / "subdir" / "file.tif")

            self.assertEqual(resolve_extraction_path(temp_dir, "subdir/file.tif"), expected_path)

    def test_resolve_extraction_path_rejects_parent_traversal(self):
        with self.assertRaisesRegex(ValueError, "unsafe path"):
            resolve_extraction_path("/tmp/medgemma-data", "../escape.txt")


if __name__ == "__main__":
    unittest.main()
