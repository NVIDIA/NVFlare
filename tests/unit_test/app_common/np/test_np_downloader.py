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

"""Unit tests for ArrayDownloadable.

Note: Deep copy protection is now handled at broadcast level in WFCommServer,
not in ArrayDownloadable itself. These tests verify the Downloadable's basic behavior.
"""

import numpy as np

from nvflare.app_common.np.np_downloader import ArrayDownloadable


class TestArrayDownloadableBasic:
    """Test basic ArrayDownloadable functionality."""

    def test_basic_functionality(self):
        """Verify basic Downloadable creation and data access."""
        # Create arrays
        arrays = {
            "weights": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "bias": np.array([0.5, 1.5]),
        }

        # Create downloadable
        downloadable = ArrayDownloadable(arrays=arrays, max_chunk_size=1024)

        # Verify basic properties
        assert downloadable.size == 2
        assert set(downloadable.keys) == {"weights", "bias"}
        assert np.allclose(downloadable.base_obj["weights"], arrays["weights"])
        assert np.allclose(downloadable.base_obj["bias"], arrays["bias"])

    def test_shares_memory_with_original(self):
        """Verify that Downloadable references original arrays (no copy at this level)."""
        original = {"layer": np.array([1.0, 2.0, 3.0])}

        downloadable = ArrayDownloadable(arrays=original, max_chunk_size=1024)

        # Should share memory (snapshot is done at broadcast level, not here)
        assert np.shares_memory(
            downloadable.base_obj["layer"], original["layer"]
        ), "Downloadable should share memory with original (copy is done at broadcast level)"

    def test_modification_affects_downloadable(self):
        """Verify that modifications to original DO affect Downloadable (by design).

        Note: Protection against this is now handled at broadcast level in WFCommServer.
        """
        arrays = {"model": np.array([1.0, 2.0])}

        downloadable = ArrayDownloadable(arrays=arrays, max_chunk_size=1024)

        # Modify original
        arrays["model"][0] = 999.0

        # Downloadable IS affected (this is expected - protection is at broadcast level)
        assert downloadable.base_obj["model"][0] == 999.0
