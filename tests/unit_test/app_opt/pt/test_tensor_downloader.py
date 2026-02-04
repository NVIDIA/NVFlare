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

"""Unit tests for TensorDownloadable.

Note: Deep copy protection is now handled at broadcast level in WFCommServer,
not in TensorDownloadable itself. These tests verify the Downloadable's basic behavior.
"""

import torch

from nvflare.app_opt.pt.tensor_downloader import TensorDownloadable


class TestTensorDownloadableBasic:
    """Test basic TensorDownloadable functionality."""

    def test_basic_functionality(self):
        """Verify basic Downloadable creation and data access."""
        # Create tensors
        tensors = {
            "weights": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "bias": torch.tensor([0.5, 1.5]),
        }

        # Create downloadable
        downloadable = TensorDownloadable(tensors=tensors, max_chunk_size=1024)

        # Verify basic properties
        assert downloadable.size == 2
        assert set(downloadable.keys) == {"weights", "bias"}
        assert torch.allclose(downloadable.base_obj["weights"], tensors["weights"])
        assert torch.allclose(downloadable.base_obj["bias"], tensors["bias"])

    def test_shares_memory_with_original(self):
        """Verify that Downloadable references original tensors (no copy at this level)."""
        original = {"layer": torch.tensor([1.0, 2.0, 3.0])}

        downloadable = TensorDownloadable(tensors=original, max_chunk_size=1024)

        # Should share memory (snapshot is done at broadcast level, not here)
        assert (
            downloadable.base_obj["layer"].data_ptr() == original["layer"].data_ptr()
        ), "Downloadable should share memory with original (copy is done at broadcast level)"

    def test_modification_affects_downloadable(self):
        """Verify that modifications to original DO affect Downloadable (by design).

        Note: Protection against this is now handled at broadcast level in WFCommServer.
        """
        tensors = {"model": torch.tensor([1.0, 2.0])}

        downloadable = TensorDownloadable(tensors=tensors, max_chunk_size=1024)

        # Modify original
        tensors["model"][0] = 999.0

        # Downloadable IS affected (this is expected - protection is at broadcast level)
        assert downloadable.base_obj["model"][0].item() == 999.0
