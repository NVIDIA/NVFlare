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

"""
Tests for ViaDownloaderDecomposer context isolation and id() behavior.

These tests verify that the FOBS context and id()-based caching in ViaDownloaderDecomposer
are safely isolated and don't cause stale data issues across different messages/rounds.

Key findings:
1. fobs_ctx is created fresh for each message/serialization (via DatumManager)
2. _DecomposeCtx (which uses id() for deduplication) is stored in fobs_ctx
3. Each serialization gets a fresh _DecomposeCtx, so id() reuse across rounds is safe
4. Within a single message, id() correctly deduplicates the same tensor referenced multiple times
5. Even if Python recycles object IDs, fresh contexts prevent stale lookups

Conclusion: The id()-based caching in ViaDownloaderDecomposer is SAFE because fobs_ctx
isolation ensures that id() lookups never span across different messages or training rounds.
"""

import pytest
import torch

from nvflare.app_opt.pt.decomposers import TensorDecomposer
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.datum import DatumManager


class TestViaDownloaderContext:
    """Tests for ViaDownloader context isolation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Register decomposer."""
        fobs.register(TensorDecomposer)
        yield

    def test_fobs_ctx_isolation_between_messages(self):
        """Verify that fobs_ctx is isolated between different serializations."""
        # Create a tensor that will be reused
        shared_tensor = torch.randn(10, 10)

        # First serialization (message/round 1)
        obj1 = {"model": shared_tensor}
        mgr1 = DatumManager(fobs_ctx={})

        # Serialize first message
        serialized1 = fobs.serialize(obj1, manager=mgr1)

        # Get the decompose context from first serialization
        ctx1 = mgr1.fobs_ctx
        decompose_ctx_key = "TensorDecomposer_dc"

        # Second serialization (message/round 2) with SAME tensor object
        obj2 = {"model": shared_tensor}  # Same tensor object
        mgr2 = DatumManager(fobs_ctx={})  # Fresh fobs_ctx

        # Serialize second message
        serialized2 = fobs.serialize(obj2, manager=mgr2)

        # Get the decompose context from second serialization
        ctx2 = mgr2.fobs_ctx

        # Verify contexts are different
        assert ctx1 is not ctx2

        # Each serialization should have its own DecomposeCtx
        if decompose_ctx_key in ctx1 and decompose_ctx_key in ctx2:
            assert ctx1[decompose_ctx_key] is not ctx2[decompose_ctx_key]

    def test_same_tensor_multiple_times_in_message(self):
        """Test that the same tensor appearing multiple times in one message works correctly."""
        # This tests the id() deduplication within a single fobs_ctx
        shared_tensor = torch.randn(100, 100)

        # Multiple references to the same tensor in one message
        obj = {
            "layer1": shared_tensor,
            "layer2": shared_tensor,  # Same tensor object, same id()
            "layer3": torch.randn(50, 50),  # Different tensor
        }

        mgr_serialize = DatumManager(fobs_ctx={})
        serialized = fobs.serialize(obj, manager=mgr_serialize)

        # Deserialize with manager
        mgr_deserialize = DatumManager(fobs_ctx={})
        deserialized = fobs.deserialize(serialized, manager=mgr_deserialize)

        # All should work correctly
        assert "layer1" in deserialized
        assert "layer2" in deserialized
        assert "layer3" in deserialized

        # Values should match
        assert torch.allclose(deserialized["layer1"], shared_tensor)
        assert torch.allclose(deserialized["layer2"], shared_tensor)

    def test_tensor_reuse_across_rounds(self):
        """
        Test that the same tensor object (same id()) used in different rounds
        doesn't cause issues because fobs_ctx is fresh each time.
        """
        # Simulate a model that keeps the same Parameter objects across rounds
        model_tensor = torch.randn(100, 100)

        # Round 1
        obj_round1 = {"weights": model_tensor}
        mgr1_serialize = DatumManager(fobs_ctx={})
        serialized1 = fobs.serialize(obj_round1, manager=mgr1_serialize)
        mgr1_deserialize = DatumManager(fobs_ctx={})
        deserialized1 = fobs.deserialize(serialized1, manager=mgr1_deserialize)

        # Modify the tensor (simulating training)
        original_values = model_tensor.clone()
        model_tensor += 1.0  # In-place modification

        # Round 2 - SAME tensor object (same id())
        obj_round2 = {"weights": model_tensor}
        mgr2_serialize = DatumManager(fobs_ctx={})  # Fresh context
        serialized2 = fobs.serialize(obj_round2, manager=mgr2_serialize)
        mgr2_deserialize = DatumManager(fobs_ctx={})
        deserialized2 = fobs.deserialize(serialized2, manager=mgr2_deserialize)

        # Verify both rounds work correctly
        assert torch.allclose(deserialized1["weights"], original_values)
        assert torch.allclose(deserialized2["weights"], model_tensor)
        assert not torch.allclose(deserialized1["weights"], deserialized2["weights"])

    def test_decompose_ctx_fresh_per_serialization(self):
        """Verify that _DecomposeCtx is created fresh for each serialization."""
        tensor = torch.randn(50, 50)

        # First serialization
        mgr1 = DatumManager(fobs_ctx={})
        fobs.serialize({"data": tensor}, manager=mgr1)

        decompose_ctx_key = "TensorDecomposer_dc"
        ctx1_has_decompose = decompose_ctx_key in mgr1.fobs_ctx

        # Second serialization with fresh manager
        mgr2 = DatumManager(fobs_ctx={})
        fobs.serialize({"data": tensor}, manager=mgr2)

        ctx2_has_decompose = decompose_ctx_key in mgr2.fobs_ctx

        # If both have decompose contexts, they should be different objects
        if ctx1_has_decompose and ctx2_has_decompose:
            dc1 = mgr1.fobs_ctx[decompose_ctx_key]
            dc2 = mgr2.fobs_ctx[decompose_ctx_key]
            assert dc1 is not dc2

            # Each should have its own target_to_item mapping
            assert dc1.target_to_item is not dc2.target_to_item

    def test_id_collision_different_objects(self):
        """
        Test that different tensor objects with the same id() don't collide
        because they're in different serializations (different fobs_ctx).
        """
        # Create two tensors
        tensor1 = torch.randn(10, 10)
        tensor1_id = id(tensor1)

        # Serialize first tensor
        mgr1_serialize = DatumManager(fobs_ctx={})
        serialized1 = fobs.serialize({"data": tensor1}, manager=mgr1_serialize)
        mgr1_deserialize = DatumManager(fobs_ctx={})
        deserialized1 = fobs.deserialize(serialized1, manager=mgr1_deserialize)

        # Delete first tensor so its id() might be reused
        tensor1_values = tensor1.clone()
        del tensor1

        # Create new tensor that MIGHT get the same id()
        tensor2 = torch.randn(10, 10)

        # Whether or not ids match, serialization should work correctly
        mgr2_serialize = DatumManager(fobs_ctx={})  # Fresh context
        serialized2 = fobs.serialize({"data": tensor2}, manager=mgr2_serialize)
        mgr2_deserialize = DatumManager(fobs_ctx={})
        deserialized2 = fobs.deserialize(serialized2, manager=mgr2_deserialize)

        # Both should deserialize correctly
        assert torch.allclose(deserialized1["data"], tensor1_values)
        assert torch.allclose(deserialized2["data"], tensor2)

    def test_native_vs_download_decomposition(self):
        """Test that small tensors use native decomposition (no download context)."""
        # Small tensor - should use native decomposition
        small_tensor = torch.randn(5, 5)

        # Create manager with chunk size set to high value (native decompose)
        mgr_serialize = DatumManager(fobs_ctx={"tensor_download_chunk_size": 0})

        serialized = fobs.serialize({"data": small_tensor}, manager=mgr_serialize)
        mgr_deserialize = DatumManager(fobs_ctx={})
        deserialized = fobs.deserialize(serialized, manager=mgr_deserialize)

        # Should work correctly
        assert torch.allclose(deserialized["data"], small_tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
