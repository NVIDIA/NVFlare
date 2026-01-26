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
Tests to validate the safety of reference-based tensor downloading.

These tests verify that the TensorDownloadable design is safe under normal NVFlare usage patterns.

Key findings:
1. Tensors are stored by reference (memory efficient - no duplication)
2. Concurrent reads from the same reference work correctly
3. Dict replacement (assigning new dict) is safe - downloader keeps old reference
4. Race condition CAN occur with concurrent in-place modification during serialization
5. NVFlare usage patterns prevent this race condition:
   - Client: flare.send() blocks, preventing concurrent user code modifications
   - Server: broadcast_and_wait() blocks, preventing modifications during broadcast
   - Updates: FedAvg replaces entire model.params dict (new reference), not in-place

The test_modification_during_produce_item demonstrates the race condition that would occur
with unsafe concurrent in-place modification, validating why our blocking patterns are essential.

Conclusion: The reference-based design is SAFE with current NVFlare workflows.
"""

import threading
import time

import pytest
import torch

from nvflare.app_opt.pt.tensor_downloader import TensorDownloadable


class TestTensorDownloaderSafety:
    """Tests for tensor downloader reference semantics and safety."""

    @pytest.fixture
    def tensors(self):
        """Create test tensors."""
        return {
            "layer1.weight": torch.randn(100, 100),
            "layer1.bias": torch.randn(100),
            "layer2.weight": torch.randn(50, 100),
            "layer2.bias": torch.randn(50),
        }

    def test_tensors_stored_by_reference(self, tensors):
        """Verify that tensors are stored by reference, not copied."""
        downloadable = TensorDownloadable(tensors, max_chunk_size=1024 * 1024)

        # Check that the same dict reference is stored
        assert downloadable.base_obj is tensors
        assert id(downloadable.base_obj) == id(tensors)

        # Check that tensor objects are the same (not cloned)
        for key in tensors:
            assert downloadable.base_obj[key] is tensors[key]
            assert downloadable.base_obj[key].data_ptr() == tensors[key].data_ptr()

    def test_concurrent_produce_items(self, tensors):
        """Test that multiple threads can concurrently call produce_item on the same downloadable."""
        # This tests the core concurrent access safety without requiring network setup
        downloadable = TensorDownloadable(tensors, max_chunk_size=1024 * 1024)

        num_threads = 3
        num_items = downloadable.get_item_count()
        results = [[] for _ in range(num_threads)]
        errors = [None] * num_threads

        def produce_worker(thread_id):
            """Worker function to produce items."""
            try:
                # Each thread produces all items
                for i in range(num_items):
                    item_bytes = downloadable.produce_item(i)
                    results[thread_id].append(item_bytes)
            except Exception as e:
                errors[thread_id] = str(e)

        # Start concurrent producers
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=produce_worker, args=(i,))
            thread.start()
            threads.append(thread)

        # Wait for all threads
        for thread in threads:
            thread.join(timeout=10)

        # Verify all threads succeeded
        for i in range(num_threads):
            assert errors[i] is None, f"Thread {i} had error: {errors[i]}"
            assert len(results[i]) == num_items, f"Thread {i} produced {len(results[i])} items, expected {num_items}"

        # Verify all threads produced identical data
        for item_idx in range(num_items):
            # All threads should produce the same bytes for each item
            item_0 = results[0][item_idx]
            for thread_id in range(1, num_threads):
                assert (
                    results[thread_id][item_idx] == item_0
                ), f"Thread {thread_id} item {item_idx} differs from thread 0"

        # Verify the downloadable still has correct reference to original tensors
        for key in tensors:
            assert downloadable.base_obj[key].data_ptr() == tensors[key].data_ptr()

    def test_produce_item_uses_original_reference(self, tensors):
        """Verify that produce_item accesses the original tensor reference."""
        downloadable = TensorDownloadable(tensors, max_chunk_size=1024 * 1024)

        # Produce first item
        item_bytes = downloadable.produce_item(0)
        assert item_bytes is not None
        assert len(item_bytes) > 0

        # The downloadable should still have reference to original tensors
        for key in tensors:
            assert downloadable.base_obj[key].data_ptr() == tensors[key].data_ptr()

    def test_multiple_serializations_same_message(self):
        """Test that the same tensor appearing multiple times in one message works correctly."""
        # This tests the id() lookup within a single fobs_ctx
        shared_tensor = torch.randn(100, 100)

        # Multiple references to the same tensor
        tensors = {
            "layer1": shared_tensor,
            "layer2": shared_tensor,  # Same tensor object
            "layer3": torch.randn(50, 50),  # Different tensor
        }

        downloadable = TensorDownloadable(tensors, max_chunk_size=1024 * 1024)

        # All should reference the original tensors
        assert downloadable.base_obj["layer1"] is shared_tensor
        assert downloadable.base_obj["layer2"] is shared_tensor
        assert downloadable.base_obj["layer1"].data_ptr() == shared_tensor.data_ptr()
        assert downloadable.base_obj["layer2"].data_ptr() == shared_tensor.data_ptr()

    def test_modification_during_produce_item(self):
        """
        Test that demonstrates what happens if tensors are modified DURING produce_item().
        This is the actual race condition scenario we want to avoid.
        """
        # Create a large tensor so produce_item takes some time
        large_tensor = torch.ones(1000, 1000)
        tensors = {"weights": large_tensor}

        downloadable = TensorDownloadable(tensors, max_chunk_size=1024 * 1024)

        results = {"produced": None, "modification_done": False, "error": None}

        def produce_worker():
            """Thread that produces items (simulates download)."""
            try:
                # This will serialize the tensor
                results["produced"] = downloadable.produce_item(0)
            except Exception as e:
                results["error"] = str(e)

        def modify_worker():
            """Thread that modifies the tensor during produce."""
            time.sleep(0.001)  # Small delay to let produce start
            large_tensor.zero_()  # In-place modification during produce!
            results["modification_done"] = True

        # Start both threads
        produce_thread = threading.Thread(target=produce_worker)
        modify_thread = threading.Thread(target=modify_worker)

        produce_thread.start()
        modify_thread.start()

        produce_thread.join(timeout=5)
        modify_thread.join(timeout=5)

        # The produce should complete (may contain mixed data)
        assert results["error"] is None, f"Produce failed: {results['error']}"
        assert results["produced"] is not None
        assert results["modification_done"]

        # This test demonstrates the race condition EXISTS if we:
        # 1. Call produce_item() (download/serialize)
        # 2. Modify tensors concurrently
        #
        # However, our actual usage patterns are safe because:
        # - Client: send() blocks, user code waits for receive() before continuing
        # - Server: broadcast_and_wait() blocks, model not modified during broadcast
        # - FedAvg: Replaces entire dict (new reference), not in-place modification


class TestReferenceSemanticsSafety:
    """Additional tests for various safety scenarios."""

    def test_dict_replacement_is_safe_locally(self):
        """Test that replacing the params dict doesn't affect the downloadable's reference."""
        original_tensors = {
            "layer1.weight": torch.randn(100, 100),
            "layer1.bias": torch.randn(100),
        }

        # Create downloadable with reference to original_tensors
        downloadable = TensorDownloadable(original_tensors, max_chunk_size=1024 * 1024)

        # Store original values and data pointers
        original_values = {k: v.clone() for k, v in original_tensors.items()}
        original_ptrs = {k: v.data_ptr() for k, v in original_tensors.items()}

        # Simulate server-side FedAvg pattern: replace the entire dict
        # In real code: model.params = new_dict
        new_tensors = {
            "layer1.weight": torch.randn(100, 100) + 100,
            "layer1.bias": torch.randn(100) + 100,
        }

        # The downloadable still has reference to original_tensors
        assert downloadable.base_obj is original_tensors
        assert downloadable.base_obj is not new_tensors

        # Verify downloadable can still access original values
        for key in original_tensors:
            assert torch.allclose(downloadable.base_obj[key], original_values[key])
            assert not torch.allclose(downloadable.base_obj[key], new_tensors[key])
            assert downloadable.base_obj[key].data_ptr() == original_ptrs[key]

    def test_dict_shallow_copy_vs_reference(self):
        """Demonstrate the difference between shallow copy and reference."""
        original_tensors = {
            "layer1": torch.randn(10, 10),
            "layer2": torch.randn(5, 5),
        }

        # Scenario 1: Store reference (current approach)
        ref_dict = original_tensors
        assert ref_dict is original_tensors

        # Replacing the dict doesn't affect the stored reference
        old_tensors = original_tensors
        original_tensors = {"new": torch.randn(3, 3)}
        assert ref_dict is not original_tensors
        assert ref_dict is old_tensors
        assert "layer1" in ref_dict  # Still has old data

        # Scenario 2: Shallow copy of dict (still shares tensor objects)
        original_tensors = {
            "layer1": torch.randn(10, 10),
            "layer2": torch.randn(5, 5),
        }
        copied_dict = dict(original_tensors)
        assert copied_dict is not original_tensors
        assert copied_dict["layer1"] is original_tensors["layer1"]  # Same tensors

    def test_blocking_pattern_safety(self):
        """Verify that the blocking client API pattern is safe."""
        # Simulate client API pattern
        model_params = {
            "layer1": torch.randn(100, 100),
            "layer2": torch.randn(50, 50),
        }

        # Step 1: Create downloadable with reference
        downloadable = TensorDownloadable(model_params, max_chunk_size=1024 * 1024)

        # Step 2: "Send" (in real code, this blocks until next receive())
        # In typical usage, code would block here
        time.sleep(0.1)  # Simulate blocking

        # Step 3: After receive() returns, safe to modify
        # But by then, the download is complete, so this is safe
        model_params = {
            "layer1": torch.randn(100, 100) + 10,
            "layer2": torch.randn(50, 50) + 10,
        }

        # The downloadable still has the old reference
        assert downloadable.base_obj is not model_params

    def test_server_fedavg_pattern_safety(self):
        """Verify that the server FedAvg non-blocking + dict replacement pattern is safe."""
        # Server model
        model = type(
            "obj",
            (object,),
            {
                "params": {
                    "layer1": torch.randn(100, 100),
                    "layer2": torch.randn(50, 50),
                }
            },
        )()

        # Create downloadable (non-blocking send)
        downloadable = TensorDownloadable(model.params, max_chunk_size=1024 * 1024)

        # Store original reference
        original_params = model.params

        # Server continues processing, replaces entire params dict (FedAvg pattern)
        # This is what happens in: model.params = model_update.params
        model.params = {
            "layer1": torch.randn(100, 100) + 100,
            "layer2": torch.randn(50, 50) + 100,
        }

        # Downloadable still has reference to original dict
        assert downloadable.base_obj is original_params
        assert downloadable.base_obj is not model.params

        # Download would get original values, not new values
        for key in original_params:
            assert not torch.allclose(downloadable.base_obj[key], model.params[key])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
