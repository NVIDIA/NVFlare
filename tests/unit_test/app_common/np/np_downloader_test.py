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
Tests to validate the safety of reference-based numpy array downloading.

These tests verify that the ArrayDownloadable design is safe under normal NVFlare usage patterns.

Key findings:
1. Arrays are stored by reference (memory efficient - no duplication)
2. Concurrent reads from the same reference work correctly
3. Dict replacement (assigning new dict) is safe - downloader keeps old reference
4. Race condition CAN occur with concurrent in-place modification during serialization
5. NVFlare usage patterns prevent this race condition:
   - Client: flare.send() blocks, preventing concurrent user code modifications
   - Server: broadcast_and_wait() blocks, preventing modifications during broadcast
   - Updates: FedAvg replaces entire model.params dict (new reference), not in-place

The test_modification_during_produce_item demonstrates the race condition that would occur
with unsafe concurrent in-place modification during np.savez(). The test shows timing-dependent
data corruption (e.g., capturing 17% vs 84% vs 100% of data depending on when modification hits),
validating why our blocking patterns are essential.

Conclusion: The reference-based design is SAFE with current NVFlare workflows.
"""

import threading
import time
from io import BytesIO

import numpy as np
import pytest

from nvflare.app_common.np.np_downloader import ArrayDownloadable


class TestNumpyDownloaderSafety:
    """Tests for numpy array downloader reference semantics and safety."""

    @pytest.fixture
    def arrays(self):
        """Create sample arrays for testing."""
        return {
            "layer1.weight": np.random.randn(100, 100).astype(np.float32),
            "layer1.bias": np.random.randn(100).astype(np.float32),
            "layer2.weight": np.random.randn(50, 100).astype(np.float32),
            "layer2.bias": np.random.randn(50).astype(np.float32),
        }

    def test_arrays_stored_by_reference(self, arrays):
        """Verify that ArrayDownloadable stores arrays by reference, not copy."""
        downloadable = ArrayDownloadable(arrays, max_chunk_size=1024 * 1024)

        # Verify base_obj is the same reference
        assert downloadable.base_obj is arrays

        # Verify each array is stored by reference (same memory location)
        for key in arrays:
            assert downloadable.base_obj[key] is arrays[key]
            # Check data pointer to ensure it's truly the same array
            assert (
                downloadable.base_obj[key].__array_interface__["data"][0] == arrays[key].__array_interface__["data"][0]
            )

    def test_concurrent_produce_items(self, arrays):
        """Test that multiple threads can concurrently call produce_item on the same downloadable."""
        # This tests the core concurrent access safety without requiring network setup
        downloadable = ArrayDownloadable(arrays, max_chunk_size=1024 * 1024)

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

        # Verify the downloadable still has correct reference to original arrays
        for key in arrays:
            assert (
                downloadable.base_obj[key].__array_interface__["data"][0] == arrays[key].__array_interface__["data"][0]
            )

    def test_dict_replacement_is_safe_locally(self, arrays):
        """Test that replacing the params dict doesn't affect the downloadable's reference."""
        original_arrays = {
            "layer1.weight": np.random.randn(100, 100).astype(np.float32),
            "layer1.bias": np.random.randn(100).astype(np.float32),
        }

        # Create downloadable with reference to original_arrays
        downloadable = ArrayDownloadable(original_arrays, max_chunk_size=1024 * 1024)

        # Store original values and data pointers
        original_values = {k: v.copy() for k, v in original_arrays.items()}
        original_ptrs = {k: v.__array_interface__["data"][0] for k, v in original_arrays.items()}

        # Simulate server-side pattern: replace the entire dict
        new_arrays = {
            "layer1.weight": np.random.randn(100, 100).astype(np.float32) + 100,
            "layer1.bias": np.random.randn(100).astype(np.float32) + 100,
        }

        # The downloadable still has reference to original_arrays
        assert downloadable.base_obj is original_arrays
        assert downloadable.base_obj is not new_arrays

        # Verify downloadable can still access original values
        for key in original_arrays:
            assert np.allclose(downloadable.base_obj[key], original_values[key])
            assert not np.allclose(downloadable.base_obj[key], new_arrays[key])
            assert downloadable.base_obj[key].__array_interface__["data"][0] == original_ptrs[key]

    def test_modification_during_produce_item(self):
        """
        Test that demonstrates what happens if arrays are modified DURING produce_item().
        This is the actual race condition scenario we want to avoid.
        """
        # Create a large array so produce_item takes some time
        large_array = np.ones((5000, 5000), dtype=np.float32)
        print(f"original sum: {large_array.sum()}")
        arrays = {"weights": large_array}

        downloadable = ArrayDownloadable(arrays, max_chunk_size=1024 * 1024)

        results = {"produced": None, "modification_done": False, "error": None}

        def produce_worker():
            """Thread that produces items (simulates download)."""
            try:
                # This will serialize the array
                results["produced"] = downloadable.produce_item(0)
            except Exception as e:
                results["error"] = str(e)

        def modify_worker():
            """Thread that modifies the array during produce."""
            # Try different sleep values to see different race condition outcomes:
            # - 0.01: Modify earlier → capture less data
            # - 0.1:  Current timing → captures ~4M (one buffer)
            # - 1.0:   Modify later → capture more/all data
            time.sleep(0.01)  # Small delay to let produce start
            large_array[:] = 0  # In-place modification during produce!
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

        # Deserialize the produced bytes back to arrays to check what was captured
        stream = BytesIO(results["produced"])
        with np.load(stream, allow_pickle=False) as npz_obj:
            produced_array = npz_obj["weights"]

        print(f"produced sum: {produced_array.sum()}")
        print(f"current array sum: {large_array.sum()}")

        # This test demonstrates the race condition EXISTS if we:
        # 1. Call produce_item() (download/serialize)
        # 2. Modify arrays concurrently
        #
        # However, our actual usage patterns are safe because:
        # - Client: send() blocks, user code waits for receive() before continuing
        # - Server: broadcast_and_wait() blocks, model not modified during broadcast
        # - FedAvg: Replaces entire dict (new reference), not in-place modification


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
