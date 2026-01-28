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
Test to validate the critical difference between += (in-place) and = + (assignment)
in the context of NumPy array downloading with slow clients.

This test demonstrates:
1. SAFE:   model.params["key"] = model.params["key"] + update  (creates new array)
2. UNSAFE: model.params["key"] += update  (in-place modification)

These tests validate that our dict snapshot fix protects against #1 but NOT #2.
"""

import threading
import time
from io import BytesIO

import numpy as np
import pytest

from nvflare.app_common.np.np_downloader import ArrayDownloadable


class TestInPlaceVsAssignment:
    """Test the critical difference between += and = + for NumPy array downloading."""

    def test_assignment_is_safe_with_dict_snapshot(self):
        """
        Test SAFE pattern: model.params["key"] = model.params["key"] + update
        This creates a NEW array, so dict snapshot protects slow clients.
        """
        # Round N model
        model_params = {
            "layer1": np.ones((100, 100), dtype=np.float32),
            "layer2": np.ones((50, 50), dtype=np.float32),
        }

        # Create downloadable (simulates broadcast to all clients)
        downloadable = ArrayDownloadable(model_params, max_chunk_size=1024 * 1024)

        # Verify snapshot happened
        assert downloadable.base_obj is not model_params, "Dict should be snapshotted"

        # Fast clients respond, server updates model for Round N+1
        # SAFE PATTERN: Assignment creates NEW array
        update = np.ones((100, 100), dtype=np.float32) * 10
        model_params["layer1"] = model_params["layer1"] + update  # Creates NEW array!

        # Slow client downloads (still getting Round N model)
        item0 = downloadable.produce_item(0)  # layer1
        stream = BytesIO(item0)
        with np.load(stream, allow_pickle=False) as npz:
            downloaded = npz["layer1"]

        # Should get original value (1.0), not updated value (11.0)
        assert np.allclose(downloaded, np.ones((100, 100), dtype=np.float32))
        print("✓ SAFE: Assignment (= +) protected by dict snapshot")
        print(f"  Downloaded value: {downloaded[0, 0]}")
        print(f"  Current model value: {model_params['layer1'][0, 0]}")

    def test_inplace_is_unsafe_despite_dict_snapshot(self):
        """
        Test UNSAFE pattern: model.params["key"] += update
        This modifies existing array IN-PLACE, dict snapshot does NOT protect!

        This test demonstrates PARTIAL CORRUPTION: one layer modified, another unchanged.
        This is the worst-case scenario - model has mixed Round N and N+1 data.
        """
        # Round N model
        model_params = {
            "layer1": np.ones((100, 100), dtype=np.float32),
            "layer2": np.ones((50, 50), dtype=np.float32),
        }

        # Create downloadable (simulates broadcast to all clients)
        downloadable = ArrayDownloadable(model_params, max_chunk_size=1024 * 1024)

        # Verify snapshot happened
        assert downloadable.base_obj is not model_params, "Dict should be snapshotted"

        # IMPORTANT: Verify that downloadable still references the same ARRAYS
        layer1_ptr = downloadable.base_obj["layer1"].__array_interface__["data"][0]
        layer2_ptr = downloadable.base_obj["layer2"].__array_interface__["data"][0]
        assert layer1_ptr == model_params["layer1"].__array_interface__["data"][0]
        assert layer2_ptr == model_params["layer2"].__array_interface__["data"][0]

        # Fast clients respond, server updates model for Round N+1
        # UNSAFE PATTERN: In-place operation on layer1 ONLY
        update = np.ones((100, 100), dtype=np.float32) * 10
        model_params["layer1"] += update  # Modifies existing array!
        # layer2 is NOT modified

        # Verify layer1 is still same memory (in-place modification)
        assert model_params["layer1"].__array_interface__["data"][0] == layer1_ptr

        # Slow client downloads BOTH layers
        item0 = downloadable.produce_item(0)  # layer1
        item1 = downloadable.produce_item(1)  # layer2

        # Load layer1
        stream0 = BytesIO(item0)
        with np.load(stream0, allow_pickle=False) as npz:
            downloaded_layer1 = npz["layer1"]

        # Load layer2
        stream1 = BytesIO(item1)
        with np.load(stream1, allow_pickle=False) as npz:
            downloaded_layer2 = npz["layer2"]

        # PARTIAL CORRUPTION DETECTED:
        # - layer1: Gets MODIFIED value (11.0) - CORRUPTED!
        # - layer2: Gets original value (1.0) - OK
        # Result: Model has MIX of Round N and N+1 data!
        assert np.allclose(
            downloaded_layer1, np.ones((100, 100), dtype=np.float32) * 11
        ), "layer1 should be corrupted (modified)"
        assert np.allclose(
            downloaded_layer2, np.ones((50, 50), dtype=np.float32)
        ), "layer2 should be unchanged (original)"

        print("✗ UNSAFE: In-place (+=) causes PARTIAL CORRUPTION")
        print(f"  layer1 (modified): {downloaded_layer1[0, 0]} (expected 1.0, got 11.0) ← CORRUPTED!")
        print(f"  layer2 (unchanged): {downloaded_layer2[0, 0]} (expected 1.0, got 1.0) ← OK")
        print("  → Model is PARTIALLY CORRUPTED (mix of Round N and N+1)!")

    def test_partial_corruption_scenario_comparison(self):
        """
        Compare SAFE vs UNSAFE patterns when updating only ONE layer.
        This clearly demonstrates the partial corruption problem.
        """
        print("\n" + "=" * 70)
        print("SCENARIO: Server updates layer1, but slow client still downloading")
        print("=" * 70)

        # --- SAFE Pattern: Assignment (= +) ---
        print("\n✓ SAFE PATTERN: layer1 = layer1 + update")
        model_params_safe = {
            "layer1": np.ones((100, 100), dtype=np.float32),
            "layer2": np.ones((50, 50), dtype=np.float32),
        }
        downloadable_safe = ArrayDownloadable(model_params_safe, max_chunk_size=1024 * 1024)

        # Server updates layer1 with SAFE pattern
        model_params_safe["layer1"] = model_params_safe["layer1"] + np.ones((100, 100), dtype=np.float32) * 10

        # Slow client downloads
        safe_layer1 = np.load(BytesIO(downloadable_safe.produce_item(0)), allow_pickle=False)["layer1"]
        safe_layer2 = np.load(BytesIO(downloadable_safe.produce_item(1)), allow_pickle=False)["layer2"]

        print(f"  Slow client gets layer1: {safe_layer1[0, 0]} (expected: 1.0)")
        print(f"  Slow client gets layer2: {safe_layer2[0, 0]} (expected: 1.0)")
        print("  ✓ Model is CONSISTENT (both from Round N)")

        assert np.allclose(safe_layer1, np.ones((100, 100), dtype=np.float32))
        assert np.allclose(safe_layer2, np.ones((50, 50), dtype=np.float32))

        # --- UNSAFE Pattern: In-place (+=) ---
        print("\n✗ UNSAFE PATTERN: layer1 += update")
        model_params_unsafe = {
            "layer1": np.ones((100, 100), dtype=np.float32),
            "layer2": np.ones((50, 50), dtype=np.float32),
        }
        downloadable_unsafe = ArrayDownloadable(model_params_unsafe, max_chunk_size=1024 * 1024)

        # Server updates layer1 with UNSAFE pattern
        model_params_unsafe["layer1"] += np.ones((100, 100), dtype=np.float32) * 10

        # Slow client downloads
        unsafe_layer1 = np.load(BytesIO(downloadable_unsafe.produce_item(0)), allow_pickle=False)["layer1"]
        unsafe_layer2 = np.load(BytesIO(downloadable_unsafe.produce_item(1)), allow_pickle=False)["layer2"]

        print(f"  Slow client gets layer1: {unsafe_layer1[0, 0]} (expected: 1.0, got: 11.0) ← CORRUPTED!")
        print(f"  Slow client gets layer2: {unsafe_layer2[0, 0]} (expected: 1.0)")
        print("  ✗ Model is INCONSISTENT (layer1 from Round N+1, layer2 from Round N)")

        assert np.allclose(unsafe_layer1, np.ones((100, 100), dtype=np.float32) * 11)  # Corrupted!
        assert np.allclose(unsafe_layer2, np.ones((50, 50), dtype=np.float32))  # Original

        print("\n" + "=" * 70)
        print("CONCLUSION:")
        print("  SAFE (=):  Both layers consistent (Round N)")
        print("  UNSAFE (+=): Partial corruption (mixed Round N and N+1)")
        print("=" * 70)

    def test_concurrent_update_safe_pattern(self):
        """
        Test that assignment pattern (= +) is safe even with concurrent downloads.
        Simulates slow client downloading while server updates model.
        """
        # Round N model
        model_params = {
            "weights": np.ones((1000, 1000), dtype=np.float32),
        }

        downloadable = ArrayDownloadable(model_params, max_chunk_size=1024 * 1024)

        results = {"downloaded": None, "update_done": False, "error": None}

        def slow_client_download():
            """Simulates slow client downloading."""
            try:
                time.sleep(0.01)  # Slow client delay
                results["downloaded"] = downloadable.produce_item(0)
            except Exception as e:
                results["error"] = str(e)

        def server_update_safe():
            """Server updates model with SAFE pattern."""
            time.sleep(0.005)  # Let download start
            # SAFE: Creates new array
            model_params["weights"] = model_params["weights"] + np.ones((1000, 1000), dtype=np.float32) * 10
            results["update_done"] = True

        # Start concurrent operations
        download_thread = threading.Thread(target=slow_client_download)
        update_thread = threading.Thread(target=server_update_safe)

        download_thread.start()
        update_thread.start()

        download_thread.join(timeout=5)
        update_thread.join(timeout=5)

        # Verify no errors
        assert results["error"] is None
        assert results["downloaded"] is not None
        assert results["update_done"]

        # Verify downloaded data is original (1.0), not updated (11.0)
        stream = BytesIO(results["downloaded"])
        with np.load(stream, allow_pickle=False) as npz:
            downloaded = npz["weights"]
        assert np.allclose(downloaded, np.ones((1000, 1000), dtype=np.float32))
        print("✓ SAFE: Concurrent assignment (= +) works correctly")

    def test_concurrent_update_unsafe_pattern(self):
        """
        Test that in-place pattern (+=) shares the same memory reference.
        This demonstrates WHY it's unsafe even if we can't reliably trigger corruption in tests.
        """
        # Round N model
        model_params = {
            "weights": np.ones((1000, 1000), dtype=np.float32),
        }

        downloadable = ArrayDownloadable(model_params, max_chunk_size=1024 * 1024)

        # Verify downloadable references the same array (not a copy)
        original_ptr = downloadable.base_obj["weights"].__array_interface__["data"][0]
        model_ptr = model_params["weights"].__array_interface__["data"][0]
        assert original_ptr == model_ptr, "Array should be referenced, not copied"

        # UNSAFE: In-place modification
        model_params["weights"] += np.ones((1000, 1000), dtype=np.float32) * 10

        # Verify it's still the same memory (in-place)
        new_ptr = model_params["weights"].__array_interface__["data"][0]
        assert new_ptr == original_ptr, "Array modified in-place (same memory)"

        # The downloadable's array is also modified (same reference)
        assert downloadable.base_obj["weights"][0, 0] == 11.0, "Downloadable sees modified data"

        print("✗ UNSAFE: In-place (+=) modifies shared memory")
        print(f"  Downloadable array value: {downloadable.base_obj['weights'][0, 0]}")
        print(f"  Model array value: {model_params['weights'][0, 0]}")
        print(f"  Both point to same memory: {original_ptr == new_ptr}")

    def test_compare_array_identity_safe_vs_unsafe(self):
        """
        Compare array identity to demonstrate the difference between safe and unsafe patterns.
        """
        # Test 1: Assignment (SAFE)
        print("\n--- SAFE Pattern: a = a + b ---")
        model_params = {"layer": np.ones((10, 10), dtype=np.float32)}
        downloadable = ArrayDownloadable(model_params, max_chunk_size=1024 * 1024)

        original_array = downloadable.base_obj["layer"]
        original_id = id(original_array)
        original_ptr = original_array.__array_interface__["data"][0]

        print(f"Before update: id={original_id}, data_ptr={original_ptr}")

        # SAFE: Assignment
        model_params["layer"] = model_params["layer"] + np.ones((10, 10), dtype=np.float32) * 10

        new_id = id(model_params["layer"])
        new_ptr = model_params["layer"].__array_interface__["data"][0]

        print(f"After update:  id={new_id}, data_ptr={new_ptr}")
        print(f"ID changed: {new_id != original_id}")
        print(f"Data pointer changed: {new_ptr != original_ptr}")

        # Downloadable still references original
        assert id(downloadable.base_obj["layer"]) == original_id
        assert downloadable.base_obj["layer"].__array_interface__["data"][0] == original_ptr
        print("✓ Downloadable still has original array")

        # Test 2: In-place (UNSAFE)
        print("\n--- UNSAFE Pattern: a += b ---")
        model_params = {"layer": np.ones((10, 10), dtype=np.float32)}
        downloadable = ArrayDownloadable(model_params, max_chunk_size=1024 * 1024)

        original_array = downloadable.base_obj["layer"]
        original_id = id(original_array)
        original_ptr = original_array.__array_interface__["data"][0]

        print(f"Before update: id={original_id}, data_ptr={original_ptr}")

        # UNSAFE: In-place
        model_params["layer"] += np.ones((10, 10), dtype=np.float32) * 10

        new_id = id(model_params["layer"])
        new_ptr = model_params["layer"].__array_interface__["data"][0]

        print(f"After update:  id={new_id}, data_ptr={new_ptr}")
        print(f"ID changed: {new_id != original_id}")
        print(f"Data pointer changed: {new_ptr != original_ptr}")

        # Downloadable references the SAME array (now modified!)
        assert id(downloadable.base_obj["layer"]) == original_id
        assert downloadable.base_obj["layer"].__array_interface__["data"][0] == original_ptr
        print("✗ Downloadable has MODIFIED array (corrupted!)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
