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

import threading

import pytest
import torch
from safetensors.torch import load as load_tensors
from safetensors.torch import save_file

from nvflare.app_opt.pt.lazy_tensor_dict import _LazyRef, safetensors_refs
from nvflare.app_opt.pt.tensor_downloader import TensorDownloadable
from nvflare.fuel.f3.streaming.download_service import ProduceRC


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

    def test_prefetches_next_tensor(self):
        tensors = {
            "first": torch.tensor([1.0]),
            "second": torch.tensor([2.0]),
        }
        downloadable = TensorDownloadable(tensors=tensors, max_chunk_size=1)

        rc, first_items, state = downloadable.produce({}, "receiver")

        assert rc == ProduceRC.OK
        assert load_tensors(first_items[0])["first"].item() == 1.0
        assert 1 in downloadable._prefetch_futures

        rc, second_items, state = downloadable.produce(state, "receiver")

        assert rc == ProduceRC.OK
        assert load_tensors(second_items[0])["second"].item() == 2.0
        assert not downloadable._prefetch_futures

    def test_prefetch_does_not_queue_two_oversized_tensors(self):
        tensors = {
            "first": torch.tensor([1.0]),
            "second": torch.zeros(1024),
            "third": torch.zeros(1024),
        }
        downloadable = TensorDownloadable(tensors=tensors, max_chunk_size=1)

        downloadable.produce({}, "receiver")

        assert 1 in downloadable._prefetch_futures
        assert 2 not in downloadable._prefetch_futures
        downloadable.release()

    def test_release_disables_prefetch_and_produce(self):
        tensors = {
            "first": torch.tensor([1.0]),
            "second": torch.tensor([2.0]),
        }
        downloadable = TensorDownloadable(tensors=tensors, max_chunk_size=1)

        downloadable.release()

        assert downloadable.base_obj is None
        downloadable.prefetch_item(1)
        assert not downloadable._prefetch_futures
        assert downloadable.get_item_size(0) is None
        with pytest.raises(RuntimeError, match="released"):
            downloadable.produce_item(0)

    @pytest.mark.parametrize("max_chunk_size", [1, 1024])
    def test_lazy_refs_roundtrip_without_prefetch_or_cache(self, tmp_path, max_chunk_size):
        tensors = {"a": torch.arange(4.0), "b": torch.ones(2)}
        save_file(tensors, tmp_path / "model.safetensors")
        downloadable = TensorDownloadable(
            tensors=safetensors_refs(str(tmp_path / "model.safetensors")), max_chunk_size=max_chunk_size
        )

        assert downloadable.cache is None
        assert downloadable.get_item_size(0) == 16
        rc, items, state = downloadable.produce({}, "receiver")

        assert rc == ProduceRC.OK
        assert len(items) == (1 if max_chunk_size == 1 else 2)
        if max_chunk_size == 1:
            rc, remaining, _ = downloadable.produce(state, "receiver")
            assert rc == ProduceRC.OK
            items += remaining
        restored = {key: value for item in items for key, value in load_tensors(item).items()}
        assert set(restored) == set(tensors)
        assert all(torch.equal(restored[key], value) for key, value in tensors.items())
        assert not downloadable._prefetch_futures
        downloadable.release()

    def test_lazy_refs_serve_concurrent_receivers_without_serializing_them(self, tmp_path, monkeypatch):
        barrier = threading.Barrier(2)
        save_file({"only": torch.ones(1)}, tmp_path / "model.safetensors")
        original = _LazyRef.materialize

        def materialize(ref):
            barrier.wait(timeout=2.0)
            return original(ref)

        monkeypatch.setattr(_LazyRef, "materialize", materialize)
        downloadable = TensorDownloadable(safetensors_refs(str(tmp_path / "model.safetensors")), max_chunk_size=1)
        results = []
        threads = [
            threading.Thread(target=lambda client=client: results.append(downloadable.produce({}, client)))
            for client in ("site-1", "site-2")
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3.0)

        assert all(not thread.is_alive() for thread in threads)
        assert len(results) == 2
        assert all(rc == ProduceRC.OK and len(items) == 1 for rc, items, _ in results)
        downloadable.release()

    def test_lazy_refs_reject_native_serialization(self, tmp_path):
        from nvflare.app_opt.pt.decomposers import TensorDecomposer

        save_file({"w": torch.ones(1)}, tmp_path / "model.safetensors")
        ref = safetensors_refs(str(tmp_path / "model.safetensors"))["w"]
        with pytest.raises(ValueError, match="tensor streaming"):
            TensorDecomposer().native_decompose(ref)
