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

import os
import threading
from types import SimpleNamespace

import psutil
import pytest
import torch
from safetensors.torch import load as load_tensors
from safetensors.torch import save_file

import nvflare.app_opt.pt.tensor_downloader as tensor_downloader
from nvflare.app_opt.pt.lazy_tensor_dict import safetensors_refs
from nvflare.app_opt.pt.tensor_downloader import TensorDownloadable
from nvflare.fuel.f3.streaming.download_service import ProduceRC
from tests.timing_utils import WAIT_TIMEOUT, ThreadGroup


def _open_paths():
    return {file.path for file in psutil.Process().open_files()}


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
    def test_lazy_refs_reuse_reader_without_payload_cache(self, tmp_path, monkeypatch, max_chunk_size):
        tensors = {"a": torch.arange(4.0), "b": torch.ones(2)}
        save_file(tensors, tmp_path / "model.safetensors")
        refs = safetensors_refs(str(tmp_path / "model.safetensors"))
        # FOBS item IDs differ from checkpoint tensor names.
        items_by_id = {f"T{i}": ref for i, ref in enumerate(refs.values())}
        downloadable = TensorDownloadable(tensors=items_by_id, max_chunk_size=max_chunk_size)
        real_open = tensor_downloader.safe_open
        opens = []

        def count_open(*args, **kwargs):
            opens.append(kwargs["backend"])
            return real_open(*args, **kwargs)

        monkeypatch.setattr(tensor_downloader, "safe_open", count_open)

        assert downloadable.cache is None
        first = tensors[next(iter(refs))]
        assert downloadable.get_item_size(0) == first.numel() * first.element_size()
        for client in ("site-1", "site-2"):
            state, restored = {}, {}
            for _ in range(len(tensors) + 1):
                rc, items, state = downloadable.produce(state, client)
                if rc == ProduceRC.EOF:
                    break
                assert rc == ProduceRC.OK
                restored.update({key: value for item in items for key, value in load_tensors(item).items()})
            assert rc == ProduceRC.EOF
            assert set(restored) == set(items_by_id)
            for key, ref in items_by_id.items():
                torch.testing.assert_close(restored[key], tensors[ref.key])
        assert opens == ["pread"]
        assert not downloadable._prefetch_futures
        downloadable.release()

    def test_readers_reused_across_files_until_release(self, tmp_path, monkeypatch):
        refs = {}
        paths = []
        for i in range(2):
            path = str(tmp_path / f"{i}.safetensors")
            paths.append(path)
            save_file({str(i): torch.tensor(i)}, path)
            refs.update(safetensors_refs(path))
        downloadable = TensorDownloadable(refs, max_chunk_size=1)
        real_open = tensor_downloader.safe_open
        opens = []

        def count_open(path, **kwargs):
            opens.append(path)
            return real_open(path, **kwargs)

        monkeypatch.setattr(tensor_downloader, "safe_open", count_open)
        try:
            for _ in range(2):
                for i in range(len(paths)):
                    assert load_tensors(downloadable.produce_item(i))[str(i)].item() == i
            assert opens == paths
            assert _open_paths().intersection(paths) == set(paths)
        finally:
            downloadable.release()
        assert not _open_paths().intersection(paths)

    def test_concurrent_reads_survive_release(self, tmp_path, monkeypatch):
        path = str(tmp_path / "model.safetensors")
        save_file({"a": torch.tensor(1), "b": torch.tensor(2)}, path)
        downloadable = TensorDownloadable(safetensors_refs(path), max_chunk_size=1)
        entered = {key: threading.Event() for key in ("a", "b")}
        finish = threading.Event()
        released = threading.Event()
        real_open = tensor_downloader.safe_open

        def blocking_open(*args, **kwargs):
            reader = real_open(*args, **kwargs)

            def get_tensor(key):
                entered[key].set()
                finish.wait()
                return reader.get_tensor(key)

            return SimpleNamespace(metadata=reader.metadata, get_tensor=get_tensor)

        monkeypatch.setattr(tensor_downloader, "safe_open", blocking_open)
        results = []

        def release():
            downloadable.release()
            released.set()

        with ThreadGroup(finish) as workers:
            for index in range(2):
                workers.start(lambda i=index: results.append(downloadable.produce_item(i)))
            assert all(event.wait(WAIT_TIMEOUT) for event in entered.values())
            workers.start(release)
            assert released.wait(WAIT_TIMEOUT)
            with pytest.raises(RuntimeError, match="released"):
                downloadable.produce_item(0)
            assert path in _open_paths()  # In-flight reads still own the reader.
        restored = {key: value.item() for item in results for key, value in load_tensors(item).items()}
        assert restored == {"a": 1, "b": 2}
        assert path not in _open_paths()

    @pytest.mark.parametrize("during_open", [False, True])
    def test_rejects_replacement_when_initializing_reader(self, tmp_path, monkeypatch, during_open):
        path, replacement = tmp_path / "model.safetensors", tmp_path / "next.safetensors"
        save_file({"w": torch.ones(1)}, path, metadata={"nvflare_model_id": "old"})
        refs = safetensors_refs(str(path))
        save_file({"w": torch.zeros(1)}, replacement, metadata={"nvflare_model_id": "new"})
        downloadable = TensorDownloadable(refs, max_chunk_size=1)
        if during_open:
            real_open = tensor_downloader.safe_open

            def open_then_replace(*args, **kwargs):
                reader = real_open(*args, **kwargs)
                os.replace(replacement, path)
                return reader

            monkeypatch.setattr(tensor_downloader, "safe_open", open_then_replace)
        else:
            os.replace(replacement, path)
        with pytest.raises(RuntimeError, match="model was replaced"):
            downloadable.produce_item(0)
        downloadable.release()
        assert str(path) not in _open_paths()

    def test_lazy_refs_reject_native_serialization(self, tmp_path):
        from nvflare.app_opt.pt.decomposers import TensorDecomposer

        save_file({"w": torch.ones(1)}, tmp_path / "model.safetensors")
        ref = safetensors_refs(str(tmp_path / "model.safetensors"))["w"]
        with pytest.raises(ValueError, match="tensor streaming"):
            TensorDecomposer().native_decompose(ref)
