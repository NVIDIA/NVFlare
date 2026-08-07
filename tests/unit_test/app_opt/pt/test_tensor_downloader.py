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
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor

import pytest
import torch
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from nvflare.app_opt.pt import tensor_downloader as tensor_downloader_module
from nvflare.app_opt.pt.tensor_downloader import TensorDownloadable
from nvflare.fuel.f3.streaming import cacheable as cacheable_module
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

    def test_prefetch_future_is_shared_by_concurrent_demand(self, monkeypatch):
        tensors = {"first": torch.tensor([1.0])}
        downloadable = TensorDownloadable(tensors=tensors, max_chunk_size=1)
        result = save_tensors({"first": tensors["first"]})
        entered = threading.Event()
        allow_result = threading.Event()

        class BlockingFuture:
            def __init__(self):
                self.result_calls = 0

            def result(self):
                self.result_calls += 1
                entered.set()
                assert allow_result.wait(timeout=5.0)
                return result

        class DemandFuture(Future):
            def __init__(self):
                super().__init__()
                self.waiter_count = 0
                self.waiter_lock = threading.Lock()
                self.two_waiters = threading.Event()

            def result(self, timeout=None):
                with self.waiter_lock:
                    self.waiter_count += 1
                    if self.waiter_count >= 2:
                        self.two_waiters.set()
                return super().result(timeout=timeout)

        future = BlockingFuture()
        demand_future = DemandFuture()
        monkeypatch.setattr(cacheable_module, "Future", lambda: demand_future)
        with downloadable._prefetch_lock:
            downloadable._prefetch_futures[0] = future

        with ThreadPoolExecutor(max_workers=3) as executor:
            produced = [executor.submit(downloadable._get_item, 0, f"receiver{i}") for i in range(3)]
            assert entered.wait(timeout=5.0)
            assert demand_future.two_waiters.wait(timeout=5.0)
            with downloadable._prefetch_lock:
                assert downloadable._prefetch_futures[0] is future
            assert future.result_calls == 1
            allow_result.set()

        assert [item.result() for item in produced] == [result] * 3
        assert not downloadable._prefetch_futures
        assert not downloadable._production_futures
        assert downloadable.cache[0][0] == result

    def test_prefetch_does_not_duplicate_in_flight_demand(self, monkeypatch):
        tensors = {"first": torch.tensor([1.0])}
        downloadable = TensorDownloadable(tensors=tensors, max_chunk_size=1)
        save_started = threading.Event()
        allow_save = threading.Event()
        real_save_tensors = tensor_downloader_module.save_tensors

        def blocking_save_tensors(items):
            save_started.set()
            assert allow_save.wait(timeout=5.0)
            return real_save_tensors(items)

        class RecordingPool:
            def __init__(self):
                self.submit_calls = 0

            def submit(self, *args, **kwargs):
                self.submit_calls += 1
                return None

        pool = RecordingPool()
        monkeypatch.setattr(tensor_downloader_module, "save_tensors", blocking_save_tensors)
        monkeypatch.setattr(tensor_downloader_module, "stream_thread_pool", pool)

        with ThreadPoolExecutor(max_workers=1) as executor:
            produced = executor.submit(downloadable._get_item, 0, "receiver")
            assert save_started.wait(timeout=5.0)
            downloadable.prefetch_item(0)
            assert pool.submit_calls == 0
            allow_save.set()

        assert load_tensors(produced.result())["first"].item() == 1.0
        assert not downloadable._production_futures

    def test_failed_prefetch_is_removed_for_retry(self):
        tensors = {"first": torch.tensor([1.0])}
        downloadable = TensorDownloadable(tensors=tensors, max_chunk_size=1)
        future = Future()
        future.set_exception(RuntimeError("prefetch failed"))
        with downloadable._prefetch_lock:
            downloadable._prefetch_futures[0] = future

        with pytest.raises(RuntimeError, match="prefetch failed"):
            downloadable.produce_item(0)

        assert not downloadable._prefetch_futures
        assert load_tensors(downloadable.produce_item(0))["first"].item() == 1.0

    def test_release_cancels_prefetch_and_wakes_demand_waiters(self, monkeypatch):
        class ObservedFuture(Future):
            def __init__(self, expected_waiters):
                super().__init__()
                self.expected_waiters = expected_waiters
                self.waiter_count = 0
                self.waiter_lock = threading.Lock()
                self.waiters_ready = threading.Event()

            def result(self, timeout=None):
                with self.waiter_lock:
                    self.waiter_count += 1
                    if self.waiter_count >= self.expected_waiters:
                        self.waiters_ready.set()
                return super().result(timeout=timeout)

        generic_future = ObservedFuture(expected_waiters=2)
        monkeypatch.setattr(cacheable_module, "Future", lambda: generic_future)
        tensors = {"first": torch.tensor([1.0])}
        downloadable = TensorDownloadable(tensors=tensors, max_chunk_size=1)
        prefetch_future = ObservedFuture(expected_waiters=1)
        with downloadable._prefetch_lock:
            downloadable._prefetch_futures[0] = prefetch_future

        with ThreadPoolExecutor(max_workers=3) as executor:
            results = [executor.submit(downloadable._get_item, 0, f"receiver{i}") for i in range(3)]
            assert prefetch_future.waiters_ready.wait(timeout=5.0)
            assert generic_future.waiters_ready.wait(timeout=5.0)
            downloadable.release()

        for result in results:
            with pytest.raises(CancelledError):
                result.result()
        assert not downloadable._prefetch_futures
        assert not downloadable._production_futures

    def test_demand_cannot_start_while_release_is_cancelling_prefetch(self, monkeypatch):
        class BlockingCancelFuture(Future):
            def __init__(self):
                super().__init__()
                self.cancel_entered = threading.Event()
                self.allow_cancel = threading.Event()

            def cancel(self):
                self.cancel_entered.set()
                assert self.allow_cancel.wait(timeout=5.0)
                return super().cancel()

        save_calls = 0

        def counted_save_tensors(items):
            nonlocal save_calls
            save_calls += 1
            return save_tensors(items)

        monkeypatch.setattr(tensor_downloader_module, "save_tensors", counted_save_tensors)
        tensors = {"first": torch.tensor([1.0])}
        downloadable = TensorDownloadable(tensors=tensors, max_chunk_size=1)
        prefetch_future = BlockingCancelFuture()
        with downloadable._prefetch_lock:
            downloadable._prefetch_futures[0] = prefetch_future

        with ThreadPoolExecutor(max_workers=1) as executor:
            releasing = executor.submit(downloadable.release)
            assert prefetch_future.cancel_entered.wait(timeout=5.0)
            try:
                with pytest.raises(RuntimeError, match="released"):
                    downloadable._get_item(0, "receiver")
            finally:
                prefetch_future.allow_cancel.set()
            releasing.result()

        assert save_calls == 0
        assert downloadable.base_obj is None
        assert not downloadable._prefetch_futures
        assert not downloadable._production_futures

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
