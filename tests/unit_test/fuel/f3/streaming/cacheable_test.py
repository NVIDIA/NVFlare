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

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, List

import pytest

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.streaming import cacheable as cacheable_module
from nvflare.fuel.f3.streaming.cacheable import CacheableObject, ItemConsumer
from nvflare.fuel.f3.streaming.download_service import DownloadService, ProduceRC
from nvflare.fuel.utils.network_utils import get_open_ports


class MockCacheableObject(CacheableObject):
    """Mock cacheable object for testing."""

    def __init__(self, items: list, max_chunk_size: int):
        self.items = items
        super().__init__(items, max_chunk_size)

    def get_item_count(self) -> int:
        return len(self.items)

    def produce_item(self, index: int) -> bytes:
        return self.items[index].encode() if isinstance(self.items[index], str) else self.items[index]


class BlockingCacheableObject(MockCacheableObject):
    """Cacheable object that exposes concurrent item production to tests."""

    def __init__(self, items: list, max_chunk_size: int):
        self.produce_started = threading.Event()
        self.two_productions_started = threading.Event()
        self.allow_produce = threading.Event()
        self._produce_calls_lock = threading.Lock()
        self.produce_calls = 0
        self.fail_next_production = False
        super().__init__(items, max_chunk_size)

    def produce_item(self, index: int) -> bytes:
        with self._produce_calls_lock:
            self.produce_calls += 1
            if self.produce_calls >= 2:
                self.two_productions_started.set()
        self.produce_started.set()
        assert self.allow_produce.wait(timeout=5.0)
        if self.fail_next_production:
            self.fail_next_production = False
            raise RuntimeError("production failed")
        return super().produce_item(index)


class ObservedFuture(Future):
    """Future that signals after two receivers are waiting for its result."""

    def __init__(self):
        super().__init__()
        self._waiter_lock = threading.Lock()
        self._waiter_count = 0
        self.two_waiters = threading.Event()

    def result(self, timeout=None):
        with self._waiter_lock:
            self._waiter_count += 1
            if self._waiter_count >= 2:
                self.two_waiters.set()
        return super().result(timeout=timeout)


class MockItemConsumer(ItemConsumer):
    """Mock item consumer for testing."""

    def __init__(self):
        super().__init__()
        self.consumed_items = []

    def consume_items(self, items: List[Any], result: Any) -> Any:
        if result is None:
            result = []
        result.extend(items)
        self.consumed_items.extend(items)
        return result


class TestCacheableObject:
    """Test suite for CacheableObject."""

    @pytest.fixture
    def port(self):
        return get_open_ports(1)[0]

    @pytest.fixture
    def cell(self, port, request):
        """Create a unique cell for each test."""
        test_name = request.node.name
        cell_name = f"test_cell_{test_name}_{port}"
        listening_url = f"tcp://localhost:{port}"
        cell = CoreCell(cell_name, listening_url, secure=False, credentials={})
        cell.start()
        yield cell
        cell.stop()

    def test_cacheable_object_initialization(self):
        """Test CacheableObject initialization."""
        items = ["item1", "item2", "item3"]
        obj = MockCacheableObject(items, max_chunk_size=100)

        assert obj.size == 3
        assert len(obj.cache) == 3
        assert obj.num_receivers == 0

    def test_cacheable_object_produce_first_request(self):
        """Test producing chunks on first request."""
        items = ["item1", "item2", "item3"]
        obj = MockCacheableObject(items, max_chunk_size=100)

        # First request (empty state)
        rc, data, state = obj.produce({}, "receiver1")

        assert rc == ProduceRC.OK
        assert isinstance(data, list)
        assert len(data) == 3  # All items fit in one chunk
        assert state["start"] == 0
        assert state["count"] == 3

    def test_cacheable_object_produce_subsequent_request(self):
        """Test producing chunks on subsequent requests."""
        items = ["item1", "item2", "item3", "item4"]
        obj = MockCacheableObject(items, max_chunk_size=10)  # Small chunk size

        # First request
        rc1, data1, state1 = obj.produce({}, "receiver1")
        assert rc1 == ProduceRC.OK
        assert state1["start"] == 0
        assert state1["count"] > 0

        # Second request
        rc2, data2, state2 = obj.produce(state1, "receiver1")
        assert rc2 == ProduceRC.OK
        assert state2["start"] == state1["start"] + state1["count"]

    def test_cacheable_object_produce_eof(self):
        """Test that EOF is returned when all items are sent."""
        items = ["item1", "item2"]
        obj = MockCacheableObject(items, max_chunk_size=100)

        # Get all items
        rc1, data1, state1 = obj.produce({}, "receiver1")
        assert rc1 == ProduceRC.OK

        # Next request should return EOF
        rc2, data2, state2 = obj.produce(state1, "receiver1")
        assert rc2 == ProduceRC.EOF
        assert data2 is None

    def test_cacheable_object_caching(self):
        """Test that items are cached for multiple receivers."""
        items = ["item1", "item2"]
        obj = MockCacheableObject(items, max_chunk_size=100)

        # First receiver requests
        rc1, data1, state1 = obj.produce({}, "receiver1")
        assert rc1 == ProduceRC.OK

        # Cache should contain the items
        with obj.lock:
            assert obj.cache[0][0] is not None
            assert obj.cache[1][0] is not None

        # Second receiver requests - should use cache
        rc2, data2, state2 = obj.produce({}, "receiver2")
        assert rc2 == ProduceRC.OK
        assert data2 == data1  # Same data from cache

    def test_concurrent_receivers_share_in_flight_item_production(self, monkeypatch):
        monkeypatch.setattr(cacheable_module, "Future", ObservedFuture)
        obj = BlockingCacheableObject(["item1"], max_chunk_size=100)

        with ThreadPoolExecutor(max_workers=3) as executor:
            results = [executor.submit(obj.produce, {}, f"receiver{i}") for i in range(3)]
            assert obj.produce_started.wait(timeout=5.0)
            with obj.lock:
                production_future = obj._production_futures[0]
            assert production_future.two_waiters.wait(timeout=5.0)
            obj.allow_produce.set()

        produced = [result.result() for result in results]
        assert [data for rc, data, state in produced] == [[b"item1"]] * 3
        assert all(rc == ProduceRC.OK and state == {"start": 0, "count": 1} for rc, data, state in produced)
        assert obj.produce_calls == 1
        assert not obj._production_futures
        assert obj._get_item(0, "receiver3") == b"item1"
        assert obj.produce_calls == 1

    def test_failed_item_production_reaches_waiters_and_is_retryable(self, monkeypatch):
        monkeypatch.setattr(cacheable_module, "Future", ObservedFuture)
        obj = BlockingCacheableObject(["item1"], max_chunk_size=100)
        obj.fail_next_production = True

        with ThreadPoolExecutor(max_workers=3) as executor:
            results = [executor.submit(obj._get_item, 0, f"receiver{i}") for i in range(3)]
            assert obj.produce_started.wait(timeout=5.0)
            with obj.lock:
                production_future = obj._production_futures[0]
            assert production_future.two_waiters.wait(timeout=5.0)
            obj.allow_produce.set()

        for result in results:
            with pytest.raises(RuntimeError, match="production failed"):
                result.result()

        assert not obj._production_futures
        assert obj._get_item(0, "receiver2") == b"item1"
        assert obj.produce_calls == 2

    def test_different_items_can_be_produced_concurrently(self):
        obj = BlockingCacheableObject(["item1", "item2"], max_chunk_size=100)

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = [executor.submit(obj._get_item, i, f"receiver{i}") for i in range(2)]
            assert obj.two_productions_started.wait(timeout=5.0)
            obj.allow_produce.set()

        assert [result.result() for result in results] == [b"item1", b"item2"]
        assert obj.produce_calls == 2
        assert not obj._production_futures

    def test_cleanup_does_not_strand_in_flight_waiters(self, monkeypatch):
        monkeypatch.setattr(cacheable_module, "Future", ObservedFuture)
        obj = BlockingCacheableObject(["item1"], max_chunk_size=100)

        with ThreadPoolExecutor(max_workers=3) as executor:
            results = [executor.submit(obj._get_item, 0, f"receiver{i}") for i in range(3)]
            assert obj.produce_started.wait(timeout=5.0)
            with obj.lock:
                production_future = obj._production_futures[0]
            assert production_future.two_waiters.wait(timeout=5.0)
            obj.clear_cache()
            obj.release()
            obj.allow_produce.set()

        assert [result.result() for result in results] == [b"item1"] * 3
        assert not obj._production_futures
        with pytest.raises(RuntimeError, match="released"):
            obj._get_item(0, "late_receiver")

    def test_cache_publication_failure_does_not_strand_waiters(self, monkeypatch):
        monkeypatch.setattr(cacheable_module, "Future", ObservedFuture)
        obj = BlockingCacheableObject(["item1"], max_chunk_size=100)
        original_debug = obj.logger.debug

        def fail_cache_publication(message):
            if message.startswith("created and cached"):
                raise RuntimeError("cache publication failed")
            original_debug(message)

        monkeypatch.setattr(obj.logger, "debug", fail_cache_publication)

        with ThreadPoolExecutor(max_workers=3) as executor:
            results = [executor.submit(obj._get_item, 0, f"receiver{i}") for i in range(3)]
            assert obj.produce_started.wait(timeout=5.0)
            with obj.lock:
                production_future = obj._production_futures[0]
            assert production_future.two_waiters.wait(timeout=5.0)
            obj.allow_produce.set()

        for result in results:
            with pytest.raises(RuntimeError, match="cache publication failed"):
                result.result()
        assert not obj._production_futures

    def test_uncached_flight_remains_visible_until_settled(self, monkeypatch):
        class BlockingSetFuture(Future):
            def __init__(self):
                super().__init__()
                self.set_entered = threading.Event()
                self.allow_set = threading.Event()
                self.result_entered = threading.Event()

            def set_result(self, result):
                self.set_entered.set()
                assert self.allow_set.wait(timeout=5.0)
                super().set_result(result)

            def result(self, timeout=None):
                self.result_entered.set()
                return super().result(timeout=timeout)

        production_future = BlockingSetFuture()
        monkeypatch.setattr(cacheable_module, "Future", lambda: production_future)
        obj = BlockingCacheableObject(["item1"], max_chunk_size=100)
        obj.clear_cache()
        obj.allow_produce.set()

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(obj._get_item, 0, "receiver1")
            assert production_future.set_entered.wait(timeout=5.0)
            second = executor.submit(obj._get_item, 0, "receiver2")
            try:
                assert production_future.result_entered.wait(timeout=5.0)
                assert obj.produce_calls == 1
            finally:
                production_future.allow_set.set()

        assert first.result() == b"item1"
        assert second.result() == b"item1"
        assert not obj._production_futures

    def test_cacheable_object_cache_clearing_per_item(self, cell):
        """Test that cache is cleared for each item after all receivers get it."""
        items = ["item1", "item2"]
        obj = MockCacheableObject(items, max_chunk_size=100)

        # Set up transaction with 2 receivers
        tx_id = DownloadService.new_transaction(cell=cell, timeout=10.0, num_receivers=2)
        obj.set_transaction(tx_id, "ref1")

        # First receiver gets all items
        rc1, data1, state1 = obj.produce({}, "receiver1")
        assert rc1 == ProduceRC.OK

        # Acknowledge reception
        obj._adjust_cache(state1["start"], state1["count"])

        # Cache should still have items (only 1 receiver received them)
        with obj.lock:
            assert obj.cache[0][1] == 1  # num_received counter

        # Second receiver gets all items
        rc2, data2, state2 = obj.produce({}, "receiver2")
        assert rc2 == ProduceRC.OK

        # Acknowledge reception
        obj._adjust_cache(state2["start"], state2["count"])

        # Cache should be cleared now (both receivers received)
        with obj.lock:
            assert obj.cache[0][0] is None
            assert obj.cache[1][0] is None
            assert obj.cache[0][1] == 2  # num_received counter
            assert obj.cache[1][1] == 2

        # Cleanup
        DownloadService.delete_transaction(tx_id)

    def test_cacheable_object_downloaded_to_all(self, cell):
        """Test downloaded_to_all clears the chunk cache (C1 fix).

        downloaded_to_all() calls clear_cache(), which now only clears self.cache.
        base_obj is released via release() after the transaction_done_cb fires —
        not here.  So base_obj must still be valid after downloaded_to_all().
        """
        items = ["item1", "item2"]
        obj = MockCacheableObject(items, max_chunk_size=100)

        # Set up transaction
        tx_id = DownloadService.new_transaction(cell=cell, timeout=10.0, num_receivers=2)
        obj.set_transaction(tx_id, "ref1")

        # Produce some items
        obj.produce({}, "receiver1")

        # Call downloaded_to_all
        obj.downloaded_to_all()

        # Cache must be cleared; base_obj is intact until release() is called (C1 fix).
        assert obj.cache is None
        assert obj.base_obj is not None, (
            "downloaded_to_all() calls clear_cache() which must not touch base_obj (C1 fix). "
            "base_obj is released via release() after the transaction_done_cb."
        )

        # Cleanup
        DownloadService.delete_transaction(tx_id)

    def test_cacheable_object_transaction_done(self):
        """Test transaction_done clears the chunk cache only (C1 fix).

        transaction_done() calls clear_cache(), which now only nulls self.cache.
        base_obj is released via release() — called by _Transaction.transaction_done()
        after the callback — not by clear_cache() directly.
        """
        items = ["item1", "item2"]
        obj = MockCacheableObject(items, max_chunk_size=100)

        # Produce some items
        obj.produce({}, "receiver1")

        # Call transaction_done (as _Transaction would invoke it)
        obj.transaction_done("tx123", "finished")

        # Cache must be cleared; base_obj is intact until release() is called (C1 fix).
        assert obj.cache is None
        assert obj.base_obj is not None, (
            "transaction_done() must not clear base_obj (C1 fix). "
            "base_obj is released via release() after the transaction_done_cb."
        )

        # Explicitly calling release() (as _Transaction does after the callback)
        # must then clear base_obj.
        obj.release()
        assert obj.base_obj is None

    def test_clear_cache_only_nulls_cache(self):
        """C1 fix: clear_cache() must only null self.cache, not base_obj.

        Before C1 fix, clear_cache() nulled both cache and base_obj.  That caused
        a race: a concurrent _get_item() saw cache=None and fell through to
        produce_item() with base_obj already None → crash.

        With C1 fix, clear_cache() only nulls self.cache.  base_obj is released
        separately via release() after the transaction_done_cb fires.
        """
        items = ["item1", "item2"]
        obj = MockCacheableObject(items, max_chunk_size=100)
        original_base_obj = obj.base_obj
        assert original_base_obj is not None

        obj.clear_cache()

        assert obj.cache is None
        assert obj.base_obj is original_base_obj, "clear_cache() must not touch base_obj (C1 fix)."

    def test_release_nulls_base_obj(self):
        """C1 fix: release() must set base_obj to None."""
        items = ["item1"]
        obj = MockCacheableObject(items, max_chunk_size=100)
        assert obj.base_obj is not None

        obj.release()

        assert obj.base_obj is None

    def test_clear_cache_is_idempotent(self):
        """Calling clear_cache() twice must not raise."""
        items = ["item1"]
        obj = MockCacheableObject(items, max_chunk_size=100)

        obj.clear_cache()
        obj.clear_cache()  # must not raise

        assert obj.cache is None
        # base_obj is NOT touched by clear_cache (C1 fix)
        assert obj.base_obj is not None

    def test_cacheable_object_produce_after_cache_cleared(self):
        """Producing after clear_cache() regenerates items via produce_item().

        After clear_cache(): cache=None, base_obj still valid.  _get_item() falls
        through to produce_item() since base_obj is not None — items are regenerated.
        (This differs from after release(), where base_obj=None causes RuntimeError.)
        """
        items = ["item1", "item2"]
        obj = MockCacheableObject(items, max_chunk_size=100)

        # Produce items
        rc1, data1, state1 = obj.produce({}, "receiver1")
        assert rc1 == ProduceRC.OK

        # Clear cache only — base_obj remains valid (C1 fix)
        obj.clear_cache()
        assert obj.base_obj is not None

        # Produce again — should regenerate items since base_obj is still valid
        rc2, data2, state2 = obj.produce({}, "receiver2")
        assert rc2 == ProduceRC.OK
        assert data2 is not None

    def test_set_transaction_info(self, cell):
        """Test that set_transaction correctly retrieves num_receivers."""
        items = ["item1"]
        obj = MockCacheableObject(items, max_chunk_size=100)

        # Create transaction with 3 receivers
        tx_id = DownloadService.new_transaction(cell=cell, timeout=10.0, num_receivers=3)
        ref_id = DownloadService.add_object(tx_id, obj)

        # Check that num_receivers was set
        assert obj.num_receivers == 3

        # Cleanup
        DownloadService.delete_transaction(tx_id)


class TestItemConsumer:
    """Test suite for ItemConsumer."""

    def test_item_consumer_consume_items(self):
        """Test ItemConsumer consume_items method."""
        consumer = MockItemConsumer()

        items1 = [b"item1", b"item2"]
        result1 = consumer.consume_items(items1, None)

        assert result1 == items1
        assert consumer.consumed_items == items1

        items2 = [b"item3", b"item4"]
        result2 = consumer.consume_items(items2, result1)

        assert result2 == items1 + items2
        assert consumer.consumed_items == items1 + items2

    def test_item_consumer_consume(self):
        """Test ItemConsumer consume method."""
        consumer = MockItemConsumer()

        state = {"key": "value"}
        data = [b"item1", b"item2"]

        new_state = consumer.consume("ref1", state, data)

        assert new_state == state
        assert consumer.result == data

    def test_item_consumer_download_completed(self):
        """Test ItemConsumer download_completed callback."""
        consumer = MockItemConsumer()

        # Should not raise error
        consumer.download_completed("ref1")

    def test_item_consumer_download_failed(self):
        """Test ItemConsumer download_failed callback."""
        consumer = MockItemConsumer()

        consumer.download_failed("ref1", "test reason")

        assert consumer.error == "test reason"
        assert consumer.result is None
