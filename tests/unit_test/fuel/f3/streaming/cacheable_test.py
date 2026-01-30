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

from typing import Any, List

import pytest

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
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
        """Test downloaded_to_all callback clears cache."""
        items = ["item1", "item2"]
        obj = MockCacheableObject(items, max_chunk_size=100)

        # Set up transaction
        tx_id = DownloadService.new_transaction(cell=cell, timeout=10.0, num_receivers=2)
        obj.set_transaction(tx_id, "ref1")

        # Produce some items
        obj.produce({}, "receiver1")

        # Call downloaded_to_all
        obj.downloaded_to_all()

        # Cache should be cleared
        assert obj.cache is None

        # Cleanup
        DownloadService.delete_transaction(tx_id)

    def test_cacheable_object_transaction_done(self):
        """Test transaction_done callback clears cache."""
        items = ["item1", "item2"]
        obj = MockCacheableObject(items, max_chunk_size=100)

        # Produce some items
        obj.produce({}, "receiver1")

        # Call transaction_done
        obj.transaction_done("tx123", "finished")

        # Cache should be cleared
        assert obj.cache is None

    def test_cacheable_object_produce_after_cache_cleared(self):
        """Test that producing after cache is cleared still works."""
        items = ["item1", "item2"]
        obj = MockCacheableObject(items, max_chunk_size=100)

        # Produce items
        rc1, data1, state1 = obj.produce({}, "receiver1")
        assert rc1 == ProduceRC.OK

        # Clear cache
        obj.clear_cache()

        # Produce again - should regenerate items
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
