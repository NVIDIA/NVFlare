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

import time
from typing import Any, Tuple
from unittest.mock import Mock

import pytest

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.streaming.download_service import (
    Consumer,
    Downloadable,
    DownloadService,
    DownloadStatus,
    ProduceRC,
    TransactionDoneStatus,
)
from nvflare.fuel.utils.network_utils import get_open_ports


class MockDownloadable(Downloadable):
    """Mock downloadable for testing."""

    def __init__(self, data_chunks: list, fail_on_chunk: int = -1):
        super().__init__(data_chunks)
        self.data_chunks = data_chunks
        self.fail_on_chunk = fail_on_chunk
        self.current_chunk = 0
        self.downloaded_to_one_calls = []
        self.downloaded_to_all_called = False
        self.transaction_done_calls = []
        self.tx_id = None
        self.ref_id = None

    def set_transaction(self, tx_id: str, ref_id: str):
        self.tx_id = tx_id
        self.ref_id = ref_id

    def produce(self, state: dict, requester: str) -> Tuple[str, Any, dict]:
        if not state:
            chunk_idx = 0
        else:
            chunk_idx = state.get("chunk_idx", 0)

        if self.fail_on_chunk >= 0 and chunk_idx == self.fail_on_chunk:
            return ProduceRC.ERROR, None, {}

        if chunk_idx >= len(self.data_chunks):
            return ProduceRC.EOF, None, {}

        return ProduceRC.OK, self.data_chunks[chunk_idx], {"chunk_idx": chunk_idx + 1}

    def downloaded_to_one(self, to_receiver: str, status: str):
        self.downloaded_to_one_calls.append((to_receiver, status))

    def downloaded_to_all(self):
        self.downloaded_to_all_called = True

    def transaction_done(self, transaction_id: str, status: str):
        self.transaction_done_calls.append((transaction_id, status))


class MockConsumer(Consumer):
    """Mock consumer for testing."""

    def __init__(self):
        super().__init__()
        self.consumed_data = []
        self.completed = False
        self.failed = False
        self.failure_reason = None
        self.ref_id = None

    def consume(self, ref_id: str, state: dict, data: Any) -> dict:
        self.ref_id = ref_id
        self.consumed_data.append(data)
        return state

    def download_completed(self, ref_id: str):
        self.ref_id = ref_id
        self.completed = True

    def download_failed(self, ref_id: str, reason: str):
        self.ref_id = ref_id
        self.failed = True
        self.failure_reason = reason


class TestDownloadService:
    """Test suite for DownloadService."""

    @pytest.fixture
    def port(self):
        return get_open_ports(1)[0]

    @pytest.fixture
    def cell(self, port, request):
        """Create a unique cell for each test."""
        # Use test name to create unique cell name
        test_name = request.node.name
        cell_name = f"test_cell_{test_name}_{port}"
        listening_url = f"tcp://localhost:{port}"
        cell = CoreCell(cell_name, listening_url, secure=False, credentials={})
        cell.start()
        yield cell
        cell.stop()

    def test_new_transaction(self, cell):
        """Test creating a new transaction."""
        tx_id = DownloadService.new_transaction(cell=cell, timeout=10.0, num_receivers=2)

        assert tx_id is not None
        assert tx_id.startswith("T")

        # Verify transaction info
        tx_info = DownloadService.get_transaction_info(tx_id)
        assert tx_info is not None
        assert tx_info.timeout == 10.0
        assert tx_info.num_receivers == 2
        assert len(tx_info.objects) == 0

        # Clean up
        DownloadService.delete_transaction(tx_id)

    def test_add_object_to_transaction(self, cell):
        """Test adding objects to a transaction."""
        tx_id = DownloadService.new_transaction(cell=cell, timeout=10.0, num_receivers=2)

        # Create mock downloadable
        data_chunks = [b"chunk1", b"chunk2", b"chunk3"]
        obj = MockDownloadable(data_chunks)

        # Add object
        ref_id = DownloadService.add_object(tx_id, obj)

        assert ref_id is not None
        assert ref_id.startswith("R")
        assert obj.tx_id == tx_id
        assert obj.ref_id == ref_id

        # Verify transaction info
        tx_info = DownloadService.get_transaction_info(tx_id)
        assert len(tx_info.objects) == 1
        assert tx_info.objects[0] == obj

        # Clean up
        DownloadService.delete_transaction(tx_id)

    def test_downloaded_to_one_callback(self, cell):
        """Test that downloaded_to_one is called with correct parameters."""
        tx_id = DownloadService.new_transaction(cell=cell, timeout=10.0, num_receivers=2)

        data_chunks = [b"chunk1"]
        obj = MockDownloadable(data_chunks)
        ref_id = DownloadService.add_object(tx_id, obj)

        # Simulate download completion
        from nvflare.fuel.f3.streaming.download_service import _Ref

        ref = DownloadService._ref_table.get(ref_id)
        assert isinstance(ref, _Ref)

        # Call obj_downloaded with to_receiver parameter
        ref.obj_downloaded(to_receiver="receiver1", status=DownloadStatus.SUCCESS)

        # Verify callback was called with correct parameters
        assert len(obj.downloaded_to_one_calls) == 1
        to_receiver, status = obj.downloaded_to_one_calls[0]
        assert to_receiver == "receiver1"
        assert status == DownloadStatus.SUCCESS

        # Clean up
        DownloadService.delete_transaction(tx_id)

    def test_downloaded_to_all_callback(self, cell):
        """Test that downloaded_to_all is called when all receivers complete."""
        num_receivers = 3
        tx_id = DownloadService.new_transaction(cell=cell, timeout=10.0, num_receivers=num_receivers)

        data_chunks = [b"chunk1"]
        obj = MockDownloadable(data_chunks)
        ref_id = DownloadService.add_object(tx_id, obj)

        from nvflare.fuel.f3.streaming.download_service import _Ref

        ref = DownloadService._ref_table.get(ref_id)
        assert isinstance(ref, _Ref)

        # Simulate downloads from multiple receivers
        for i in range(num_receivers - 1):
            ref.obj_downloaded(to_receiver=f"receiver{i}", status=DownloadStatus.SUCCESS)
            assert not obj.downloaded_to_all_called

        # Last receiver should trigger downloaded_to_all
        ref.obj_downloaded(to_receiver=f"receiver{num_receivers - 1}", status=DownloadStatus.SUCCESS)
        assert obj.downloaded_to_all_called

        # Clean up
        DownloadService.delete_transaction(tx_id)

    def test_downloaded_to_all_not_called_with_unlimited_receivers(self, cell):
        """Test that downloaded_to_all is NOT called when num_receivers is 0 (unlimited)."""
        tx_id = DownloadService.new_transaction(cell=cell, timeout=10.0, num_receivers=0)

        data_chunks = [b"chunk1"]
        obj = MockDownloadable(data_chunks)
        ref_id = DownloadService.add_object(tx_id, obj)

        from nvflare.fuel.f3.streaming.download_service import _Ref

        ref = DownloadService._ref_table.get(ref_id)
        assert isinstance(ref, _Ref)

        # Download to multiple receivers
        for i in range(5):
            ref.obj_downloaded(to_receiver=f"receiver{i}", status=DownloadStatus.SUCCESS)

        # downloaded_to_all should NOT be called because num_receivers is 0
        assert not obj.downloaded_to_all_called

        # Clean up
        DownloadService.delete_transaction(tx_id)

    def test_transaction_done_on_delete(self, cell):
        """Test that transaction_done is called when transaction is deleted."""
        tx_id = DownloadService.new_transaction(cell=cell, timeout=10.0, num_receivers=2)

        data_chunks = [b"chunk1"]
        obj = MockDownloadable(data_chunks)
        DownloadService.add_object(tx_id, obj)

        # Delete transaction
        DownloadService.delete_transaction(tx_id)

        # Verify transaction_done was called
        assert len(obj.transaction_done_calls) == 1
        transaction_id, status = obj.transaction_done_calls[0]
        assert transaction_id == tx_id
        assert status == TransactionDoneStatus.DELETED

    def test_transaction_done_on_timeout(self, cell):
        """Test that transaction times out after inactivity."""
        # Use very short timeout for testing
        tx_id = DownloadService.new_transaction(cell=cell, timeout=0.1, num_receivers=2)

        data_chunks = [b"chunk1"]
        obj = MockDownloadable(data_chunks)
        DownloadService.add_object(tx_id, obj)

        # Wait for timeout (with buffer)
        time.sleep(0.5)

        # Wait for monitor thread to process (it runs every 5 seconds, but processes expired ones)
        # Since we can't wait 5 seconds, we'll manually trigger timeout check
        from nvflare.fuel.f3.streaming.download_service import _Transaction

        with DownloadService._tx_lock:
            tx = DownloadService._tx_table.get(tx_id)
            if tx:
                assert isinstance(tx, _Transaction)
                assert time.time() - tx.last_active_time > tx.timeout
                # Simulate what monitor does
                tx.transaction_done(TransactionDoneStatus.TIMEOUT)
                DownloadService._delete_tx(tx)

        # Verify transaction_done was called with TIMEOUT status
        assert len(obj.transaction_done_calls) == 1
        transaction_id, status = obj.transaction_done_calls[0]
        assert transaction_id == tx_id
        assert status == TransactionDoneStatus.TIMEOUT

    def test_transaction_done_on_completion(self, cell):
        """Test that transaction_done is called when all objects downloaded to all receivers."""
        num_receivers = 2
        tx_id = DownloadService.new_transaction(cell=cell, timeout=10.0, num_receivers=num_receivers)

        data_chunks = [b"chunk1"]
        obj = MockDownloadable(data_chunks)
        ref_id = DownloadService.add_object(tx_id, obj)

        from nvflare.fuel.f3.streaming.download_service import _Transaction

        ref = DownloadService._ref_table.get(ref_id)

        # Simulate all receivers downloading
        for i in range(num_receivers):
            ref.obj_downloaded(to_receiver=f"receiver{i}", status=DownloadStatus.SUCCESS)

        # Check if transaction is finished
        with DownloadService._tx_lock:
            tx = DownloadService._tx_table.get(tx_id)
            if tx:
                assert isinstance(tx, _Transaction)
                if tx.is_finished():
                    tx.transaction_done(TransactionDoneStatus.FINISHED)
                    DownloadService._delete_tx(tx)

        # Verify transaction_done was called
        assert len(obj.transaction_done_calls) == 1
        transaction_id, status = obj.transaction_done_calls[0]
        assert transaction_id == tx_id
        assert status == TransactionDoneStatus.FINISHED

    def test_get_transaction_id_from_ref_id(self, cell):
        """Test retrieving transaction ID from reference ID."""
        tx_id = DownloadService.new_transaction(cell=cell, timeout=10.0, num_receivers=2)

        data_chunks = [b"chunk1"]
        obj = MockDownloadable(data_chunks)
        ref_id = DownloadService.add_object(tx_id, obj)

        # Get transaction ID from ref ID
        retrieved_tx_id = DownloadService.get_transaction_id(ref_id)
        assert retrieved_tx_id == tx_id

        # Clean up
        DownloadService.delete_transaction(tx_id)

    def test_multiple_objects_in_transaction(self, cell):
        """Test transaction with multiple objects."""
        num_receivers = 2
        tx_id = DownloadService.new_transaction(cell=cell, timeout=10.0, num_receivers=num_receivers)

        # Add multiple objects
        obj1 = MockDownloadable([b"obj1_chunk1"])
        obj2 = MockDownloadable([b"obj2_chunk1"])
        obj3 = MockDownloadable([b"obj3_chunk1"])

        ref_id1 = DownloadService.add_object(tx_id, obj1)
        ref_id2 = DownloadService.add_object(tx_id, obj2)
        ref_id3 = DownloadService.add_object(tx_id, obj3)

        # Verify all objects in transaction
        tx_info = DownloadService.get_transaction_info(tx_id)
        assert len(tx_info.objects) == 3
        assert obj1 in tx_info.objects
        assert obj2 in tx_info.objects
        assert obj3 in tx_info.objects

        # Simulate downloads
        for ref_id in [ref_id1, ref_id2, ref_id3]:
            ref = DownloadService._ref_table.get(ref_id)
            for i in range(num_receivers):
                ref.obj_downloaded(to_receiver=f"receiver{i}", status=DownloadStatus.SUCCESS)

        # All objects should have downloaded_to_all called
        assert obj1.downloaded_to_all_called
        assert obj2.downloaded_to_all_called
        assert obj3.downloaded_to_all_called

        # Clean up
        DownloadService.delete_transaction(tx_id)

    def test_num_receivers_done_tracking(self, cell):
        """Test that num_receivers_done is correctly tracked."""
        num_receivers = 3
        tx_id = DownloadService.new_transaction(cell=cell, timeout=10.0, num_receivers=num_receivers)

        data_chunks = [b"chunk1"]
        obj = MockDownloadable(data_chunks)
        ref_id = DownloadService.add_object(tx_id, obj)

        ref = DownloadService._ref_table.get(ref_id)
        assert ref.num_receivers_done == 0

        # Download to receivers one by one
        for i in range(num_receivers):
            ref.obj_downloaded(to_receiver=f"receiver{i}", status=DownloadStatus.SUCCESS)
            assert ref.num_receivers_done == i + 1

        # Clean up
        DownloadService.delete_transaction(tx_id)

    def test_custom_ref_id(self, cell):
        """Test adding object with custom ref_id."""
        tx_id = DownloadService.new_transaction(cell=cell, timeout=10.0, num_receivers=2)

        data_chunks = [b"chunk1"]
        obj = MockDownloadable(data_chunks)
        custom_ref_id = "CUSTOM_REF_123"

        ref_id = DownloadService.add_object(tx_id, obj, ref_id=custom_ref_id)

        assert ref_id == custom_ref_id
        assert obj.ref_id == custom_ref_id

        # Clean up
        DownloadService.delete_transaction(tx_id)

    def test_transaction_done_callback(self, cell):
        """Test transaction_done callback is invoked."""
        callback_mock = Mock()

        tx_id = DownloadService.new_transaction(
            cell=cell, timeout=10.0, num_receivers=2, transaction_done_cb=callback_mock, test_arg="test_value"
        )

        data_chunks = [b"chunk1"]
        obj = MockDownloadable(data_chunks)
        DownloadService.add_object(tx_id, obj)

        # Delete transaction
        DownloadService.delete_transaction(tx_id)

        # Verify callback was called
        callback_mock.assert_called_once()
        args, kwargs = callback_mock.call_args
        assert args[0] == tx_id  # transaction_id
        assert args[1] == TransactionDoneStatus.DELETED  # status
        assert len(args[2]) == 1  # objects list
        assert kwargs.get("test_arg") == "test_value"
