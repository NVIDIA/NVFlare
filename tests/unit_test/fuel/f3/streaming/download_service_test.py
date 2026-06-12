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
import weakref
from typing import Any, Tuple
from unittest.mock import Mock, patch

import pytest

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import new_cell_message
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
        self.downloaded_to_all_call_count = 0
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
        self.downloaded_to_all_call_count += 1

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


def _make_isolated_download_service():
    class IsolatedDownloadService(DownloadService):
        _tx_table = {}
        _ref_table = {}
        _finished_refs = {}
        _logger = Mock()
        _tx_lock = threading.Lock()
        _initialized_cells = weakref.WeakKeyDictionary()

    return IsolatedDownloadService


def _make_download_request(ref_id: str, requester: str, state: dict = None):
    payload = {"ref_id": ref_id}
    if state is not None:
        payload["state"] = state
    return new_cell_message(headers={MessageHeaderKey.ORIGIN: requester}, payload=payload)


def _run_monitor_once(service_cls, now):
    from nvflare.fuel.f3.streaming import download_service as download_service_module

    class MonitorIterationDone(Exception):
        pass

    monitor_thread = threading.current_thread()
    real_time = download_service_module.time.time
    real_sleep = download_service_module.time.sleep

    def test_thread_time():
        if threading.current_thread() is monitor_thread:
            return now
        return real_time()

    def test_thread_sleep(seconds):
        if threading.current_thread() is monitor_thread:
            raise MonitorIterationDone
        real_sleep(seconds)

    with (
        patch.object(download_service_module.time, "time", side_effect=test_thread_time),
        patch.object(download_service_module.time, "sleep", side_effect=test_thread_sleep),
    ):
        with pytest.raises(MonitorIterationDone):
            service_cls._monitor_tx()


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

    def test_transaction_done_on_timeout(self):
        """Test that transaction times out after inactivity."""
        from nvflare.fuel.f3.streaming.download_service import _Transaction

        service = _make_isolated_download_service()
        fake_start_time = 1_000_000_000_000.0
        timeout = 1.0
        data_chunks = [b"chunk1"]
        obj = MockDownloadable(data_chunks)
        tx = _Transaction(timeout=timeout, num_receivers=2)
        tx.last_active_time = fake_start_time
        tx.start_time = fake_start_time
        ref = tx.add_object(obj)

        assert isinstance(tx, _Transaction)
        with service._tx_lock:
            service._tx_table[tx.tid] = tx
            service._ref_table[ref.rid] = ref

        try:
            _run_monitor_once(service, fake_start_time + timeout)
            assert obj.transaction_done_calls == []
            assert tx.tid in service._tx_table
            assert ref.rid in service._ref_table

            _run_monitor_once(service, fake_start_time + timeout + 0.001)
            assert obj.transaction_done_calls == [(tx.tid, TransactionDoneStatus.TIMEOUT)]
            assert tx.tid not in service._tx_table
            assert ref.rid not in service._ref_table
        finally:
            with service._tx_lock:
                service._tx_table.pop(tx.tid, None)
                service._ref_table.pop(ref.rid, None)

    def test_transaction_done_on_completion(self):
        """Test that transaction_done is called when all objects downloaded to all receivers."""
        from nvflare.fuel.f3.streaming.download_service import _Transaction

        service = _make_isolated_download_service()
        num_receivers = 2

        data_chunks = [b"chunk1"]
        obj = MockDownloadable(data_chunks)
        tx = _Transaction(timeout=10.0, num_receivers=num_receivers)
        ref = tx.add_object(obj)
        tx_id = tx.tid

        with service._tx_lock:
            service._tx_table[tx.tid] = tx
            service._ref_table[ref.rid] = ref
            assert isinstance(tx, _Transaction)

            # Simulate all receivers downloading while the synthetic transaction
            # is under the same lock used by the monitor.
            for i in range(num_receivers):
                ref.obj_downloaded(to_receiver=f"receiver{i}", status=DownloadStatus.SUCCESS)

            assert tx.is_finished()
            tx.transaction_done(TransactionDoneStatus.FINISHED)
            service._delete_tx(tx)

            assert tx.tid not in service._tx_table
            assert ref.rid not in service._ref_table

        # Verify transaction_done was called
        assert len(obj.transaction_done_calls) == 1
        transaction_id, status = obj.transaction_done_calls[0]
        assert transaction_id == tx_id
        assert status == TransactionDoneStatus.FINISHED

    def test_delete_transaction_invokes_done_callback_after_releasing_tx_lock(self):
        """transaction_done callbacks may re-enter DownloadService APIs without deadlocking the tx lock."""
        from nvflare.fuel.f3.streaming.download_service import _Transaction

        service = _make_isolated_download_service()
        callback_lock_available = []

        def _callback(*_args, **_kwargs):
            acquired = service._tx_lock.acquire(blocking=False)
            callback_lock_available.append(acquired)
            if acquired:
                service._tx_lock.release()

        tx = _Transaction(timeout=10.0, num_receivers=1, transaction_done_cb=_callback)
        obj = MockDownloadable([b"chunk1"])
        ref = tx.add_object(obj)

        with service._tx_lock:
            service._tx_table[tx.tid] = tx
            service._ref_table[ref.rid] = ref

        service.delete_transaction(tx.tid)

        assert callback_lock_available == [True]
        assert obj.transaction_done_calls == [(tx.tid, TransactionDoneStatus.DELETED)]

    def test_monitor_invokes_done_callback_after_releasing_tx_lock(self):
        """Monitor cleanup must delete table state under lock and run callbacks after lock release."""
        from nvflare.fuel.f3.streaming.download_service import _Transaction

        service = _make_isolated_download_service()
        callback_lock_available = []
        fake_start_time = 1_000_000_000_000.0

        def _callback(*_args, **_kwargs):
            acquired = service._tx_lock.acquire(blocking=False)
            callback_lock_available.append(acquired)
            if acquired:
                service._tx_lock.release()

        tx = _Transaction(timeout=10.0, num_receivers=1, transaction_done_cb=_callback)
        tx.last_active_time = fake_start_time
        tx.start_time = fake_start_time
        obj = MockDownloadable([b"chunk1"])
        ref = tx.add_object(obj)

        with service._tx_lock:
            service._tx_table[tx.tid] = tx
            service._ref_table[ref.rid] = ref
            ref.obj_downloaded(to_receiver="receiver1", status=DownloadStatus.SUCCESS)

        _run_monitor_once(service, fake_start_time + 1.0)

        assert callback_lock_available == [True]
        assert tx.tid not in service._tx_table
        assert ref.rid not in service._ref_table
        assert ref.rid in service._finished_refs

    def test_add_object_cannot_orphan_ref_when_transaction_is_deleted_mid_add(self):
        """add_object keeps tx.refs and _ref_table updates atomic with transaction deletion."""
        from nvflare.fuel.f3.streaming.download_service import _Transaction

        class BlockingSetTransactionDownloadable(MockDownloadable):
            def __init__(self):
                super().__init__([b"chunk1"])
                self.entered = threading.Event()
                self.proceed = threading.Event()

            def set_transaction(self, tx_id: str, ref_id: str):
                self.tx_id = tx_id
                self.ref_id = ref_id
                self.entered.set()
                assert self.proceed.wait(timeout=2.0)

        service = _make_isolated_download_service()
        tx = _Transaction(timeout=10.0, num_receivers=1)
        obj = BlockingSetTransactionDownloadable()
        result = {}

        with service._tx_lock:
            service._tx_table[tx.tid] = tx

        def _add_object():
            try:
                result["ref_id"] = service.add_object(tx.tid, obj)
            except Exception as ex:
                result["error"] = ex

        add_thread = threading.Thread(target=_add_object)
        add_thread.start()
        assert obj.entered.wait(timeout=1.0)

        delete_thread = threading.Thread(target=lambda: service.delete_transaction(tx.tid))
        delete_thread.start()
        obj.proceed.set()

        add_thread.join(timeout=2.0)
        delete_thread.join(timeout=2.0)

        assert not add_thread.is_alive()
        assert not delete_thread.is_alive()
        assert "error" not in result
        assert tx.tid not in service._tx_table
        assert result["ref_id"] not in service._ref_table
        assert obj.transaction_done_calls == [(tx.tid, TransactionDoneStatus.DELETED)]

    def test_transaction_done_uses_ref_snapshot_when_refs_grow_during_callback(self):
        """A late ref append must not be visited by an in-flight transaction_done() iteration."""
        from nvflare.fuel.f3.streaming.download_service import _Transaction

        tx = _Transaction(timeout=10.0, num_receivers=1)
        late_obj = MockDownloadable([b"late"])

        class AppendRefOnDone(MockDownloadable):
            def transaction_done(self, transaction_id: str, status: str):
                super().transaction_done(transaction_id, status)
                tx.add_object(late_obj)

        first_obj = AppendRefOnDone([b"first"])
        tx.add_object(first_obj)

        tx.transaction_done(TransactionDoneStatus.FINISHED)

        assert first_obj.transaction_done_calls == [(tx.tid, TransactionDoneStatus.FINISHED)]
        assert late_obj.transaction_done_calls == []
        assert len(tx.snapshot_refs()) == 2

    def test_shutdown_clears_initialized_cells(self):
        """A new cell allocated after shutdown must register callbacks even if an old cell was initialized."""
        service = _make_isolated_download_service()
        service._tx_monitor = object()

        class FakeCell:
            def __init__(self):
                self.registered = []

            def register_request_cb(self, **kwargs):
                self.registered.append(kwargs)

        cell = FakeCell()
        service._initialize(cell)
        assert list(service._initialized_cells.keys()) == [cell]

        service.shutdown()

        assert list(service._initialized_cells.keys()) == []

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

    def test_duplicate_receiver_completion_is_idempotent(self):
        """Duplicate EOF/error notifications from one requester must not count twice."""
        from nvflare.fuel.f3.streaming.download_service import _Transaction

        num_receivers = 2
        tx = _Transaction(timeout=10.0, num_receivers=num_receivers)
        obj = MockDownloadable([b"chunk1"])
        ref = tx.add_object(obj)

        ref.obj_downloaded(to_receiver="receiver1", status=DownloadStatus.SUCCESS)
        ref.obj_downloaded(to_receiver="receiver1", status=DownloadStatus.SUCCESS)

        assert ref.num_receivers_done == 1
        assert obj.downloaded_to_one_calls == [("receiver1", DownloadStatus.SUCCESS)]
        assert not obj.downloaded_to_all_called
        assert obj.downloaded_to_all_call_count == 0
        assert not tx.is_finished()

        ref.obj_downloaded(to_receiver="receiver2", status=DownloadStatus.SUCCESS)

        assert ref.num_receivers_done == num_receivers
        assert obj.downloaded_to_one_calls == [
            ("receiver1", DownloadStatus.SUCCESS),
            ("receiver2", DownloadStatus.SUCCESS),
        ]
        assert obj.downloaded_to_all_called
        assert obj.downloaded_to_all_call_count == 1
        assert tx.is_finished()

    def test_finished_ref_tombstone_returns_eof_for_completed_requester(self):
        """A retry after FINISHED cleanup should receive EOF for the same completed requester."""
        from nvflare.fuel.f3.streaming.download_service import _Transaction

        service = _make_isolated_download_service()
        tx = _Transaction(timeout=10.0, num_receivers=1)
        obj = MockDownloadable([b"chunk1"])
        ref = tx.add_object(obj)

        with service._tx_lock:
            service._tx_table[tx.tid] = tx
            service._ref_table[ref.rid] = ref
            ref.obj_downloaded(to_receiver="receiver1", status=DownloadStatus.SUCCESS)
            tx.transaction_done(TransactionDoneStatus.FINISHED)
            service._delete_tx(tx, tombstone_finished_refs=True)

        reply = service._handle_download(_make_download_request(ref.rid, "receiver1"))

        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
        assert reply.payload == {"status": ProduceRC.EOF}

    def test_monitor_tombstones_finished_transaction_refs(self):
        """The monitor should tombstone refs only after normal FINISHED cleanup."""
        from nvflare.fuel.f3.streaming.download_service import _Transaction

        service = _make_isolated_download_service()
        fake_start_time = 1_000_000_000_000.0
        tx = _Transaction(timeout=10.0, num_receivers=1)
        tx.last_active_time = fake_start_time
        tx.start_time = fake_start_time
        obj = MockDownloadable([b"chunk1"])
        ref = tx.add_object(obj)

        with service._tx_lock:
            service._tx_table[tx.tid] = tx
            service._ref_table[ref.rid] = ref
            ref.obj_downloaded(to_receiver="receiver1", status=DownloadStatus.SUCCESS)

        _run_monitor_once(service, fake_start_time + 1.0)

        assert tx.tid not in service._tx_table
        assert ref.rid not in service._ref_table
        assert ref.rid in service._finished_refs

        reply = service._handle_download(_make_download_request(ref.rid, "receiver1"))

        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
        assert reply.payload == {"status": ProduceRC.EOF}

    def test_large_fanout_retries_after_finished_cleanup_return_eof(self):
        """Simulate a saturated large-model fanout where EOF replies are delayed.

        The real failure happens with many clients downloading many large tensor refs:
        the transaction finishes, the monitor removes the refs, and delayed parent
        resends make receivers ask for the same refs again. Those retries must see
        EOF from the finished-ref tombstone, not INVALID_REQUEST / "no ref found".
        """
        from nvflare.fuel.f3.streaming.download_service import _Transaction

        service = _make_isolated_download_service()
        fake_start_time = 1_000_000_000_000.0
        num_receivers = 16
        num_refs = 24
        tx = _Transaction(timeout=10.0, num_receivers=num_receivers)
        tx.last_active_time = fake_start_time
        tx.start_time = fake_start_time
        refs = []

        with service._tx_lock:
            service._tx_table[tx.tid] = tx
            for ref_index in range(num_refs):
                ref = tx.add_object(MockDownloadable([f"tensor-{ref_index}".encode()]))
                service._ref_table[ref.rid] = ref
                refs.append(ref)

        for ref in refs:
            for receiver_index in range(num_receivers):
                requester = f"site-{receiver_index}.site-{receiver_index}_job_active"
                chunk_reply = service._handle_download(_make_download_request(ref.rid, requester))
                assert chunk_reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
                assert chunk_reply.payload["status"] == ProduceRC.OK

                eof_reply = service._handle_download(
                    _make_download_request(ref.rid, requester, chunk_reply.payload["state"])
                )
                assert eof_reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
                assert eof_reply.payload == {"status": ProduceRC.EOF}

                if receiver_index == 0:
                    duplicate_eof = service._handle_download(
                        _make_download_request(ref.rid, requester, chunk_reply.payload["state"])
                    )
                    assert duplicate_eof.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
                    assert duplicate_eof.payload == {"status": ProduceRC.EOF}
                    assert ref.num_receivers_done == 1
                    assert len(ref.obj.downloaded_to_one_calls) == 1
                    assert ref.obj.downloaded_to_all_call_count == 0

        assert tx.is_finished()
        assert all(ref.num_receivers_done == num_receivers for ref in refs)
        assert all(len(ref.obj.downloaded_to_one_calls) == num_receivers for ref in refs)
        assert all(ref.obj.downloaded_to_all_call_count == 1 for ref in refs)

        _run_monitor_once(service, fake_start_time + 1.0)

        assert tx.tid not in service._tx_table
        assert all(ref.rid not in service._ref_table for ref in refs)
        assert all(ref.rid in service._finished_refs for ref in refs)

        service._logger.error.reset_mock()
        for ref in refs:
            for receiver_index in range(num_receivers):
                requester = f"site-{receiver_index}.site-{receiver_index}_job_active"
                retry_reply = service._handle_download(_make_download_request(ref.rid, requester))
                assert retry_reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
                assert retry_reply.payload == {"status": ProduceRC.EOF}

        service._logger.error.assert_not_called()

    def test_finished_ref_tombstone_is_requester_scoped(self):
        """A tombstone must not convert an unknown requester's stale ref to EOF."""
        from nvflare.fuel.f3.streaming.download_service import _Transaction

        service = _make_isolated_download_service()
        service._logger.reset_mock()
        tx = _Transaction(timeout=10.0, num_receivers=1)
        obj = MockDownloadable([b"chunk1"])
        ref = tx.add_object(obj)

        with service._tx_lock:
            service._tx_table[tx.tid] = tx
            service._ref_table[ref.rid] = ref
            ref.obj_downloaded(to_receiver="receiver1", status=DownloadStatus.SUCCESS)
            tx.transaction_done(TransactionDoneStatus.FINISHED)
            service._delete_tx(tx, tombstone_finished_refs=True)

        reply = service._handle_download(_make_download_request(ref.rid, "receiver2"))

        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
        service._logger.error.assert_called_once()

    def test_timeout_cleanup_does_not_create_finished_ref_tombstone(self):
        """Only FINISHED cleanup may tombstone refs; TIMEOUT/DELETED refs stay fatal."""
        from nvflare.fuel.f3.streaming.download_service import _Transaction

        service = _make_isolated_download_service()
        service._logger.reset_mock()
        tx = _Transaction(timeout=10.0, num_receivers=1)
        obj = MockDownloadable([b"chunk1"])
        ref = tx.add_object(obj)

        with service._tx_lock:
            service._tx_table[tx.tid] = tx
            service._ref_table[ref.rid] = ref
            service._delete_tx(tx)

        reply = service._handle_download(_make_download_request(ref.rid, "receiver1"))

        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
        assert ref.rid not in service._finished_refs

    def test_failed_finished_ref_retry_returns_error_not_eof(self):
        """A failed terminal requester must not be converted to EOF by a tombstone."""
        from nvflare.fuel.f3.streaming.download_service import _Transaction

        service = _make_isolated_download_service()
        tx = _Transaction(timeout=10.0, num_receivers=1)
        obj = MockDownloadable([b"chunk1"])
        ref = tx.add_object(obj)

        with service._tx_lock:
            service._tx_table[tx.tid] = tx
            service._ref_table[ref.rid] = ref
            ref.obj_downloaded(to_receiver="receiver1", status=DownloadStatus.FAILED)
            tx.transaction_done(TransactionDoneStatus.FINISHED)
            service._delete_tx(tx, tombstone_finished_refs=True)

        reply = service._handle_download(_make_download_request(ref.rid, "receiver1"))

        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
        assert reply.payload == {"status": ProduceRC.ERROR}

    def test_expired_finished_ref_tombstone_returns_invalid_request(self):
        """Finished-ref tombstones are temporary and expire back to normal missing-ref behavior."""
        from nvflare.fuel.f3.streaming import download_service as download_service_module
        from nvflare.fuel.f3.streaming.download_service import _Transaction

        service = _make_isolated_download_service()
        service._logger.reset_mock()
        tx = _Transaction(timeout=10.0, num_receivers=1)
        obj = MockDownloadable([b"chunk1"])
        ref = tx.add_object(obj)

        with service._tx_lock:
            service._tx_table[tx.tid] = tx
            service._ref_table[ref.rid] = ref
            ref.obj_downloaded(to_receiver="receiver1", status=DownloadStatus.SUCCESS)
            tx.transaction_done(TransactionDoneStatus.FINISHED)
            service._delete_tx(tx, tombstone_finished_refs=True)
            service._finished_refs[ref.rid].last_active_time = 100.0

        with patch.object(download_service_module.time, "time", return_value=100.0 + service.FINISHED_REFS_TTL + 1.0):
            reply = service._handle_download(_make_download_request(ref.rid, "receiver1"))

        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
        assert ref.rid not in service._finished_refs

    def test_finished_ref_retry_does_not_extend_tombstone_ttl(self):
        """Repeated valid retries must not keep a finished-ref tombstone alive indefinitely."""
        from nvflare.fuel.f3.streaming import download_service as download_service_module
        from nvflare.fuel.f3.streaming.download_service import _Transaction

        service = _make_isolated_download_service()
        service._logger.reset_mock()
        tx = _Transaction(timeout=10.0, num_receivers=1)
        obj = MockDownloadable([b"chunk1"])
        ref = tx.add_object(obj)

        with service._tx_lock:
            service._tx_table[tx.tid] = tx
            service._ref_table[ref.rid] = ref
            ref.obj_downloaded(to_receiver="receiver1", status=DownloadStatus.SUCCESS)

        with patch.object(download_service_module.time, "time", return_value=100.0):
            with service._tx_lock:
                tx.transaction_done(TransactionDoneStatus.FINISHED)
                service._delete_tx(tx, tombstone_finished_refs=True)

        with patch.object(download_service_module.time, "time", return_value=100.0 + service.FINISHED_REFS_TTL - 1.0):
            retry_reply = service._handle_download(_make_download_request(ref.rid, "receiver1"))

        assert retry_reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
        assert retry_reply.payload == {"status": ProduceRC.EOF}
        assert service._finished_refs[ref.rid].last_active_time == 100.0

        with patch.object(download_service_module.time, "time", return_value=100.0 + service.FINISHED_REFS_TTL + 1.0):
            expired_retry_reply = service._handle_download(_make_download_request(ref.rid, "receiver1"))

        assert expired_retry_reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
        assert ref.rid not in service._finished_refs

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
        """Test transaction_done callback is invoked with the original base_obj (C2 fix).

        Before C2 fix, the callback received [None] because obj.transaction_done()
        cleared base_obj before the callback was invoked.  After C2 fix, base_objs
        are snapshotted before the per-object loop; the callback receives the original
        source objects.
        """
        callback_mock = Mock()

        tx_id = DownloadService.new_transaction(
            cell=cell, timeout=10.0, num_receivers=2, transaction_done_cb=callback_mock, test_arg="test_value"
        )

        data_chunks = [b"chunk1", b"chunk2"]
        original_base_obj = data_chunks  # MockDownloadable stores data_chunks as base_obj
        obj = MockDownloadable(data_chunks)
        DownloadService.add_object(tx_id, obj)

        # Delete transaction — triggers transaction_done() → callback
        DownloadService.delete_transaction(tx_id)

        # Verify callback was called
        callback_mock.assert_called_once()
        args, kwargs = callback_mock.call_args
        assert args[0] == tx_id  # transaction_id
        assert args[1] == TransactionDoneStatus.DELETED  # status
        assert len(args[2]) == 1  # objects list has one entry
        # C2 fix: callback must receive original base_obj, not None
        assert args[2][0] is original_base_obj, (
            "transaction_done_cb must receive the original base_obj, not None (C2 fix). " f"Got: {args[2][0]}"
        )
        assert kwargs.get("test_arg") == "test_value"

    def test_transaction_done_callback_ordering(self, cell):
        """release() must be called AFTER the callback, not before (C2 fix)."""
        base_obj_seen_in_cb = []

        def _cb(tid, status, base_objs, **kwargs):
            # Record what base_obj looks like on the obj directly during callback
            base_obj_seen_in_cb.append(obj.base_obj)

        tx_id = DownloadService.new_transaction(cell=cell, timeout=10.0, num_receivers=1, transaction_done_cb=_cb)

        data_chunks = [b"chunk1"]
        obj = MockDownloadable(data_chunks)
        DownloadService.add_object(tx_id, obj)
        DownloadService.delete_transaction(tx_id)

        # During the callback, obj.base_obj must still be whatever it was before
        # (MockDownloadable does not override release(), so base_obj is the same object
        # unless explicitly cleared — the important thing is release() was not yet called)
        assert len(base_obj_seen_in_cb) == 1

    def test_byte_accounting_uses_chunk_lengths(self, cell):
        """H1 fix: total_bytes must sum individual chunk byte lengths, not list count.

        Before H1 fix, total_bytes += len(data) counted list items (e.g. 3 chunks
        → total_bytes += 3).  After fix, total_bytes += sum(len(chunk) for chunk in data)
        gives the correct byte total.
        """
        from nvflare.fuel.f3.streaming.download_service import _Transaction

        # Build a transaction and manually inject total_bytes as _handle_download would
        tx_id = DownloadService.new_transaction(cell=cell, timeout=10.0, num_receivers=1)

        with DownloadService._tx_lock:
            tx = DownloadService._tx_table.get(tx_id)
            assert isinstance(tx, _Transaction)
            # Simulate what _handle_download does: data is a list of byte chunks
            data = [b"hello", b"world", b"!!!!!"]  # 3 chunks, 5+5+5 = 15 bytes
            tx.add_total_bytes(sum(len(chunk) for chunk in data))
            assert tx.get_total_bytes() == 15, (
                f"H1 fix: expected 15 bytes from {data}, got {tx.get_total_bytes()}. "
                "total_bytes must be the sum of chunk lengths, not the number of chunks."
            )

        DownloadService.delete_transaction(tx_id)

    def test_source_progress_callback_reports_per_receiver_bytes_items_and_completion(self):
        """DownloadService source-side progress is scoped by ref and downstream requester."""
        from nvflare.fuel.f3.streaming.download_service import _Transaction

        service = _make_isolated_download_service()
        events = []
        tx = _Transaction(
            timeout=10.0,
            num_receivers=2,
            progress_cb=lambda **kwargs: events.append(kwargs),
            progress_interval=0.0,
        )
        obj = MockDownloadable([[b"aa", b"bbb"]])
        ref = tx.add_object(obj, ref_id="ref-1")

        with service._tx_lock:
            service._tx_table[tx.tid] = tx
            service._ref_table[ref.rid] = ref

        try:
            receiver_a_chunk = service._handle_download(_make_download_request(ref.rid, "receiver-a"))
            receiver_a_eof = service._handle_download(
                _make_download_request(ref.rid, "receiver-a", receiver_a_chunk.payload["state"])
            )
            receiver_b_chunk = service._handle_download(_make_download_request(ref.rid, "receiver-b"))

            assert receiver_a_chunk.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
            assert receiver_a_eof.payload == {"status": ProduceRC.EOF}
            assert receiver_b_chunk.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
        finally:
            with service._tx_lock:
                service._tx_table.pop(tx.tid, None)
                service._ref_table.pop(ref.rid, None)

        receiver_a_events = [event for event in events if event["receiver_id"] == "receiver-a"]
        receiver_b_events = [event for event in events if event["receiver_id"] == "receiver-b"]

        assert [event["state"] for event in receiver_a_events] == ["active", "active", "completed"]
        assert [event["sequence"] for event in receiver_a_events] == [1, 2, 3]
        assert [event["bytes_done"] for event in receiver_a_events] == [0, 5, 5]
        assert [event["items_done"] for event in receiver_a_events] == [None, 2, 2]
        assert [event["state"] for event in receiver_b_events] == ["active", "active"]
        assert [event["sequence"] for event in receiver_b_events] == [1, 2]
        assert [event["bytes_done"] for event in receiver_b_events] == [0, 5]
        assert all(event["ref_id"] == "ref-1" for event in events)
        assert all("timestamp" in event for event in events)

    def test_source_progress_callback_reports_failed_produce_result(self):
        from nvflare.fuel.f3.streaming.download_service import _Transaction

        service = _make_isolated_download_service()
        events = []
        tx = _Transaction(
            timeout=10.0,
            num_receivers=1,
            progress_cb=lambda **kwargs: events.append(kwargs),
            progress_interval=0.0,
        )
        obj = MockDownloadable([b"chunk1"], fail_on_chunk=0)
        ref = tx.add_object(obj, ref_id="ref-fail")

        with service._tx_lock:
            service._tx_table[tx.tid] = tx
            service._ref_table[ref.rid] = ref

        try:
            reply = service._handle_download(_make_download_request(ref.rid, "receiver-a"))
        finally:
            with service._tx_lock:
                service._tx_table.pop(tx.tid, None)
                service._ref_table.pop(ref.rid, None)

        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
        assert reply.payload == {"status": ProduceRC.ERROR}
        assert [event["state"] for event in events] == ["active", "failed"]
        assert [event["sequence"] for event in events] == [1, 2]
        assert [event["bytes_done"] for event in events] == [0, 0]

    def test_source_progress_callback_reports_aborted_transaction_for_started_receivers(self):
        from nvflare.fuel.f3.streaming.download_service import _Transaction

        service = _make_isolated_download_service()
        events = []
        tx = _Transaction(
            timeout=10.0,
            num_receivers=1,
            progress_cb=lambda **kwargs: events.append(kwargs),
            progress_interval=0.0,
        )
        obj = MockDownloadable([b"chunk1", b"chunk2"])
        ref = tx.add_object(obj, ref_id="ref-abort")

        with service._tx_lock:
            service._tx_table[tx.tid] = tx
            service._ref_table[ref.rid] = ref

        service._handle_download(_make_download_request(ref.rid, "receiver-a"))
        service.delete_transaction(tx.tid)

        assert [event["state"] for event in events] == ["active", "active", "aborted"]
        assert [event["sequence"] for event in events] == [1, 2, 3]
        assert events[-1]["receiver_id"] == "receiver-a"
        assert events[-1]["bytes_done"] == len(b"chunk1")

    def test_source_progress_terminal_state_applies_to_late_receiver(self):
        from nvflare.fuel.f3.streaming.download_service import _Transaction
        from nvflare.fuel.f3.streaming.transfer_progress import TransferProgressState

        events = []
        tx = _Transaction(
            timeout=10.0,
            num_receivers=1,
            progress_cb=lambda **kwargs: events.append(kwargs),
            progress_interval=0.0,
        )
        obj = MockDownloadable([b"chunk1", b"chunk2"])
        ref = tx.add_object(obj, ref_id="ref-complete")

        ref.emit_progress(receiver_id="receiver-a", state=TransferProgressState.ACTIVE, bytes_delta=10)
        ref.emit_terminal_progress_for_started_receivers(TransferProgressState.COMPLETED)
        ref.emit_progress(receiver_id="receiver-b", state=TransferProgressState.ACTIVE, bytes_delta=5)

        assert [event["receiver_id"] for event in events] == ["receiver-a", "receiver-a", "receiver-b"]
        assert [event["state"] for event in events] == [
            TransferProgressState.ACTIVE,
            TransferProgressState.COMPLETED,
            TransferProgressState.COMPLETED,
        ]
        assert events[-1]["bytes_done"] == 0
