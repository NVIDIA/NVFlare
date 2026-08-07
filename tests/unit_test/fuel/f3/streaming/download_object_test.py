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

from typing import Any
from unittest.mock import MagicMock

import pytest

import nvflare.fuel.f3.streaming.download_service as download_service
from nvflare.apis.signal import Signal
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.download_service import Consumer, ProduceRC, _encode_direct_control, download_object


class MockConsumer(Consumer):
    """Mock consumer for testing."""

    supports_pipelining = True

    def __init__(self, consume_exc: Exception = None):
        super().__init__()
        self.consumed_data = []
        self.completed = False
        self.failed = False
        self.failure_reason = None
        self.ref_id = None
        self._consume_exc = consume_exc

    def consume(self, ref_id: str, state: dict, data: Any) -> dict:
        self.ref_id = ref_id
        if self._consume_exc:
            raise self._consume_exc
        self.consumed_data.append(data)
        return state

    def download_completed(self, ref_id: str):
        self.ref_id = ref_id
        self.completed = True

    def download_failed(self, ref_id: str, reason: str):
        self.ref_id = ref_id
        self.failed = True
        self.failure_reason = reason


def _make_reply(rc: str, status=None, data=None, state=None) -> Message:
    """Build a Message that mimics what cell.send_request returns."""
    payload = {}
    if status is not None:
        payload["status"] = status
    if data is not None:
        payload["data"] = data
    if state is not None:
        payload["state"] = state

    msg = Message()
    msg.set_header(MessageHeaderKey.RETURN_CODE, rc)
    msg.payload = payload
    return msg


class TestDownloadObject:
    """Test suite for the download_object function."""

    @pytest.fixture(autouse=True)
    def _disable_sleep(self, monkeypatch):
        """Disable real sleep to keep unit tests fast."""
        monkeypatch.setattr(download_service.time, "sleep", lambda *_args, **_kwargs: None)

    @pytest.fixture
    def cell(self):
        """Create a mock cell whose send_request we control per-test."""
        return MagicMock()

    @pytest.fixture
    def consumer(self):
        return MockConsumer()

    def test_single_chunk_download(self, cell, consumer):
        """Test download completes after one data chunk then EOF."""
        cell.send_request.side_effect = [
            _make_reply(ReturnCode.OK, status=ProduceRC.OK, data=b"chunk1", state={"start": 0, "count": 1}),
            _make_reply(ReturnCode.OK, status=ProduceRC.EOF),
        ]

        download_object("server.site-1", "ref-001", 10.0, cell, consumer)

        assert consumer.completed
        assert not consumer.failed
        assert consumer.consumed_data == [b"chunk1"]
        assert cell.send_request.call_count == 2

    def test_direct_chunk_uses_initial_capability_and_protected_control(self, cell):
        class DirectConsumer(MockConsumer):
            def get_initial_state(self):
                return {"direct_capability": "v1"}

            def consume_direct_chunk(self, data):
                return bytes(data)

        consumer = DirectConsumer()
        state = {"start": 0, "count": 1, "direct_capability": "v1"}
        direct_reply = Message(
            headers={MessageHeaderKey.RETURN_CODE: ReturnCode.OK, "direct_download": True},
            payload=bytearray(_encode_direct_control(ProduceRC.OK, state) + b"chunk1"),
        )
        cell.send_request.side_effect = [direct_reply, _make_reply(ReturnCode.OK, status=ProduceRC.EOF)]

        download_object("server.site-1", "ref-001", 10.0, cell, consumer)

        assert consumer.completed
        assert consumer.consumed_data == [b"chunk1"]
        first_request = cell.send_request.call_args_list[0].kwargs["request"]
        assert first_request.payload["state"] == {"direct_capability": "v1"}
        second_request = cell.send_request.call_args_list[1].kwargs["request"]
        assert second_request.payload["state"] == state

    def test_direct_chunk_rejects_non_data_status_before_materializing(self, cell):
        class DirectConsumer(MockConsumer):
            def __init__(self):
                super().__init__()
                self.direct_calls = 0

            def consume_direct_chunk(self, data):
                self.direct_calls += 1
                return bytes(data)

        consumer = DirectConsumer()
        direct_reply = Message(
            headers={MessageHeaderKey.RETURN_CODE: ReturnCode.OK, "direct_download": True},
            payload=bytearray(_encode_direct_control(ProduceRC.EOF, {}) + b"unexpected"),
        )
        cell.send_request.return_value = direct_reply

        download_object("server.site-1", "ref-001", 10.0, cell, consumer)

        assert consumer.failed
        assert "invalid direct download status" in consumer.failure_reason
        assert consumer.direct_calls == 0

    def test_multi_chunk_download(self, cell, consumer):
        """Test download with multiple chunks before EOF."""
        cell.send_request.side_effect = [
            _make_reply(ReturnCode.OK, status=ProduceRC.OK, data=b"c1", state={"start": 0, "count": 1}),
            _make_reply(ReturnCode.OK, status=ProduceRC.OK, data=b"c2", state={"start": 1, "count": 1}),
            _make_reply(ReturnCode.OK, status=ProduceRC.OK, data=b"c3", state={"start": 2, "count": 1}),
            _make_reply(ReturnCode.OK, status=ProduceRC.EOF),
        ]

        download_object("server.site-1", "ref-001", 10.0, cell, consumer)

        assert consumer.completed
        assert consumer.consumed_data == [b"c1", b"c2", b"c3"]

    def test_value_equal_new_state_does_not_resubmit(self, cell, consumer):
        """Test a fresh but value-equivalent consumer state keeps the speculative request."""

        def consume_with_fresh_state(ref_id, state, data):
            consumer.consumed_data.append(data)
            return dict(state)

        consumer.consume = consume_with_fresh_state
        cell.send_request.side_effect = [
            _make_reply(ReturnCode.OK, status=ProduceRC.OK, data=b"c1", state={"received_bytes": 2}),
            _make_reply(ReturnCode.OK, status=ProduceRC.EOF),
        ]

        download_object("server.site-1", "ref-001", 10.0, cell, consumer)

        assert consumer.completed
        assert consumer.consumed_data == [b"c1"]
        assert cell.send_request.call_count == 2

    def test_state_transforming_consumer_uses_stop_and_wait(self, cell):
        class TransformingConsumer(MockConsumer):
            supports_pipelining = False

            def consume(self, ref_id, state, data):
                self.consumed_data.append(data)
                return {"start": 99, "count": 1}

        consumer = TransformingConsumer()
        cell.send_request.side_effect = [
            _make_reply(ReturnCode.OK, status=ProduceRC.OK, data=b"c1", state={"start": 0, "count": 1}),
            _make_reply(ReturnCode.OK, status=ProduceRC.EOF),
        ]

        download_object("server.site-1", "ref-001", 10.0, cell, consumer)

        assert consumer.completed
        assert consumer.consumed_data == [b"c1"]
        assert cell.send_request.call_count == 2
        assert cell.send_request.call_args_list[1].kwargs["request"].payload["state"] == {"start": 99, "count": 1}

    def test_pipelined_consumer_cannot_hide_in_place_state_mutation(self, cell, consumer):
        def mutate_state(ref_id, state, data):
            consumer.consumed_data.append(data)
            state["start"] = 99
            return state

        consumer.consume = mutate_state
        cell.send_request.side_effect = [
            _make_reply(ReturnCode.OK, status=ProduceRC.OK, data=b"c1", state={"start": 0, "count": 1}),
            _make_reply(ReturnCode.OK, status=ProduceRC.EOF),
        ]

        download_object("server.site-1", "ref-001", 10.0, cell, consumer)

        assert consumer.failed
        assert "changed state despite enabling download pipelining" in consumer.failure_reason
        assert cell.send_request.call_count <= 2

    def test_progress_callback_reports_start_progress_and_completion(self, cell, consumer):
        """Test download progress callback emits monotonic bytes/items and terminal completion."""
        events = []
        cell.send_request.side_effect = [
            _make_reply(ReturnCode.OK, status=ProduceRC.OK, data=[b"c1", b"c2"], state={"start": 0, "count": 2}),
            _make_reply(ReturnCode.OK, status=ProduceRC.OK, data=[b"c3"], state={"start": 2, "count": 1}),
            _make_reply(ReturnCode.OK, status=ProduceRC.EOF),
        ]

        download_object(
            "server.site-1",
            "ref-001",
            10.0,
            cell,
            consumer,
            progress_cb=lambda **kwargs: events.append(kwargs),
            progress_interval=0.0,
        )

        assert [event["state"] for event in events] == ["start", "active", "active", "completed"]
        assert [event["sequence"] for event in events] == [1, 2, 3, 4]
        assert [event["bytes_done"] for event in events] == [0, 4, 6, 6]
        assert [event["items_done"] for event in events] == [None, 2, 3, 3]

    def test_progress_callback_reports_failure(self, cell, consumer):
        """Test download progress callback emits terminal failure when the producer errors."""
        events = []
        cell.send_request.side_effect = [
            _make_reply(ReturnCode.OK, status=ProduceRC.ERROR),
        ]

        download_object(
            "server.site-1",
            "ref-001",
            10.0,
            cell,
            consumer,
            progress_cb=lambda **kwargs: events.append(kwargs),
            progress_interval=0.0,
        )

        assert consumer.failed
        assert [event["state"] for event in events] == ["start", "failed"]

    def test_immediate_eof(self, cell, consumer):
        """Test producer has nothing to send — EOF on first request."""
        cell.send_request.side_effect = [
            _make_reply(ReturnCode.OK, status=ProduceRC.EOF),
        ]

        download_object("server.site-1", "ref-001", 10.0, cell, consumer)

        assert consumer.completed
        assert consumer.consumed_data == []

    def test_single_timeout_then_recovery(self, cell, consumer):
        """Test one TIMEOUT followed by successful response triggers recovery."""
        cell.send_request.side_effect = [
            _make_reply(ReturnCode.TIMEOUT),
            _make_reply(ReturnCode.OK, status=ProduceRC.OK, data=b"c1", state={"start": 0, "count": 1}),
            _make_reply(ReturnCode.OK, status=ProduceRC.EOF),
        ]

        download_object("server.site-1", "ref-001", 10.0, cell, consumer)

        assert consumer.completed
        assert not consumer.failed
        assert consumer.consumed_data == [b"c1"]
        # 1 timeout + 1 retry success + 1 EOF = 3 calls
        assert cell.send_request.call_count == 3

    def test_multiple_timeouts_then_recovery(self, cell, consumer):
        """Test two consecutive TIMEOUTs then recovery within max_retries=3."""
        cell.send_request.side_effect = [
            _make_reply(ReturnCode.TIMEOUT),
            _make_reply(ReturnCode.TIMEOUT),
            _make_reply(ReturnCode.OK, status=ProduceRC.OK, data=b"c1", state={"start": 0, "count": 1}),
            _make_reply(ReturnCode.OK, status=ProduceRC.EOF),
        ]

        download_object("server.site-1", "ref-001", 10.0, cell, consumer)

        assert consumer.completed
        assert consumer.consumed_data == [b"c1"]

    def test_max_retries_exhausted(self, cell, consumer):
        """Test all retries exhausted causes download failure."""
        cell.send_request.side_effect = [
            _make_reply(ReturnCode.TIMEOUT),
            _make_reply(ReturnCode.TIMEOUT),
            _make_reply(ReturnCode.TIMEOUT),
            _make_reply(ReturnCode.TIMEOUT),
        ]

        download_object("server.site-1", "ref-001", 10.0, cell, consumer, max_retries=3)

        assert consumer.failed
        assert not consumer.completed
        # 1 initial + 3 retries = 4 calls
        assert cell.send_request.call_count == 4

    def test_max_retries_zero_no_retry(self, cell, consumer):
        """Test with max_retries=0, first TIMEOUT fails immediately."""
        cell.send_request.side_effect = [
            _make_reply(ReturnCode.TIMEOUT),
        ]

        download_object("server.site-1", "ref-001", 10.0, cell, consumer, max_retries=0)

        assert consumer.failed
        assert cell.send_request.call_count == 1

    def test_timeout_mid_download_then_recovery(self, cell, consumer):
        """Test timeout after some successful chunks, then recovery."""
        cell.send_request.side_effect = [
            _make_reply(ReturnCode.OK, status=ProduceRC.OK, data=b"c1", state={"start": 0, "count": 1}),
            _make_reply(ReturnCode.OK, status=ProduceRC.OK, data=b"c2", state={"start": 1, "count": 1}),
            _make_reply(ReturnCode.TIMEOUT),
            _make_reply(ReturnCode.OK, status=ProduceRC.OK, data=b"c3", state={"start": 2, "count": 1}),
            _make_reply(ReturnCode.OK, status=ProduceRC.EOF),
        ]

        download_object("server.site-1", "ref-001", 10.0, cell, consumer)

        assert consumer.completed
        assert consumer.consumed_data == [b"c1", b"c2", b"c3"]

    def test_consecutive_timeout_counter_resets_after_success(self, cell, consumer):
        """Test retry counter resets after successful recovery, allowing future retries."""
        cell.send_request.side_effect = [
            _make_reply(ReturnCode.OK, status=ProduceRC.OK, data=b"c1", state={"start": 0, "count": 1}),
            _make_reply(ReturnCode.TIMEOUT),  # timeout 1/2
            _make_reply(ReturnCode.TIMEOUT),  # timeout 2/2
            _make_reply(ReturnCode.OK, status=ProduceRC.OK, data=b"c2", state={"start": 1, "count": 1}),
            _make_reply(ReturnCode.TIMEOUT),  # new timeout 1/2 (counter reset)
            _make_reply(ReturnCode.OK, status=ProduceRC.OK, data=b"c3", state={"start": 2, "count": 1}),
            _make_reply(ReturnCode.OK, status=ProduceRC.EOF),
        ]

        download_object("server.site-1", "ref-001", 10.0, cell, consumer, max_retries=2)

        assert consumer.completed
        assert consumer.consumed_data == [b"c1", b"c2", b"c3"]

    def test_retry_resends_same_state(self, cell, consumer):
        """Test retry resends the same state so producer re-generates the same chunk.

        The pipelined implementation speculatively launches the next request as soon
        as a good reply arrives (before consume).  For ItemConsumer, consume() returns
        the producer's `state` unchanged, so the speculative request carries the
        correct state and the TIMEOUT+retry pair both carry that same state.
        """
        next_state = {"start": 0, "count": 1}
        cell.send_request.side_effect = [
            _make_reply(ReturnCode.OK, status=ProduceRC.OK, data=b"c1", state=next_state),
            _make_reply(ReturnCode.TIMEOUT),  # speculative req for chunk 2 times out
            _make_reply(ReturnCode.OK, status=ProduceRC.OK, data=b"c2", state={"start": 1, "count": 1}),
            _make_reply(ReturnCode.OK, status=ProduceRC.EOF),
        ]

        download_object("server.site-1", "ref-001", 10.0, cell, consumer)

        assert consumer.completed
        assert consumer.consumed_data == [b"c1", b"c2"]
        calls = cell.send_request.call_args_list
        assert len(calls) == 4

        # calls[0]: initial request (no state)
        # calls[1]: speculative request for chunk 2 — got TIMEOUT
        # calls[2]: retry of calls[1] — must carry the SAME state as calls[1]
        # calls[3]: speculative request for chunk 3 (returns EOF)
        payload_timed_out = calls[1].kwargs["request"].payload
        payload_retry = calls[2].kwargs["request"].payload
        # Core contract: the timed-out request and its retry carry identical state
        assert payload_timed_out.get("state") == payload_retry.get("state") == next_state

    def test_non_timeout_error_fails_immediately(self, cell, consumer):
        """Test non-TIMEOUT errors are not retried."""
        cell.send_request.side_effect = [
            _make_reply(ReturnCode.COMM_ERROR),
        ]

        download_object("server.site-1", "ref-001", 10.0, cell, consumer)

        assert consumer.failed
        assert not consumer.completed
        assert cell.send_request.call_count == 1

    def test_producer_error(self, cell, consumer):
        """Test ProduceRC.ERROR from producer causes download failure."""
        cell.send_request.side_effect = [
            _make_reply(ReturnCode.OK, status=ProduceRC.ERROR),
        ]

        download_object("server.site-1", "ref-001", 10.0, cell, consumer)

        assert consumer.failed
        assert "producer error" in consumer.failure_reason

    def test_abort_signal(self, cell, consumer):
        """Test abort signal causes download failure."""
        signal = Signal()
        signal.trigger("test abort")

        cell.send_request.side_effect = [
            _make_reply(ReturnCode.OK, status=ProduceRC.OK, data=b"c1", state={"start": 0, "count": 1}),
        ]

        download_object("server.site-1", "ref-001", 10.0, cell, consumer, abort_signal=signal)

        assert consumer.failed
        assert "aborted" in consumer.failure_reason

    def test_abort_after_consume(self, cell, consumer):
        """Test abort signal triggered after consuming a chunk causes download failure."""
        signal = Signal()

        # Wrap consume to trigger abort after processing the first chunk
        original_consume = consumer.consume

        def consume_and_abort(ref_id, state, data):
            result = original_consume(ref_id, state, data)
            signal.trigger("abort after consume")
            return result

        consumer.consume = consume_and_abort

        cell.send_request.side_effect = [
            _make_reply(ReturnCode.OK, status=ProduceRC.OK, data=b"c1", state={"start": 0, "count": 1}),
        ]

        download_object("server.site-1", "ref-001", 10.0, cell, consumer, abort_signal=signal)

        assert consumer.failed
        assert "aborted" in consumer.failure_reason

    def test_consumer_exception(self, cell):
        """Test exception in consumer.consume causes download failure."""
        consumer = MockConsumer(consume_exc=ValueError("bad data"))

        cell.send_request.side_effect = [
            _make_reply(ReturnCode.OK, status=ProduceRC.OK, data=b"c1", state={"start": 0, "count": 1}),
        ]

        download_object("server.site-1", "ref-001", 10.0, cell, consumer)

        assert consumer.failed
        assert "exception" in consumer.failure_reason

    def test_consumer_returns_non_dict(self, cell):
        """Test consumer returning non-dict state causes download failure."""
        consumer = MockConsumer()
        consumer.consume = lambda ref_id, state, data: "not_a_dict"

        cell.send_request.side_effect = [
            _make_reply(ReturnCode.OK, status=ProduceRC.OK, data=b"c1", state={"start": 0, "count": 1}),
        ]

        download_object("server.site-1", "ref-001", 10.0, cell, consumer)

        assert consumer.failed
        assert "dict" in consumer.failure_reason

    def test_negative_max_retries_raises(self, cell, consumer):
        """Test that negative max_retries raises ValueError."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            download_object("server.site-1", "ref-001", 10.0, cell, consumer, max_retries=-1)
