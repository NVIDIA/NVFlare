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

from nvflare.apis.signal import Signal
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message
import nvflare.fuel.f3.streaming.download_service as download_service
from nvflare.fuel.f3.streaming.download_service import (
    Consumer,
    ProduceRC,
    download_object,
)


class MockConsumer(Consumer):
    """Mock consumer for testing."""

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
        """Test retry resends the same state so producer re-generates the same chunk."""
        cell.send_request.side_effect = [
            _make_reply(ReturnCode.OK, status=ProduceRC.OK, data=b"c1", state={"start": 0, "count": 1}),
            _make_reply(ReturnCode.TIMEOUT),
            _make_reply(ReturnCode.OK, status=ProduceRC.OK, data=b"c2", state={"start": 1, "count": 1}),
            _make_reply(ReturnCode.OK, status=ProduceRC.EOF),
        ]

        download_object("server.site-1", "ref-001", 10.0, cell, consumer)

        assert consumer.completed
        calls = cell.send_request.call_args_list

        # calls[0]: initial request (no state)
        # calls[1]: request after consuming c1, carries state from c1 (got TIMEOUT)
        # calls[2]: retry of calls[1], should carry the SAME state
        payload_before_timeout = calls[1].kwargs["request"].payload
        payload_retry = calls[2].kwargs["request"].payload
        assert payload_before_timeout.get("state") == payload_retry.get("state")
        # Verify the retried state matches the state returned by c1
        assert payload_retry.get("state") == {"start": 0, "count": 1}

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
