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
from unittest.mock import Mock

import pytest

from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.streaming.download_service import Consumer, DownloadStatus, ProduceRC, download_object
from nvflare.fuel.f3.streaming.stream_types import DownloadUnavailable
from tests.unit_test.fuel.f3.streaming.download_test_utils import (
    MockDownloadable,
    cancel_request,
    confirm_request,
    make_service_no_monitor,
    pull_request,
    run_monitor_once,
    serve_nonce,
)


def _source():
    service = make_service_no_monitor()
    tx_id = service.new_transaction(cell=Mock(), timeout=300.0, num_receivers=1)
    tx = service._tx_table[tx_id]
    ref = tx.add_object(MockDownloadable([b"a", b"b", b"c"]))
    service._ref_table[ref.rid] = ref
    return service, tx, ref


def _finish(service, ref, *, confirm, retire):
    state = None
    chunks = []
    while True:
        reply = service._handle_download(pull_request(ref.rid, "receiver", confirm_capable=confirm, state=state))
        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
        if reply.payload["status"] == ProduceRC.EOF:
            break
        chunks.append(reply.payload["data"])
        state = reply.payload["state"]
    if confirm:
        # Synchronous settlement keeps the test independent of worker scheduling.
        service._submit_finished_settlement = service._settle_finished_transaction
        if not retire:
            service._finish_transaction_if_complete = Mock(return_value=False)
        service._handle_download(confirm_request(ref.rid, "receiver", DownloadStatus.SUCCESS, serve_nonce(reply)))
    elif retire:
        run_monitor_once(service, time.time())
    assert chunks == [b"a", b"b", b"c"]
    assert (ref.rid not in service._ref_table) is retire
    return state


@pytest.mark.parametrize("confirm", [False, True])
@pytest.mark.parametrize("retire", [False, True])
@pytest.mark.parametrize("state", [None, {"chunk_idx": 1}])
def test_completed_receiver_does_not_certify_an_incomplete_overlapping_download(confirm, retire, state):
    service, tx, ref = _source()
    # A retry starts before the first attempt reaches EOF, with the same receiver identity.
    partial = service._handle_download(pull_request(ref.rid, "receiver"))
    assert partial.payload["data"] == b"a"
    terminal_state = _finish(service, ref, confirm=confirm, retire=retire)

    reply = service._handle_download(pull_request(ref.rid, "receiver", state=state))
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.SERVICE_UNAVAILABLE
    assert reply.payload == {"retryable": True}
    # A lost EOF reply, in contrast, remains safely replayable after retirement.
    eof = service._handle_download(pull_request(ref.rid, "receiver", state=terminal_state))
    assert eof.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    assert eof.payload == {"status": ProduceRC.EOF}
    assert ref.snapshot_receiver_statuses() == {"receiver": DownloadStatus.SUCCESS}
    assert ref.obj.downloaded_to_one_calls == [("receiver", DownloadStatus.SUCCESS)]
    assert tx.is_finished()


def test_downloader_cleans_up_duplicate_without_cancelling_the_successful_receiver():
    service, tx, ref = _source()
    cell = Mock()

    def send_request(**kwargs):
        request = kwargs["request"]
        request.set_header(MessageHeaderKey.ORIGIN, "receiver")
        return service._handle_download(request)

    cell.send_request.side_effect = send_request

    class PartialConsumer(Consumer):
        def __init__(self):
            super().__init__()
            self.completed = False
            self.failure = None
            self.partial = []

        def consume(self, ref_id, state, data):
            self.partial.append(data)
            _finish(service, ref, confirm=True, retire=True)
            return state

        def download_completed(self, ref_id):
            self.completed = True

        def download_failed(self, ref_id, reason):
            self.failure = reason
            self.partial.clear()

    consumer = PartialConsumer()
    with pytest.raises(DownloadUnavailable, match="another attempt"):
        download_object("producer", ref.rid, 5.0, cell, consumer)
    assert consumer.failure is not None
    assert consumer.partial == []
    assert not consumer.completed
    cell.fire_and_forget.assert_not_called()
    assert tx._settlement_complete
    assert ref.obj.released


def test_terminal_request_is_copied_before_producer_mutates_it():
    service, _, ref = _source()
    original = ref.obj.produce

    def mutating_produce(state, requester):
        result = original(state, requester)
        if state is not None:
            state["chunk_idx"] = -1
        return result

    ref.obj.produce = mutating_produce
    _finish(service, ref, confirm=False, retire=True)
    eof = service._handle_download(pull_request(ref.rid, "receiver", state={"chunk_idx": 3}))
    assert eof.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    assert eof.payload == {"status": ProduceRC.EOF}


@pytest.mark.parametrize("num_receivers", [0, 2])
def test_live_source_with_receivers_remaining_can_still_serve_a_second_download(num_receivers):
    service, tx, ref = _source()
    tx.num_receivers = num_receivers
    for _ in range(2):
        state = None
        chunks = []
        while True:
            reply = service._handle_download(pull_request(ref.rid, "receiver", state=state))
            assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
            if reply.payload["status"] == ProduceRC.EOF:
                break
            chunks.append(reply.payload["data"])
            state = reply.payload["state"]
        assert chunks == [b"a", b"b", b"c"]
    assert not tx.is_finished()
    assert ref.obj.downloaded_to_one_calls == [("receiver", DownloadStatus.SUCCESS)]


@pytest.mark.parametrize("raise_after_completion", [False, True])
def test_other_attempt_completing_during_produce_is_recoverable(raise_after_completion):
    service, _, ref = _source()
    original = ref.obj.produce
    completed = False

    def racing_produce(state, requester):
        nonlocal completed
        result = original(state, requester)
        if not completed:
            completed = True
            _finish(service, ref, confirm=True, retire=False)
            if raise_after_completion:
                raise RuntimeError("source removed by completion callback")
        return result

    ref.obj.produce = racing_produce
    reply = service._handle_download(pull_request(ref.rid, "receiver"))
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.SERVICE_UNAVAILABLE
    assert reply.payload == {"retryable": True}


def test_unknown_and_expired_refs_still_fail_as_invalid_requests():
    service, _, ref = _source()
    _finish(service, ref, confirm=False, retire=True)
    unknown = service._handle_download(pull_request(ref.rid, "other-receiver"))
    assert unknown.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
    service._finished_refs[ref.rid].last_active_time = time.time() - service.FINISHED_REFS_TTL - 1
    expired = service._handle_download(pull_request(ref.rid, "receiver"))
    assert expired.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST


@pytest.mark.parametrize("confirm", [False, True])
def test_legacy_duplicate_cancel_cannot_fail_an_unfinished_sibling(confirm):
    service, tx, first = _source()
    sibling = tx.add_object(MockDownloadable([b"later"]))
    service._ref_table[sibling.rid] = sibling
    state = None
    for _ in range(4):
        terminal = service._handle_download(pull_request(first.rid, "receiver", confirm_capable=confirm, state=state))
        state = terminal.payload.get("state")
    assert terminal.payload["status"] == ProduceRC.EOF
    if confirm:
        service._handle_download(confirm_request(first.rid, "receiver", DownloadStatus.SUCCESS, serve_nonce(terminal)))
    assert first.snapshot_receiver_statuses() == {"receiver": DownloadStatus.SUCCESS}

    data = service._handle_download(pull_request(sibling.rid, "receiver", confirm_capable=confirm))
    assert data.payload["data"] == b"later"
    rejected = service._handle_download(pull_request(first.rid, "receiver"))
    assert rejected.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.SERVICE_UNAVAILABLE
    # A pre-fix receiver treats this response as a generic failure and sends CANCEL.
    service._handle_download(cancel_request(first.rid, "receiver"))
    assert sibling.snapshot_receiver_statuses() == {}
    assert tx.tid in service._tx_table

    terminal = service._handle_download(
        pull_request(sibling.rid, "receiver", confirm_capable=confirm, state=data.payload["state"])
    )
    assert terminal.payload["status"] == ProduceRC.EOF
    if confirm:
        service._submit_finished_settlement = service._settle_finished_transaction
        service._handle_download(
            confirm_request(sibling.rid, "receiver", DownloadStatus.SUCCESS, serve_nonce(terminal))
        )
    else:
        run_monitor_once(service, time.time())
    assert service.get_transaction_outcome(tx.tid).completed
