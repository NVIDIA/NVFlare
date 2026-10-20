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

"""Regression checks for external-trainer accounting without network services.

These tests are controlled checks, not external-process qualification.
Serialization, counter state, and DownloadService completion
are real; transport acceptance is driven explicitly by the test.
"""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from nvflare.apis.fl_constant import ServerCommandNames
from nvflare.app_common.decomposers.numpy_decomposers import NumpyArrayDecomposer
from nvflare.fuel.f3.cellnet.cell import Adapter, Cell
from nvflare.fuel.f3.cellnet.defs import CellChannel, MessagePropKey
from nvflare.fuel.f3.cellnet.fqcn import FqcnInfo
from nvflare.fuel.f3.cellnet.utils import buffer_len, decode_payload
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.send_accounting import SOURCE_CONTRIBUTION_CONTEXT_KEY, SourceContributionAccumulator
from nvflare.fuel.f3.streaming import cacheable, obj_downloader
from nvflare.fuel.f3.streaming.stream_const import StreamHeaderKey
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.decomposers.via_downloader import LazyDownloadRef, LazyDownloadRefDecomposer
from nvflare.private.fed.resource_stats.external_transfer import (
    ExternalTransferManager,
    make_authenticated_consumer_resolver,
)
from nvflare.private.fed.resource_stats.f3_bindings import attach_f3_context
from nvflare.private.fed.resource_stats.f3_counter import F3Counter, F3TrafficClass
from tests.unit_test.fuel.f3.streaming.download_test_utils import make_service_no_monitor, pull_request

_SERVER = "server.job-1"
_CLIENT = "site-1.job-1"
_TRAINER = "site-1.job-1.trainer"


def _cell(fqcn):
    cell = Cell.__new__(Cell)
    context = {fobs.FOBSContextKey.CELL: cell}
    cell.core_cell = SimpleNamespace(get_fobs_context=lambda: dict(context))
    cell.my_info = FqcnInfo(fqcn)
    cell.get_fqcn = lambda: fqcn
    cell.logger = logging.getLogger(__name__)
    cell.register_request_cb = Mock()
    cell.send_blob = Mock(return_value=Mock())
    cell._send_one_request = Mock()
    return cell


@pytest.fixture
def downloads(monkeypatch):
    fobs.register(NumpyArrayDecomposer)
    fobs.register(LazyDownloadRefDecomposer)
    service = make_service_no_monitor()
    monkeypatch.setattr(obj_downloader, "DownloadService", service)
    monkeypatch.setattr(cacheable, "DownloadService", service)
    yield service
    for transaction_id in list(service._tx_table):
        service.delete_transaction(transaction_id)


def _serve(service, ref_id, requester, source=None, prepare_request=None):
    """Serve the real downloadable to EOF using the legacy completion protocol."""
    state = None
    payload_bytes = 0
    transaction = service._ref_table[ref_id].tx
    waiter = service.get_transfer_waiter(transaction.tid) if source is not None else None
    for _ in range(20):
        request = pull_request(ref_id, requester, state=state)
        if prepare_request is not None:
            prepare_request(request)
        reply = service._handle_download(request)
        assert reply.payload["status"] in {"ok", "eof"}
        if reply.payload["status"] == "eof":
            if source is not None:
                # The production monitor settles legacy EOF only after its reply
                # leaves the callback. This isolated service has no monitor.
                service._finish_transaction_if_complete(transaction)
                # Settlement runs on the existing callback pool. Observe its
                # actual outcome instead of racing its accounting callback.
                assert waiter.wait(timeout=2.0) is not None
            return payload_bytes
        payload_bytes += buffer_len(reply.payload["data"])
        if source is not None:
            attempt = reply.get_logical_send_context().try_begin(source, requester)
            assert attempt is not None
            attempt.accepted(buffer_len(reply.payload["data"]))
        state = reply.payload["state"]
    raise AssertionError("small download did not reach EOF")


@pytest.mark.parametrize("accounted", [False, True])
def test_task_response_accounting_preserves_download_completion_for_a_forwarding_client(downloads, accounted):
    server = _cell(_SERVER)
    response = Message(payload={"weights": np.arange(512, dtype=np.float32)})
    counter = F3Counter()
    if accounted:
        assert attach_f3_context(response, counter, F3TrafficClass.TASK_RESPONSE, recipient_name="site-1")

    Adapter(Mock(), server.my_info, server)._send_response(
        response, "stream-request", "request", CellChannel.SERVER_COMMAND, ServerCommandNames.GET_TASK, _CLIENT, False
    )

    assert len(downloads._tx_table) == 1
    transaction = next(iter(downloads._tx_table.values()))
    assert transaction.receiver_ids is None
    ref_id = next(iter(downloads._ref_table))
    if accounted:
        attempt = response.get_logical_send_context().try_begin(_SERVER, _CLIENT)
        assert attempt is not None
        attempt.accepted(buffer_len(response.payload))

    # The external backend locally enables GET_TASK pass-through. Its trainer,
    # rather than the first-hop client job, downloads the server's source data.
    assert _serve(downloads, ref_id, _TRAINER) > 0
    assert transaction.is_finished()
    if accounted:
        assert counter.freeze()["status"] == "partial"


def test_authenticated_task_source_joins_bytes_without_changing_legacy_completion(downloads):
    server = _cell(_SERVER)
    response = Message(payload={"weights": np.arange(512, dtype=np.float32)})
    counter = F3Counter()
    assert attach_f3_context(
        response,
        counter,
        F3TrafficClass.TASK_RESPONSE,
        recipient_name="site-1",
        consumer_resolver=make_authenticated_consumer_resolver(_CLIENT, "site-1"),
    )
    Adapter(Mock(), server.my_info, server)._send_response(
        response, "stream-request", "request", CellChannel.SERVER_COMMAND, ServerCommandNames.GET_TASK, _CLIENT, False
    )
    transaction = next(iter(downloads._tx_table.values()))
    assert transaction.receiver_ids is None
    ref_id = next(iter(downloads._ref_table))
    envelope_bytes = buffer_len(response.payload)
    response.get_logical_send_context().try_begin(_SERVER, _CLIENT).accepted(envelope_bytes)

    def direct_proof(request):
        request.set_prop(MessagePropKey.ENDPOINT, SimpleNamespace(name=_TRAINER))
        request.set_prop(MessagePropKey.COMMON_NAME, "site-1")

    source_bytes = _serve(downloads, ref_id, _TRAINER, source=_SERVER, prepare_request=direct_proof)
    assert transaction.is_finished()
    assert counter.freeze() == {
        "sent_to": [
            {"participant_name": "site-1", "payload_bytes": str(envelope_bytes + source_bytes), "messages": "1"}
        ]
    }


def test_unmanaged_forwarded_external_result_marks_missing_source_bytes_partial(downloads):
    trainer = _cell(_TRAINER)
    result = Message(payload={"weights": np.arange(512, dtype=np.float32)})
    trainer._encode_message(result, None, receiver_ids=(_SERVER,))
    transaction = next(iter(downloads._tx_table.values()))
    ref_id = next(iter(downloads._ref_table))
    assert transaction.receiver_ids == (_SERVER,)
    assert transaction.send_accounting_context is None

    client = _cell(_CLIENT)
    decode_payload(
        result,
        StreamHeaderKey.PAYLOAD_ENCODING,
        fobs_ctx=client.get_fobs_context({fobs.FOBSContextKey.PASS_THROUGH: True}),
    )
    assert isinstance(result.payload["weights"], LazyDownloadRef)
    assert result.payload["weights"].fqcn == _TRAINER

    forwarded = Message(payload=result.payload)
    counter = F3Counter()
    assert attach_f3_context(forwarded, counter, F3TrafficClass.TASK_RESULT, recipient_name="server")
    client._encode_message(forwarded, None, receiver_ids=(_SERVER,))
    assert len(downloads._tx_table) == 1  # Re-emitted references create no client-owned transaction.
    envelope_bytes = buffer_len(forwarded.payload)
    context = forwarded.get_logical_send_context()
    attempt = context.try_begin(_CLIENT, _SERVER)
    assert attempt is not None
    attempt.accepted(envelope_bytes)

    source_bytes = _serve(downloads, ref_id, _SERVER)
    assert source_bytes > envelope_bytes
    assert transaction.is_finished()
    assert counter.freeze() == {
        "status": "partial",
        "issues": ["counter_gap"],
        "sent_to": [{"participant_name": "server", "payload_bytes": str(envelope_bytes), "messages": "1"}],
    }


def test_explicit_gap_marker_keeps_external_result_subtotal_but_removes_complete_claim(downloads):
    # The low-risk interim safeguard can retain known envelope bytes, while
    # honestly declaring that direct-source tensor bytes are not yet attributed.
    counter = F3Counter()
    message = Message(payload=b"forwarded-envelope")
    assert attach_f3_context(message, counter, F3TrafficClass.TASK_RESULT, recipient_name="server")
    attempt = message.get_logical_send_context().try_begin(_CLIENT, _SERVER)
    attempt.accepted(len(message.payload))
    assert counter.mark_counter_gap()

    assert counter.freeze() == {
        "status": "partial",
        "issues": ["counter_gap"],
        "sent_to": [{"participant_name": "server", "payload_bytes": str(len(message.payload)), "messages": "1"}],
    }


def test_managed_forwarded_result_joins_real_serialized_source_bytes(downloads):
    trainer = _cell(_TRAINER)
    source = SourceContributionAccumulator(_TRAINER, (_SERVER,))
    result = Message(payload={"weights": np.arange(512, dtype=np.float32)})
    trainer._encode_message(
        result, None, receiver_ids=(_SERVER,), fobs_ctx_props={SOURCE_CONTRIBUTION_CONTEXT_KEY: source}
    )
    ref_id = next(iter(downloads._ref_table))
    client = _cell(_CLIENT)
    decode_payload(
        result,
        StreamHeaderKey.PAYLOAD_ENCODING,
        fobs_ctx=client.get_fobs_context({fobs.FOBSContextKey.PASS_THROUGH: True}),
    )
    counter = F3Counter()
    manager = ExternalTransferManager(counter)
    manager.register_result_source("session", "task", _TRAINER, (ref_id,), (_SERVER,), True)
    forwarded = Message(payload=result.payload)
    attach_f3_context(forwarded, counter, F3TrafficClass.TASK_RESULT, recipient_name="server")
    context = forwarded.get_logical_send_context()
    manager.bind_context(context, _SERVER)
    context.prepare_destination(_CLIENT, _SERVER)
    client._encode_message(forwarded, None)
    envelope_bytes = buffer_len(forwarded.payload)
    context.try_begin(_CLIENT, _SERVER).accepted(envelope_bytes)
    assert counter.pending_count == 1
    served = _serve(downloads, ref_id, _SERVER, source=_TRAINER)
    block = source.snapshot()
    assert block["groups"][0]["complete"]
    assert block["groups"][0]["accepted_payload_bytes"] == str(served)
    assert manager.accept_summary("session", "task", _TRAINER, block)
    assert counter.freeze() == {
        "sent_to": [{"participant_name": "server", "payload_bytes": str(envelope_bytes + served), "messages": "1"}]
    }
