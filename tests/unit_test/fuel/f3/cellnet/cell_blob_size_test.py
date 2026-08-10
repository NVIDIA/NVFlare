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

from types import SimpleNamespace

import pytest

from nvflare.fuel.f3.cellnet import cell as cell_module
from nvflare.fuel.f3.cellnet.cell import Adapter
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.stream_types import BlobSizeError, StreamError, StreamFuture


def _oversized_response_cell():
    def send_blob(*args, **kwargs):
        raise BlobSizeError("Blob size 5 exceeds configured limit 4 (streaming_max_blob_size)")

    return SimpleNamespace(send_blob=send_blob, get_fobs_context=lambda: {})


def _async_response_cell(reply_future):
    return SimpleNamespace(send_blob=lambda *args, **kwargs: reply_future, get_fobs_context=lambda: {})


def test_oversized_server_job_response_exits_job_process(monkeypatch):
    adapter = Adapter(None, SimpleNamespace(fqcn="server.job-id"), _oversized_response_cell())
    monkeypatch.setattr(cell_module, "encode_payload", lambda *args, **kwargs: None)
    monkeypatch.setattr(cell_module.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))

    with pytest.raises(SystemExit) as exc_info:
        adapter._send_response(
            Message(payload=b"payload"), "stream-id", "request-id", "channel", "topic", "client", False, False
        )

    assert exc_info.value.code == 1


def test_oversized_non_server_job_response_propagates_error(monkeypatch):
    adapter = Adapter(None, SimpleNamespace(fqcn="site-1.job-id"), _oversized_response_cell())
    monkeypatch.setattr(cell_module, "encode_payload", lambda *args, **kwargs: None)

    with pytest.raises(BlobSizeError, match=r"limit 4 \(streaming_max_blob_size\)"):
        adapter._send_response(
            Message(payload=b"payload"), "stream-id", "request-id", "channel", "topic", "peer", False, False
        )


def test_asymmetric_receiver_limit_exits_server_job_after_async_rejection(monkeypatch):
    reply_future = StreamFuture(stream_id=1)
    adapter = Adapter(None, SimpleNamespace(fqcn="server.job-id"), _async_response_cell(reply_future))
    monkeypatch.setattr(cell_module, "encode_payload", lambda *args, **kwargs: None)
    monkeypatch.setattr(cell_module.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))

    adapter._send_response(
        Message(payload=b"payload"), "stream-id", "request-id", "channel", "topic", "client", False, False
    )

    with pytest.raises(SystemExit) as exc_info:
        reply_future.set_exception(BlobSizeError("receiver limit 4 is smaller than response size 5"))

    assert exc_info.value.code == 1


def test_async_stream_error_does_not_exit_non_server_job(monkeypatch):
    reply_future = StreamFuture(stream_id=2)
    adapter = Adapter(None, SimpleNamespace(fqcn="site-1.job-id"), _async_response_cell(reply_future))
    monkeypatch.setattr(cell_module, "encode_payload", lambda *args, **kwargs: None)
    exit_calls = []
    monkeypatch.setattr(cell_module.os, "_exit", exit_calls.append)

    adapter._send_response(
        Message(payload=b"payload"), "stream-id", "request-id", "channel", "topic", "peer", False, False
    )
    reply_future.set_exception(StreamError("receiver stopped stream"))

    assert exit_calls == []
