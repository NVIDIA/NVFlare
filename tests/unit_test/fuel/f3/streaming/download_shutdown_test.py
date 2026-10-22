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

"""Local download-service shutdown must remain distinct from source failure."""

from unittest.mock import Mock

import pytest

from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply, new_cell_message
from nvflare.fuel.f3.streaming import download_service
from nvflare.fuel.f3.streaming.download_service import ProduceRC, _SourceFailureKey, download_object
from nvflare.fuel.f3.streaming.stream_types import DownloadCancelled
from tests.unit_test.fuel.f3.streaming.download_test_utils import make_service_no_monitor


@pytest.fixture
def service(monkeypatch):
    service = make_service_no_monitor()
    monkeypatch.setattr(download_service, "DownloadService", service)
    try:
        yield service
    finally:
        service.shutdown()


@pytest.mark.parametrize("shutdown_at", ["registration", "request", "consume"])
def test_shutdown_cancels_download_and_notifies_consumer(service, monkeypatch, shutdown_at):
    cell = Mock()
    consumer = Mock(supports_pipelining=False)
    consumer.consume.return_value = {}
    progress = []
    if shutdown_at == "registration":
        register = service._register_source_download

        def register_then_shutdown(*args, **kwargs):
            signal = register(*args, **kwargs)
            service.shutdown()
            return signal

        monkeypatch.setattr(service, "_register_source_download", register_then_shutdown)
    elif shutdown_at == "request":

        def request_then_shutdown(**kwargs):
            assert not kwargs["abort_signal"].triggered
            service.shutdown()
            return make_reply(ReturnCode.TIMEOUT)

        cell.send_request.side_effect = request_then_shutdown
    else:
        cell.send_request.return_value = new_cell_message(
            headers={MessageHeaderKey.RETURN_CODE: ReturnCode.OK},
            payload={"status": ProduceRC.OK, "data": b"chunk", "state": {}},
        )

        def consume_then_shutdown(*args):
            service.shutdown()
            return {}

        consumer.consume.side_effect = consume_then_shutdown

    with pytest.raises(DownloadCancelled, match="download service shut down"):
        download_object(
            "site-3.job-1.trainer", "ref-1", 10.0, cell, consumer, progress_cb=lambda **p: progress.append(p)
        )

    consumer.download_failed.assert_called_once_with("ref-1", "download service shut down")
    consumer.download_completed.assert_not_called()
    assert not service._active_source_downloads
    assert cell.send_request.call_count == (0 if shutdown_at == "registration" else 1)
    assert [p["state"] for p in progress] == ([] if shutdown_at == "registration" else ["start", "aborted"])


@pytest.mark.parametrize("reason", ["trainer exited rc=-9", "download service shut down"])
@pytest.mark.parametrize("shutdown_after_failure", [False, True])
def test_source_failure_is_not_reclassified_as_local_cancellation(service, reason, shutdown_after_failure):
    cell = Mock()
    consumer = Mock(supports_pipelining=False)

    def request_then_fail(**kwargs):
        response = service._handle_source_failure(
            new_cell_message(
                headers={MessageHeaderKey.ORIGIN: "site-3.job-1"},
                payload={
                    _SourceFailureKey.SOURCE_FQCN: "site-3.job-1.trainer",
                    _SourceFailureKey.REF_IDS: ["ref-1"],
                    _SourceFailureKey.REASON: reason,
                },
            )
        )
        assert response.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
        if shutdown_after_failure:
            service.shutdown()
        return make_reply(ReturnCode.TIMEOUT)

    cell.send_request.side_effect = request_then_fail

    download_object("site-3.job-1.trainer", "ref-1", 10.0, cell, consumer)

    consumer.download_failed.assert_called_once_with("ref-1", reason)
    consumer.download_completed.assert_not_called()
    assert cell.send_request.call_count == 1
    assert not service._active_source_downloads


def test_shutdown_does_not_replace_source_failure_registered_from_tombstone(service, monkeypatch):
    cell = Mock()
    consumer = Mock(supports_pipelining=False)
    service._handle_source_failure(
        new_cell_message(
            headers={MessageHeaderKey.ORIGIN: "site-3.job-1"},
            payload={
                _SourceFailureKey.SOURCE_FQCN: "site-3.job-1.trainer",
                _SourceFailureKey.REF_IDS: ["ref-1"],
                _SourceFailureKey.REASON: "trainer exited rc=-9",
            },
        )
    )
    register = service._register_source_download

    def register_then_shutdown(*args, **kwargs):
        signal = register(*args, **kwargs)
        service.shutdown()
        return signal

    monkeypatch.setattr(service, "_register_source_download", register_then_shutdown)

    download_object("site-3.job-1.trainer", "ref-1", 10.0, cell, consumer)

    consumer.download_failed.assert_called_once_with("ref-1", "trainer exited rc=-9")
    consumer.download_completed.assert_not_called()
    cell.send_request.assert_not_called()
    assert not service._active_source_downloads
