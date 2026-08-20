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

import logging
import multiprocessing
import time
import traceback

import pytest

from nvflare.fuel.f3.cellnet.cell import Adapter
from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.stream_cell import StreamCell
from nvflare.fuel.f3.streaming.stream_const import STREAM_CHANNEL, STREAM_DATA_TOPIC, StreamHeaderKey
from nvflare.fuel.f3.streaming.stream_types import StreamTargetUnreachable
from nvflare.fuel.utils.network_utils import get_open_ports

_CONNECT_TIMEOUT = 10.0
_CHANNEL = "routed_stream_error_test"
_TOPIC = "late_reply"
_MISSING_RECEIVER = "site-2.dead-job"


class _ErrorCapture(logging.Handler):
    def __init__(self):
        super().__init__(logging.ERROR)
        self.messages = []

    def emit(self, record):
        self.messages.append(self.format(record))


def _wait_for_connection(cell, peer_fqcn):
    deadline = time.time() + _CONNECT_TIMEOUT
    while time.time() < deadline:
        if cell.is_cell_connected(peer_fqcn):
            return
        time.sleep(0.05)
    raise RuntimeError(f"{cell.get_fqcn()} did not connect to {peer_fqcn}")


def _wait_for_peer_state(cell, peer_fqcn, connected):
    deadline = time.time() + _CONNECT_TIMEOUT
    while time.time() < deadline:
        if cell.is_cell_connected(peer_fqcn) is connected:
            return
        time.sleep(0.05)
    state = "connect" if connected else "disconnect"
    raise RuntimeError(f"{peer_fqcn} did not {state} from {cell.get_fqcn()}")


def _run_server(root_url, status_queue, stop_event):
    cell = None
    capture = _ErrorCapture()
    logging.getLogger().addHandler(capture)
    try:
        cell = CoreCell("server", root_url, secure=False, credentials={})
        StreamCell(cell)
        cell.start()
        status_queue.put({"ready": True})
        _wait_for_peer_state(cell, _MISSING_RECEIVER, connected=True)
        status_queue.put({"receiver_connected": True})
        _wait_for_peer_state(cell, _MISSING_RECEIVER, connected=False)
        status_queue.put({"receiver_removed": True})
        stop_event.wait(30)
    except Exception:
        status_queue.put({"error": traceback.format_exc()})
        raise
    finally:
        if cell:
            cell.stop()
        status_queue.put({"errors": capture.messages})


def _run_receiver(root_url, status_queue, stop_event):
    cell = None
    try:
        cell = CoreCell(_MISSING_RECEIVER, root_url, secure=False, credentials={})
        cell.start()
        _wait_for_connection(cell, "server")
        status_queue.put({"ready": True})
        stop_event.wait(30)
    except Exception:
        status_queue.put({"error": traceback.format_exc()})
        raise
    finally:
        if cell:
            cell.stop()


def _run_sender(root_url, result_queue):
    cell = None
    capture = _ErrorCapture()
    logging.getLogger().addHandler(capture)
    try:
        cell = CoreCell("site-1", root_url, secure=False, credentials={})
        stream_cell = StreamCell(cell)
        cell.start()
        _wait_for_connection(cell, "server")

        sends = []
        original_send = cell._send_to_endpoint

        def record_send(endpoint, message):
            result = original_send(endpoint, message)
            if message.get_header(StreamHeaderKey.TOPIC) == _TOPIC:
                sends.append(
                    {
                        "endpoint": endpoint.name,
                        "result": result,
                        "stream_topic": message.get_header(StreamHeaderKey.TOPIC),
                    }
                )
            return result

        cell._send_to_endpoint = record_send
        future = stream_cell.send_blob(
            _CHANNEL,
            _TOPIC,
            _MISSING_RECEIVER,
            Message(payload=b"late response"),
            optional=True,
            reliable=True,
        )
        adapter = Adapter(lambda _request: None, cell.my_info, stream_cell)
        future.add_done_callback(adapter._handle_reply_stream_done, future, True)
        error = future.exception(timeout=10)
        result_queue.put(
            {
                "error_type": type(error).__name__,
                "stream_id": future.stream_id,
                "sends": sends,
                "errors": capture.messages,
            }
        )
    except Exception:
        result_queue.put({"error": traceback.format_exc()})
        raise
    finally:
        if cell:
            cell.stop()


def _run_recovery_server(root_url, status_queue, stop_event):
    cell = None
    try:
        cell = CoreCell("server", root_url, secure=False, credentials={})
        stream_cell = StreamCell(cell)
        original_handler = stream_cell.byte_streamer._forward_error_handler

        def report_route_failure(message, error):
            status_queue.put({"route_error": error})
            original_handler(message, error)

        cell.add_error_handler(STREAM_CHANNEL, STREAM_DATA_TOPIC, report_route_failure)
        cell.start()
        status_queue.put({"ready": True})
        stop_event.wait(30)
    except Exception:
        status_queue.put({"error": traceback.format_exc()})
        raise
    finally:
        if cell:
            cell.stop()


def _run_recovering_sender(root_url, result_queue):
    cell = None
    try:
        import nvflare.fuel.f3.streaming.byte_streamer as byte_streamer_module

        byte_streamer_module.STREAM_RETRY_WAIT = 0.2
        byte_streamer_module.STREAM_RETRY_TIMEOUT = 3.0
        cell = CoreCell("site-1", root_url, secure=False, credentials={})
        stream_cell = StreamCell(cell)
        cell.start()
        _wait_for_connection(cell, "server")
        payload = b"recover after transient route failure"
        future = stream_cell.send_blob(
            _CHANNEL,
            _TOPIC,
            _MISSING_RECEIVER,
            Message(payload=payload),
            optional=False,
            reliable=True,
        )
        result_queue.put({"bytes_sent": future.result(timeout=10), "payload": payload})
    except Exception:
        result_queue.put({"error": traceback.format_exc()})
        raise
    finally:
        if cell:
            cell.stop()


def _run_recovery_receiver(root_url, status_queue, result_queue, stop_event):
    cell = None
    try:
        cell = CoreCell(_MISSING_RECEIVER, root_url, secure=False, credentials={})
        stream_cell = StreamCell(cell)

        def receive_blob(future):
            result_queue.put({"payload": bytes(future.result())})

        stream_cell.register_blob_cb(_CHANNEL, _TOPIC, receive_blob)
        cell.start()
        _wait_for_connection(cell, "server")
        status_queue.put({"ready": True})
        stop_event.wait(30)
    except Exception:
        status_queue.put({"error": traceback.format_exc()})
        raise
    finally:
        if cell:
            cell.stop()


def _queue_result(queue):
    result = queue.get(timeout=30)
    assert "error" not in result, result.get("error")
    return result


@pytest.mark.timeout(60)
def test_routed_optional_stream_reports_downstream_route_removal_without_error_logs():
    """The sender's first hop succeeds, then the server reports the missing final receiver."""
    context = multiprocessing.get_context("spawn")
    root_url = f"tcp://localhost:{get_open_ports(1)[0]}"
    server_status = context.Queue()
    receiver_status = context.Queue()
    result_queue = context.Queue()
    server_stop = context.Event()
    receiver_stop = context.Event()
    server = context.Process(target=_run_server, args=(root_url, server_status, server_stop))
    receiver = context.Process(target=_run_receiver, args=(root_url, receiver_status, receiver_stop))
    sender = context.Process(target=_run_sender, args=(root_url, result_queue))

    try:
        server.start()
        _queue_result(server_status)
        receiver.start()
        _queue_result(receiver_status)
        assert _queue_result(server_status)["receiver_connected"] is True
        receiver_stop.set()
        receiver.join(timeout=15)
        assert receiver.exitcode == 0
        assert _queue_result(server_status)["receiver_removed"] is True
        sender.start()
        result = _queue_result(result_queue)
        sender.join(timeout=15)
    finally:
        receiver_stop.set()
        server_stop.set()
        receiver.join(timeout=10)
        sender.join(timeout=10)
        server.join(timeout=10)
        for process in (receiver, sender, server):
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    server_result = _queue_result(server_status)
    assert sender.exitcode == 0
    assert server.exitcode == 0
    assert result["error_type"] == StreamTargetUnreachable.__name__
    assert result["sends"]
    assert result["sends"][0] == {
        "endpoint": "server",
        "result": "",
        "stream_topic": _TOPIC,
    }
    assert result["errors"] == []
    assert server_result["errors"] == []


@pytest.mark.timeout(60)
def test_non_optional_reliable_stream_recovers_after_transient_downstream_route_failure():
    context = multiprocessing.get_context("spawn")
    root_url = f"tcp://localhost:{get_open_ports(1)[0]}"
    server_status = context.Queue()
    receiver_status = context.Queue()
    sender_result = context.Queue()
    receiver_result = context.Queue()
    server_stop = context.Event()
    receiver_stop = context.Event()
    server = context.Process(target=_run_recovery_server, args=(root_url, server_status, server_stop))
    sender = context.Process(target=_run_recovering_sender, args=(root_url, sender_result))
    receiver = context.Process(
        target=_run_recovery_receiver,
        args=(root_url, receiver_status, receiver_result, receiver_stop),
    )

    try:
        server.start()
        _queue_result(server_status)
        sender.start()
        route_failure = _queue_result(server_status)
        assert route_failure["route_error"]

        # Heal the downstream route while the non-optional sender is still within
        # retry_timeout. The next reliable retry must reach the new receiver.
        receiver.start()
        _queue_result(receiver_status)
        send_result = _queue_result(sender_result)
        receive_result = _queue_result(receiver_result)
        sender.join(timeout=15)
    finally:
        receiver_stop.set()
        server_stop.set()
        receiver.join(timeout=10)
        sender.join(timeout=10)
        server.join(timeout=10)
        for process in (receiver, sender, server):
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert sender.exitcode == 0
    assert receiver.exitcode == 0
    assert server.exitcode == 0
    assert send_result["bytes_sent"] == len(send_result["payload"])
    assert receive_result["payload"] == send_result["payload"]
