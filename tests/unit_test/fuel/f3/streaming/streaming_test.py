# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import multiprocessing
import threading
import time
import traceback
from unittest.mock import MagicMock, patch

import pytest

import nvflare.fuel.f3.cellnet.cell as cell_module
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.core_cell import CoreCell, make_reply
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.stream_cell import StreamCell
from nvflare.fuel.f3.streaming.stream_const import STREAM_CHANNEL, STREAM_DATA_TOPIC, StreamHeaderKey
from nvflare.fuel.f3.streaming.stream_types import StreamError, StreamFuture
from nvflare.fuel.f3.streaming.tools.utils import TEST_CHANNEL, TEST_TOPIC, make_buffer

_STREAM_RX_CELL = "stream_test_server"
_STREAM_TX_CELL = "stream_test_sender"
from nvflare.fuel.utils.network_utils import get_open_ports

WAIT_SEC = 10


class State:
    def __init__(self):
        self.done = threading.Event()
        self.result = None


class TestStreamCell:
    @pytest.fixture(scope="session")
    def port(self):
        return get_open_ports(1)[0]

    @pytest.fixture(scope="session")
    def state(self):
        return State()

    @pytest.fixture(scope="session")
    def server_cell(self, port, state):
        # Patch STREAM_ACK_WAIT in the byte_streamer module
        with patch("nvflare.fuel.f3.streaming.byte_streamer.STREAM_ACK_WAIT", 400):
            listening_url = f"tcp://localhost:{port}"
            cell = CoreCell(_STREAM_RX_CELL, listening_url, secure=False, credentials={})
            stream_cell = StreamCell(cell)
            stream_cell.register_blob_cb(TEST_CHANNEL, TEST_TOPIC, self.blob_cb, state=state)
            cell.start()

            yield stream_cell
            cell.stop()

    @pytest.fixture(scope="session")
    def client_cell(self, port, state):
        with patch("nvflare.fuel.f3.streaming.byte_streamer.STREAM_ACK_WAIT", 400):
            connect_url = f"tcp://localhost:{port}"
            cell = CoreCell(_STREAM_TX_CELL, connect_url, secure=False, credentials={})
            stream_cell = StreamCell(cell)
            cell.start()

            yield stream_cell
            cell.stop()

    def test_streaming_blob(self, server_cell, client_cell, state):

        size = 64 * 1024 * 1024 + 123
        buffer = make_buffer(size)

        send_future = client_cell.send_blob(TEST_CHANNEL, TEST_TOPIC, _STREAM_RX_CELL, Message(None, buffer))
        bytes_sent = send_future.result()
        assert bytes_sent == len(buffer)

        if not state.done.wait(timeout=30):
            raise Exception("Data not received after 30 seconds")

        assert buffer == state.result

    def test_streaming_buffer_list(self, server_cell, client_cell, state):

        size = 64 * 1024 * 1024 + 123
        buffer = make_buffer(size)
        buf_list = []
        interval = int(size / 4)
        buf_list.append(buffer[0:interval])
        buf_list.append(buffer[interval : 2 * interval])
        buf_list.append(buffer[2 * interval : 3 * interval])
        buf_list.append(buffer[3 * interval : size])

        send_future = client_cell.send_blob(TEST_CHANNEL, TEST_TOPIC, _STREAM_RX_CELL, Message(None, buf_list))
        bytes_sent = send_future.result()
        assert bytes_sent == len(buffer)

        if not state.done.wait(timeout=30):
            raise Exception("Data not received after 30 seconds")

        assert buffer == state.result

    def blob_cb(self, future: StreamFuture, **kwargs):
        state = kwargs.get("state")
        state.result = future.result()
        state.done.set()


def test_incoming_filter_rejection_fails_stream_sender_immediately():
    port = get_open_ports(1)[0]
    server_name = "stream_filter_reject_server"
    sender_name = "stream_filter_reject_sender"
    server = CoreCell(server_name, f"tcp://localhost:{port}", secure=False, credentials={})
    sender = CoreCell(sender_name, f"tcp://localhost:{port}", secure=False, credentials={})
    StreamCell(server)
    sender_stream = StreamCell(sender)
    server.add_incoming_filter(
        STREAM_CHANNEL,
        STREAM_DATA_TOPIC,
        lambda _message: make_reply(ReturnCode.UNAUTHENTICATED, error="missing client name"),
    )
    server.start()
    sender.start()
    try:
        future = sender_stream.send_blob(TEST_CHANNEL, TEST_TOPIC, server_name, Message(None, b"payload"))

        with pytest.raises(StreamError, match="missing client name"):
            future.result(timeout=2.0)
    finally:
        sender.stop()
        server.stop()


def test_streamed_request_decode_failure_returns_process_exception(monkeypatch):
    port = get_open_ports(1)[0]
    server_name = "stream_decode_failure_server"
    sender_name = "stream_decode_failure_sender"
    root_url = f"tcp://localhost:{port}"
    server = Cell(server_name, root_url, secure=False, credentials={})
    sender = Cell(sender_name, root_url, secure=False, credentials={})
    topic = "decode_failure"
    callback = MagicMock(return_value=make_reply(ReturnCode.OK))
    server.register_request_cb(channel=TEST_CHANNEL, topic=topic, cb=callback)
    original_decode_payload = cell_module.decode_payload

    def fail_request_decode(message, encoding_key, fobs_ctx):
        if (
            message.get_header(StreamHeaderKey.CHANNEL) == TEST_CHANNEL
            and message.get_header(StreamHeaderKey.TOPIC) == topic
        ):
            raise RuntimeError("nested DownloadService request failed")
        return original_decode_payload(message, encoding_key, fobs_ctx)

    monkeypatch.setattr(cell_module, "decode_payload", fail_request_decode)
    server.start()
    sender.start()
    try:
        _wait_for_connection(sender, server_name)
        started = time.monotonic()
        reply = sender.send_request(
            channel=TEST_CHANNEL,
            topic=topic,
            target=server_name,
            request=Message({}, {"task": "payload"}),
            timeout=2.0,
            reliable=True,
        )

        assert time.monotonic() - started < 2.0
        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.PROCESS_EXCEPTION
        assert reply.get_header(MessageHeaderKey.ERROR) == "failed to decode streamed request payload"
        callback.assert_not_called()
    finally:
        sender.stop()
        server.stop()


def _wait_for_connection(cell, peer_fqcn):
    deadline = time.time() + WAIT_SEC
    while time.time() < deadline:
        if cell.is_cell_connected(peer_fqcn):
            return
        time.sleep(0.05)
    raise RuntimeError(f"{cell.get_fqcn()} did not connect to {peer_fqcn}")


def _run_filtering_router(root_url, status_queue, stop_event):
    cell = None
    try:
        cell = CoreCell("server", root_url, secure=False, credentials={})
        cell.add_incoming_filter(
            STREAM_CHANNEL,
            STREAM_DATA_TOPIC,
            lambda _message: make_reply(ReturnCode.UNAUTHENTICATED, error="transit missing client name"),
        )
        cell.start()
        status_queue.put({"ready": True})
        stop_event.wait(30)
    except Exception:
        status_queue.put({"traceback": traceback.format_exc()})
        raise
    finally:
        if cell:
            cell.stop()


def _run_routed_receiver(root_url, status_queue, stop_event):
    cell = None
    try:
        cell = CoreCell("site-b", root_url, secure=False, credentials={})
        StreamCell(cell)
        cell.start()
        _wait_for_connection(cell, "server")
        status_queue.put({"ready": True})
        stop_event.wait(30)
    except Exception:
        status_queue.put({"traceback": traceback.format_exc()})
        raise
    finally:
        if cell:
            cell.stop()


def _run_routed_sender(root_url, result_queue):
    cell = None
    try:
        cell = CoreCell("site-a", root_url, secure=False, credentials={})
        stream_cell = StreamCell(cell)
        cell.start()
        _wait_for_connection(cell, "server")
        future = stream_cell.send_blob(
            TEST_CHANNEL,
            TEST_TOPIC,
            "site-b",
            Message(None, b"payload"),
            reliable=True,
        )
        try:
            result_queue.put({"result": future.result(timeout=5.0)})
        except Exception as ex:
            result_queue.put({"error_type": type(ex).__name__, "error": str(ex)})
    except Exception:
        result_queue.put({"traceback": traceback.format_exc()})
        raise
    finally:
        if cell:
            cell.stop()


def _get_process_result(queue):
    result = queue.get(timeout=30)
    assert "traceback" not in result, result.get("traceback")
    return result


@pytest.mark.timeout(60)
def test_routed_incoming_filter_rejection_fails_stream_sender_immediately():
    context = multiprocessing.get_context("spawn")
    root_url = f"tcp://localhost:{get_open_ports(1)[0]}"
    router_status = context.Queue()
    receiver_status = context.Queue()
    result_queue = context.Queue()
    router_stop = context.Event()
    receiver_stop = context.Event()
    router = context.Process(target=_run_filtering_router, args=(root_url, router_status, router_stop))
    receiver = context.Process(target=_run_routed_receiver, args=(root_url, receiver_status, receiver_stop))
    sender = context.Process(target=_run_routed_sender, args=(root_url, result_queue))
    started_processes = []
    try:
        router.start()
        started_processes.append(router)
        _get_process_result(router_status)
        receiver.start()
        started_processes.append(receiver)
        _get_process_result(receiver_status)
        sender.start()
        started_processes.append(sender)
        result = _get_process_result(result_queue)
        sender.join(timeout=10)

        assert result.get("error_type") == StreamError.__name__
        assert "Received error from server" in result["error"]
        assert "transit missing client name" in result["error"]
    finally:
        receiver_stop.set()
        router_stop.set()
        for process in reversed(started_processes):
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert all(process.exitcode == 0 for process in started_processes)
