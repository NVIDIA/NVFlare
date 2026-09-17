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

import multiprocessing
import time
import traceback

import pytest

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message
from nvflare.fuel.utils.network_utils import get_open_ports

_CHANNEL = "server_transit_test"
_TOPIC = "ping"
_CONNECT_TIMEOUT = 10.0


def _mark_server_transit(message):
    message.set_header(MessageHeaderKey.SERVER_TRANSIT_REQUIRED, True)


def _wait_for_connection(cell, peer_fqcn):
    deadline = time.time() + _CONNECT_TIMEOUT
    while time.time() < deadline:
        if cell.is_cell_connected(peer_fqcn):
            return
        time.sleep(0.05)
    raise RuntimeError(f"{cell.get_fqcn()} did not connect to {peer_fqcn}")


def _run_server(root_url, status_queue, stop_event):
    cell = None
    try:
        cell = CoreCell("server", root_url, secure=False, credentials={})
        cell.start()
        status_queue.put({"ready": True})
        stop_event.wait(30)
    except Exception:
        status_queue.put({"error": traceback.format_exc()})
        raise
    finally:
        if cell:
            cell.stop()


def _run_relay(root_url, status_queue, stop_event):
    cell = None
    try:
        cell = CoreCell("relay-1", root_url, secure=False, credentials={}, create_internal_listener=True)
        cell.start()
        _wait_for_connection(cell, "server")
        status_queue.put({"parent_url": cell.get_internal_listener_url()})
        stop_event.wait(30)
    except Exception:
        status_queue.put({"error": traceback.format_exc()})
        raise
    finally:
        if cell:
            cell.stop()


def _run_destination(parent_url, status_queue, stop_event):
    cell = None
    try:
        cell = CoreCell("relay-1.site-b", None, secure=False, credentials={}, parent_url=parent_url)
        cell.add_outgoing_reply_filter("*", "*", _mark_server_transit)

        def reply_with_route(request):
            route = request.get_header(MessageHeaderKey.ROUTE, [])
            return Message(
                payload={
                    "marker": request.get_header(MessageHeaderKey.SERVER_TRANSIT_REQUIRED),
                    "route": [hop[0] for hop in route],
                }
            )

        cell.register_request_cb(_CHANNEL, _TOPIC, reply_with_route)
        cell.start()
        _wait_for_connection(cell, "relay-1")
        status_queue.put({"ready": True})
        stop_event.wait(30)
    except Exception:
        status_queue.put({"error": traceback.format_exc()})
        raise
    finally:
        if cell:
            cell.stop()


def _run_origin(parent_url, result_queue):
    cell = None
    try:
        cell = CoreCell("relay-1.site-a", None, secure=False, credentials={}, parent_url=parent_url)
        cell.add_outgoing_request_filter("*", "*", _mark_server_transit)
        cell.start()
        _wait_for_connection(cell, "relay-1")
        reply = cell.send_request(
            _CHANNEL,
            _TOPIC,
            "relay-1.site-b",
            Message(payload="hello"),
            timeout=10.0,
        )
        route = reply.get_header(MessageHeaderKey.ROUTE, [])
        result_queue.put(
            {
                "return_code": reply.get_header(MessageHeaderKey.RETURN_CODE),
                "payload": reply.payload,
                "reply_marker": reply.get_header(MessageHeaderKey.SERVER_TRANSIT_REQUIRED),
                "reply_route": [hop[0] for hop in route],
            }
        )
    except Exception:
        result_queue.put({"error": traceback.format_exc()})
        raise
    finally:
        if cell:
            cell.stop()


def _queue_result(queue):
    result = queue.get(timeout=30)
    assert "error" not in result, result.get("error")
    return result


@pytest.mark.timeout(90)
def test_non_secure_cross_site_request_and_reply_cross_server_under_shared_relay():
    context = multiprocessing.get_context("spawn")
    root_url = f"tcp://localhost:{get_open_ports(1)[0]}"
    server_status = context.Queue()
    relay_status = context.Queue()
    destination_status = context.Queue()
    result_queue = context.Queue()
    server_stop = context.Event()
    relay_stop = context.Event()
    destination_stop = context.Event()

    server = context.Process(target=_run_server, args=(root_url, server_status, server_stop))
    relay = context.Process(target=_run_relay, args=(root_url, relay_status, relay_stop))
    destination = None
    origin = None
    started_processes = []
    try:
        server.start()
        started_processes.append(server)
        _queue_result(server_status)
        relay.start()
        started_processes.append(relay)
        parent_url = _queue_result(relay_status)["parent_url"]

        destination = context.Process(
            target=_run_destination,
            args=(parent_url, destination_status, destination_stop),
        )
        destination.start()
        started_processes.append(destination)
        _queue_result(destination_status)

        origin = context.Process(target=_run_origin, args=(parent_url, result_queue))
        origin.start()
        started_processes.append(origin)
        result = _queue_result(result_queue)
        origin.join(timeout=15)

        assert result["return_code"] == ReturnCode.OK
        assert result["payload"]["marker"] is True
        assert result["payload"]["route"].count("relay-1") == 2
        assert "server" in result["payload"]["route"]
        assert result["reply_marker"] is True
        assert result["reply_route"].count("relay-1") == 2
        assert "server" in result["reply_route"]
    finally:
        destination_stop.set()
        relay_stop.set()
        server_stop.set()
        for process in reversed(started_processes):
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert all(process.exitcode == 0 for process in started_processes)
