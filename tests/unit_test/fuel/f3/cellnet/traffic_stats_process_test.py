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

"""Real TCP exchanges: server, relay and sender run in separate processes."""

import multiprocessing
import traceback

import pytest

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message
from nvflare.fuel.utils.network_utils import get_open_ports
from tests.unit_test.fuel.f3.cellnet.server_transit_e2e_test import (
    _mark_server_transit,
    _queue_result,
    _run_relay,
    _wait_for_connection,
)

_CHANNEL = "traffic_counter_test"
_TOPIC = "echo"
_REPEATS = 3


def _traffic_server(root_url, streaming, status_queue, capture_event, stop_event):
    cell = None
    try:
        cell_type = Cell if streaming else CoreCell
        cell = cell_type("server", root_url, secure=False, credentials={})
        cell.add_outgoing_reply_filter("*", "*", _mark_server_transit)
        cell.register_request_cb(_CHANNEL, _TOPIC, lambda request: Message(payload=request.payload))
        cell.start()
        status_queue.put({"ready": True})
        if not capture_event.wait(30):
            raise TimeoutError("No request to capture server traffic")
        status_queue.put(cell.get_traffic_stats())
        stop_event.wait(30)
    except Exception:
        status_queue.put({"error": traceback.format_exc()})
        raise
    finally:
        if cell:
            cell.stop()


def _traffic_sender(parent_url, streaming, payload_size, result_queue):
    cell = None
    try:
        cell_type = Cell if streaming else CoreCell
        cell = cell_type("relay-1.site-a", None, secure=False, credentials={}, parent_url=parent_url)
        cell.add_outgoing_request_filter("*", "*", _mark_server_transit)
        cell.start()
        _wait_for_connection(cell, "relay-1")
        payload = b"x" * payload_size
        snapshots = []
        for _ in range(_REPEATS):
            reply = cell.send_request(_CHANNEL, _TOPIC, "server", Message(payload=payload), timeout=10.0)
            assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
            assert reply.payload == payload
            snapshots.append(cell.get_traffic_stats())
        result_queue.put({"snapshots": snapshots})
    except Exception:
        result_queue.put({"error": traceback.format_exc()})
        raise
    finally:
        if cell:
            cell.stop()


@pytest.mark.timeout(90)
@pytest.mark.parametrize("streaming,payload_size", [(False, 4096), (True, 2 * 1024 * 1024)], ids=["core", "streamed"])
def test_real_tcp_repeated_request_reply_counts_actual_adjacent_peers(streaming, payload_size):
    context = multiprocessing.get_context("spawn")
    root_url = f"tcp://localhost:{get_open_ports(1)[0]}"
    server_status, relay_status, sender_result = (context.Queue() for _ in range(3))
    capture_event, stop_event = context.Event(), context.Event()
    processes = []
    try:
        server = context.Process(
            target=_traffic_server, args=(root_url, streaming, server_status, capture_event, stop_event)
        )
        server.start()
        processes.append(server)
        assert _queue_result(server_status)["ready"]

        relay = context.Process(target=_run_relay, args=(root_url, relay_status, stop_event))
        relay.start()
        processes.append(relay)
        parent_url = _queue_result(relay_status)["parent_url"]

        sender = context.Process(target=_traffic_sender, args=(parent_url, streaming, payload_size, sender_result))
        sender.start()
        processes.append(sender)
        result = _queue_result(sender_result)
        capture_event.set()
        server_snapshot = _queue_result(server_status)

        for index, snapshot in enumerate(result["snapshots"], start=1):
            assert snapshot["cell"] == "relay-1.site-a"
            # The final destination is server, but this sender's actual peer is its relay.
            assert set(snapshot["peers"]) == {"relay-1"}
            counters = snapshot["peers"]["relay-1"]
            assert counters["sent_bytes"] >= index * payload_size
            assert counters["received_bytes"] >= index * payload_size

        assert server_snapshot["cell"] == "server"
        assert set(server_snapshot["peers"]) == {"relay-1"}
        server_counters = server_snapshot["peers"]["relay-1"]
        assert server_counters["sent_bytes"] >= _REPEATS * payload_size
        assert server_counters["received_bytes"] >= _REPEATS * payload_size
        if streaming:
            # Allow stream-control overhead, but not an extra count of the reconstructed blob.
            upper_bound = _REPEATS * payload_size * 1.25 + 65536
            for snapshot in (result["snapshots"][-1], server_snapshot):
                assert all(value < upper_bound for value in snapshot["peers"]["relay-1"].values())
    finally:
        capture_event.set()
        stop_event.set()
        for process in reversed(processes):
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert all(process.exitcode == 0 for process in processes)
