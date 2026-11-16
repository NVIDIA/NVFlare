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

"""Process targets and bounded waits shared by live Cell transit tests."""

import time
import traceback

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey

_CONNECT_TIMEOUT = 10.0


def _mark_server_transit(message):
    message.set_header(MessageHeaderKey.SERVER_TRANSIT_REQUIRED, True)


def _wait_for_connection(cell, peer_fqcn):
    deadline = time.monotonic() + _CONNECT_TIMEOUT
    while time.monotonic() < deadline:
        if cell.is_cell_connected(peer_fqcn):
            return
        time.sleep(0.05)
    raise RuntimeError(f"{cell.get_fqcn()} did not connect to {peer_fqcn}")


def _run_relay(root_url, status_queue, stop_event):
    cell = None
    try:
        cell = CoreCell("relay-1", root_url, secure=False, credentials={}, create_internal_listener=True)
        cell.start()
        _wait_for_connection(cell, "server")
        status_queue.put({"parent_url": cell.get_internal_listener_url()})
        stop_event.wait()
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
