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

"""A replacement must be reachable before the old connection reports its close."""

import json
import multiprocessing as mp
from queue import Empty

import pytest

from nvflare.fuel.f3.cellnet.defs import ReturnCode
from tests.cell_reconnect_test_utils import RECONNECT_TEST_TIMEOUT, run_child, run_parent
from tests.timing_utils import PROCESS_STARTUP_TIMEOUT, stop_processes


def _receive(control, timeout=15):
    assert control.poll(timeout), "Cell role did not answer"
    response = control.recv()
    assert "error" not in response, response
    return response


def _wait_for_child_ready(ready, incarnation):
    try:
        return ready.get(timeout=PROCESS_STARTUP_TIMEOUT)
    except Empty:
        pytest.fail(f"Cell child {incarnation} did not report a connection")


@pytest.mark.timeout(RECONNECT_TEST_TIMEOUT)
def test_graceful_child_replacement_before_old_transport_disconnect(tmp_path):
    config = {
        "backbone_conn_gen": 1,
        "allow_adhoc_conns": False,
        "internal": {
            "scheme": "shared-file",
            "resources": {"root_dir": str(tmp_path / "channel"), "connection_security": "clear"},
        },
        "shared-file": {"poll_interval": 0.002, "max_poll_interval": 0.01, "lease_interval": 0.05},
    }
    (tmp_path / "comm_config.json").write_text(json.dumps(config))
    ctx = mp.get_context("spawn")
    control, parent_control = ctx.Pipe()
    goodbye, close_entered, close_released = (ctx.Event() for _ in range(3))
    ready = ctx.Queue()
    stops = [ctx.Event(), ctx.Event()]
    parent = ctx.Process(
        target=run_parent, args=(str(tmp_path), parent_control, goodbye, close_entered, close_released)
    )
    children = []
    try:
        parent.start()
        url = _receive(control, timeout=PROCESS_STARTUP_TIMEOUT)["url"]
        old = ctx.Process(target=run_child, args=(str(tmp_path), url, 1, ready, stops[0]))
        children.append(old)
        old.start()
        assert _wait_for_child_ready(ready, 1) == {"connected": 1}
        control.send("arm")
        assert _receive(control) == {"armed": True}
        control.send("request")
        assert _receive(control) == {"rc": ReturnCode.OK, "payload": 1}

        stops[0].set()
        assert goodbye.wait(15), "Old Cell did not send its graceful goodbye"
        assert close_entered.wait(15), "Old transport did not reach the close-notification boundary"
        old.join(10)
        assert old.exitcode == 0

        replacement = ctx.Process(target=run_child, args=(str(tmp_path), url, 2, ready, stops[1]))
        children.append(replacement)
        replacement.start()
        assert _wait_for_child_ready(ready, 2) == {"connected": 2}
        assert not close_released.is_set(), "Old close notification must stay gated during replacement"
        control.send("request")
        assert _receive(control) == {"rc": ReturnCode.OK, "payload": 2}
    finally:
        close_released.set()
        for stop in stops:
            stop.set()
        if parent.is_alive():
            control.send("stop")
        stop_processes(*children, parent)
        control.close()
        parent_control.close()
