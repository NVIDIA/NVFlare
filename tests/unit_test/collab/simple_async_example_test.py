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

"""Exercise the example scheduler through real Collab group calls and callbacks."""

import importlib
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from nvflare.apis.signal import Signal
from nvflare.apis.workspace import Workspace
from nvflare.collab.api.app import ServerApp
from nvflare.collab.api.constants import FL_CONTEXT_PROP
from nvflare.collab.api.proxy import Proxy


@pytest.fixture
def workflow(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples" / "advanced"))
    module = importlib.import_module("collab.simple_async.job")
    server = module.AsyncServer(num_updates=3)
    app = ServerApp(server)
    pending = queue.Queue()
    dispatched = []
    signal = Signal()
    (tmp_path / "startup").mkdir()
    (tmp_path / "local").mkdir()
    workspace = Workspace(str(tmp_path), "server")
    run_dir = Path(workspace.get_run_dir("test"))
    run_dir.mkdir(parents=True)
    app.set_prop(FL_CONTEXT_PROP, SimpleNamespace(get_job_id=lambda: "test"))

    def dispatch(gcc, func_name, **kwargs):
        assert func_name == "train"
        dispatched.append(gcc)
        pending.put((gcc, kwargs))

    clients = [
        Proxy(
            app,
            name,
            name,
            MagicMock(call_target_in_group=dispatch),
            {"train": ["weights", "version"]},
        )
        for name in ("site-1", "site-2")
    ]
    app.setup(workspace, MagicMock(), clients, signal)

    def run():
        with app.new_context("server", "server"):
            return server.run()

    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            yield SimpleNamespace(
                start=lambda: executor.submit(run),
                pending=pending,
                signal=signal,
                run_dir=run_dir,
                server=server,
            )
        finally:
            signal.trigger(True)
            for gcc in dispatched:
                gcc.set_exception(RuntimeError("test cleanup"))
                gcc.call_completed()


def complete(call, weights):
    gcc, kwargs = call
    gcc.set_result((np.array(weights, dtype=float), kwargs["version"]))
    gcc.call_completed()


@pytest.mark.parametrize("num_updates", [2, 3])
def test_refill_precedes_slow_response_and_respects_budget(workflow, num_updates):
    workflow.server.num_updates = num_updates
    future = workflow.start()
    initial = [workflow.pending.get(timeout=5) for _ in range(2)]
    by_site = {call[0].target_name: call for call in initial}
    complete(by_site["site-1"], [2, 2])

    if num_updates == 3:
        # The slow initial call has not completed. Getting this assignment
        # proves there is no group barrier before the fast client's refill.
        refill = workflow.pending.get(timeout=5)
        assert refill[0].target_name == "site-1"
        assert refill[1]["version"] == 1
        np.testing.assert_array_equal(refill[1]["weights"], [1, 1])
        complete(refill, [3, 3])
        expected = [3, 3]
    else:
        expected = [2.5, 2.5]

    # The slow client still holds its original snapshot, even after refill.
    assert by_site["site-2"][1]["version"] == 0
    np.testing.assert_array_equal(by_site["site-2"][1]["weights"], [0, 0])
    complete(by_site["site-2"], [4, 4])
    np.testing.assert_array_equal(future.result(timeout=5), expected)
    np.testing.assert_array_equal(np.load(workflow.run_dir / "final_model.npy"), expected)
    assert workflow.pending.empty()


@pytest.mark.parametrize("failure", [ValueError("training failed"), TimeoutError("client timed out")])
def test_failure_without_success_callback_terminates(workflow, failure):
    future = workflow.start()
    gcc, _ = workflow.pending.get(timeout=5)
    gcc.set_exception(failure)
    gcc.call_completed()
    with pytest.raises(RuntimeError, match="Client training failed"):
        future.result(timeout=5)
    assert not (workflow.run_dir / "final_model.npy").exists()


def test_abort_while_waiting_terminates(workflow):
    future = workflow.start()
    workflow.pending.get(timeout=5)
    workflow.signal.trigger(True)
    with pytest.raises(RuntimeError, match="Training aborted"):
        future.result(timeout=5)
