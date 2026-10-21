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

"""Explicitly scheduled real-GPU scope checks; federation controls run separately.

NVFLARE_PROCESS_GPU_ALLOCATOR names a JSON file containing the approved allocator arguments, including its canonical persistent ledger directory. Never substitute
a fresh ledger to bypass an existing reservation.
"""

import json
import os
import sqlite3
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil
import pytest

from nvflare.apis.task_launcher_spec import TaskLaunchRequest, TaskResourceRequest
from nvflare.app_common.resource_managers.gpu_reservation_ledger import GpuReservationLedger
from nvflare.app_common.task_launcher.process_launcher import ProcessTaskLauncher

pytestmark = pytest.mark.skipif(
    not os.environ.get("NVFLARE_PROCESS_GPU_ALLOCATOR"), reason="requires explicitly scheduled real GPU capacity"
)

COMPUTE = """
import json, os, subprocess, sys, time
from pathlib import Path
import torch
output, mode = Path(sys.argv[1]), sys.argv[2]
torch.set_num_threads(1)
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
model = torch.nn.Linear(1, 1, bias=False).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
x, y = torch.ones(64, 1, device='cuda'), torch.full((64, 1), 3., device='cuda')
for _ in range(40):
    optimizer.zero_grad()
    loss = (model(x) - y).square().mean()
    loss.backward()
    optimizer.step()
torch.cuda.synchronize()
assert loss.item() < 1e-4
torch.save({key: value.cpu() for key, value in model.state_dict().items()}, output.with_suffix('.pt'))
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']) if mode == 'child' else None
output.write_text(json.dumps(dict(pid=os.getpid(), descendant_pid=child.pid if child else None,
    visible=os.environ['CUDA_VISIBLE_DEVICES'], loss=loss.item(),
    allocated_bytes=torch.cuda.max_memory_allocated(), computed_at=time.time())))
if mode in ('hold', 'child'):
    time.sleep(120)
if mode == 'exception':
    raise RuntimeError('deliberate post-CUDA worker exception')
if mode == 'abrupt':
    os._exit(17)
"""


@pytest.fixture
def gpu_launcher():
    config = json.loads(Path(os.environ["NVFLARE_PROCESS_GPU_ALLOCATOR"]).read_text())
    launcher = ProcessTaskLauncher(gpu_allocator=GpuReservationLedger(**config))
    launcher.gpu_allocator.validate_host()
    return launcher


def _launch(launcher, output, mode="normal"):
    return launcher.launch_task(
        TaskLaunchRequest(
            "gpu-qualification",
            "site-probe",
            "compute",
            str(uuid.uuid4()),
            (sys.executable, "-c", COMPUTE, str(output), mode),
            environment={"PATH": os.environ["PATH"]},
            resources=TaskResourceRequest(gpu_count=1),
        )
    )


def _ready(path):
    deadline = time.monotonic() + 60
    while not path.exists():
        assert time.monotonic() < deadline
        time.sleep(0.05)
    return json.loads(path.read_text())


def _no_live_pid(pid):
    try:
        assert psutil.Process(pid).status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
    except psutil.NoSuchProcess:
        pass


@pytest.mark.timeout(120)
def test_real_cuda_reuse_while_previous_result_publication_is_blocked(gpu_launcher, tmp_path):
    first = _launch(gpu_launcher, tmp_path / "first.json")
    try:
        assert first.wait_for_settlement(60).succeeded
        first_facts = _ready(tmp_path / "first.json")
        saved = (tmp_path / "first.pt").read_bytes()
        publishing, finish_publication = threading.Event(), threading.Event()

        def publish_saved_result():
            publishing.set()
            assert finish_publication.wait(60)
            assert (tmp_path / "first.pt").read_bytes() == saved
            return time.time()

        with ThreadPoolExecutor(max_workers=1) as executor:
            upload = executor.submit(publish_saved_result)
            assert publishing.wait(5)
            second = None
            try:
                second = _launch(gpu_launcher, tmp_path / "second.json")
                assert second.wait_for_settlement(60).succeeded
                assert not upload.done()
                second_facts = _ready(tmp_path / "second.json")
                assert second_facts["visible"] == first_facts["visible"]
            finally:
                finish_publication.set()
                if second:
                    second.cancel()
            published_at = upload.result(5)
        assert second_facts["computed_at"] < published_at
        # The first immutable result remains reusable after another execution.
        assert (tmp_path / "first.pt").read_bytes() == saved
    finally:
        first.cancel()


@pytest.mark.timeout(120)
def test_real_cuda_contention_and_descendant_cancellation(gpu_launcher, tmp_path):
    handle = _launch(gpu_launcher, tmp_path / "held.json", "child")
    try:
        facts = _ready(tmp_path / "held.json")
        # Use a one-device view of the SAME ledger to test that assigned device.
        selected = json.loads(Path(os.environ["NVFLARE_PROCESS_GPU_ALLOCATOR"]).read_text())
        selected["devices"] = [facts["visible"]]
        other = ProcessTaskLauncher(gpu_allocator=GpuReservationLedger(**selected))
        with pytest.raises(RuntimeError, match="capacity"):
            _launch(other, tmp_path / "must-not-start.json")
        assert not (tmp_path / "must-not-start.json").exists()
        assert handle.cancel().settled
        _no_live_pid(facts["pid"])
        _no_live_pid(facts["descendant_pid"])
        replacement = _launch(other, tmp_path / "replacement.json")
        try:
            assert replacement.wait_for_settlement(60).succeeded
        finally:
            replacement.cancel()
    finally:
        handle.cancel()


@pytest.mark.timeout(120)
@pytest.mark.parametrize("mode", ["exception", "abrupt"])
def test_real_cuda_worker_failure_releases_only_after_settlement(gpu_launcher, tmp_path, mode):
    handle = _launch(gpu_launcher, tmp_path / "failure.json", mode)
    try:
        status = handle.wait_for_settlement(60)
        assert status.settled and not status.succeeded
        facts = _ready(tmp_path / "failure.json")
        _no_live_pid(facts["pid"])
        with sqlite3.connect(gpu_launcher.gpu_allocator.path) as connection:
            record = connection.execute(
                "SELECT released,execution_id FROM attempts WHERE identity=?", (handle.reservation.identity,)
            ).fetchone()
        assert record == (1, handle.execution_id)
    finally:
        handle.cancel()


@pytest.mark.timeout(120)
def test_real_cuda_uncertain_scope_retains_reservation(gpu_launcher, tmp_path, monkeypatch):
    handle = _launch(gpu_launcher, tmp_path / "uncertain.json")
    original = handle.process._group_exists
    try:
        facts = _ready(tmp_path / "uncertain.json")
        # Controlled loss of settlement evidence. CUDA and the execution remain
        # real; only the scope observation is injected as unavailable.
        monkeypatch.setattr(handle.process, "_group_exists", lambda: True)
        with pytest.raises(TimeoutError):
            handle.wait_for_settlement(0.2)
        selected = json.loads(Path(os.environ["NVFLARE_PROCESS_GPU_ALLOCATOR"]).read_text())
        selected["devices"] = [facts["visible"]]
        other = ProcessTaskLauncher(gpu_allocator=GpuReservationLedger(**selected))
        with pytest.raises(RuntimeError, match="capacity"):
            _launch(other, tmp_path / "must-not-start.json")
        with sqlite3.connect(gpu_launcher.gpu_allocator.path) as connection:
            released = connection.execute(
                "SELECT released FROM attempts WHERE identity=?", (handle.reservation.identity,)
            ).fetchone()[0]
        assert released == 0
    finally:
        monkeypatch.setattr(handle.process, "_group_exists", original)
        handle.cancel()
