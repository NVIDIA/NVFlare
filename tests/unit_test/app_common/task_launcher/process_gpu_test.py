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

"""Reservation tests do not qualify CUDA; real GPU qualification runs separately."""

import json
import os
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import Mock

import pytest

from nvflare.apis.task_launcher_spec import TaskLaunchError, TaskLaunchRequest, TaskResourceRequest, TaskSettlementError
from nvflare.app_common.task_launcher.process_launcher import ProcessTaskLauncher

GPU = "GPU-00000000-0000-0000-0000-000000000001"
GPU2 = "GPU-00000000-0000-0000-0000-000000000002"


def _partition(tmp_path):
    return {"id": "task-only", "devices": [GPU], "exclusive": True, "reservation_root": str(tmp_path / "pool")}


def _request(tmp_path, attempt="a", code="pass"):
    return TaskLaunchRequest(
        "job",
        "site",
        "task",
        attempt,
        (sys.executable, "-c", code),
        cwd=str(tmp_path),
        environment={"CUDA_VISIBLE_DEVICES": "all"},
        resources=TaskResourceRequest(gpu_count=1),
    )


def test_pool_reservations_survive_owner_exit_and_reject_overcommit(tmp_path):
    from nvflare.app_common.task_launcher.process_gpu import ProcessGpuPool

    config = _partition(tmp_path)
    script = (
        "import json,sys; from nvflare.app_common.task_launcher.process_gpu import ProcessGpuPool; "
        "pool=ProcessGpuPool(json.loads(sys.argv[1])); pool.acquire(('j','s','t','a'),1)"
    )
    subprocess.run([sys.executable, "-c", script, json.dumps(config)], check=True)
    pool = ProcessGpuPool(config)
    with pytest.raises(RuntimeError, match="capacity"):
        pool.acquire(("other", "s", "t", "a"), 1)


def test_released_attempt_cannot_be_physically_relaunched(tmp_path):
    from nvflare.app_common.task_launcher.process_gpu import ProcessGpuPool

    pool = ProcessGpuPool(_partition(tmp_path))
    identity = ("j", "s", "t", "a")
    reservation = pool.acquire(identity, 1)
    assert reservation.devices == (GPU,)
    reservation.release()
    reservation.release()
    with pytest.raises(RuntimeError, match="already"):
        ProcessGpuPool(_partition(tmp_path)).acquire(identity, 1)
    next_reservation = pool.acquire(("j2", "s", "t", "a"), 1)
    next_reservation.release()


def test_multi_gpu_admission_is_atomic_across_overlapping_partitions(tmp_path):
    from nvflare.app_common.task_launcher.process_gpu import ProcessGpuPool

    config = _partition(tmp_path)
    first_pool = ProcessGpuPool(config)
    reservation = first_pool.acquire(("j", "s", "t", "a"), 1)
    second_pool = ProcessGpuPool({**config, "id": "other", "devices": [GPU, GPU2]})
    with pytest.raises(RuntimeError, match="capacity"):
        second_pool.acquire(("j2", "s", "t", "a"), 2)
    # Failure did not acquire GPU2 or consume the physical-attempt identity.
    second = second_pool.acquire(("j2", "s", "t", "a"), 1)
    assert second.devices == (GPU2,)
    reservation.release()
    second.release()


def test_concurrent_admission_has_one_winner(tmp_path):
    from nvflare.app_common.task_launcher.process_gpu import ProcessGpuPool

    pools = [ProcessGpuPool(_partition(tmp_path)) for _ in range(8)]

    def acquire(item):
        index, pool = item
        try:
            return pool.acquire((str(index), "s", "t", "a"), 1)
        except RuntimeError as error:
            assert "capacity" in str(error)
            return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        winners = [result for result in executor.map(acquire, enumerate(pools)) if result]
    assert len(winners) == 1
    winners[0].release()


@pytest.mark.parametrize(
    "field,value",
    [
        ("devices", ["0"]),
        ("devices", ["MIG-foo"]),
        ("devices", [GPU, GPU]),
        ("exclusive", False),
        ("reservation_root", "relative"),
    ],
)
def test_pool_rejects_unsupported_or_ambiguous_configuration(tmp_path, field, value):
    from nvflare.app_common.task_launcher.process_gpu import ProcessGpuPool

    with pytest.raises(ValueError):
        ProcessGpuPool({**_partition(tmp_path), field: value})


def test_pool_rejects_missing_or_mig_enabled_host_device(tmp_path, monkeypatch):
    from nvflare.app_common.task_launcher.process_gpu import ProcessGpuPool

    pool = ProcessGpuPool(_partition(tmp_path))
    monkeypatch.setattr(subprocess, "run", Mock(return_value=Mock(stdout=GPU2 + ", Disabled\n")))
    with pytest.raises(RuntimeError, match="whole GPUs"):
        pool.validate_host()
    monkeypatch.setattr(subprocess, "run", Mock(return_value=Mock(stdout=GPU + ", Enabled\n")))
    with pytest.raises(RuntimeError, match="whole GPUs"):
        pool.validate_host()


def test_release_refuses_changed_ownership(tmp_path):
    from nvflare.app_common.task_launcher.process_gpu import ProcessGpuPool

    pool = ProcessGpuPool(_partition(tmp_path))
    reservation = pool.acquire(("j", "s", "t", "a"), 1)
    with sqlite3.connect(pool.path) as connection:
        connection.execute("UPDATE allocations SET token='different'")
    with pytest.raises(RuntimeError, match="ownership changed"):
        reservation.release()
    with pytest.raises(RuntimeError, match="release is unconfirmed"):
        pool.acquire(("j2", "s", "t", "a"), 1)


@pytest.mark.parametrize("commit_applied", [False, True])
def test_uncertain_release_commit_blocks_other_cjs(tmp_path, monkeypatch, commit_applied):
    from nvflare.app_common.task_launcher.process_gpu import ProcessGpuPool

    pool = ProcessGpuPool(_partition(tmp_path))
    reservation = pool.acquire(("j", "s", "t", "a"), 1)
    connect = sqlite3.connect

    class FailedReleaseCommit(sqlite3.Connection):
        def commit(self):
            if commit_applied:
                super().commit()
            raise sqlite3.OperationalError("release commit confirmation unavailable")

    with monkeypatch.context() as fault:
        fault.setattr(sqlite3, "connect", lambda *a, **k: connect(*a, factory=FailedReleaseCommit, **k))
        with pytest.raises(RuntimeError, match="retain ownership"):
            reservation.release()
    assert not reservation.released
    with pytest.raises(RuntimeError, match="release is unconfirmed"):
        ProcessGpuPool(_partition(tmp_path)).acquire(("other", "s", "t", "a"), 1)


@pytest.mark.skipif(not hasattr(os, "waitid"), reason="requires waitid/WNOWAIT")
def test_worker_visibility_and_release_before_publication(tmp_path, monkeypatch):
    from nvflare.app_common.task_launcher.process_gpu import ProcessGpuPool

    monkeypatch.setattr(ProcessGpuPool, "validate_host", lambda self: None)
    launcher = ProcessTaskLauncher(gpu_partition=_partition(tmp_path))
    output = tmp_path / "environment.json"
    code = f"import json,os; open({str(output)!r},'w').write(json.dumps(dict(os.environ)))"
    handle = launcher.launch_task(_request(tmp_path, code=code))
    try:
        assert handle.wait_for_settlement(10).succeeded
        assert json.loads(output.read_text())["CUDA_VISIBLE_DEVICES"] == GPU
        # No receipt/publication operation has occurred. Another CJ can admit.
        other = ProcessTaskLauncher(gpu_partition=_partition(tmp_path))
        second = other.launch_task(replace(_request(tmp_path), job_id="other"))
        try:
            assert second.wait_for_settlement(10).succeeded
        finally:
            second.cancel()
    finally:
        handle.cancel()


@pytest.mark.skipif(not hasattr(os, "waitid"), reason="requires waitid/WNOWAIT")
@pytest.mark.parametrize("code", ["raise RuntimeError('worker failed')", "import os; os._exit(17)"])
def test_failed_worker_releases_reservation_after_settlement(tmp_path, monkeypatch, code):
    from nvflare.app_common.task_launcher.process_gpu import ProcessGpuPool

    monkeypatch.setattr(ProcessGpuPool, "validate_host", lambda self: None)
    launcher = ProcessTaskLauncher(gpu_partition=_partition(tmp_path))
    handle = launcher.launch_task(_request(tmp_path, code=code))
    try:
        status = handle.wait_for_settlement(10)
        assert status.settled and not status.succeeded
        reservation = launcher.gpu_pool.acquire(("other", "s", "t", "a"), 1)
        reservation.release()
    finally:
        handle.cancel()


@pytest.mark.skipif(not hasattr(os, "waitid"), reason="requires waitid/WNOWAIT")
def test_uncertain_settlement_retains_reservation(tmp_path, monkeypatch):
    from nvflare.app_common.task_launcher.process_gpu import ProcessGpuPool

    monkeypatch.setattr(ProcessGpuPool, "validate_host", lambda self: None)
    launcher = ProcessTaskLauncher(gpu_partition=_partition(tmp_path))
    ready = tmp_path / "ready"
    code = f"import time; open({str(ready)!r},'w').close(); time.sleep(60)"
    handle = launcher.launch_task(_request(tmp_path, code=code))
    try:
        deadline = time.monotonic() + 10
        while not ready.exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        original = handle.process._group_exists
        monkeypatch.setattr(handle.process, "_group_exists", lambda: True)
        with pytest.raises(TimeoutError):
            handle.wait_for_settlement(0.01)
        with pytest.raises(RuntimeError, match="capacity"):
            ProcessGpuPool(_partition(tmp_path)).acquire(("other", "s", "t", "a"), 1)
        monkeypatch.setattr(handle.process, "_group_exists", original)
        assert handle.cancel().settled
    finally:
        monkeypatch.setattr(handle.process, "_group_exists", original)
        handle.cancel()


@pytest.mark.skipif(not hasattr(os, "waitid"), reason="requires waitid/WNOWAIT")
def test_release_failure_fences_handle_and_retains_capacity(tmp_path, monkeypatch):
    from nvflare.app_common.task_launcher.process_gpu import ProcessGpuPool

    monkeypatch.setattr(ProcessGpuPool, "validate_host", lambda self: None)
    launcher = ProcessTaskLauncher(gpu_partition=_partition(tmp_path))
    handle = launcher.launch_task(_request(tmp_path))
    release = handle.reservation.release
    monkeypatch.setattr(handle.reservation, "release", Mock(side_effect=OSError("release failed")))
    try:
        with pytest.raises(TaskSettlementError, match="unconfirmed"):
            handle.wait_for_settlement(10)
        assert not handle.poll().settled
        with pytest.raises(RuntimeError, match="capacity"):
            ProcessGpuPool(_partition(tmp_path)).acquire(("other", "s", "t", "a"), 1)
    finally:
        # Test-owned repair only after the real process scope settled.
        assert handle.process.poll().settled
        release()


@pytest.mark.skipif(not hasattr(os, "waitid"), reason="requires waitid/WNOWAIT")
def test_spawn_failure_releases_acquired_gpu_but_fences_attempt_identity(tmp_path, monkeypatch):
    from nvflare.app_common.task_launcher import process_launcher
    from nvflare.app_common.task_launcher.process_gpu import ProcessGpuPool

    monkeypatch.setattr(ProcessGpuPool, "validate_host", lambda self: None)
    monkeypatch.setattr(process_launcher, "spawn_process", Mock(side_effect=OSError("spawn failed")))
    launcher = ProcessTaskLauncher(gpu_partition=_partition(tmp_path))
    with pytest.raises(TaskLaunchError, match="GPU reservation released") as failure:
        launcher.launch_task(_request(tmp_path))
    assert failure.value.handle.poll().settled
    with pytest.raises(RuntimeError, match="already launched"):
        launcher.launch_task(_request(tmp_path))
    reservation = ProcessGpuPool(_partition(tmp_path)).acquire(("other", "s", "t", "a"), 1)
    reservation.release()


@pytest.mark.skipif(not hasattr(os, "waitid"), reason="requires waitid/WNOWAIT")
def test_cpu_attempt_on_gpu_site_cannot_inherit_gpu_visibility(tmp_path):
    output = tmp_path / "cpu-environment.json"
    code = f"import json,os; open({str(output)!r},'w').write(json.dumps(dict(os.environ)))"
    launcher = ProcessTaskLauncher(gpu_partition=_partition(tmp_path))
    handle = launcher.launch_task(replace(_request(tmp_path, code=code), resources=TaskResourceRequest()))
    try:
        assert handle.wait_for_settlement(10).succeeded
        assert json.loads(output.read_text())["CUDA_VISIBLE_DEVICES"] == ""
    finally:
        handle.cancel()
