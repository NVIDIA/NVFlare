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

"""Static GPU visibility and process cleanup; these tests do not run CUDA."""

import json
import os
import sys
from dataclasses import replace

import psutil
import pytest

from nvflare.apis.task_launcher_spec import TaskLaunchRequest, TaskResourceRequest, UnsupportedTaskResourceError
from nvflare.app_common.task_launcher.process_launcher import ProcessTaskLauncher

GPU = "GPU-00000000-0000-0000-0000-000000000001"
GPU2 = "GPU-00000000-0000-0000-0000-000000000002"
requires_waitid = pytest.mark.skipif(not hasattr(os, "waitid"), reason="requires waitid/WNOWAIT")


def _request(tmp_path, attempt="a", code="pass", count=1):
    return TaskLaunchRequest(
        "job",
        "site",
        "task",
        attempt,
        (sys.executable, "-c", code),
        cwd=str(tmp_path),
        environment={"CUDA_VISIBLE_DEVICES": "all"},
        resources=TaskResourceRequest(gpu_count=count),
    )


@pytest.mark.parametrize("devices", ["0", [0], [True], [""], ["all"], ["none"], ["-1"], ["0,1"], ["0", "0"]])
def test_invalid_static_visibility_is_rejected(devices):
    with pytest.raises(ValueError, match="gpu_devices"):
        ProcessTaskLauncher(gpu_devices=devices)


@requires_waitid
@pytest.mark.parametrize("count,expected", [(0, ""), (1, GPU2), (2, GPU2 + "," + GPU)])
def test_worker_receives_static_device_prefix_without_changing_parent(tmp_path, monkeypatch, count, expected):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent-assignment")
    output = tmp_path / "visibility.json"
    code = f"import json,os; open({str(output)!r},'w').write(json.dumps(dict(os.environ)))"
    launcher = ProcessTaskLauncher(gpu_devices=[GPU2, GPU])
    handle = launcher.launch_task(_request(tmp_path, code=code, count=count))
    try:
        assert handle.wait_for_settlement(10).succeeded
        assert json.loads(output.read_text())["CUDA_VISIBLE_DEVICES"] == expected
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "parent-assignment"
    finally:
        handle.cancel()


@requires_waitid
def test_request_exceeding_static_assignment_fails_before_spawn(tmp_path):
    launcher = ProcessTaskLauncher(gpu_devices=[GPU])
    output = tmp_path / "must-not-start"
    request = _request(tmp_path, code=f"open({str(output)!r},'w').close()", count=2)
    with pytest.raises(UnsupportedTaskResourceError, match="static visibility"):
        launcher.launch_task(request)
    assert not output.exists()
    handle = launcher.launch_task(replace(request, resources=TaskResourceRequest(gpu_count=1)))
    try:
        assert handle.wait_for_settlement(10).succeeded
    finally:
        handle.cancel()


@requires_waitid
def test_two_launchers_can_use_the_same_assignment_without_local_accounting(tmp_path):
    first = ProcessTaskLauncher(gpu_devices=[GPU]).launch_task(_request(tmp_path, code="import time; time.sleep(60)"))
    second = None
    try:
        second = ProcessTaskLauncher(gpu_devices=[GPU]).launch_task(_request(tmp_path, attempt="other"))
        assert second.wait_for_settlement(10).succeeded
        assert not first.poll().settled
    finally:
        first.cancel()
        if second:
            second.cancel()


@requires_waitid
@pytest.mark.parametrize("failure", ["exception", "abrupt"])
def test_failed_worker_and_descendant_settle_before_a_fresh_launch(tmp_path, failure):
    child_file = tmp_path / "child"
    code = (
        "import os,sys,subprocess; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        f"open({str(child_file)!r},'w').write(str(p.pid)); "
        + ("raise RuntimeError('CUDA out of memory (simulated)')" if failure == "exception" else "os._exit(17)")
    )
    launcher = ProcessTaskLauncher(gpu_devices=[GPU])
    failed = launcher.launch_task(_request(tmp_path, code=code))
    try:
        status = failed.wait_for_settlement(10)
        assert status.settled and not status.succeeded
        child = int(child_file.read_text())
        try:
            assert psutil.Process(child).status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
        except psutil.NoSuchProcess:
            pass
        healthy = launcher.launch_task(_request(tmp_path, attempt="after-failure"))
        try:
            assert healthy.wait_for_settlement(10).succeeded
        finally:
            healthy.cancel()
    finally:
        failed.cancel()
