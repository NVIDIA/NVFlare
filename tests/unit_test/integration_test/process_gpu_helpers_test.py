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

import json
import runpy
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from nvflare.apis.fl_constant import FLContextKey, ReservedKey, TaskResultReceipt
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.private.fed.client.client_runner import _TASK_CHECK_RESULT_OK, _TASK_CHECK_RESULT_TRY_AGAIN, ClientRunner
from tests.integration_test.tools.process_gpu import cuda_probe, export_job

HOOK = Path(__file__).resolve().parents[2] / "integration_test/tools/process_gpu/cj_hooks/sitecustomize.py"


def test_scheduled_publication_hook_delays_once_and_recovers_receipt_observation(tmp_path, monkeypatch):
    def sent(self, result, task_id, context):
        context.set_prop(FLContextKey.TASK_RESULT_RECEIPT, TaskResultReceipt.RECEIVED, private=True, sticky=False)
        return True

    def try_sent(self, result, task_id, context):
        sent(self, result, task_id, context)
        return _TASK_CHECK_RESULT_OK

    monkeypatch.setenv("NVFLARE_PROCESS_GPU_DELAY_ROOT", str(tmp_path))
    monkeypatch.setenv("NVFLARE_PROCESS_GPU_DELAY_SITE", "site-1")
    monkeypatch.setenv("NVFLARE_PROCESS_GPU_RETRY_ONCE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(ClientRunner, "_send_task_result", sent)
    monkeypatch.setattr(ClientRunner, "_try_send_result_once", try_sent)
    runpy.run_path(str(HOOK))
    context = FLContext()
    for key, value in (
        (ReservedKey.RUN_NUM, "job"),
        (ReservedKey.IDENTITY_NAME, "site-1"),
        (FLContextKey.TASK_ATTEMPT_ID, "attempt"),
    ):
        context.set_prop(key, value, private=True, sticky=True)
    runner = SimpleNamespace(run_abort_signal=Signal())
    with ThreadPoolExecutor(max_workers=1) as executor:
        publication = executor.submit(ClientRunner._send_task_result, runner, Shareable(), "task", context)
        try:
            deadline = time.monotonic() + 5
            while not list(tmp_path.glob("pause-*.json")):
                assert time.monotonic() < deadline
                time.sleep(0.01)
            marker = json.loads(next(tmp_path.glob("pause-*.json")).read_text())
            assert marker["identity"] == ["job", "site-1", "task", "attempt"]
            assert not publication.done()
            Path(marker["resume"]).touch()
            assert publication.result(5)
        finally:
            runner.run_abort_signal.trigger(True)
    assert ClientRunner._try_send_result_once(runner, Shareable(), "task", context) == _TASK_CHECK_RESULT_TRY_AGAIN
    assert context.get_prop(FLContextKey.TASK_RESULT_RECEIPT) is None
    assert ClientRunner._try_send_result_once(runner, Shareable(), "task", context) == _TASK_CHECK_RESULT_OK
    assert context.get_prop(FLContextKey.TASK_RESULT_RECEIPT) == TaskResultReceipt.RECEIVED


@pytest.mark.parametrize("enabled", ["neither", "root-only", "site-only"])
def test_publication_hook_requires_both_explicit_settings(tmp_path, monkeypatch, enabled):
    monkeypatch.delenv("NVFLARE_PROCESS_GPU_DELAY_ROOT", raising=False)
    monkeypatch.delenv("NVFLARE_PROCESS_GPU_DELAY_SITE", raising=False)
    if enabled == "root-only":
        monkeypatch.setenv("NVFLARE_PROCESS_GPU_DELAY_ROOT", str(tmp_path))
    if enabled == "site-only":
        monkeypatch.setenv("NVFLARE_PROCESS_GPU_DELAY_SITE", "site-1")
    send, try_send = ClientRunner._send_task_result, ClientRunner._try_send_result_once
    runpy.run_path(str(HOOK))
    assert ClientRunner._send_task_result is send
    assert ClientRunner._try_send_result_once is try_send


def test_publication_hook_does_not_delay_or_drop_receipts_for_another_site(tmp_path, monkeypatch):
    monkeypatch.setenv("NVFLARE_PROCESS_GPU_DELAY_ROOT", str(tmp_path))
    monkeypatch.setenv("NVFLARE_PROCESS_GPU_DELAY_SITE", "site-1")
    monkeypatch.setenv("NVFLARE_PROCESS_GPU_RETRY_ONCE", "1")
    calls = []

    def send(self, result, task_id, context):
        calls.append("send")
        return True

    def try_send(self, result, task_id, context):
        calls.append("try_send")
        return _TASK_CHECK_RESULT_OK

    monkeypatch.setattr(ClientRunner, "_send_task_result", send)
    monkeypatch.setattr(ClientRunner, "_try_send_result_once", try_send)
    runpy.run_path(str(HOOK))
    context = FLContext()
    context.set_prop(ReservedKey.IDENTITY_NAME, "site-2")
    runner = SimpleNamespace(run_abort_signal=Signal())
    assert ClientRunner._send_task_result(runner, Shareable(), "task", context)
    assert ClientRunner._try_send_result_once(runner, Shareable(), "task", context) == _TASK_CHECK_RESULT_OK
    assert calls == ["send", "try_send"]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("initial", [None, "{"])
def test_probe_reader_waits_for_a_complete_record(tmp_path, monkeypatch, initial):
    path = tmp_path / "probe.json"
    if initial is not None:
        path.write_text(initial)
    clock = SimpleNamespace(now=0.0)
    expected = {"pid": 123, "allocated_bytes": 64}

    def sleep(delay):
        clock.now += delay
        path.write_text(json.dumps(expected))

    monkeypatch.setattr(cuda_probe, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep))
    assert cuda_probe.wait_for_probe(path, timeout=1) == expected
    assert clock.now > 0


def test_probe_reader_bounds_a_truncated_record(tmp_path, monkeypatch):
    path = tmp_path / "probe.json"
    path.write_text("{")
    clock = SimpleNamespace(now=0.0)

    def sleep(delay):
        clock.now += delay

    monkeypatch.setattr(cuda_probe, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep))
    with pytest.raises(TimeoutError, match="complete JSON record"):
        cuda_probe.wait_for_probe(path, timeout=0.1)
    assert clock.now == pytest.approx(0.1)


@pytest.mark.parametrize("lifetime,device", [("job", "cpu"), ("task", "cuda")])
def test_exporter_packages_workload_without_vendoring_pytorch(tmp_path, monkeypatch, lifetime, device):
    argv = ["export_job", "--lifetime", lifetime, "--device", device, "--output", str(tmp_path)]
    if lifetime == "task":
        argv += ["--zero-gpu-site", "site-2"]
    monkeypatch.setattr(sys, "argv", argv)
    export_job.main()
    job = next(tmp_path.iterdir())
    config_path = next(job.rglob("config_fed_client.json"))
    config = json.loads(config_path.read_text())
    args = config["executors"][0]["executor"]["args"]
    assert args["execution_mode"] == "in_process"
    assert config.get("execution_lifetime", "job") == lifetime
    script = config_path.parents[1] / "custom" / args["task_script_path"]
    assert script.is_file()
    assert list(job.rglob("torch")) == []
    assert list(job.rglob("sympy")) == []
    meta = json.loads((job / "meta.json").read_text())
    if lifetime == "task":
        assert meta["resource_spec"] == {"@default": {"num_of_gpus": 1}, "site-2": {"num_of_gpus": 0}}
        assert meta["cj_resource_spec"] == {"@ALL": {}}
