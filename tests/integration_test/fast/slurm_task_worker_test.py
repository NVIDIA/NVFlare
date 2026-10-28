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

import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, TaskResultReceipt
from nvflare.apis.signal import Signal
from nvflare.apis.task_launcher_spec import TaskResourceRequest
from nvflare.apis.workspace import Workspace
from nvflare.app_opt.job_launcher.slurm.config import CommandResult, SubmissionResult
from nvflare.app_opt.job_launcher.slurm.task_launcher import WORKER_MODULE, SlurmTaskLauncher, _batch_script
from nvflare.app_opt.job_launcher.slurm.task_service import _CHANNEL, ENV_TASK_OWNER, ENV_TASK_TOKEN, SlurmTaskService
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.private.fed.client.client_json_config import ClientJsonConfigurator
from nvflare.private.fed.client.task_worker_executor import TaskWorkerExecutor
from nvflare.private.fed.task_worker import cell_session
from nvflare.private.fed.task_worker.supervisor import TaskSupervisor
from tests.slurm_test_utils import _accounting, _backend, _query, _supervisor_case
from tests.unit_test.app_opt.job_launcher import slurm_task_integration_test as slurm_cases

_context = slurm_cases._context
profile = slurm_cases.profile


@pytest.fixture(params=(0, 1), ids=("cpu", "gpu"))
def backend(tmp_path, monkeypatch, request):
    return _backend(tmp_path, monkeypatch, request.param)


def test_batch_quoting_and_environment_are_explicit(backend, monkeypatch):
    launcher, adapter, request = backend
    output = Path(request.cwd, "command.json")
    malicious = "literal $(touch injected); 'quoted'\nnext"
    code = "import json,os,sys; open(sys.argv[1],'w').write(json.dumps([sys.argv[2],dict(os.environ)]))"
    request = replace(
        request,
        argv=(sys.executable, "-c", code, str(output), malicious),
        environment={"PATH": "/usr/bin:/bin", "SETTING": malicious},
    )
    script = Path(request.cwd, "batch.sh")
    script.write_text(_batch_script(request, "export SETTING='site runtime'"))
    subprocess.run(
        ["/bin/bash", str(script)],
        check=True,
        timeout=20,
        env={"SLURM_JOB_ID": "42", "CJ_PRIVATE": "not forwarded", "CUDA_VISIBLE_DEVICES": "7", "SLURM_JWT": "secret"},
    )
    argument, environment = json.loads(output.read_text())
    assert argument == malicious
    assert environment["SETTING"] == "site runtime"
    assert environment["SLURM_JOB_ID"] == "42"
    assert environment["CUDA_VISIBLE_DEVICES"] == ("7" if request.resources.gpu_count else "")
    assert "CJ_PRIVATE" not in environment
    assert "SLURM_JWT" not in environment
    assert not Path(request.cwd, "injected").exists()


@pytest.mark.parametrize("visibility,returncode", [("2", 0), (None, 102)])
def test_gpu_worker_requires_allocation_visibility(backend, visibility, returncode):
    _, _, request = backend
    output = Path(request.cwd, "gpu-environment.json")
    code = "import os,sys; open(sys.argv[1],'w').write(os.environ['CUDA_VISIBLE_DEVICES'])"
    request = replace(
        request, argv=(sys.executable, "-c", code, str(output)), resources=TaskResourceRequest(gpu_count=1)
    )
    script = Path(request.cwd, "gpu-batch.sh")
    script.write_text(_batch_script(request, ""))
    environment = {"SLURM_JOB_ID": "42"}
    if visibility is not None:
        environment["CUDA_VISIBLE_DEVICES"] = visibility
    completed = subprocess.run(["/bin/bash", str(script)], env=environment, capture_output=True, timeout=20)
    assert completed.returncode == returncode
    if returncode == 0:
        assert output.read_text() == visibility
    else:
        assert not output.exists()


@pytest.mark.usefixtures("isolated_fobs")
def test_real_worker_result_survives_mock_scheduler_settlement_and_receipt(backend):
    launcher, adapter, _ = backend
    identity, store, arguments = _supervisor_case(backend)
    supervisor = TaskSupervisor()
    outcome = supervisor.run(**arguments)
    assert outcome.execution_status.succeeded
    np.testing.assert_array_equal(from_shareable(outcome.result).data["numpy_key"], np.full(3, 3.0))
    assert Path(store.attempt_dir(identity), "result.fobs").exists()
    assert not supervisor.owns_execution
    # Lost receipt resends the committed result and does not resubmit sbatch.
    supervisor.record_receipt(identity, TaskResultReceipt.RETRY)
    supervisor.read_saved_result(identity)
    assert adapter.submit.call_count == 1
    supervisor.record_receipt(identity, TaskResultReceipt.RECEIVED)
    assert not Path(store.attempt_dir(identity), "result.fobs").exists()
    assert Path(store.attempt_dir(identity), "completion.json").exists()
    assert all(Path(handle.job_dir, "scheduler.jsonl").exists() for handle in launcher.manager._handles.values())


@pytest.mark.usefixtures("isolated_fobs")
def test_end_run_during_active_result_reader_retains_payloads(backend):
    identity, store, arguments = _supervisor_case(backend)
    supervisor = TaskSupervisor()
    reading, proceed = threading.Event(), threading.Event()
    original = store.read_result

    def blocked_read(identity):
        reading.set()
        assert proceed.wait(10)
        return original(identity)

    store.read_result = blocked_read
    with ThreadPoolExecutor() as pool:
        future = pool.submit(supervisor.run, **arguments)
        try:
            assert reading.wait(20)
            supervisor.end_run()
            assert Path(store.attempt_dir(identity), "result.fobs").exists()
        finally:
            proceed.set()
        assert future.result(timeout=10).aborted
    assert Path(store.attempt_dir(identity), "result.fobs").exists()


@pytest.fixture
def cj_cell():
    from nvflare.fuel.f3.cellnet.cell import Cell
    from nvflare.fuel.f3.cellnet.core_cell import CoreCell
    from nvflare.fuel.f3.stats_pool import StatsPoolManager

    cells, pools = dict(CoreCell.ALL_CELLS), dict(StatsPoolManager.pools)
    cell = Cell("site-1.job-1", "tcp://127.0.0.1:0", secure=False, credentials={})
    cell.start()
    try:
        yield cell
    finally:
        cell.stop()
        CoreCell.ALL_CELLS.clear()
        CoreCell.ALL_CELLS.update(cells)
        StatsPoolManager.pools.clear()
        StatsPoolManager.pools.update(pools)


@pytest.mark.parametrize("gpu_count", [0, 1])
@pytest.mark.parametrize("workspace_alias", [False, True])
@pytest.mark.usefixtures("isolated_fobs")
def test_staged_cj_request_runs_authorized_worker_with_bound_native_resources(
    tmp_path, monkeypatch, gpu_count, cj_cell, workspace_alias
):
    backend = _backend(tmp_path, monkeypatch, gpu_count, worker_listener_host="127.0.0.1")
    launcher, adapter, request = backend
    launcher.python_path = sys.executable
    launcher.environment["PYTHONPATH"] = os.getcwd()
    identity, store, arguments = _supervisor_case(backend, write_result=False)
    workspace_root = launcher.manager.config.workspace_path
    if workspace_alias:
        alias = tmp_path / "workspace-link"
        alias.symlink_to(workspace_root, target_is_directory=True)
        workspace_root = str(alias)
    workspace = Workspace(workspace_root, site_name="site-1")
    meta = {
        "launcher_spec": {"site-1": {"slurm": {"gpus_per_node": gpu_count, "cpus_per_node": 3, "time": "00:03:00"}}}
    }
    launcher.handle_event(EventType.START_RUN, _context(workspace, meta))
    cell_session.listen(cj_cell, launcher.get_worker_listener_host())
    session = cell_session.CellWorkerSession(cj_cell, lambda data: None)
    arguments["bootstrap"] = replace(
        arguments["bootstrap"], workspace_root=workspace_root, cell_session=session.bootstrap_fields()
    )

    def factory(attempt, path):
        return replace(
            request,
            argv=(
                "/untrusted/cj/python",
                "-m",
                "nvflare.private.fed.app.client.task_worker_process",
                "--bootstrap",
                path,
                "--parent_pid",
                "1234",
            ),
            environment={"CJ_PRIVATE": "not forwarded"},
            resources=TaskResourceRequest(),
            cwd=workspace.get_run_dir(attempt.job_id),
        )

    def submit(argv, timeout):
        assert "--cpus-per-task=3" in argv and "--time=00:03:00" in argv
        assert ("--gres=gpu:1" if gpu_count else "--gres=none") in argv
        text = Path(argv[-1]).read_text()
        assert WORKER_MODULE in text and "--parent_pid" not in text and "/untrusted/cj/python" not in text
        assert "CJ_PRIVATE" not in text
        environment = {"SLURM_JOB_ID": "42"}
        if gpu_count:
            environment["CUDA_VISIBLE_DEVICES"] = "0"
        result = subprocess.run(["/bin/bash", argv[-1]], env=environment, capture_output=True, timeout=30)
        assert result.returncode == 0, result.stderr.decode()
        return SubmissionResult(CommandResult(0, "42", ""), "42")

    adapter.submit.side_effect = submit
    arguments["request_factory"] = factory
    try:
        outcome = TaskSupervisor().run(**arguments)
    finally:
        session.close()
    assert outcome.execution_status.succeeded
    np.testing.assert_array_equal(from_shareable(outcome.result).data["numpy_key"], np.full(3, 3.0))
    assert Path(store.attempt_dir(identity), "result.fobs").exists()


@pytest.mark.parametrize("gpu_count", [0, 1])
@pytest.mark.usefixtures("isolated_fobs")
def test_application_flag_selects_slurm_and_runs_through_normal_cj(profile, gpu_count, cj_cell, monkeypatch):
    root, workspace, site, cj, task = profile
    config_path = Path(workspace.get_client_app_config_file_path("job-1"))
    config_path.write_text(
        json.dumps(
            {
                "format_version": 3,
                "execution_lifetime": "task",
                "executors": [
                    {
                        "tasks": ["train"],
                        "executor": {
                            "path": "nvflare.app_common.np.np_trainer.NPTrainer",
                            "args": {"delta": 2},
                        },
                    }
                ],
            }
        )
    )
    meta = {"job_id": "job-1", "launcher_spec": {"site-1": {"slurm": {"gpus_per_node": gpu_count}}}}
    (root / "job-1" / "meta.json").write_text(json.dumps(meta))
    args = argparse.Namespace(
        sp_scheme="grpc",
        sp_target="localhost:8002",
        client_name="site-1",
        parent_url=None,
        job_id="job-1",
        workspace=str(root),
    )
    configurator = ClientJsonConfigurator(workspace, str(config_path), args, str(config_path.parent.parent))
    configurator.configure()
    executor = configurator.executors[0].executor
    assert type(executor) is TaskWorkerExecutor
    launcher = executor.policy.launcher
    assert type(launcher) is SlurmTaskLauncher
    assert executor.policy.worker_environment() == {}
    assert executor.policy.task_resources(meta, "site-1").is_empty()
    adapter = Mock()
    adapter.accounting_probe.return_value = CommandResult(0, "", "")
    adapter.active_by_id.return_value = _query()
    adapter.task_accounting.return_value = _accounting()
    task.manager.adapter = adapter
    task.manager.initialize()
    task.bind_job("job-1", "site-1", workspace, meta)
    service = SlurmTaskService(SimpleNamespace(get_fqcn=lambda: "site-1", register_request_cb=Mock()))
    owner = service.add_job(task, "job-1", "site-1", "test-capability")
    monkeypatch.setenv(ENV_TASK_OWNER, "site-1")
    monkeypatch.setenv(ENV_TASK_TOKEN, "test-capability")
    send_request = cj_cell.send_request

    def route_request(*args, **kwargs):
        if kwargs.get("channel") == _CHANNEL:
            message = kwargs["request"]
            message.set_header(MessageHeaderKey.ORIGIN, "site-1.job-1")
            return service._request(message)
        return send_request(*args, **kwargs)

    monkeypatch.setattr(cj_cell, "send_request", route_request)

    def submit(argv, timeout):
        assert ("--gres=gpu:1" if gpu_count else "--gres=none") in argv
        assert "--cpus-per-task=4" in argv and "--mem=2048M" in argv
        environment = {"SLURM_JOB_ID": "42"}
        if gpu_count:
            environment["CUDA_VISIBLE_DEVICES"] = "0"
        completed = subprocess.run(["/bin/bash", argv[-1]], env=environment, capture_output=True, timeout=30)
        assert completed.returncode == 0, completed.stderr.decode()
        return SubmissionResult(CommandResult(0, "42", ""), "42")

    adapter.submit.side_effect = submit
    context = _context(workspace, meta)
    context.set_prop(ReservedKey.ENGINE, SimpleNamespace(get_cell=lambda: cj_cell, fire_event=Mock()))
    context.set_prop(FLContextKey.TASK_ID, "task-1")
    context.set_prop(FLContextKey.TASK_NAME, "train")
    context.set_prop(FLContextKey.TASK_ATTEMPT_ID, "attempt-1")
    try:
        executor.handle_event(EventType.START_RUN, context)
        result = executor.execute(
            "train", DXO(DataKind.WEIGHTS, {"numpy_key": np.ones(3)}).to_shareable(), context, Signal()
        )
        np.testing.assert_array_equal(from_shareable(result).data["numpy_key"], np.full(3, 3.0))
        executor.execute("train", DXO(DataKind.WEIGHTS, {"numpy_key": np.ones(3)}).to_shareable(), context, Signal())
        assert adapter.submit.call_count == 1
        context.set_prop(FLContextKey.TASK_RESULT, result)
        context.set_prop(FLContextKey.TASK_RESULT_RECEIPT, TaskResultReceipt.RECEIVED)
        executor.handle_event(EventType.AFTER_SEND_TASK_RESULT, context)
        executor.handle_event(EventType.END_RUN, context)
    finally:
        owner.release()
        owner.thread.join(5)
        assert not owner.thread.is_alive()
