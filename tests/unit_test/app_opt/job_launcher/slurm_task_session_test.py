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
"""Slurm workers join a live CJ Cell with shared, job-scoped credential paths."""

import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import replace
from pathlib import Path

import pytest

from nvflare.apis.analytix import TrackConst
from nvflare.apis.dxo import from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.shareable import Shareable
from nvflare.apis.task_launcher_spec import TaskLauncherError, TaskResourceRequest
from nvflare.apis.workspace import Workspace
from nvflare.app_opt.job_launcher.slurm.config import CommandResult, SubmissionResult
from nvflare.app_opt.job_launcher.slurm.task_launcher import SlurmTaskLauncher
from nvflare.private.fed.task_worker.protocol import write_bootstrap
from nvflare.private.fed.task_worker.supervisor import TaskSupervisor
from nvflare.private.fed.utils.job_cert_utils import job_cert_paths
from tests.unit_test.app_opt.job_launcher.slurm_task_integration_test import _context
from tests.unit_test.app_opt.job_launcher.slurm_task_launcher_test import _backend, _supervisor_case
from tests.unit_test.fuel.f3.cellnet.job_cert_binding_live_test import _JOB_B, _free_port, _write_pki
from tests.unit_test.private.fed.conftest import isolated_fobs  # noqa: F401


def _cj_session(root, security, ready, stop, records):
    from nvflare.client.decomposers import register_framework_decomposers
    from nvflare.fuel.f3.cellnet.cell import Cell
    from nvflare.fuel.utils.config_service import ConfigService
    from nvflare.private.fed.task_worker import cell_session
    from nvflare.private.fed.utils.fed_utils import nvflare_fobs_initialize

    cell = session = None
    try:
        ConfigService.initialize(section_files={}, config_path=[str(Path(root, "local"))])
        nvflare_fobs_initialize()
        register_framework_decomposers("numpy", "numpy")
        certificate, key = job_cert_paths(str(Path(root, _JOB_B)))
        credentials = {}
        if security == "mtls":
            credentials = {
                "ca_cert": str(Path(root, "startup", "rootCA.pem")),
                "client_cert": certificate,
                "client_key": key,
                "server_cert": certificate,
                "server_key": key,
                "connection_security": "mtls",
            }
        cell = Cell(
            f"site-1.{_JOB_B}",
            f"{'stcp' if security == 'mtls' else 'tcp'}://127.0.0.1:{_free_port()}",
            secure=security == "mtls",
            credentials=credentials,
            create_internal_listener=False,
            auth_identity="site-1",
            auth_identity_map={f"site-1.{_JOB_B}": "site-1"},
        )
        cell.start()
        launcher = SlurmTaskLauncher(
            workspace_path=root,
            shared_filesystem=True,
            cgroup_enforced=True,
            worker_listener_host="127.0.0.1",  # This local test's allocations share a host.
        )
        cell_session.listen(cell, launcher.get_worker_listener_host())
        session = cell_session.CellWorkerSession(cell, lambda data: records.put(dict(data)))
        ready.put(session.bootstrap_fields())
        stop.wait(90)
    except BaseException:
        ready.put(traceback.format_exc())
    finally:
        if session is not None:
            session.close()
        if cell is not None:
            cell.stop()


@pytest.mark.parametrize("security", ["clear", "mtls", "mtls_alias"])
@pytest.mark.parametrize("gpu_count", [0, 1])
@pytest.mark.usefixtures("isolated_fobs")
def test_native_worker_hello_and_analytics_use_cj_session(tmp_path, monkeypatch, security, gpu_count):
    use_alias = security == "mtls_alias"
    security = "mtls" if use_alias else security
    backend = _backend(
        tmp_path,
        monkeypatch,
        gpu_count,
        job_id=_JOB_B,
        python_path=sys.executable,
        worker_listener_host="127.0.0.1",
    )
    launcher, adapter, request = backend
    launcher.environment["PYTHONPATH"] = os.getcwd()
    root = Path(launcher.manager.config.workspace_path)
    identity, store, arguments = _supervisor_case(backend, write_result=False)
    (root / "local" / "comm_config.json").write_text(
        json.dumps(
            {
                "internal": {
                    "scheme": "stcp" if security == "mtls" else "tcp",
                    "resources": {"host": "cp.invalid", "port": 8102, "connection_security": security},
                }
            }
        )
    )
    if security == "mtls":
        (tmp_path / "pki").mkdir()
        pki = _write_pki(str(tmp_path / "pki"))
        certificate, key = job_cert_paths(request.cwd)
        Path(certificate).parent.mkdir()
        shutil.copy(pki["rootCA.pem"], root / "startup" / "rootCA.pem")
        shutil.copy(pki["job_b.crt"], certificate)
        shutil.copy(pki["job_b.key"], key)
    script = root / _JOB_B / "app_site-1" / "custom" / "session_probe.py"
    script.parent.mkdir()
    script.write_text(
        "import nvflare.client as flare\n"
        "from nvflare.apis.analytix import AnalyticsDataType\n"
        "flare.init()\n"
        "model = flare.receive()\n"
        "flare.log('native_session', 0.5, AnalyticsDataType.SCALAR)\n"
        "flare.send(flare.FLModel(params=model.params, current_round=model.current_round))\n"
    )
    (root / _JOB_B / "meta.json").write_text(json.dumps({"byoc": True}))
    ctx = mp.get_context("spawn")
    ready, records, stop = ctx.Queue(), ctx.Queue(), ctx.Event()
    cj = ctx.Process(target=_cj_session, args=(str(root), security, ready, stop, records))
    cj.start()
    try:
        fields = ready.get(timeout=30)
        assert isinstance(fields, dict), fields
        assert fields["connect_url"].startswith(f"{'stcp' if security == 'mtls' else 'tcp'}://127.0.0.1:")
        assert not fields["connect_url"].endswith(":8102")
        if security == "mtls":
            assert fields["credentials"]["client_cert"] == fields["credentials"]["server_cert"] == certificate
            assert fields["credentials"]["client_key"] == fields["credentials"]["server_key"] == key
        if use_alias:
            alias = tmp_path / "workspace_alias"
            alias.symlink_to(root, target_is_directory=True)
            fields["credentials"] = {
                k: str(alias / Path(v).relative_to(root)) for k, v in fields["credentials"].items()
            }
        arguments["bootstrap"] = replace(
            arguments["bootstrap"],
            executor={
                "path": "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor",
                "args": {"execution_mode": "in_process", "task_script_path": "session_probe.py"},
            },
            cell_session=fields,
        )
        workspace = Workspace(str(root), site_name="site-1")
        meta = {"launcher_spec": {"site-1": {"slurm": {"gpus_per_node": gpu_count, "mem_per_node": 2048}}}}
        launcher.handle_event(EventType.START_RUN, _context(workspace, meta, job_id=_JOB_B))

        def submit(argv, timeout):
            assert ("--gres=gpu:1" if gpu_count else "--gres=none") in argv
            environment = {"SLURM_JOB_ID": "42"}
            if gpu_count:
                environment["CUDA_VISIBLE_DEVICES"] = "0"
            completed = subprocess.run(["/bin/bash", argv[-1]], env=environment, capture_output=True, timeout=45)
            assert completed.returncode == 0, completed.stderr.decode()
            return SubmissionResult(CommandResult(0, "42", ""), "42")

        adapter.submit.side_effect = submit
        arguments["request_factory"] = lambda attempt, path: replace(
            request,
            argv=(
                sys.executable,
                "-m",
                "nvflare.private.fed.app.client.task_worker_process",
                "--bootstrap",
                path,
                "--parent_pid",
                str(os.getpid()),
            ),
            resources=TaskResourceRequest(),
        )
        outcome = TaskSupervisor().run(**arguments)
        assert outcome.execution_status.succeeded
        metric = Shareable()
        metric.update(records.get(timeout=5))
        data = from_shareable(metric).data
        assert data[TrackConst.TRACK_KEY] == "native_session"
        assert data[TrackConst.TRACK_VALUE] == 0.5
    finally:
        stop.set()
        cj.join(15)
        if cj.is_alive():
            cj.kill()
            cj.join(5)


def test_default_listener_host_is_this_cj_host(tmp_path, monkeypatch):
    launcher, _, _ = _backend(tmp_path, monkeypatch, 0)
    monkeypatch.setattr("socket.getfqdn", lambda: "cj.compute.example")
    assert launcher.get_worker_listener_host() == "cj.compute.example"
    monkeypatch.setattr("socket.getfqdn", lambda: "localhost")
    with pytest.raises(RuntimeError, match="worker-reachable"):
        launcher.get_worker_listener_host()


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "bad host", "host:8002", "https://host"])
def test_invalid_advertised_host_fails_closed(tmp_path, monkeypatch, host):
    launcher, _, _ = _backend(tmp_path, monkeypatch, 0, worker_listener_host=host)
    with pytest.raises(RuntimeError):
        launcher.get_worker_listener_host()


@pytest.mark.parametrize(
    "problem", ["missing_pair", "other_job", "outside_ca", "traversal_ca", "site_cert", "symlink_key"]
)
def test_unreachable_or_non_job_credentials_fail_before_native_submission(tmp_path, monkeypatch, problem):
    backend = _backend(tmp_path, monkeypatch, 0)
    launcher, adapter, request = backend
    identity, store, arguments = _supervisor_case(backend, write_result=False)
    root = Path(launcher.manager.config.workspace_path)
    certificate, key = job_cert_paths(request.cwd)
    Path(certificate).parent.mkdir()
    for path in (certificate, key, str(root / "startup" / "rootCA.pem")):
        Path(path).write_text("fixture: rejected before TLS opens")
    credentials = {
        "ca_cert": str(root / "startup" / "rootCA.pem"),
        "client_cert": certificate,
        "client_key": key,
        "server_cert": certificate,
        "server_key": key,
    }
    if problem == "missing_pair":
        credentials.pop("server_key")
    elif problem == "other_job":
        credentials["client_cert"] = str(root / "other-job" / "job_cert" / "job.crt")
    elif problem == "outside_ca":
        credentials["ca_cert"] = str(tmp_path / "host-private-ca.pem")
    elif problem == "traversal_ca":
        outside = tmp_path / "host-private-ca.pem"
        outside.write_text("outside fixture")
        credentials["ca_cert"] = str(root / ".." / outside.name)
    elif problem == "site_cert":
        credentials["client_cert"] = str(root / "startup" / "client.crt")
    else:
        Path(key).unlink()
        Path(key).symlink_to(certificate)
    bootstrap = replace(
        arguments["bootstrap"],
        cell_session={
            "connect_url": "stcp://cj.compute.example:8002",
            "worker_fqcn": "site-1.job-1.task_worker_fixture",
            "connection_security": "mtls",
            "credentials": credentials,
        },
    )
    path = str(root / "job-1" / "bootstrap.json")
    write_bootstrap(path, bootstrap)
    launcher.handle_event(EventType.START_RUN, _context(Workspace(str(root), site_name="site-1"), {}))
    staged = replace(
        request,
        argv=(
            sys.executable,
            "-m",
            "nvflare.private.fed.app.client.task_worker_process",
            "--bootstrap",
            path,
            "--parent_pid",
            str(os.getpid()),
        ),
        resources=TaskResourceRequest(),
    )
    with pytest.raises(TaskLauncherError, match="Slurm Cell"):
        launcher.launch_task(staged)
    adapter.submit.assert_not_called()
