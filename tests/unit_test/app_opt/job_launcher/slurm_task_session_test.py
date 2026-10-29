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

import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.task_launcher_spec import TaskLauncherError, TaskResourceRequest
from nvflare.apis.workspace import Workspace
from nvflare.private.fed.task_worker.protocol import write_bootstrap
from nvflare.private.fed.utils.job_cert_utils import job_cert_paths
from tests.slurm_test_utils import _backend, _supervisor_case
from tests.unit_test.app_opt.job_launcher.slurm_task_integration_test import _context


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
