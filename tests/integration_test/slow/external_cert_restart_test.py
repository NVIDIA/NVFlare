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

"""External endpoint credentials take effect after restart, with real mTLS peers."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from nvflare.fuel.flare_api.flare_api import new_secure_session
from nvflare.fuel.sec.cert_uri import job_ca_marker_uri
from nvflare.fuel.utils.network_utils import get_open_ports
from nvflare.lighter.constants import CtxKey
from nvflare.lighter.impl.cert import CertBuilder
from nvflare.lighter.impl.signature import SignatureBuilder
from nvflare.lighter.impl.static_file import StaticFileBuilder
from nvflare.lighter.impl.workspace import WorkspaceBuilder
from nvflare.lighter.provision import prepare_project
from nvflare.lighter.provisioner import Provisioner
from nvflare.lighter.utils import Identity, generate_cert, generate_keys, serialize_cert, serialize_pri_key

pytestmark = pytest.mark.slow


def _install_pair(directory, role, name, root, root_key, key=None):
    directory.mkdir(exist_ok=True)
    if key is None:
        key = generate_keys()[0]
    cert = generate_cert(
        Identity(name), root.subject, root_key, key.public_key(), server_default_host="localhost", valid_days=90
    )
    (directory / "rootCA.pem").write_bytes(serialize_cert(root))
    (directory / f"{role}.crt").write_bytes(serialize_cert(cert))
    (directory / f"{role}.key").write_bytes(serialize_pri_key(key))
    return key


def test_secure_cpu_jobs_after_client_and_server_key_replacement(tmp_path):
    fed_port, admin_port = get_open_ports(2)
    project = prepare_project(
        {
            "api_version": 3,
            "name": "restart-test",
            "description": "external credential restart test",
            "participants": [
                {
                    "name": "localhost",
                    "type": "server",
                    "org": "test",
                    "external_cert": True,
                    "external_job_ca": True,
                    "fed_learn_port": fed_port,
                    "admin_port": admin_port,
                },
                {"name": "site-1", "type": "client", "org": "test", "external_cert": True},
                {"name": "admin@test.org", "type": "admin", "org": "test", "role": "project_admin"},
            ],
        }
    )
    ctx = Provisioner(
        str(tmp_path / "provision"),
        [WorkspaceBuilder(), StaticFileBuilder(scheme="stcp"), CertBuilder(), SignatureBuilder()],
    ).provision(project)
    assert not ctx.get(CtxKey.BUILD_ERROR)
    prod = Path(ctx[CtxKey.CURRENT_PROD_DIR])
    resources_path = prod / "localhost" / "local" / "resources.json"
    resources = json.loads(resources_path.with_suffix(".json.default").read_text())
    resources["snapshot_persistor"]["args"]["storage"]["args"]["root_dir"] = str(tmp_path / "snapshots")
    for component in resources["components"]:
        if component["id"] == "job_manager":
            component["args"]["uri_root"] = str(tmp_path / "jobs")
    resources_path.write_text(json.dumps(resources))
    root_key, root = ctx[CtxKey.ROOT_PRI_KEY], ctx[CtxKey.ROOT_CERT]
    issuer_key, issuer_pub = generate_keys()
    intermediate = generate_cert(
        Identity("issuer", "IT"), root.subject, root_key, issuer_pub, ca=True, ca_path_length=1
    )
    job_key, job_pub = generate_keys()
    job_ca = generate_cert(
        Identity("job-ca", "Federation"),
        intermediate.subject,
        issuer_key,
        job_pub,
        ca=True,
        ca_path_length=0,
        uri_names=[job_ca_marker_uri()],
    )
    startup = prod / "localhost" / "startup"
    (startup / "job_ca.crt").write_bytes(serialize_cert(job_ca) + serialize_cert(intermediate))
    (startup / "job_ca.key").write_bytes(serialize_pri_key(job_key))
    roles = {"localhost": "server", "site-1": "client"}
    for name, role in roles.items():
        _install_pair(prod / name / "startup", role, name, root, root_key)

    job = tmp_path / "numpy-job"
    shutil.copytree(Path(__file__).parents[1] / "data" / "jobs" / "hello-numpy-sag", job)
    meta = json.loads((job / "meta.json").read_text())
    meta.update(name="external-certs", min_clients=1, deploy_map={"app": ["server", "site-1"]})
    (job / "meta.json").write_text(json.dumps(meta))
    server_config = job / "app" / "config" / "config_fed_server.json"
    config = json.loads(server_config.read_text())
    config["workflows"][0]["args"].update(min_clients=1, num_rounds=2, wait_time_after_min_received=0)
    server_config.write_text(json.dumps(config))

    processes, logs = {}, []
    api = None

    def start(name):
        role = roles[name]
        log = (tmp_path / f"{role}-{len(logs)}.log").open("w")
        logs.append(log)
        processes[name] = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                f"nvflare.private.fed.app.{role}.{role}_train",
                "-m",
                str(prod / name),
                "-s",
                f"fed_{role}.json",
                "--set",
                "secure_train=true",
                f"uid={name}",
                "org=test",
                "config_folder=config",
                "heart_beat_interval=2",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def stop(name):
        process = processes.pop(name)
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)

    try:
        start("localhost")
        start("site-1")
        for restarted in (None, "site-1", "localhost"):
            if restarted:
                surviving = "localhost" if restarted == "site-1" else "site-1"
                surviving_pid = processes[surviving].pid
                stop(restarted)
                _install_pair(prod / restarted / "startup", roles[restarted], restarted, root, root_key)
                if api:
                    api.close()
                    api = None
                start(restarted)
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                assert all(p.poll() is None for p in processes.values()), "endpoint failed; inspect temporary logs"
                try:
                    if api is None:
                        api = new_secure_session("admin@test.org", str(prod / "admin@test.org"), timeout=5)
                    if api.get_system_info().client_info:
                        break
                except Exception:
                    if api:
                        api.close()
                        api = None
                time.sleep(0.5)
            else:
                pytest.fail(f"client did not register after {restarted or 'initial startup'}; inspect logs")
            job_id = api.submit_job(str(job))
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                metadata = api.get_job_meta(job_id)
                if metadata.get("status", "").startswith("FINISHED"):
                    break
                time.sleep(0.5)
            assert metadata.get("status") == "FINISHED:COMPLETED", metadata
            model = np.load(prod / "site-1" / job_id / "model" / "best_numpy.npy", allow_pickle=False)
            np.testing.assert_array_equal(model, np.arange(1, 10).reshape(3, 3) + 2)
            if restarted:
                assert processes[surviving].pid == surviving_pid
    finally:
        if api:
            api.close()
        for name in list(processes):
            stop(name)
        for log in logs:
            log.close()
