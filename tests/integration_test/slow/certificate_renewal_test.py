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

"""Real secure FL jobs survive renewal and the original endpoint certificate expiry.

Run explicitly with pytest; all kits, keys, jobs and logs are generated in tmp_path.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import pytest
from cryptography import x509

from nvflare.fuel.flare_api.api_spec import NoConnection
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
from tests.unit_test.security.certificate_renewal_test import issue


def _json(path, data):
    path.write_text(json.dumps(data))


@pytest.mark.slow
@pytest.mark.parametrize("scheme", ["stcp", "https", "grpcs"])
@pytest.mark.parametrize("external_job_ca,legacy_client", [(False, False), (True, False), (True, True)])
def test_federated_job_survives_parent_certificate_renewal(tmp_path, scheme, external_job_ca, legacy_client):
    fed_port, admin_port = get_open_ports(2)
    project = prepare_project(
        {
            "api_version": 3,
            "name": "renewal-test",
            "description": "local renewal integration test",
            "participants": [
                {
                    "name": "localhost",
                    "type": "server",
                    "org": "test",
                    "external_cert": True,
                    "external_job_ca": external_job_ca,
                    "fed_learn_port": fed_port,
                    "admin_port": admin_port,
                },
                {"name": "site-1", "type": "client", "org": "test", "external_cert": not legacy_client},
                {"name": "admin@test.org", "type": "admin", "org": "test", "role": "project_admin"},
            ],
        }
    )
    ctx = Provisioner(
        str(tmp_path / "provision"),
        [WorkspaceBuilder(), StaticFileBuilder(scheme=scheme), CertBuilder(), SignatureBuilder()],
    ).provision(project)
    assert not ctx.get(CtxKey.BUILD_ERROR)
    prod = Path(ctx[CtxKey.CURRENT_PROD_DIR])
    resources_path = prod / "localhost" / "local" / "resources.json"
    resources = json.loads(resources_path.with_suffix(".json.default").read_text())
    resources["snapshot_persistor"]["args"]["storage"]["args"]["root_dir"] = str(tmp_path / "snapshots")
    for component in resources["components"]:
        if component["id"] == "job_manager":
            component["args"]["uri_root"] = str(tmp_path / "jobs")
    _json(resources_path, resources)
    root_key, root = ctx[CtxKey.ROOT_PRI_KEY], ctx[CtxKey.ROOT_CERT]
    if external_job_ca:
        # Disposable approved hierarchy, independent of endpoint issuance. No issuer
        # service is available to the running federation; all job signing stays local.
        issuer_key, issuer_pub = generate_keys()
        issuer = generate_cert(Identity("issuer", "IT"), root.subject, root_key, issuer_pub, ca=True, ca_path_length=1)
        job_ca_key, job_ca_pub = generate_keys()
        job_ca = generate_cert(
            Identity("job-ca", "Federation"),
            issuer.subject,
            issuer_key,
            job_ca_pub,
            ca=True,
            ca_path_length=0,
            uri_names=[job_ca_marker_uri()],
        )
        startup = prod / "localhost" / "startup"
        assert not (startup / "job_ca.key").exists()
        (startup / "job_ca.crt").write_bytes(serialize_cert(job_ca) + serialize_cert(issuer))
        (startup / "job_ca.key").write_bytes(serialize_pri_key(job_ca_key))
    renewable_parents = [("localhost", "server")]
    if not legacy_client:
        renewable_parents.append(("site-1", "client"))
    keys = {}
    original_expiry = 0
    for name, role in renewable_parents:
        kit = prod / name
        key, _ = generate_keys()
        keys[name] = key
        (kit / "startup" / f"{role}.key").write_bytes(serialize_pri_key(key))
        cert = issue(root_key, root, key, name, seconds=40)
        (kit / "startup" / f"{role}.crt").write_bytes(cert)
        original_expiry = max(original_expiry, x509.load_pem_x509_certificate(cert).not_valid_after_utc.timestamp())
        _json(kit / "local" / "comm_config.json", {"certificate_renewal": True})

    job = tmp_path / "renewal-job"
    source = Path(__file__).parents[1] / "data" / "jobs" / "hello-numpy-sag"
    shutil.copytree(source, job)
    meta = json.loads((job / "meta.json").read_text())
    meta.update(name="certificate-renewal", min_clients=1, deploy_map={"app": ["server", "site-1"]})
    _json(job / "meta.json", meta)
    server_config = job / "app" / "config" / "config_fed_server.json"
    config = json.loads(server_config.read_text())
    config["workflows"][0]["args"].update(min_clients=1, num_rounds=20, wait_time_after_min_received=0)
    _json(server_config, config)
    client_config = job / "app" / "config" / "config_fed_client.json"
    config = json.loads(client_config.read_text())
    config["executors"][0]["executor"]["args"]["sleep_time"] = 2
    _json(client_config, config)

    processes, logs = [], []
    api = None
    try:
        for name, role in (("localhost", "server"), ("site-1", "client")):
            log = (tmp_path / f"{role}.log").open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(
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
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        deadline = time.time() + 60
        while time.time() < deadline:
            assert all(p.poll() is None for p in processes), "endpoint failed; inspect temporary server/client logs"
            try:
                api = new_secure_session("admin@test.org", str(prod / "admin@test.org"), timeout=5)
                if api.get_system_info().client_info:
                    break
                api.close()
                api = None
            except Exception:
                time.sleep(1)
        assert api is not None, "admin/client connection timeout; inspect temporary logs"
        job_id = api.submit_job(str(job))
        deadline = time.time() + 90
        while time.time() < deadline:
            if list((prod / "site-1" / job_id).rglob("model/*.npy")):
                break
            time.sleep(0.5)
        else:
            pytest.fail("job did not start training; inspect temporary logs")
        children = [child for p in processes for child in psutil.Process(p.pid).children(recursive=True)]
        assert children, "test must observe running job processes before renewal"
        observation_counts = {
            role: (tmp_path / f"{role}.log").read_text().count("Observed renewed endpoint certificate")
            for _, role in renewable_parents
        }
        for name, role in renewable_parents:
            path = prod / name / "startup" / f"{role}.crt"
            candidate = path.with_suffix(".next")
            candidate.write_bytes(issue(root_key, root, keys[name], name, seconds=1200))
            candidate.replace(path)
        deadline = time.time() + 15
        while time.time() < deadline:
            if all(
                (tmp_path / f"{role}.log").read_text().count("Observed renewed endpoint certificate")
                > observation_counts[role]
                for _, role in renewable_parents
            ):
                break
            time.sleep(0.2)
        else:
            pytest.fail("renewal was not observed by every renewable parent")
        assert all(p.poll() is None for p in processes)
        assert all(child.is_running() for child in children), "renewal restarted a job process"
        deadline = time.time() + 150
        status, metadata = "", {}
        while time.time() < deadline:
            try:
                metadata = api.get_job_meta(job_id)
            except NoConnection:
                # Tolerate a transient admin query failure while observing the job.
                time.sleep(1)
                continue
            status = metadata.get("status", "")
            if status.startswith("FINISHED"):
                break
            time.sleep(2)
        assert status == "FINISHED:COMPLETED", metadata
        assert time.time() > original_expiry, "job must run beyond the original endpoint certificate expiry"
        assert all(p.poll() is None for p in processes)
        model = np.load(prod / "site-1" / job_id / "model" / "best_numpy.npy", allow_pickle=False)
        np.testing.assert_array_equal(model, np.arange(1, 10).reshape(3, 3) + 20)
    finally:
        if api:
            api.close()
        for p in processes:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait(timeout=5)
        for log in logs:
            log.close()
