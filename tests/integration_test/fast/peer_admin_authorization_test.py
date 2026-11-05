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

"""An enrolled site's transport privileges do not grant administration of a peer.

Real provisioning, server, enrollment, client entry points and admin login; separate
processes/CWDs on a shared host filesystem (no OS user or mount isolation).
"""

import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import pytest

from nvflare.fuel.f3.cellnet.defs import ReturnCode
from nvflare.fuel.flare_api.flare_api import new_secure_session
from nvflare.fuel.utils.network_utils import get_open_ports
from nvflare.lighter.constants import CtxKey
from nvflare.lighter.impl.cert import CertBuilder
from nvflare.lighter.impl.signature import SignatureBuilder
from nvflare.lighter.impl.static_file import StaticFileBuilder
from nvflare.lighter.impl.workspace import WorkspaceBuilder
from nvflare.lighter.provision import prepare_project
from nvflare.lighter.provisioner import Provisioner
from nvflare.private.defs import RequestHeader
from tests.peer_admin_test_utils import _write_json
from tests.process_test_utils import OwnedProcesses


@pytest.mark.timeout(180)
@pytest.mark.parametrize("with_relay", [False, True], ids=["direct", "relay"])
def test_enrolled_peer_cannot_read_client_file(tmp_path, with_relay):
    fed_port, admin_port, relay_port = get_open_ports(3)
    participants = [
        {"name": "localhost", "type": "server", "org": "test", "fed_learn_port": fed_port, "admin_port": admin_port},
        {"name": "site-1", "type": "client", "org": "test"},
        {"name": "site-2", "type": "client", "org": "test"},
        {"name": "admin@test.org", "type": "admin", "org": "test", "role": "project_admin"},
    ]
    if with_relay:
        for participant in participants[1:3]:
            participant["connect_to"] = {"name": "relay-1"}
        participants.append(
            {
                "name": "relay-1",
                "type": "relay",
                "org": "test",
                "listening_host": {
                    "default_host": "localhost",
                    "port": relay_port,
                    "scheme": "stcp",
                    "connection_security": "mtls",
                },
            }
        )
    project = prepare_project(
        {"api_version": 3, "name": "peer-admin", "description": "local regression", "participants": participants}
    )
    ctx = Provisioner(
        str(tmp_path / "provision"),
        [WorkspaceBuilder(), StaticFileBuilder(scheme="stcp"), CertBuilder(), SignatureBuilder()],
    ).provision(project)
    assert not ctx.get(CtxKey.BUILD_ERROR)
    prod = Path(ctx[CtxKey.CURRENT_PROD_DIR])
    evidence = {}
    for name in ("localhost", "site-1", "site-2"):
        evidence[name] = tmp_path / name
        evidence[name].mkdir()
        path = prod / name / "local" / "resources.json"
        config = json.loads(path.with_suffix(".json.default").read_text())
        if name == "localhost":
            config["snapshot_persistor"]["args"]["storage"]["args"]["root_dir"] = str(tmp_path / "snapshots")
            for component in config["components"]:
                if component["id"] == "job_manager":
                    component["args"]["uri_root"] = str(tmp_path / "jobs")
        args = {"directory": str(evidence[name])}
        if name == "site-1":
            args["target"] = "relay-1.site-2" if with_relay else "site-2"
        config["components"].append({"id": "probe", "path": "tests.peer_admin_test_utils.PeerAdminProbe", "args": args})
        path.write_text(json.dumps(config))

    owner = OwnedProcesses()

    def wait_for(predicate, timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            owner.capture()
            assert all(p.poll() is None for p in owner.processes), f"federation process exited; logs: {tmp_path}"
            result = predicate()
            if result:
                return result
            time.sleep(0.1)
        pytest.fail(f"federation readiness/response timeout; logs: {tmp_path}")

    with ExitStack() as stack:
        stack.enter_context(owner)
        roles = [("localhost", "server")]
        if with_relay:
            roles.append(("relay-1", "relay"))
        roles.extend([("site-1", "client"), ("site-2", "client")])
        for name, role in roles:
            log = stack.enter_context((tmp_path / f"{name}.log").open("w"))
            owner.start(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    (
                        f"nvflare.private.fed.app.{role}.{role}_train"
                        if role != "relay"
                        else "nvflare.private.fed.app.relay.relay"
                    ),
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
                cwd=prod / name,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                },
            )
        for directory in evidence.values():
            wait_for(lambda: (directory / "ready.json").exists())
        if with_relay:
            wait_for(lambda: "Started relay" in (tmp_path / "relay-1.log").read_text())
        ready = {name: json.loads((directory / "ready.json").read_text()) for name, directory in evidence.items()}
        assert ready["localhost"]["incoming_filters"] == [
            "FederatedServer._validate_auth_headers",
            "PeerAdminProbe.observe",
        ]
        assert ready["site-1"]["outgoing_filters"] == ["add_authentication_headers", "PeerAdminProbe.forge_server"]

        admin = new_secure_session("admin@test.org", str(prod / "admin@test.org"), timeout=10)
        stack.callback(admin.close)
        wait_for(lambda: len(admin.get_system_info().client_info) == 2)
        marker = json.loads((evidence["site-2"] / "ready.json").read_text())["marker"]
        own_marker = json.loads((evidence["site-1"] / "ready.json").read_text())["marker"]
        assert marker != own_marker
        assert marker in admin.cat_target("site-2", file="peer-marker.txt")
        cases = [
            {"name": "omit_authorization"},
            {
                "name": "forge_admin_role",
                "headers": {
                    RequestHeader.ADMIN_COMMAND: "cat",
                    RequestHeader.REQUIRE_AUTHZ: "true",
                    RequestHeader.USER_NAME: "admin@test.org",
                    RequestHeader.USER_ROLE: "project_admin",
                    RequestHeader.USER_ORG: "test",
                },
            },
            {"name": "missing_auth_token", "omit_token": True},
        ]
        if with_relay:
            cases.append({"name": "forge_server_origin", "forge_origin": True})
        _write_json(evidence["site-1"] / "request.json", cases)
        result_path = evidence["site-1"] / "results.json"
        wait_for(result_path.exists)
        results = json.loads(result_path.read_text())
        assert isinstance(results, list), results
        destination_records = [
            json.loads(line) for line in (evidence["site-2"] / "received.jsonl").read_text().splitlines()
        ]
        server_records = [
            json.loads(line) for line in (evidence["localhost"] / "received.jsonl").read_text().splitlines()
        ]
        peer_origin = "relay-1.site-1" if with_relay else "site-1"
        peer_requests = [r for r in destination_records if r["origin"] == peer_origin and r["type"] == "req"]
        assert len(peer_requests) == 2, destination_records
        assert all("server" in r["route"] for r in peer_requests)
        assert len([r for r in server_records if r["origin"] == peer_origin and r["type"] == "req"]) == 2
        summary = {
            "marker": marker,
            "attacker_marker": own_marker,
            "results": results,
            "destination_requests": destination_records,
            "server_transit": server_records,
            "admin_read": "passed",
            "startup": ready,
        }
        boundary_log = tmp_path / ("relay-1.log" if with_relay else "localhost.log")
        assert "missing auth token" in boundary_log.read_text()
        assert results[2]["rc"] == ReturnCode.TIMEOUT, results[2]
        summary["missing_token_rejected_by_production_filter"] = True
        _write_json(tmp_path / "sanitized-results.json", summary)
        print(f"Sanitized evidence: {tmp_path / 'sanitized-results.json'}")
        assert marker in admin.cat_target("site-2", file="peer-marker.txt")
        assert len(admin.get_system_info().client_info) == 2

        # A real one-round CPU job exercises deployment, scheduling, CJ -> CP
        # notifications, task exchange, aggregation and completion after the attack.
        job = tmp_path / "numpy-job"
        shutil.copytree(Path(__file__).parents[1] / "data/jobs/hello-numpy-sag", job)
        config_path = job / "app/config/config_fed_server.json"
        config = json.loads(config_path.read_text())
        config["workflows"][0]["args"].update(num_rounds=1, wait_time_after_min_received=0)
        config_path.write_text(json.dumps(config))
        job_id = admin.submit_job(str(job))

        def finished():
            status = admin.get_job_meta(job_id).get("status", "")
            return status if status.startswith("FINISHED") else None

        summary["job_status"] = wait_for(finished, timeout=90)
        for name in ("site-1", "site-2"):
            notifications = [json.loads(line) for line in (evidence[name] / "received.jsonl").read_text().splitlines()]
            prefix = "relay-1." if with_relay else ""
            assert any(
                r["origin"] == f"{prefix}{name}.{job_id}" and r["topic"] == "train.notify_job_status"
                for r in notifications
            )
        summary["own_child_status_notifications"] = "observed at both clients"
        _write_json(tmp_path / "sanitized-results.json", summary)
        assert summary["job_status"] == "FINISHED:COMPLETED"
        assert all(marker not in (r["body"] or "") for r in results), results
        assert all(
            r["rc"] == ReturnCode.OK and r["body"] == "request must originate from server" for r in results[:2]
        ), results
        if with_relay:
            assert results[-1]["rc"] == ReturnCode.UNAUTHENTICATED, results[-1]
            assert results[-1]["origin"] == "relay-1"
            assert not any(r["origin"] == "server" and r["topic"] == "sys.shell" for r in destination_records)
