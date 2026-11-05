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

"""Two real jobs, distinct owners/sites, normal provisioning, enrollment and secure transport."""

import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest
import yaml

import nvflare
from nvflare.fuel.f3.cellnet.defs import ReturnCode
from nvflare.fuel.flare_api.api_spec import AuthorizationError
from nvflare.fuel.flare_api.flare_api import new_secure_session
from nvflare.fuel.utils.network_utils import get_open_ports
from nvflare.job_config.api import FedJob
from nvflare.lighter.prov_utils import prepare_builders
from nvflare.lighter.provision import prepare_project
from nvflare.lighter.provisioner import Provisioner
from nvflare.lighter.utils import update_storage_locations
from tests.process_test_utils import OwnedProcesses, process_running, wait_for_processes


def wait_for(check, description, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(0.2)
    raise AssertionError(f"timed out waiting for {description}")


@pytest.fixture(params=[False, True], ids=["direct", "relay"])
def federation(tmp_path, monkeypatch, request):
    repo = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(repo / "tests/integration_test/data/server_command"))
    module = importlib.import_module("control_job")
    source = Path(nvflare.__file__).resolve().parent.parent
    env = dict(
        os.environ, PYTHONPATH=str(source), PATH=f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"
    )
    project = yaml.safe_load((repo / "tests/integration_test/data/projects/client_api.yml").read_text())
    project["name"] = "server_command_authorization"
    project["participants"].extend(
        {"name": name, "type": "admin", "org": "nvidia", "role": "lead"}
        for name in ["victim@test.org", "attacker@test.org"]
    )
    server = project["participants"][0]
    server["fed_learn_port"], server["admin_port"] = get_open_ports(2)
    for participant in project["participants"]:
        if participant["type"] in ["server", "client"]:
            participant["listening_host"] = {
                "scheme": "stcp",
                "default_host": "localhost",
                "connection_security": "mtls",
            }
    if request.param:
        project["participants"].append(
            {
                "name": "relay-1",
                "type": "relay",
                "org": "nvidia",
                "listening_host": {
                    "scheme": "stcp",
                    "default_host": "localhost",
                    "port": get_open_ports(1)[0],
                    "connection_security": "mtls",
                },
            }
        )
        project["participants"][1]["connect_to"] = {"name": "relay-1", "connection_security": "mtls"}
    Provisioner(str(tmp_path), prepare_builders(project)).provision(prepare_project(project))
    kits = tmp_path / project["name"] / "prod_00"
    for name in ["localhost", "site-1", "site-2"]:
        internal = json.loads((kits / name / "local/comm_config.json").read_text())["internal"]
        assert internal["scheme"] == "stcp"
        assert internal["resources"]["connection_security"] == "mtls"
    update_storage_locations(str(kits / "localhost/local"), str(tmp_path))
    processes, streams, sessions = OwnedProcesses(), [], []
    try:
        sites = [("localhost", "server")]
        if request.param:
            sites.append(("relay-1", "relay"))
        sites.extend([("site-1", "client"), ("site-2", "client")])
        for name, kind in sites:
            workspace = kits / name
            stream = (tmp_path / f"{name}.stdout").open("w")
            streams.append(stream)
            command = [
                sys.executable,
                "-u",
                "-m",
                f"nvflare.private.fed.app.{kind}.{kind if kind == 'relay' else kind + '_train'}",
                "-m",
                str(workspace),
                "-s",
                f"fed_{kind}.json",
                "--set",
                "secure_train=true",
                "config_folder=config",
                "org=nvidia",
            ]
            if kind == "client":
                command.append(f"uid={name}")
            processes.start(command, cwd=workspace, env=env, stdout=stream, stderr=subprocess.STDOUT)
        for name in ["victim@test.org", "attacker@test.org"]:
            sessions.append(
                new_secure_session(name, str(kits / name), timeout=45, command_timeout=15, auto_login_max_tries=1)
            )
            processes.callbacks.append(sessions[-1].close)
        victim, attacker = sessions
        wait_for(lambda: len(victim.get_system_info().client_info) == 2, "two enrolled clients")
        yield kits, module, victim, attacker, request.param
    finally:
        try:
            processes.close()
        finally:
            for stream in streams:
                stream.close()


def submit(kits, module, session, name, site, victim=""):
    job = FedJob(name=name, min_clients=1)
    job.to(module.ControlController(victim=victim), "server")
    job.to(module.ControlExecutor(), site, tasks=["attack" if victim else "hold"])
    path = kits.parent / "exports" / name
    job.export_job(str(path))
    return session.submit_job(str(path / name))


def status(session, job):
    return session.get_job_meta(job)["status"]


def holding(kits, job, session, site="site-2"):
    run = kits / site / job
    wait_for(lambda: (run / "holding.json").exists(), f"victim task running: {job}")
    assert status(session, job) == "RUNNING"
    assert (kits / "localhost" / job / "running.json").exists()
    return run


def runner_aborted(result_dir):
    return any("asked to abort - triggered abort_signal" in p.read_text() for p in result_dir.rglob("log.txt"))


@pytest.mark.timeout(300)
def test_enrolled_job_cannot_control_another_server_job(federation, tmp_path):
    kits, module, victim_session, attacker_session, via_relay = federation
    victim = submit(kits, module, victim_session, "victim", "site-2")
    run = holding(kits, victim, victim_session)
    evidence = {
        "source": str(Path(nvflare.__file__).resolve()),
        "victim_job": victim,
        "before": status(victim_session, victim),
    }
    server_run = kits / "localhost" / victim
    server_started = json.loads((server_run / "running.json").read_text())
    client_started = json.loads((run / "holding.json").read_text())
    evidence["server_source"] = str(Path(server_started["source"]).resolve())
    evidence["victim_client_source"] = str(Path(client_started["source"]).resolve())
    assert evidence["server_source"] == evidence["victim_client_source"] == evidence["source"]
    server_pid = server_started["pid"]
    assert psutil.pid_exists(server_pid)
    # Same policy applies to both lead users: managing somebody else's job is denied.
    with pytest.raises(AuthorizationError) as denied:
        attacker_session.abort_job(victim)
    evidence["admin_denial"] = str(denied.value)
    assert status(victim_session, victim) == "RUNNING"
    attacker = submit(kits, module, attacker_session, "attacker", "site-1", victim)
    attack_file = kits / "site-1" / attacker / "attack.json"
    wait_for(attack_file.exists, "attack reply")
    attack = json.loads(attack_file.read_text())
    evidence["attack"] = attack
    assert str(Path(attack["source"]).resolve()) == evidence["source"]
    assert attack["origin"] == f"{'relay-1.' if via_relay else ''}site-1.{attacker}"
    assert ("relay-1" in attack["parent_reply_route"]) is via_relay
    assert attack["target"] == f"server.{victim}"
    assert victim != attacker
    wait_for(lambda: status(attacker_session, attacker) == "FINISHED:COMPLETED", "attacker task/update completion")
    attacker_result = Path(attacker_session.download_job_result(attacker))
    evidence["forged_origin"] = json.loads(next(attacker_result.rglob("forged_origin.json")).read_text())
    assert evidence["forged_origin"]["sender"] == f"server.{attacker}"
    rejection = "auth token for client server_job is bound to origin <missing>, not message origin server"
    evidence["forged_origin_rejected"] = rejection in (kits / "localhost/log.txt").read_text()
    rejection = "auth token for client server is bound to origin <missing>, not message origin server"
    evidence["server_credential_replay_rejected"] = rejection in (kits / "localhost/log.txt").read_text()
    if (
        attack["rc"] == ReturnCode.AUTHENTICATION_ERROR
        and evidence["forged_origin_rejected"]
        and evidence["server_credential_replay_rejected"]
    ):
        assert status(victim_session, victim) == "RUNNING"
        (run / "release").touch()
    evidence["after"] = wait_for(
        lambda: (s if (s := status(victim_session, victim)).startswith("FINISHED:") else None), "victim terminal state"
    )
    wait_for(lambda: not psutil.pid_exists(server_pid), "victim server process exit")
    evidence["server_process_exited"] = True
    result_dir = Path(victim_session.download_job_result(victim))
    evidence["victim_result_received"] = bool(list(result_dir.rglob("received.json")))
    evidence["runner_aborted"] = runner_aborted(result_dir)
    # Authorized owner -> admin server -> parent server -> server-job abort must still work.
    authorized = submit(kits, module, victim_session, "authorized-abort", "site-1")
    holding(kits, authorized, victim_session, "site-1")
    evidence["authorized_job"] = authorized
    victim_session.abort_job(authorized)
    evidence["authorized_abort"] = wait_for(
        lambda: (s if (s := status(victim_session, authorized)).startswith("FINISHED:") else None), "authorized abort"
    )
    authorized_result = Path(victim_session.download_job_result(authorized))
    evidence["authorized_runner_aborted"] = runner_aborted(authorized_result)
    (tmp_path / "evidence.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))
    assert evidence["authorized_abort"] == "FINISHED:ABORTED"
    assert evidence["authorized_runner_aborted"] is True
    assert attack["rc"] == ReturnCode.AUTHENTICATION_ERROR
    assert evidence["forged_origin_rejected"] is True
    assert evidence["server_credential_replay_rejected"] is True
    assert evidence["runner_aborted"] is False
    assert evidence["after"] == "FINISHED:COMPLETED"
    received = json.loads(next(result_dir.rglob("received.json")).read_text())
    assert received["job"] == victim and received["aborted"] is False


@pytest.mark.timeout(40)
def test_process_ownership_survives_early_parent_exit(tmp_path):
    ready = tmp_path / "child.pid"
    child = None
    try:
        with OwnedProcesses(grace=0.2) as owner:
            parent = owner.start(
                [sys.executable, str(Path(__file__).parents[1] / "data/server_command/orphan_parent.py"), str(ready)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            assert parent.wait(timeout=15) == 0
            child = psutil.Process(int(ready.read_text()))
            assert process_running(child)
            assert child.ppid() != parent.pid
        assert not process_running(child), "orphaned job survived federation cleanup"
    finally:
        if child is not None and process_running(child):
            child.kill()
            _, alive = wait_for_processes([child], timeout=5)
            assert not alive
