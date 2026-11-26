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
The existing one-round job also verifies task, workflow and model audit observations;
controlled component suites cover execution failures, duplicate/late results, and
abort semantics without adding another federation startup here.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pytest

from nvflare.apis.fl_constant import AdminCommandNames, SiteType, WorkspaceConstants
from nvflare.apis.shareable import ReturnCode as ShareableReturnCode
from nvflare.apis.storage import WORKSPACE_ZIP
from nvflare.apis.workspace import Workspace
from nvflare.app_common.streamers.file_streamer import _ChunkProducer
from nvflare.fuel.f3.cellnet.defs import ReturnCode
from nvflare.fuel.flare_api.api_spec import SessionClosed
from nvflare.fuel.flare_api.flare_api import new_secure_session
from nvflare.fuel.hci.client.api import APIStatus, ResultKey
from nvflare.fuel.hci.proto import InternalCommands
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.utils.network_utils import get_open_ports
from nvflare.lighter.constants import CtxKey
from nvflare.lighter.impl.cert import CertBuilder
from nvflare.lighter.impl.signature import SignatureBuilder
from nvflare.lighter.impl.static_file import StaticFileBuilder
from nvflare.lighter.impl.workspace import WorkspaceBuilder
from nvflare.lighter.provision import prepare_project
from nvflare.lighter.provisioner import Provisioner
from nvflare.private.defs import CellChannel, RequestHeader
from tests.peer_admin_test_utils import _write_json
from tests.process_test_utils import OwnedProcesses
from tests.task_audit_test_utils import MODEL_SOURCE_VALUE, RESULT_MARKER, TASK_MARKER


def _task_details(record):
    return {
        key: json.loads(value)
        for key, value in re.findall(
            r'(task|task_id|workflow|attempt_id|site)=("(?:\\.|[^"\\])*")', record.get("message", "")
        )
    }


def _one_task_record(records, action, **identity):
    matches = [
        record
        for record in records
        if record.get("action") == action
        and all(_task_details(record).get(key) == value for key, value in identity.items())
    ]
    assert len(matches) == 1, (action, identity, matches)
    return matches[0]


@pytest.mark.timeout(90)
def test_admin_can_reconnect_after_server_restarts_during_upload(tmp_path, monkeypatch):
    fed_port, admin_port = get_open_ports(2)
    user = "admin@test.org"
    project = prepare_project(
        {
            "api_version": 3,
            "name": "admin-restart",
            "description": "local restart regression",
            "participants": [
                {
                    "name": "localhost",
                    "type": "server",
                    "org": "test",
                    "fed_learn_port": fed_port,
                    "admin_port": admin_port,
                },
                {"name": user, "type": "admin", "org": "test", "role": "project_admin"},
            ],
        }
    )
    ctx = Provisioner(
        str(tmp_path / "provision"),
        [WorkspaceBuilder(), StaticFileBuilder(scheme="stcp"), CertBuilder(), SignatureBuilder()],
    ).provision(project)
    assert not ctx.get(CtxKey.BUILD_ERROR)
    prod = Path(ctx[CtxKey.CURRENT_PROD_DIR])
    server_root = prod / "localhost"
    admin_root = str(prod / user)
    config_path = server_root / "local" / "resources.json"
    config = json.loads(config_path.with_suffix(".json.default").read_text())
    config["snapshot_persistor"]["args"]["storage"]["args"]["root_dir"] = str(tmp_path / "snapshots")
    for component in config["components"]:
        if component["id"] == "job_manager":
            component["args"]["uri_root"] = str(tmp_path / "jobs")
    config_path.write_text(json.dumps(config))
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    job = tmp_path / "job"
    shutil.copytree(Path(__file__).parents[1] / "data/jobs/hello-numpy-sag", job)
    (job / "app" / "upload-data").write_bytes(os.urandom(2 * 1024 * 1024))

    def start_server(owner, log):
        return owner.start(
            [
                sys.executable,
                "-u",
                "-m",
                "nvflare.private.fed.app.server.server_train",
                "-m",
                str(server_root),
                "-s",
                "fed_server.json",
                "--set",
                "secure_train=true",
                "uid=localhost",
                "org=test",
                "config_folder=config",
            ],
            cwd=server_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[3]), "TMPDIR": str(uploads)},
        )

    with (
        OwnedProcesses() as original,
        OwnedProcesses() as restarted,
        (tmp_path / "server.log").open("w") as original_log,
        (tmp_path / "restarted.log").open("w") as restarted_log,
        ExitStack() as stack,
    ):
        start_server(original, original_log)
        admin = new_secure_session(user, admin_root, timeout=15)
        stack.callback(admin.close)
        assert not admin.list_jobs()
        original_cell = admin.api.cell
        process_replies = _ChunkProducer.process_replies
        acked_chunks = 0

        def restart_after_first_chunk(producer, replies, stream_ctx, fl_ctx):
            nonlocal acked_chunks
            result = process_replies(producer, replies, stream_ctx, fl_ctx)
            assert replies["server"].get_return_code() == ShareableReturnCode.OK
            acked_chunks += 1
            assert acked_chunks == 1 and result is None
            original.close()
            server = start_server(restarted, restarted_log)
            fresh_admin = new_secure_session(user, admin_root, timeout=15)
            stack.callback(fresh_admin.close)
            assert not fresh_admin.list_jobs()
            deadline = time.monotonic() + 10
            while not original_cell.core_cell.is_cell_connected("server"):
                assert server.poll() is None
                assert time.monotonic() < deadline, "original transport did not reconnect to the restarted server"
                time.sleep(0.1)
            return result

        monkeypatch.setattr(_ChunkProducer, "process_replies", restart_after_first_chunk)
        with pytest.raises(SessionClosed):
            admin.submit_job(str(job))
        assert acked_chunks == 1
        admin.try_connect(timeout=10)
        assert admin.api.cell is not original_cell
        assert not admin.list_jobs()


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
        config.setdefault("class_allow_list", []).extend(
            [
                "tests.task_audit_test_utils.TaskAuditJobTrainer",
                "tests.task_audit_test_utils.TaskAuditDataFilter",
                "tests.task_audit_test_utils.TaskAuditResultFilter",
            ]
        )
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
        if not with_relay:
            # Two certificate logins for the same user have independent origins.
            # Exercise rejection over the real streamed HCI command path.
            peer_admin = new_secure_session("admin@test.org", str(prod / "admin@test.org"), timeout=10)
            stack.callback(peer_admin.close)
            caller = peer_admin.api
            check = InternalCommands.CHECK_SESSION
            token = caller.token
            assert caller.server_execute(check)[ResultKey.STATUS] == APIStatus.SUCCESS
            caller.token = admin.api.token
            assert caller.server_execute(check)[ResultKey.STATUS] == APIStatus.ERROR_INACTIVE_SESSION
            caller.token = token
            assert caller.server_execute(check)[ResultKey.STATUS] == APIStatus.SUCCESS
            assert caller.server_execute(InternalCommands.LOGOUT)[ResultKey.STATUS] == APIStatus.SUCCESS
            assert caller.server_execute(check)[ResultKey.STATUS] == APIStatus.ERROR_INACTIVE_SESSION
            assert admin.api.server_execute(check)[ResultKey.STATUS] == APIStatus.SUCCESS
        marker = json.loads((evidence["site-2"] / "ready.json").read_text())["marker"]
        own_marker = json.loads((evidence["site-1"] / "ready.json").read_text())["marker"]
        assert marker != own_marker
        assert marker in admin.cat_target("site-2", file="peer-marker.txt")
        # Inspect the actual fresh-install audit files through the authenticated
        # admin API and real server/client command execution, not just a listing.
        for target in ("server", "site-2"):
            for command in ("cat", "head", "tail", "grep"):
                kwargs = {"file": WorkspaceConstants.AUDIT_LOG}
                if command == "grep":
                    kwargs["pattern"] = "event_id"
                output = getattr(admin, f"{command}_target")(target, **kwargs)
                records = [json.loads(line) for line in output.splitlines()]
                assert records, (target, command, output)
                assert all("event_id" in record and "timestamp" in record for record in records), output

        def audit_records(name):
            path = prod / name / WorkspaceConstants.AUDIT_LOG
            records = [json.loads(line) for line in path.read_text().splitlines()]
            for record in records:
                assert set(record) <= {
                    "event_id",
                    "reference",
                    "timestamp",
                    "user",
                    "action",
                    "job",
                    "scope",
                    "message",
                }
                assert all(isinstance(value, str) for value in record.values())
                # Filenames, shell arguments, reply contents and attacker-supplied
                # identity/reference values must not leak into administrative records.
                serialized = json.dumps(record)
                for forbidden in (
                    "peer-marker.txt",
                    "private-job-error-marker",
                    "forged-admin@test.org",
                    "forged-reference-marker",
                ):
                    assert forbidden not in serialized, record
            return records

        server_audit = audit_records("localhost")
        client_audit = audit_records("site-2")
        assert any(r["action"] == "admin.login.succeeded" and r["user"] == "admin@test.org" for r in server_audit)
        server_by_id = {r["event_id"]: r for r in server_audit}
        receipts = [
            r
            for r in client_audit
            if r["action"] == "client.command.received" and "command=sys.shell" in r.get("message", "")
        ]
        assert receipts, client_audit
        for receipt in receipts:
            assert receipt["user"] == "admin@test.org"
            assert "identity_source=trusted_server" in receipt["message"]
            request = server_by_id[receipt["reference"]]
            assert request["action"] in ("cat", "head", "tail", "grep")
            assert request["user"] == receipt["user"]
            outcomes = [
                r
                for r in client_audit
                if r.get("reference") == receipt["event_id"] and r["action"] == "client.command.handler_returned"
            ]
            assert len(outcomes) == 1, (receipt, client_audit)
        rejection_count = sum(r["action"] == "client.origin.rejected" for r in client_audit)
        cases = [
            {"name": "omit_authorization"},
            {
                "name": "forge_admin_role",
                "headers": {
                    RequestHeader.ADMIN_COMMAND: "cat",
                    RequestHeader.REQUIRE_AUTHZ: "true",
                    RequestHeader.USER_NAME: "forged-admin@test.org",
                    ConnProps.EVENT_ID: "forged-reference-marker",
                    RequestHeader.USER_ROLE: "project_admin",
                    RequestHeader.USER_ORG: "test",
                },
            },
            {"name": "missing_auth_token", "omit_token": True},
        ]
        if with_relay:
            cases.append({"name": "forge_server_origin", "forge_origin": True})
        boundary_log = tmp_path / ("relay-1.log" if with_relay else "localhost.log")
        origin_rejection_log = "rejected message: origin does not belong to sending peer"
        before_initial_rejections = boundary_log.read_text().count(origin_rejection_log)
        _write_json(evidence["site-1"] / "request.json", cases)
        result_path = evidence["site-1"] / "results.json"
        wait_for(result_path.exists)
        results = json.loads(result_path.read_text())
        assert isinstance(results, list), results
        if with_relay:
            assert boundary_log.read_text().count(origin_rejection_log) >= before_initial_rejections + 1
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
        client_audit = audit_records("site-2")
        rejections = [r for r in client_audit if r["action"] == "client.origin.rejected"]
        assert len(rejections) - rejection_count == 2, rejections
        for rejection in rejections:
            assert rejection["user"] == "unknown"
            assert "reference" not in rejection
            assert "job" not in rejection
            assert rejection["message"] == "reason=untrusted_origin"
        summary = {
            "marker": marker,
            "attacker_marker": own_marker,
            "results": results,
            "destination_requests": destination_records,
            "server_transit": server_records,
            "admin_read": "passed",
            "startup": ready,
        }
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
        model_source_name = "private-model-source-marker.npy"
        model_output_dir = "private-model-output-marker"
        model_source_dir = job / "app" / WorkspaceConstants.CUSTOM_FOLDER_NAME
        model_source_dir.mkdir(exist_ok=True)
        np.save(
            model_source_dir / model_source_name, np.array([[MODEL_SOURCE_VALUE]], dtype=np.float32), allow_pickle=False
        )
        persistor = next(component for component in config["components"] if component["id"] == "persistor")
        persistor["args"].update(
            source_ckpt_file_full_name=model_source_name,
            model_dir=model_output_dir,
            model=[[7.0]],
        )
        config["workflows"][0]["args"].update(num_rounds=1, wait_time_after_min_received=0)
        children = tmp_path / "children"
        config["task_data_filters"] = [
            {"tasks": ["train"], "filters": [{"path": "tests.task_audit_test_utils.TaskAuditDataFilter"}]}
        ]
        config["task_result_filters"] = [
            {
                "tasks": ["train"],
                "filters": [
                    {
                        "path": "tests.task_audit_test_utils.TaskAuditResultFilter",
                        "args": {"directory": str(children)},
                    }
                ],
            }
        ]
        config_path.write_text(json.dumps(config))
        config_path = job / "app/config/config_fed_client.json"
        config = json.loads(config_path.read_text())
        config["executors"][0]["executor"] = {
            "path": "tests.task_audit_test_utils.TaskAuditJobTrainer",
            "args": {"directory": str(children)},
        }
        config_path.write_text(json.dumps(config))
        job_id = admin.submit_job(str(job))
        for name in ("site-1", "site-2"):
            wait_for(lambda name=name: (children / f"{name}.json").exists())
        target = json.loads((children / "site-2.json").read_text())["fqcn"]
        assert "private-job-error-marker" in json.dumps(admin.show_errors(job_id, "client", ["site-2"]))
        job_cases = [{"name": "peer_job_errors"}]
        if with_relay:
            job_cases.extend(
                [
                    {"name": "forged_server_job_errors", "forge_origin": True},
                    {"name": "forged_parent_job_errors", "forge_origin": target.rsplit(".", 1)[0]},
                ]
            )
        for case in job_cases:
            case.update(channel=CellChannel.CLIENT_COMMAND, topic=AdminCommandNames.SHOW_ERRORS, target=target)
        result_path.unlink()
        before_rejections = boundary_log.read_text().count(origin_rejection_log)
        _write_json(evidence["site-1"] / "request.json", job_cases)
        wait_for(result_path.exists)
        job_results = json.loads(result_path.read_text())
        assert job_results[0]["rc"] == ReturnCode.UNAUTHENTICATED, job_results
        if with_relay:
            # Forged origins stop at the relay's incoming auth filter before
            # CellNet can generate a protocol reply. Prove the production gate
            # rejected both attempts rather than accepting arbitrary timeouts.
            assert all(r["rc"] == ReturnCode.TIMEOUT for r in job_results[1:]), job_results
            assert boundary_log.read_text().count(origin_rejection_log) >= before_rejections + 2
        assert all("private-job-error-marker" not in json.dumps(r) for r in job_results)
        assert "private-job-error-marker" in json.dumps(admin.show_errors(job_id, "client", ["site-2"]))
        summary["job_commands"] = job_results
        children.joinpath("release").write_text("finish")

        def finished():
            status = admin.get_job_meta(job_id).get("status", "")
            return status if status.startswith("FINISHED") else None

        summary["job_status"] = wait_for(finished, timeout=90)

        # The server archives and removes its job workspace before FINISHED.
        # Read its actual persisted audit file through the authenticated full
        # download. An outputs-only download deliberately omits audit files.
        downloaded = admin.download_job_result(job_id, destination=str(tmp_path / "job-downloads"))
        assert downloaded
        archived_workspace = Path(downloaded) / Path(WORKSPACE_ZIP).stem
        assert archived_workspace.is_dir(), downloaded
        saved_models = list(archived_workspace.rglob(f"{model_output_dir}/server.npy"))
        assert len(saved_models) == 1, saved_models
        # Each client adds one to the supplied weight. Both contributions have
        # equal weight, so averaging preserves that increment.
        expected_weight = MODEL_SOURCE_VALUE + 1
        np.testing.assert_array_equal(np.load(saved_models[0], allow_pickle=False), [[expected_weight]])
        for name in ("site-1", "site-2"):
            assert json.loads((children / f"model-source-{name}.json").read_text()) == {"source_model_seen": True}
        job_audit_paths = {}
        for name in ("localhost", "site-1", "site-2"):
            site_name = SiteType.SERVER if name == "localhost" else name
            workspace = Workspace(str(prod / name), site_name=site_name)
            # Workspace honors the same environment-configured audit roots as
            # the job processes. The full archive combines those source roots.
            path = Path(workspace.get_audit_file_path(job_id))
            if name == "localhost":
                archived = list(archived_workspace.rglob(path.name))
                assert len(archived) == 1, archived
                path = archived[0]
            job_audit_paths[name] = path

        def completed_job_audits():
            try:
                records = {
                    name: [json.loads(line) for line in path.read_text().splitlines()]
                    for name, path in job_audit_paths.items()
                }
            except (FileNotFoundError, json.JSONDecodeError):
                # A different process may be finishing its last append/flush.
                return None
            if not any(event.get("action") == "workflow.finalization.returned" for event in records["localhost"]):
                return None
            if not all(
                any(event.get("action") == "task.publication.received" for event in records[name])
                for name in ("site-1", "site-2")
            ):
                return None
            return records

        job_records = wait_for(completed_job_audits, timeout=30)
        for name in ("localhost", "site-1", "site-2"):
            for event in job_records[name]:
                assert set(event) <= {"event_id", "reference", "timestamp", "user", "action", "job", "scope", "message"}
                assert all(isinstance(value, str) for value in event.values()), event
                assert event["timestamp"].endswith("+00:00"), event
                if event.get("action", "").startswith(("task.", "workflow.")):
                    assert event["job"] == job_id, event
                    assert event["user"] == "unknown", event
                    serialized = json.dumps(event)
                    for forbidden in (
                        TASK_MARKER,
                        RESULT_MARKER,
                        "private-job-error-marker",
                        marker,
                        own_marker,
                        "forged-admin@test.org",
                        "forged-reference-marker",
                        str(tmp_path),
                    ):
                        assert forbidden not in serialized, event
        records_by_site = {name: {event["event_id"]: event for event in events} for name, events in job_records.items()}
        for name, events in job_records.items():
            for event in events:
                if event.get("action", "").startswith(("task.", "workflow.")) and event.get("reference"):
                    assert event["reference"] in records_by_site[name], event

        server_job_audit = job_records["localhost"]
        server_by_id = records_by_site["localhost"]
        model_records = [record for record in server_job_audit if record["action"].startswith("model.")]
        for record in model_records:
            assert record["job"] == job_id, record
            assert record["user"] == "unknown", record
            assert not record["action"].endswith((".error", ".stage_error", ".skipped")), record
            if record.get("reference"):
                assert record["reference"] in server_by_id, record
            for forbidden in (
                model_source_name,
                model_output_dir,
                str(MODEL_SOURCE_VALUE),
                str(expected_weight),
                str(tmp_path),
            ):
                assert forbidden not in record.get("message", ""), record
        for operation, file_stage in (("load", "source_read"), ("save", "write")):
            prefix = f"model.{operation}."
            matching = [
                record
                for record in model_records
                if record["action"] in {prefix + "started", prefix + "returned"}
                or (
                    record["action"] == prefix + "stage_returned"
                    and f"stage={file_stage}" in record.get("message", "").split()
                )
            ]
            assert [record["action"] for record in matching] == [
                prefix + "started",
                prefix + "stage_returned",
                prefix + "returned",
            ], model_records
            if operation == "load":
                assert "status=value_returned" in matching[-1].get("message", "").split(), matching[-1]
        summary["model_audit_records"] = "verified source consumption, saved model, and private model data exclusion"
        workflow_id = "scatter_and_gather"
        initialized = _one_task_record(server_job_audit, "workflow.execution.initialized", workflow=workflow_id)
        control = _one_task_record(server_job_audit, "workflow.control.returned", workflow=workflow_id)
        finalized = _one_task_record(server_job_audit, "workflow.finalization.returned", workflow=workflow_id)
        assert control["reference"] == finalized["reference"] == initialized["event_id"]
        assert server_job_audit.index(initialized) < server_job_audit.index(control) < server_job_audit.index(finalized)

        for name in ("site-1", "site-2"):
            # These markers actually crossed both runtime boundaries. Merely
            # omitting an unused sentinel from the audit output would prove less.
            assert json.loads((children / f"result-{name}.json").read_text()) == {"private_result_seen": True}
            client_job_audit = job_records[name]
            client_by_id = records_by_site[name]
            assignment = _one_task_record(server_job_audit, "task.assignment.prepared", site=name, task="train")
            identity = _task_details(assignment)
            assert identity["workflow"] == workflow_id
            assert identity["task_id"] and identity["attempt_id"]
            legacy_assignment = server_by_id[assignment["reference"]]
            assert legacy_assignment["action"] == "train#" + identity["task_id"]
            assert legacy_assignment["message"] == f'sent task to client "{name}"'
            received = _one_task_record(client_job_audit, "task.assignment.received", task_id=identity["task_id"])
            started = _one_task_record(client_job_audit, "task.execution.started", task_id=identity["task_id"])
            returned = _one_task_record(client_job_audit, "task.execution.returned", task_id=identity["task_id"])
            prepared = _one_task_record(client_job_audit, "task.result.prepared", task_id=identity["task_id"])
            published = _one_task_record(client_job_audit, "task.publication.received", task_id=identity["task_id"])
            legacy_client_receipt = client_by_id[received["reference"]]
            assert legacy_client_receipt["action"] == legacy_assignment["action"]
            assert legacy_client_receipt["message"] == "received task from server"
            assert legacy_client_receipt["reference"] == legacy_assignment["event_id"]
            assert legacy_client_receipt["job"] == job_id
            for event in (received, started, returned):
                assert event["reference"] == legacy_client_receipt["event_id"], event
            for event in (received, started, returned, prepared, published):
                details = _task_details(event)
                for key in ("task", "task_id", "workflow", "attempt_id"):
                    assert details[key] == identity[key], (event, identity)
                assert details["site"] == name, event
            # The training executor returns a result without an explicit status.
            for event in (returned, prepared):
                assert "status=unspecified" in event["message"].split(), event
            legacy_result = client_by_id[prepared["reference"]]
            assert legacy_result["action"] == legacy_assignment["action"]
            assert legacy_result["reference"] == legacy_assignment["event_id"]
            assert legacy_result["job"] == job_id
            assert published["reference"] == legacy_result["event_id"]
            assert "reason=receipt_confirmed" in published["message"]
            assert (
                client_job_audit.index(legacy_client_receipt)
                < client_job_audit.index(received)
                < client_job_audit.index(started)
                < client_job_audit.index(returned)
            )
            assert (
                client_job_audit.index(returned) < client_job_audit.index(prepared) < client_job_audit.index(published)
            )
            result = _one_task_record(server_job_audit, "task.result.received", task_id=identity["task_id"])
            callback = _one_task_record(server_job_audit, "task.result_callback.returned", task_id=identity["task_id"])
            manager = _one_task_record(server_job_audit, "task.result_manager.returned", task_id=identity["task_id"])
            handling = _one_task_record(server_job_audit, "task.result.handling_returned", task_id=identity["task_id"])
            legacy_server_receipt = server_by_id[result["reference"]]
            assert legacy_server_receipt["action"] == legacy_result["action"]
            assert legacy_server_receipt["message"] == f"received result from client '{name}'"
            assert legacy_server_receipt["reference"] == legacy_result["event_id"]
            assert legacy_server_receipt["job"] == job_id
            for event in (result, callback, manager, handling):
                assert event["reference"] == legacy_server_receipt["event_id"], event
                details = _task_details(event)
                assert details["site"] == name, event
                for key in ("task", "task_id", "workflow", "attempt_id"):
                    assert details[key] == identity[key], (event, identity)
            assert "reason=complete_result" in result["message"]
            assert (
                server_job_audit.index(legacy_server_receipt)
                < server_job_audit.index(result)
                < server_job_audit.index(manager)
                < server_job_audit.index(callback)
                < server_job_audit.index(handling)
            )
        summary["task_workflow_audit_records"] = (
            "verified job-process records, legacy references and private payload exclusion"
        )
        for name in ("site-1", "site-2"):
            notifications = [json.loads(line) for line in (evidence[name] / "received.jsonl").read_text().splitlines()]
            prefix = "relay-1." if with_relay else ""
            assert any(
                r["origin"] == f"{prefix}{name}.{job_id}" and r["topic"] == "train.notify_job_status"
                for r in notifications
            )
        summary["own_child_status_notifications"] = "observed at both clients"
        for name in ("site-1", "site-2"):
            # These legitimate internal notifications must not create new
            # administrator receipts, outcomes or falsely attributed actors.
            assert not any(
                r["action"] == "train.notify_job_status" or "command=train.notify_job_status" in r.get("message", "")
                for r in audit_records(name)
            )
        summary["security_audit_records"] = "verified server/client actors, references, rejections and child silence"
        _write_json(tmp_path / "sanitized-results.json", summary)
        assert summary["job_status"] == "FINISHED:COMPLETED"
        assert all(marker not in (r["body"] or "") for r in results), results
        assert all(
            r["rc"] == ReturnCode.OK and r["body"] == "request must originate from server" for r in results[:2]
        ), results
        if with_relay:
            assert results[-1]["rc"] == ReturnCode.TIMEOUT, results[-1]
            assert results[-1]["origin"] is None
            assert not any(r["origin"] == "server" and r["topic"] == "sys.shell" for r in destination_records)
