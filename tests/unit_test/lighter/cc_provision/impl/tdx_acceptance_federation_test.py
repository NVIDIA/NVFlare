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

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[5] / "examples/devops/coco/acceptance/verify_federation.py"
SPEC = importlib.util.spec_from_file_location("tdx_federation", PATH)
federation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(federation)


def append(path, text):
    with path.open("a") as stream:
        stream.write(text)


def round_lines():
    return (
        "2026 INFO CCManager - Site site-observer triggering periodic cross-site validation\n"
        "2026 INFO CCManager - Cross-site validation passed\n"
    )


@pytest.mark.parametrize("timeout,soak", [(0, 900), (601, 900), (float("nan"), 900), (600, 899), (600, float("inf"))])
def test_invalid_bounds(timeout, soak):
    with pytest.raises(federation.AcceptanceError):
        federation.validate_options(timeout, soak)


@pytest.mark.parametrize("soak", [0, 900, 1200])
def test_valid_bounds(soak):
    federation.validate_options(600, soak)


def test_only_new_complete_observer_periodic_rounds(tmp_path):
    path = tmp_path / "observer.log"
    path.write_text(round_lines())
    log = federation.FreshObserverLog(path)
    try:
        assert log.poll(1) == 0
        append(path, "INFO CCManager - Cross-site validation passed\n")
        assert log.poll(2) == 0
        append(path, "INFO CCManager - Site site-1 triggering periodic cross-site validation\n")
        append(path, "INFO CCManager - Cross-site validation passed\n")
        assert log.poll(3) == 0
        append(path, round_lines().rstrip("\n"))
        assert log.poll(4) == 0
        append(path, "\n")
        assert log.poll(5) == 1
        assert log.last_success == 5
    finally:
        log.close()


def test_old_incomplete_line_not_counted(tmp_path):
    path = tmp_path / "observer.log"
    path.write_text("INFO CCManager - ")
    log = federation.FreshObserverLog(path)
    try:
        append(path, "Site site-observer triggering periodic cross-site validation\n")
        append(path, "INFO CCManager - Cross-site validation passed\n")
        assert log.poll(1) == 0
        append(path, round_lines())
        assert log.poll(2) == 1
    finally:
        log.close()


@pytest.mark.parametrize("mutation", ["rotation", "truncate", "rewrite"])
def test_log_integrity(tmp_path, mutation):
    path = tmp_path / "observer.log"
    path.write_text("initial trusted log\n")
    log = federation.FreshObserverLog(path)
    try:
        if mutation == "rotation":
            path.rename(tmp_path / "old.log")
            path.write_text(round_lines())
        elif mutation == "truncate":
            path.write_text("")
        else:
            path.write_text(round_lines())
        with pytest.raises(federation.AcceptanceError):
            log.poll(1)
    finally:
        log.close()


def test_failure_not_ignored(tmp_path):
    path = tmp_path / "observer.log"
    path.touch()
    log = federation.FreshObserverLog(path)
    try:
        append(path, "CCManager - Cross-site validation failed: sensitive contents\n")
        with pytest.raises(federation.AcceptanceError) as exc:
            log.poll(1)
        assert "sensitive" not in str(exc.value)
    finally:
        log.close()


@pytest.mark.parametrize(
    "members",
    [["site-1", "site-2"], ["site-1", "site-2", "intruder"], ["site-1", "site-2", "site-observer", "site-observer"]],
)
def test_membership_exact(members):
    session = SimpleNamespace(
        get_system_info=lambda: SimpleNamespace(client_info=[SimpleNamespace(name=n) for n in members])
    )
    with pytest.raises(federation.AcceptanceError):
        federation.verify_membership(session)


def result_file(root):
    result = {
        "schema_version": 1,
        "status": "passed",
        "job_id": "current-job",
        "nonce": "current-nonce",
        "values": {"site-1": 3, "site-2": 7},
        "aggregate": 10,
        "errors": [],
    }
    path = root / "tdx_acceptance_result.json"
    path.write_text(json.dumps(result))
    return path, result


@pytest.mark.parametrize(
    "field,value",
    [
        ("nonce", "old-nonce"),
        ("job_id", "other-job"),
        ("aggregate", 9),
        ("aggregate", 10.0),
        ("values", {"site-1": 3}),
        ("values", {"site-1": 3.0, "site-2": 7}),
        ("status", "failed"),
    ],
)
def test_results_bound_to_current_job_and_exact_values(tmp_path, field, value):
    path, result = result_file(tmp_path)
    result[field] = value
    path.write_text(json.dumps(result))
    with pytest.raises(federation.AcceptanceError):
        federation.verify_result(tmp_path, "current-nonce", "current-job")


def test_result_valid_and_duplicate_rejected(tmp_path):
    path, result = result_file(tmp_path)
    assert len(federation.verify_result(tmp_path, "current-nonce", "current-job")) == 64
    (tmp_path / "duplicate").mkdir()
    (tmp_path / "duplicate" / path.name).write_text(json.dumps(result))
    with pytest.raises(federation.AcceptanceError):
        federation.verify_result(tmp_path, "current-nonce", "current-job")


def test_job_rejects_byoc(tmp_path):
    (tmp_path / "custom").mkdir()
    (tmp_path / "custom" / "payload.py").write_text("print('payload')")
    with pytest.raises(federation.AcceptanceError):
        federation.verify_job_definition(tmp_path, "nonce")


def test_observation_timeout_without_new_rounds():
    now = [0]
    session = SimpleNamespace(
        get_system_info=lambda: SimpleNamespace(client_info=[SimpleNamespace(name=n) for n in federation.MEMBERS])
    )
    log = SimpleNamespace(rounds=0, last_success=None, poll=lambda _: None)
    with pytest.raises(federation.AcceptanceError):
        federation.observe(session, log, 0, 4, clock=lambda: now[0], sleep=lambda n: now.__setitem__(0, now[0] + n))


def test_two_rounds_observed_and_soak_requires_duration():
    now = [0]
    session = SimpleNamespace(
        get_system_info=lambda: SimpleNamespace(client_info=[SimpleNamespace(name=n) for n in federation.MEMBERS])
    )
    log = SimpleNamespace(rounds=0, last_success=None)

    def poll(t):
        log.rounds += 1
        log.last_success = t

    log.poll = poll
    federation.observe(session, log, 10, 4, clock=lambda: now[0], sleep=lambda n: now.__setitem__(0, now[0] + n))
    assert now[0] >= 10


def make_job(root):
    nonce = "current-nonce-1234"
    config = root / "app/config"
    config.mkdir(parents=True)
    (root / "meta.json").write_text(
        json.dumps(
            {
                "mandatory_clients": ["site-1", "site-2"],
                "min_clients": 2,
                "deploy_map": {"app": ["server", "site-1", "site-2"]},
            }
        )
    )
    server = {
        "format_version": 2,
        "components": [],
        "task_data_filters": [],
        "task_result_filters": [],
        "workflows": [{"id": "acceptance", "path": "tdx_acceptance.AcceptanceController", "args": {"nonce": nonce}}],
    }
    client = {
        "format_version": 2,
        "components": [],
        "task_data_filters": [],
        "task_result_filters": [],
        "executors": [
            {
                "tasks": ["tdx_acceptance"],
                "executor": {"path": "tdx_acceptance.AcceptanceExecutor", "args": {"nonce": nonce}},
            }
        ],
    }
    (config / "config_fed_server.json").write_text(json.dumps(server))
    (config / "config_fed_client.json").write_text(json.dumps(client))
    return nonce, config, server


def test_reviewed_job_and_injected_component(tmp_path):
    nonce, config, server = make_job(tmp_path)
    assert len(federation.verify_job_definition(tmp_path, nonce)) == 64
    server["components"] = [{"path": "another.Component"}]
    (config / "config_fed_server.json").write_text(json.dumps(server))
    with pytest.raises(federation.AcceptanceError):
        federation.verify_job_definition(tmp_path, nonce)


def test_job_requires_matching_nonce_and_all_clients(tmp_path):
    nonce, _, _ = make_job(tmp_path)
    with pytest.raises(federation.AcceptanceError):
        federation.verify_job_definition(tmp_path, "old-nonce")
    meta = tmp_path / "meta.json"
    data = json.loads(meta.read_text())
    data["mandatory_clients"] = ["site-1"]
    meta.write_text(json.dumps(data))
    with pytest.raises(federation.AcceptanceError):
        federation.verify_job_definition(tmp_path, nonce)


def test_reviewed_job_root_symlink_rejected(tmp_path):
    original = tmp_path / "original"
    nonce, _, _ = make_job(original)
    link = tmp_path / "linked-job"
    link.symlink_to(original, target_is_directory=True)
    with pytest.raises(federation.AcceptanceError):
        federation.verify_job_definition(link, nonce)


def test_job_digest_rechecked_before_submission(tmp_path):
    nonce, _, _ = make_job(tmp_path)
    digest = federation.verify_job_definition(tmp_path, nonce)
    federation.verify_unchanged_job(tmp_path, nonce, digest)
    # A valid metadata change still invalidates the original reviewed bytes.
    path = tmp_path / "meta.json"
    data = json.loads(path.read_text())
    data["name"] = "changed-before-submit"
    path.write_text(json.dumps(data))
    with pytest.raises(federation.AcceptanceError, match="changed before submission"):
        federation.verify_unchanged_job(tmp_path, nonce, digest)


@pytest.mark.parametrize("change_before_submit", [False, True])
def test_runner_keeps_original_review_and_allows_submission_metadata(tmp_path, monkeypatch, change_before_submit):
    job = tmp_path / "job"
    nonce, _, _ = make_job(job)
    digest = federation.verify_job_definition(job, nonce)
    observer = tmp_path / "observer"
    observer.mkdir()
    observer_config(observer, "A")
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    result_path, result = result_file(result_dir)
    result["nonce"] = nonce
    result_path.write_text(json.dumps(result))
    submitted = []
    log = SimpleNamespace(rounds=2, poll=lambda _: None, close=lambda: None)
    monkeypatch.setattr(federation, "FreshObserverLog", lambda _: log)

    def observe(*_):
        if change_before_submit:
            with (job / "meta.json").open("a") as stream:
                stream.write("\n")

    def submit(path):
        submitted.append(path)
        # The submission API can add metadata to the caller's job directory.
        (job / "job_id.json").write_text(json.dumps({"job_id": "current-job"}))
        return "current-job"

    monkeypatch.setattr(federation, "observe", observe)
    session = SimpleNamespace(
        get_system_info=lambda: SimpleNamespace(client_info=[SimpleNamespace(name=n) for n in federation.MEMBERS]),
        submit_job=submit,
        get_job_meta=lambda _: {"status": "FINISHED:COMPLETED"},
        download_job_result=lambda *_: result_dir,
    )
    args = SimpleNamespace(
        timeout=600,
        soak=0,
        observer_local=observer,
        topology="A",
        job=job,
        nonce=nonce,
        observer_log=tmp_path / "observer.log",
        download_dir=result_dir,
    )
    if change_before_submit:
        with pytest.raises(federation.AcceptanceError, match="changed before submission"):
            federation.run(args, session)
        assert not submitted
    else:
        receipt = federation.run(args, session)
        assert submitted == [job]
        assert receipt["functional_evidence"]["job_sha256"] == digest
        assert receipt["functional_evidence"]["application_execution"] == "passed"


def observer_config(root, topology):
    required = federation.CLIENTS | ({"server"} if topology == "B" else set())
    args = {
        "cc_issuers_conf": [],
        "cc_verifier_ids": ["coco_authorizer"],
        "cc_enabled_sites": sorted(required),
        "required_site_verifier_ids": {n: ["coco_authorizer"] for n in required},
        "require_site_binding": True,
        "verify_frequency": 120,
    }
    (root / "cc_manager__p_resources.json").write_text(
        json.dumps(
            {
                "components": [
                    {
                        "id": "cc_manager",
                        "path": "nvflare.app_opt.confidential_computing.cc_manager.CCManager",
                        "args": args,
                    }
                ]
            }
        )
    )
    (root / "coco_authorizer__p_resources.json").write_text(
        json.dumps(
            {
                "components": [
                    {
                        "id": "coco_authorizer",
                        "path": "nvflare.app_opt.confidential_computing.coco_authorizer.CoCoAuthorizer",
                        "args": {},
                    }
                ]
            }
        )
    )
    return args


@pytest.mark.parametrize("topology", ["A", "B"])
def test_observer_complete_bound_configuration(tmp_path, topology):
    observer_config(tmp_path, topology)
    federation.verify_observer_config(tmp_path, topology)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cc_issuers_conf", [{"issuer_id": "coco_authorizer"}]),
        ("require_site_binding", False),
        ("cc_enabled_sites", ["site-1", "site-2"]),
        ("required_site_verifier_ids", {"site-1": ["coco_authorizer"]}),
    ],
)
def test_observer_rejects_incomplete_or_issuing_config(tmp_path, field, value):
    args = observer_config(tmp_path, "B")
    args[field] = value
    path = tmp_path / "cc_manager__p_resources.json"
    data = json.loads(path.read_text())
    data["components"][0]["args"] = args
    path.write_text(json.dumps(data))
    with pytest.raises(federation.AcceptanceError):
        federation.verify_observer_config(tmp_path, "B")
