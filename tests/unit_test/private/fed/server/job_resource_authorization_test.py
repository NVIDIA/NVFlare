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

"""Study queries must not expose jobs forbidden by the single-job policy."""

import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from zipfile import ZipFile

import pytest
import yaml

from nvflare.apis.fl_constant import AdminCommandNames, JobConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey, RunStatus
from nvflare.apis.job_def_manager_spec import JobDefManagerSpec
from nvflare.apis.storage import StorageException
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import MetaKey, MetaStatusValue, ProtoKey
from nvflare.fuel.hci.server.authz import AuthzFilter
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.hci.server.reg import ServerCommandRegister
from nvflare.fuel.sec.authz import AuthorizationService
from nvflare.private.fed.resource_stats.contract import (
    KIND_RESOURCE_SUMMARY,
    SCHEMA_VERSION,
    canonical_json_bytes,
    validate_record,
)
from nvflare.private.fed.server import job_cmds as job_cmds_module
from nvflare.private.fed.server.job_cmds import JobCommandModule
from nvflare.private.fed.server.job_meta_validator import JobMetaValidator
from nvflare.security.security import FLAuthorizer


@pytest.fixture
def policy(monkeypatch):
    authorizer = FLAuthorizer(
        "server-org",
        {
            "format_version": "1.0",
            "permissions": {
                "owner": {"get_study_resources": "all", "get_job_resources": "name:submitter"},
                "category_owner": {"view": "name:submitter", "get_study_resources": "all"},
                "colleague": {"get_study_resources": "all", "get_job_resources": "org:submitter"},
                "study_only": {"get_study_resources": "all", "get_job_resources": "none"},
                "job_only": {"get_study_resources": "none", "get_job_resources": "all"},
                "project_admin": "all",
            },
        },
    )
    monkeypatch.setattr(AuthorizationService, "the_authorizer", authorizer)
    return authorizer


def _job(number, *, submitter="alice", org="org-a", status=RunStatus.FINISHED_COMPLETED, study="default"):
    job_id = f"123e4567-e89b-42d3-a456-{number:012d}"
    return SimpleNamespace(
        job_id=job_id,
        meta={
            JobMetaKey.JOB_ID.value: job_id,
            JobMetaKey.JOB_NAME.value: f"job-{number}",
            JobMetaKey.STATUS.value: status,
            JobMetaKey.STUDY.value: study,
            JobMetaKey.SUBMITTER_NAME.value: submitter,
            JobMetaKey.SUBMITTER_ORG.value: org,
            JobMetaKey.SUBMITTER_ROLE.value: "submitter",
        },
    )


def _summary(job_id):
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND_RESOURCE_SUMMARY,
        "job_id": job_id,
        "participants": [
            {
                "duration_seconds": "1",
                "participant_name": "server",
                "role": "server",
                "received_at": "2026-10-08T12:00:00Z",
                "resource_capacity": {
                    "cpu": {"groups": [{"units": "2"}]},
                    "memory": {"bytes": "1024"},
                    "gpu": {"groups": []},
                },
                "resource_time": {
                    "cpu": {"groups": [{"unit_seconds": "2"}]},
                    "memory": {"byte_seconds": "1024"},
                    "gpu": {"groups": []},
                },
                "resource_utilization": {
                    "cpu_consumed": {"cpu_seconds": "0"},
                    "retained_content": {"bytes": "3"},
                    "message_traffic": {"sent_bytes": "0", "received_bytes": "0"},
                },
            }
        ],
    }
    validate_record(summary)
    return summary


def _connection(jobs, *, role="owner", unavailable=(), study="default"):
    manager = MagicMock(spec=JobDefManagerSpec)
    manager.get_all_jobs.return_value = jobs
    manager.get_job.side_effect = lambda jid, fl_ctx: next((job for job in jobs if job.job_id == jid), None)

    def stage(jid, download_dir, component, download_file, fl_ctx):
        if jid in unavailable:
            raise StorageException("workspace not retained")
        archive_path = Path(download_dir) / jid / download_file
        archive_path.parent.mkdir(parents=True)
        with ZipFile(archive_path, "w") as archive:
            archive.writestr("resource_stats/resource_summary.json", canonical_json_bytes(_summary(jid)))
            # Study queries validate the summary and inventory without reading participant bodies.
            archive.writestr("resource_stats/participants/server.json", b"unused by study queries")

    manager.get_storage_for_download.side_effect = stage
    engine = SimpleNamespace(job_def_manager=manager, new_context=FLContext)
    conn = Connection(
        app_ctx=engine,
        props={
            ConnProps.ACTIVE_STUDY: study,
            ConnProps.USER_NAME: "alice",
            ConnProps.USER_ORG: "org-a",
            ConnProps.USER_ROLE: role,
            # Stale connection state must not grant access to someone else's job.
            ConnProps.SUBMITTER_NAME: "alice",
            ConnProps.SUBMITTER_ORG: "org-a",
            ConnProps.SUBMITTER_ROLE: "project_admin",
        },
    )
    return conn


def _dispatch(conn, command=AdminCommandNames.GET_STUDY_RESOURCES):
    register = ServerCommandRegister(conn.app_ctx)
    register.register_module(JobCommandModule())
    register.finalize()
    register.add_filter(AuthzFilter())
    register.process_command(conn, command)
    return conn.buffer.output


def _result(output):
    assert output[ProtoKey.META][MetaKey.STATUS] == MetaStatusValue.OK
    assert len(output[ProtoKey.DATA]) == 1
    assert output[ProtoKey.DATA][0][ProtoKey.TYPE] == ProtoKey.DICT
    result = output[ProtoKey.DATA][0][ProtoKey.DATA]
    validate_record(result)
    return result


@pytest.mark.parametrize("command", ["get_study_resources", "job_mgmt.get_study_resources"])
def test_study_applies_single_job_policy_before_reading_or_exposing_denied_jobs(policy, command):
    allowed = _job(1)
    denied = [
        _job(2, submitter="bob"),
        _job(3, submitter="bob", status=RunStatus.RUNNING),
        _job(4, submitter="bob"),
    ]
    conn = _connection([allowed, *denied], unavailable=[denied[-1].job_id])
    result = _result(_dispatch(conn, command))

    assert [row["job_id"] for row in result["jobs"]] == [allowed.job_id]
    assert result["coverage"] == {
        "selected_jobs": "1",
        "included_jobs": "1",
        "unavailable_jobs": "0",
        "nonterminal_jobs": "0",
    }
    assert result["totals"]["resource_utilization"]["retained_content"] == {"bytes": "3"}
    assert [call.kwargs["jid"] for call in conn.app_ctx.job_def_manager.get_storage_for_download.call_args_list] == [
        allowed.job_id
    ]
    for job in denied:
        assert job.job_id not in json.dumps(conn.buffer.output)
        assert job.meta[JobMetaKey.JOB_NAME.value] not in json.dumps(conn.buffer.output)


@pytest.mark.parametrize(
    "role, expected", [("owner", [1]), ("category_owner", [1]), ("colleague", [1, 2]), ("project_admin", [1, 2, 3])]
)
def test_real_policy_uses_each_jobs_submitter_and_callers_certificate_role(policy, role, expected):
    jobs = [_job(1), _job(2, submitter="bob"), _job(3, submitter="bob", org="org-b")]
    conn = _connection(jobs, role=role)
    result = _result(_dispatch(conn))
    assert [row["job_id"] for row in result["jobs"]] == [jobs[number - 1].job_id for number in expected]
    assert result["coverage"]["selected_jobs"] == str(len(expected))
    assert result["totals"]["resource_utilization"]["retained_content"] == {"bytes": str(3 * len(expected))}
    assert conn.get_prop(ConnProps.USER_ROLE) == role
    assert conn.get_prop(ConnProps.SUBMITTER_NAME) == "alice"


@pytest.mark.parametrize("command", ["get_job_resources", "job_mgmt.get_job_resources"])
@pytest.mark.parametrize("submitter, authorized", [("alice", True), ("bob", False)])
def test_registered_single_job_command_matches_study_policy(policy, command, submitter, authorized):
    job = _job(1, submitter=submitter)
    conn = _connection([job])
    output = _dispatch(conn, f"{command} {job.job_id}")
    assert (output[ProtoKey.META][MetaKey.STATUS] == MetaStatusValue.OK) is authorized
    assert conn.app_ctx.job_def_manager.get_storage_for_download.call_count == int(authorized)


def test_all_denied_is_indistinguishable_from_an_empty_study(policy):
    denied = _job(1, submitter="bob")
    conn = _connection([denied])
    result = _result(_dispatch(conn))
    empty = _result(_dispatch(_connection([])))
    result.pop("generated_at")
    empty.pop("generated_at")
    assert result == empty
    assert result["jobs"] == []
    assert result["totals"]["resource_utilization"]["retained_content"]["status"] == "unavailable"
    conn.app_ctx.job_def_manager.get_storage_for_download.assert_not_called()


def test_study_permission_does_not_grant_single_job_access(policy):
    conn = _connection([_job(1)], role="study_only")
    result = _result(_dispatch(conn))
    assert result["coverage"]["selected_jobs"] == "0"
    conn.app_ctx.job_def_manager.get_storage_for_download.assert_not_called()


def test_missing_submitter_metadata_does_not_default_to_the_callers_identity(policy):
    job = _job(1)
    for key in (JobMetaKey.SUBMITTER_NAME, JobMetaKey.SUBMITTER_ORG, JobMetaKey.SUBMITTER_ROLE):
        job.meta.pop(key.value)
    conn = _connection([job])
    assert _result(_dispatch(conn))["coverage"]["selected_jobs"] == "0"
    conn.app_ctx.job_def_manager.get_storage_for_download.assert_not_called()


def test_single_job_permission_does_not_bypass_study_command_gate(policy):
    conn = _connection([_job(1)], role="job_only")
    output = _dispatch(conn)
    assert output[ProtoKey.META][MetaKey.STATUS] == MetaStatusValue.NOT_AUTHORIZED
    conn.app_ctx.job_def_manager.get_all_jobs.assert_not_called()
    conn.app_ctx.job_def_manager.get_storage_for_download.assert_not_called()


def test_accessible_running_and_missing_reports_keep_their_existing_coverage(policy):
    running = _job(1, status=RunStatus.RUNNING)
    unavailable = _job(2)
    conn = _connection([running, unavailable], unavailable=[unavailable.job_id])
    result = _result(_dispatch(conn))
    assert [row["resource_data"] for row in result["jobs"]] == ["nonterminal", "unavailable"]
    assert result["coverage"] == {
        "selected_jobs": "2",
        "included_jobs": "0",
        "unavailable_jobs": "1",
        "nonterminal_jobs": "1",
    }
    assert conn.app_ctx.job_def_manager.get_storage_for_download.call_count == 1


def test_other_study_jobs_are_not_authorized_or_read(policy, monkeypatch):
    allowed = _job(1)
    other_study = _job(2, study="another-study")
    checked = []
    original = policy.authorize

    def authorize(ctx):
        if ctx.right == AdminCommandNames.GET_JOB_RESOURCES:
            checked.append(ctx.submitter.name)
        return original(ctx)

    monkeypatch.setattr(policy, "authorize", authorize)
    conn = _connection([allowed, other_study])
    result = _result(_dispatch(conn))
    assert checked == ["alice"]
    assert len(result["jobs"]) == 1
    assert other_study.job_id not in json.dumps(conn.buffer.output)
    assert conn.app_ctx.job_def_manager.get_storage_for_download.call_count == 1


def test_denied_jobs_do_not_affect_limits(policy, monkeypatch):
    monkeypatch.setattr(job_cmds_module, "MAX_STUDY_JOBS", 1)
    allowed = _job(1)
    denied = [_job(number, submitter="bob") for number in range(2, 7)]
    result = _result(_dispatch(_connection([allowed, *denied])))
    assert result["coverage"]["selected_jobs"] == "1"

    conn = _connection([allowed, _job(7), *denied])
    output = _dispatch(conn)
    assert output[ProtoKey.META][MetaKey.STATUS] == MetaStatusValue.ERROR
    assert output[ProtoKey.DATA][0][ProtoKey.DATA] == "study contains more than 1 accessible retained jobs"
    conn.app_ctx.job_def_manager.get_storage_for_download.assert_not_called()


@pytest.mark.parametrize("permission", [(True, "error"), (False, ""), (False, "denied")])
def test_only_an_explicit_error_free_permission_grants_access(monkeypatch, permission):
    monkeypatch.setattr(AuthorizationService, "authorize", lambda ctx: permission)
    conn = _connection([_job(1)])
    JobCommandModule().get_study_resources(conn, [AdminCommandNames.GET_STUDY_RESOURCES])
    assert _result(conn.buffer.output)["coverage"]["selected_jobs"] == "0"
    conn.app_ctx.job_def_manager.get_storage_for_download.assert_not_called()


@pytest.mark.parametrize(
    "permission",
    [
        None,
        True,
        (),
        (True,),
        (True, "", "unexpected"),
        [True, ""],
        {True: "", "": True},
        "yes",
        (True, None),
        (True, False),
        (None, ""),
        (1, ""),
        ("yes", ""),
    ],
)
def test_malformed_permission_results_fail_closed_without_details(monkeypatch, permission):
    monkeypatch.setattr(AuthorizationService, "authorize", lambda ctx: permission)
    conn = _connection([_job(1)])
    JobCommandModule().get_study_resources(conn, [AdminCommandNames.GET_STUDY_RESOURCES])
    output = conn.buffer.output
    assert output[ProtoKey.META][MetaKey.STATUS] == MetaStatusValue.NOT_AUTHORIZED
    assert output[ProtoKey.DATA] == [
        {ProtoKey.TYPE: ProtoKey.ERROR, ProtoKey.DATA: "unable to authorize study resource statistics"}
    ]
    conn.app_ctx.job_def_manager.get_storage_for_download.assert_not_called()


def test_policy_exception_checks_every_job_before_any_archive_read(policy, monkeypatch):
    allowed, failed = _job(1), _job(2, submitter="bob")
    original = policy.authorize

    def authorize(ctx):
        if ctx.right == AdminCommandNames.GET_JOB_RESOURCES and ctx.submitter.name == "bob":
            raise RuntimeError(f"private policy detail for {failed.job_id}")
        return original(ctx)

    monkeypatch.setattr(policy, "authorize", authorize)
    conn = _connection([allowed, failed])
    output = _dispatch(conn)
    assert output[ProtoKey.META][MetaKey.STATUS] == MetaStatusValue.NOT_AUTHORIZED
    assert output[ProtoKey.DATA] == [
        {ProtoKey.TYPE: ProtoKey.ERROR, ProtoKey.DATA: "unable to authorize study resource statistics"}
    ]
    conn.app_ctx.job_def_manager.get_storage_for_download.assert_not_called()
    assert allowed.job_id not in json.dumps(output)
    assert failed.job_id not in json.dumps(output)


def test_no_authorizer_preserves_existing_unrestricted_poc_behavior(monkeypatch):
    monkeypatch.setattr(AuthorizationService, "the_authorizer", None)
    conn = _connection([_job(1, submitter="bob")])
    result = _result(_dispatch(conn))
    assert result["coverage"]["selected_jobs"] == "1"


@pytest.mark.parametrize("policy_name", ["default_authz", "cc_authz"])
@pytest.mark.parametrize("name", ["Test Job 1", "肺部训练 — étude 🚀", "Example\x1b[2Jjob\nnext\u202e"])
def test_shipped_view_policy_and_registered_queries_preserve_existing_display_names(monkeypatch, policy_name, name):
    template = Path(__file__).resolve().parents[5] / "nvflare/lighter/templates/master_template.yml"
    policies = yaml.safe_load(template.read_text(encoding="utf-8"))
    authorizer = FLAuthorizer("server-org", json.loads(policies[policy_name]))
    monkeypatch.setattr(AuthorizationService, "the_authorizer", authorizer)
    job = _job(1, submitter="bob")
    job.meta[JobMetaKey.JOB_NAME.value] = name
    job_data = BytesIO()
    with ZipFile(job_data, "w") as archive:
        archive.writestr("job1/app/config/", "")
        archive.writestr(f"job1/{JobConstants.META_FILE}", json.dumps({"name": name, "deploy_map": {"app": ["@ALL"]}}))
        archive.writestr(f"job1/app/config/{JobConstants.SERVER_JOB_CONFIG}", "{}")
        archive.writestr(f"job1/app/config/{JobConstants.CLIENT_JOB_CONFIG}", "{}")
    accepted, error, metadata = JobMetaValidator().validate("job1", job_data.getvalue())
    assert accepted, error
    assert metadata[JobMetaKey.JOB_NAME.value] == name
    conn = _connection([job], role="member")
    # Normal sessions have no previous single-job submitter fields.
    for key in (ConnProps.SUBMITTER_NAME, ConnProps.SUBMITTER_ORG, ConnProps.SUBMITTER_ROLE):
        conn.props.pop(key, None)

    result = _result(_dispatch(conn))

    assert result["jobs"][0]["job_name"] == name
    assert result["coverage"]["included_jobs"] == "1"
    assert json.loads(canonical_json_bytes(result))["jobs"][0]["job_name"] == name
    output = _dispatch(_connection([job], role="member"), f"get_job_resources {job.job_id}")
    assert output[ProtoKey.META][MetaKey.STATUS] == MetaStatusValue.OK
    data = output[ProtoKey.DATA][0][ProtoKey.DATA]
    assert data["job_name"] == name
    assert "job_name" not in data["resource_summary"]


@pytest.mark.parametrize(
    "name, folder, expected",
    [
        (None, None, None),
        ("", "legacy folder", "legacy folder"),
        ([], "legacy folder", "legacy folder"),
        ({"invalid": True}, None, None),
        ("é" * 300, None, "é" * 255),
        (None, "folder" * 60, ("folder" * 60)[:255]),
    ],
)
def test_display_metadata_fallback_and_bound_do_not_poison_neighboring_jobs(policy, name, folder, expected):
    first, second = _job(1), _job(2)
    first.meta[JobMetaKey.JOB_NAME.value] = name
    first.meta[JobMetaKey.JOB_FOLDER_NAME.value] = folder
    conn = _connection([first, second])

    result = _result(_dispatch(conn))

    assert result["jobs"][0]["job_name"] == (expected or first.job_id)
    assert result["jobs"][1]["job_name"] == "job-2"
    assert result["coverage"]["included_jobs"] == "2"
    assert first.meta[JobMetaKey.JOB_NAME.value] == name
    assert first.meta[JobMetaKey.JOB_FOLDER_NAME.value] == folder


@pytest.mark.parametrize("status", [None, [], "", "FINISHED:\nBAD", "FINISHED:" + "x" * 64])
def test_invalid_lifecycle_metadata_is_not_fabricated_as_terminal(policy, status):
    invalid, valid = _job(1, status=status), _job(2)
    conn = _connection([invalid, valid])

    result = _result(_dispatch(conn))

    assert result["jobs"][0] == {
        "job_id": invalid.job_id,
        "job_name": "job-1",
        "job_status": "UNKNOWN",
        "resource_data": "nonterminal",
    }
    assert result["coverage"]["included_jobs"] == "1"
    assert result["coverage"]["nonterminal_jobs"] == "1"
    assert [call.kwargs["jid"] for call in conn.app_ctx.job_def_manager.get_storage_for_download.call_args_list] == [
        valid.job_id
    ]


def test_invalid_job_totals_leave_one_unavailable_row_and_preserve_neighbor(policy, monkeypatch):
    invalid, valid = _job(1), _job(2)
    original = job_cmds_module.derive_job_totals

    def derive(participants):
        if participants[0]["resource_utilization"]["retained_content"]["bytes"] == "4":
            raise job_cmds_module.ContractError("invalid measurement total")
        return original(participants)

    original_summary = _summary

    def summary(job_id):
        value = original_summary(job_id)
        if job_id == invalid.job_id:
            value["participants"][0]["resource_utilization"]["retained_content"]["bytes"] = "4"
        return value

    monkeypatch.setattr(job_cmds_module, "derive_job_totals", derive)
    monkeypatch.setitem(globals(), "_summary", summary)
    result = _result(_dispatch(_connection([invalid, valid])))

    assert [row["resource_data"] for row in result["jobs"]] == ["unavailable", "included"]
    assert result["coverage"]["unavailable_jobs"] == "1"
    assert result["totals"]["resource_utilization"]["retained_content"] == {"bytes": "3"}


def test_study_derivation_failure_returns_bounded_error_instead_of_escaping_handler(policy, monkeypatch):
    def fail(rows):
        raise job_cmds_module.ContractError("study totals cannot be derived")

    monkeypatch.setattr(job_cmds_module, "derive_study_totals", fail)
    output = _dispatch(_connection([_job(1)]))

    assert output[ProtoKey.META][MetaKey.STATUS] == MetaStatusValue.INTERNAL_ERROR
    assert output[ProtoKey.DATA][0][ProtoKey.TYPE] == ProtoKey.ERROR
