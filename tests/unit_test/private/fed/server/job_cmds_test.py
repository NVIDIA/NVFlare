# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, WorkspaceConstants
from nvflare.apis.job_def import JobMetaKey, RunStatus
from nvflare.fuel.hci.proto import MetaKey
from nvflare.fuel.hci.server.authz import PreAuthzReturnCode
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.private.fed.server import cmd_utils as cmd_utils_module
from nvflare.private.fed.server import job_cmds as job_cmds_module
from nvflare.private.fed.server.job_cmds import (
    JobCommandModule,
    _create_get_job_log_cmd_parser,
    _create_list_job_cmd_parser,
)

TEST_CASES = [
    (
        ["-d", "-u", "12345", "-n", "hello_", "-m", "3"],
        Namespace(u=True, d=True, job_id="12345", m=3, n="hello_", r=False),
    ),
    (
        ["12345", "-d", "-u", "-n", "hello_", "-m", "3"],
        Namespace(u=True, d=True, job_id="12345", m=3, n="hello_", r=False),
    ),
    (["-d", "-u", "-n", "hello_", "-m", "3"], Namespace(u=True, d=True, job_id=None, m=3, n="hello_", r=False)),
    (["-u", "-n", "hello_", "-m", "5"], Namespace(u=True, d=False, job_id=None, m=5, n="hello_", r=False)),
    (["-u"], Namespace(u=True, d=False, job_id=None, m=None, n=None, r=False)),
    (["-r"], Namespace(u=False, d=False, job_id=None, m=None, n=None, r=True)),
    (["nvflare"], Namespace(u=False, d=False, job_id="nvflare", m=None, n=None, r=False)),
]


class TestListJobCmdParser:
    @pytest.mark.parametrize("args, expected_args", TEST_CASES)
    def test_parse_args(self, args: list[str], expected_args):
        parser = _create_list_job_cmd_parser()
        parsed_args = parser.parse_args(args)
        assert parsed_args == expected_args


class TestGetJobLogCmdParser:
    def test_parse_args(self):
        parser = _create_get_job_log_cmd_parser()
        parsed_args = parser.parse_args(["job-123", "-n", "10", "-g", "ERROR"])
        assert parsed_args == Namespace(job_id="job-123", tail_lines=10, grep_pattern="ERROR")


class _MockConnection:
    def __init__(self, cmd_props=None, app_ctx=None, props=None):
        self._props = dict(props or {})
        self._props.setdefault(ConnProps.CMD_PROPS, cmd_props)
        self.app_ctx = app_ctx
        self.errors = []
        self.strings = []
        self.successes = []
        self.dicts = []
        self.tables = []

    def get_prop(self, key, default=None):
        return self._props.get(key, default)

    def set_prop(self, key, value):
        self._props[key] = value

    def append_error(self, msg, meta=None):
        self.errors.append((msg, meta))

    def append_string(self, msg, meta=None):
        self.strings.append((msg, meta))

    def append_success(self, msg, meta=None):
        self.successes.append((msg, meta))

    def append_dict(self, data, meta=None):
        self.dicts.append((data, meta))

    def append_table(self, headers, name=None):
        table = _MockTable(headers=headers, name=name)
        self.tables.append(table)
        return table


class _MockTable:
    def __init__(self, headers, name=None):
        self.headers = headers
        self.name = name
        self.rows = []

    def add_row(self, row, meta=None):
        self.rows.append((row, meta))


class _FakeJobMetaValidator:
    def validate(self, folder_name, zip_file_name):
        assert folder_name == "job_folder"
        assert zip_file_name == "job.zip"
        return True, "", {}


class _FakeJobDefManager:
    def __init__(self):
        self.created_meta = None
        self.cloned_meta = None

    def create(self, meta, uploaded_content, fl_ctx):
        self.created_meta = dict(meta)
        result = dict(meta)
        result[JobMetaKey.JOB_ID.value] = "new-job-id"
        return result

    def clone(self, from_jid, meta, fl_ctx):
        self.cloned_meta = dict(meta)
        result = dict(meta)
        result[JobMetaKey.JOB_ID.value] = "cloned-job-id"
        return result


class _FakeEngine:
    def __init__(self):
        self.job_def_manager = _FakeJobDefManager()
        self.submit_event_meta = None

    def new_context(self):
        from nvflare.apis.fl_context import FLContext

        return FLContext()

    def fire_event(self, event_type, fl_ctx):
        assert event_type == EventType.SUBMIT_JOB
        self.submit_event_meta = dict(fl_ctx.get_prop(FLContextKey.JOB_META, {}))


class _FakeListedJob:
    def __init__(self, meta):
        self.meta = meta


class _FakeListJobDefManager:
    def __init__(self, jobs):
        self.jobs = jobs

    def get_all_jobs(self, fl_ctx):
        return self.jobs

    def get_job(self, jid, fl_ctx):
        for job in self.jobs:
            if job.meta.get(JobMetaKey.JOB_ID.value) == jid:
                return job
        return None


class _FakeListEngine:
    def __init__(self, jobs):
        self.job_def_manager = _FakeListJobDefManager(jobs)

    def new_context(self):
        from nvflare.apis.fl_context import FLContext

        return FLContext()


class _FakeWorkspace:
    def __init__(self, root_dir):
        self.root_dir = str(root_dir)

    def get_log_root(self, job_id=None):
        path = Path(self.root_dir) / str(job_id)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)


class _FakeServerEngine:
    def __init__(self, workspace):
        self.workspace = workspace
        self.job_def_manager = MagicMock()
        self.configure_job_log = MagicMock(return_value=None)

    def get_workspace(self):
        return self.workspace

    def new_context(self):
        from nvflare.apis.fl_context import FLContext

        return FLContext()


class _FakeStudyRegistry:
    def __init__(self, roles=None, sites=None):
        self.roles = roles or {}
        self.sites = sites or {}

    def has_study(self, study):
        return study in self.roles or study in self.sites

    def get_role(self, user_name, study):
        return self.roles.get(study, {}).get(user_name)

    def get_sites(self, study):
        return self.sites.get(study)


class _FakeStudyRegistryService:
    registry = None

    @staticmethod
    def get_registry():
        return _FakeStudyRegistryService.registry


class _FakeJobMetaValidatorWithMeta:
    def __init__(self, meta):
        self.meta = meta

    def validate(self, folder_name, zip_file_name):
        assert folder_name == "job_folder"
        assert zip_file_name == "job.zip"
        return True, "", dict(self.meta)


def test_submit_job_exposes_study_in_submit_event(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobMetaValidator", _FakeJobMetaValidator)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)

    engine = _FakeEngine()
    conn = _MockConnection(
        app_ctx=engine,
        props={
            ConnProps.FILE_LOCATION: "job.zip",
            ConnProps.ACTIVE_STUDY: "cancer-research",
            ConnProps.USER_NAME: "submitter",
            ConnProps.USER_ORG: "org",
            ConnProps.USER_ROLE: "role",
        },
    )

    JobCommandModule().submit_job(conn, ["submit_job", "job_folder"])

    assert conn.errors == []
    assert len(conn.successes) == 1
    assert engine.submit_event_meta == {JobMetaKey.STUDY.value: "cancer-research"}
    assert engine.job_def_manager.created_meta[JobMetaKey.STUDY.value] == "cancer-research"


def test_submit_job_defaults_study_when_cmd_props_missing(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobMetaValidator", _FakeJobMetaValidator)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)

    engine = _FakeEngine()
    conn = _MockConnection(
        app_ctx=engine,
        props={
            ConnProps.FILE_LOCATION: "job.zip",
            ConnProps.USER_NAME: "submitter",
            ConnProps.USER_ORG: "org",
            ConnProps.USER_ROLE: "role",
        },
    )

    JobCommandModule().submit_job(conn, ["submit_job", "job_folder"])

    assert conn.errors == []
    assert engine.submit_event_meta == {JobMetaKey.STUDY.value: "default"}
    assert engine.job_def_manager.created_meta[JobMetaKey.STUDY.value] == "default"


def test_clone_job_preserves_source_study(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", object)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)

    source_job = _FakeListedJob(
        {
            JobMetaKey.JOB_ID.value: "source-job",
            JobMetaKey.JOB_NAME.value: "source",
            JobMetaKey.STUDY.value: "cancer-research",
        }
    )
    engine = _FakeEngine()
    conn = _MockConnection(
        app_ctx=engine,
        props={
            JobCommandModule.JOB: source_job,
            JobCommandModule.JOB_ID: "source-job",
            ConnProps.ACTIVE_STUDY: "multiple-sclerosis",
            ConnProps.USER_NAME: "submitter",
            ConnProps.USER_ORG: "org",
            ConnProps.USER_ROLE: "role",
        },
    )

    JobCommandModule().clone_job(conn, ["clone_job", "source-job"])

    assert conn.errors == []
    assert engine.job_def_manager.cloned_meta[JobMetaKey.STUDY.value] == "cancer-research"


def test_list_jobs_filters_legacy_jobs_into_default_study(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    jobs = [
        _FakeListedJob({JobMetaKey.JOB_ID.value: "legacy-job", JobMetaKey.JOB_NAME.value: "legacy"}),
        _FakeListedJob(
            {
                JobMetaKey.JOB_ID.value: "study-job",
                JobMetaKey.JOB_NAME.value: "study",
                JobMetaKey.STUDY.value: "cancer-research",
            }
        ),
    ]
    conn = _MockConnection(app_ctx=_FakeListEngine(jobs), props={ConnProps.ACTIVE_STUDY: "default"})

    JobCommandModule().list_jobs(conn, ["list_jobs"])

    assert conn.errors == []
    assert len(conn.tables) == 1
    assert len(conn.tables[0].rows) == 1
    assert conn.tables[0].rows[0][1][JobMetaKey.STUDY.value] == "default"
    assert conn.tables[0].rows[0][1][MetaKey.JOB_ID] == "legacy-job"


def test_list_jobs_defaults_to_default_study_when_session_study_missing(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    jobs = [
        _FakeListedJob({JobMetaKey.JOB_ID.value: "legacy-job", JobMetaKey.JOB_NAME.value: "legacy"}),
        _FakeListedJob(
            {
                JobMetaKey.JOB_ID.value: "study-job",
                JobMetaKey.JOB_NAME.value: "study",
                JobMetaKey.STUDY.value: "cancer-research",
            }
        ),
    ]
    conn = _MockConnection(app_ctx=_FakeListEngine(jobs))

    JobCommandModule().list_jobs(conn, ["list_jobs"])

    assert conn.errors == []
    assert conn.strings == []
    assert len(conn.tables) == 1
    assert len(conn.tables[0].rows) == 1
    assert conn.tables[0].rows[0][1][JobMetaKey.STUDY.value] == "default"
    assert len(conn.successes) == 1


def test_get_job_meta_normalizes_legacy_job_study(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    jobs = [_FakeListedJob({JobMetaKey.JOB_ID.value: "legacy-job", JobMetaKey.JOB_NAME.value: "legacy"})]
    conn = _MockConnection(app_ctx=_FakeListEngine(jobs), props={JobCommandModule.JOB_ID: "legacy-job"})

    JobCommandModule().get_job_meta(conn, ["get_job_meta", "legacy-job"])

    assert conn.errors == []
    assert len(conn.dicts) == 1
    assert conn.dicts[0][0][JobMetaKey.STUDY.value] == "default"


def test_get_job_log_tail_uses_bounded_lines(tmp_path):
    job_cmds_module.ServerEngine = _FakeServerEngine
    workspace = _FakeWorkspace(tmp_path)
    engine = _FakeServerEngine(workspace)
    conn = _MockConnection(app_ctx=engine, props={JobCommandModule.JOB_ID: "job-1"})
    log_file = Path(workspace.get_log_root("job-1")) / WorkspaceConstants.LOG_FILE_NAME
    log_file.write_text("line1\nline2\nline3\n", encoding="utf-8")

    JobCommandModule().get_job_log(conn, ["get_job_log", "job-1", "-n", "2"])

    payload, _meta = conn.dicts[0]
    assert payload["logs"]["server"] == "line2\nline3\n"


def test_get_job_log_truncates_large_output(tmp_path, monkeypatch):
    job_cmds_module.ServerEngine = _FakeServerEngine
    workspace = _FakeWorkspace(tmp_path)
    engine = _FakeServerEngine(workspace)
    conn = _MockConnection(app_ctx=engine, props={JobCommandModule.JOB_ID: "job-1"})
    log_file = Path(workspace.get_log_root("job-1")) / WorkspaceConstants.LOG_FILE_NAME
    log_file.write_text("a" * 12 + "\n" + "b" * 12 + "\n", encoding="utf-8")

    monkeypatch.setattr(JobCommandModule, "MAX_RETURNED_JOB_LOG_BYTES", 16)

    JobCommandModule().get_job_log(conn, ["get_job_log", "job-1"])

    payload, _meta = conn.dicts[0]
    assert "truncated after 16 bytes" in payload["logs"]["server"]


def test_configure_job_log_all_targets_server_and_clients(tmp_path, monkeypatch):
    workspace = _FakeWorkspace(tmp_path)
    engine = _FakeServerEngine(workspace)
    engine.job_def_manager.get_job.return_value = _FakeListedJob({JobMetaKey.STATUS.value: RunStatus.RUNNING})
    conn = _MockConnection(app_ctx=engine)
    module = JobCommandModule()
    client_replies = [object()]
    monkeypatch.setattr(module, "send_request_to_clients", lambda conn, message: client_replies)
    processed = []
    monkeypatch.setattr(module, "process_replies_to_table", lambda conn, replies: processed.append(replies))

    module.configure_job_log(conn, ["configure_job_log", "job-1", "all", "DEBUG"])

    engine.configure_job_log.assert_called_once_with("job-1", "DEBUG")
    assert processed == [client_replies]
    assert any("successfully configured server job job-1 log" in msg for msg, _meta in conn.strings)


def test_authorize_job_id_hides_jobs_from_other_studies(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    monkeypatch.setattr(job_cmds_module, "StudyRegistryService", _FakeStudyRegistryService, raising=False)
    monkeypatch.setattr(
        _FakeStudyRegistryService,
        "registry",
        _FakeStudyRegistry(
            roles={"cancer-research": {"submitter": "study_lead"}}, sites={"cancer-research": {"site1"}}
        ),
        raising=False,
    )

    job_id = "123e4567-e89b-42d3-a456-426614174000"
    study_job = _FakeListedJob(
        {
            JobMetaKey.JOB_ID.value: job_id,
            JobMetaKey.JOB_NAME.value: "study",
            JobMetaKey.STUDY.value: "cancer-research",
            JobMetaKey.SUBMITTER_NAME.value: "submitter",
            JobMetaKey.SUBMITTER_ORG.value: "org",
            JobMetaKey.SUBMITTER_ROLE.value: "study_lead",
        }
    )
    conn = _MockConnection(
        app_ctx=_FakeListEngine([study_job]),
        props={ConnProps.ACTIVE_STUDY: "multiple-sclerosis"},
    )

    rc = JobCommandModule().authorize_job_id(conn, ["authorize_job_id", job_id])

    assert rc == PreAuthzReturnCode.ERROR
    assert conn.errors and conn.errors[0][0] == f"Job with ID {job_id} doesn't exist"
    assert conn.get_prop(JobCommandModule.JOB) is None
    assert conn.get_prop(ConnProps.SUBMITTER_ROLE) is None


def test_authorize_job_id_hides_non_default_jobs_from_default_session_without_registry(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    monkeypatch.setattr(job_cmds_module, "StudyRegistryService", _FakeStudyRegistryService, raising=False)
    monkeypatch.setattr(_FakeStudyRegistryService, "registry", None, raising=False)

    job_id = "123e4567-e89b-42d3-a456-426614174002"
    study_job = _FakeListedJob(
        {
            JobMetaKey.JOB_ID.value: job_id,
            JobMetaKey.JOB_NAME.value: "study",
            JobMetaKey.STUDY.value: "legacy-study",
            JobMetaKey.SUBMITTER_NAME.value: "submitter",
            JobMetaKey.SUBMITTER_ORG.value: "org",
            JobMetaKey.SUBMITTER_ROLE.value: "study_lead",
        }
    )
    conn = _MockConnection(
        app_ctx=_FakeListEngine([study_job]),
        props={ConnProps.ACTIVE_STUDY: "default"},
    )

    rc = JobCommandModule().authorize_job_id(conn, ["authorize_job_id", job_id])

    assert rc == PreAuthzReturnCode.ERROR
    assert conn.errors and conn.errors[0][0] == f"Job with ID {job_id} doesn't exist"
    assert conn.get_prop(JobCommandModule.JOB) is None
    assert conn.get_prop(ConnProps.SUBMITTER_ROLE) is None


def test_authorize_job_id_resolves_study_role_before_authz(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    monkeypatch.setattr(job_cmds_module, "StudyRegistryService", _FakeStudyRegistryService, raising=False)
    monkeypatch.setattr(cmd_utils_module, "StudyRegistryService", _FakeStudyRegistryService, raising=False)
    monkeypatch.setattr(
        _FakeStudyRegistryService,
        "registry",
        _FakeStudyRegistry(
            roles={"cancer-research": {"admin@nvidia.com": "study_lead"}}, sites={"cancer-research": {"site1"}}
        ),
        raising=False,
    )

    job_id = "123e4567-e89b-42d3-a456-426614174001"
    study_job = _FakeListedJob(
        {
            JobMetaKey.JOB_ID.value: job_id,
            JobMetaKey.JOB_NAME.value: "study",
            JobMetaKey.STUDY.value: "cancer-research",
            JobMetaKey.SUBMITTER_NAME.value: "submitter",
            JobMetaKey.SUBMITTER_ORG.value: "org",
            JobMetaKey.SUBMITTER_ROLE.value: "study_lead",
        }
    )
    conn = _MockConnection(
        app_ctx=_FakeListEngine([study_job]),
        props={
            ConnProps.ACTIVE_STUDY: "cancer-research",
            ConnProps.USER_NAME: "admin@nvidia.com",
            ConnProps.USER_ROLE: "project_admin",
        },
    )

    rc = JobCommandModule().authorize_job_id(conn, ["authorize_job_id", job_id])

    assert rc == PreAuthzReturnCode.REQUIRE_AUTHZ
    assert conn.get_prop(ConnProps.USER_ROLE) == "study_lead"


def test_submit_job_persists_effective_study_submitter_role(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    monkeypatch.setattr(job_cmds_module, "StudyRegistryService", _FakeStudyRegistryService, raising=False)
    monkeypatch.setattr(cmd_utils_module, "StudyRegistryService", _FakeStudyRegistryService, raising=False)
    monkeypatch.setattr(
        _FakeStudyRegistryService,
        "registry",
        _FakeStudyRegistry(
            roles={"cancer-research": {"submitter": "study_lead"}}, sites={"cancer-research": {"site1"}}
        ),
        raising=False,
    )
    monkeypatch.setattr(
        job_cmds_module,
        "JobMetaValidator",
        lambda: _FakeJobMetaValidatorWithMeta(
            {
                JobMetaKey.JOB_NAME.value: "study-job",
                JobMetaKey.DEPLOY_MAP.value: {"app1": ["server", "site1"]},
            }
        ),
    )

    engine = _FakeEngine()
    conn = _MockConnection(
        app_ctx=engine,
        props={
            ConnProps.FILE_LOCATION: "job.zip",
            ConnProps.ACTIVE_STUDY: "cancer-research",
            ConnProps.USER_NAME: "submitter",
            ConnProps.USER_ORG: "org",
            ConnProps.USER_ROLE: "cert_admin",
        },
    )

    rc = JobCommandModule().command_authz_required(conn, ["submit_job"])
    assert rc == PreAuthzReturnCode.REQUIRE_AUTHZ

    JobCommandModule().submit_job(conn, ["submit_job", "job_folder"])

    assert conn.errors == []
    assert len(conn.successes) == 1
    assert engine.job_def_manager.created_meta[JobMetaKey.SUBMITTER_ROLE.value] == "study_lead"


def test_submit_job_rejects_deploy_map_sites_outside_study(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    monkeypatch.setattr(job_cmds_module, "StudyRegistryService", _FakeStudyRegistryService, raising=False)
    monkeypatch.setattr(
        _FakeStudyRegistryService,
        "registry",
        _FakeStudyRegistry(sites={"cancer-research": {"site1", "site2"}}),
        raising=False,
    )
    monkeypatch.setattr(
        job_cmds_module,
        "JobMetaValidator",
        lambda: _FakeJobMetaValidatorWithMeta(
            {
                JobMetaKey.JOB_NAME.value: "study-job",
                JobMetaKey.DEPLOY_MAP.value: {"app1": ["server", "site1", "site3"]},
            }
        ),
    )

    engine = _FakeEngine()
    conn = _MockConnection(
        app_ctx=engine,
        props={
            ConnProps.FILE_LOCATION: "job.zip",
            ConnProps.ACTIVE_STUDY: "cancer-research",
            ConnProps.USER_NAME: "submitter",
            ConnProps.USER_ORG: "org",
            ConnProps.USER_ROLE: "cert_admin",
        },
    )

    JobCommandModule().submit_job(conn, ["submit_job", "job_folder"])

    assert conn.errors
    assert "site 'site3' is not enrolled in study 'cancer-research'" in conn.errors[0][0]


def test_get_job_log_returns_server_log(monkeypatch, tmp_path):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", object)

    workspace = _FakeWorkspace(tmp_path)
    log_file = Path(workspace.get_log_root("job-123")) / WorkspaceConstants.LOG_FILE_NAME
    log_file.write_text("line1\nERROR line2\nline3\n", encoding="utf-8")

    conn = _MockConnection(
        app_ctx=_FakeServerEngine(workspace),
        props={JobCommandModule.JOB_ID: "job-123"},
    )

    JobCommandModule().get_job_log(conn, ["get_job_log", "job-123"])

    assert conn.errors == []
    assert conn.dicts[0][0] == {"logs": {"server": "line1\nERROR line2\nline3\n"}}
    assert conn.dicts[0][1][MetaKey.STATUS].lower() == "ok"


def test_get_job_log_applies_grep_and_tail(monkeypatch, tmp_path):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", object)

    workspace = _FakeWorkspace(tmp_path)
    log_file = Path(workspace.get_log_root("job-123")) / WorkspaceConstants.LOG_FILE_NAME
    log_file.write_text("INFO line1\nERROR line2\nERROR line3\n", encoding="utf-8")

    conn = _MockConnection(
        app_ctx=_FakeServerEngine(workspace),
        props={JobCommandModule.JOB_ID: "job-123"},
    )

    JobCommandModule().get_job_log(conn, ["get_job_log", "job-123", "-g", "ERROR", "-n", "1"])

    assert conn.errors == []
    assert conn.dicts[0][0] == {"logs": {"server": "ERROR line3\n"}}


def test_get_job_log_tail_zero_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", object)

    workspace = _FakeWorkspace(tmp_path)
    log_file = Path(workspace.get_log_root("job-123")) / WorkspaceConstants.LOG_FILE_NAME
    log_file.write_text("line1\nline2\n", encoding="utf-8")

    conn = _MockConnection(
        app_ctx=_FakeServerEngine(workspace),
        props={JobCommandModule.JOB_ID: "job-123"},
    )

    JobCommandModule().get_job_log(conn, ["get_job_log", "job-123", "-n", "0"])

    assert conn.errors == []
    assert conn.dicts[0][0] == {"logs": {"server": ""}}


def test_get_job_log_missing_file_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", object)

    workspace = _FakeWorkspace(tmp_path)
    conn = _MockConnection(
        app_ctx=_FakeServerEngine(workspace),
        props={JobCommandModule.JOB_ID: "job-123"},
    )

    JobCommandModule().get_job_log(conn, ["get_job_log", "job-123"])

    assert conn.errors == []
    assert conn.dicts[0][0] == {"logs": {"server": ""}}
    assert conn.successes == []


def test_submit_job_reports_all_deploy_map_sites_outside_study(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    monkeypatch.setattr(job_cmds_module, "StudyRegistryService", _FakeStudyRegistryService, raising=False)
    monkeypatch.setattr(
        _FakeStudyRegistryService,
        "registry",
        _FakeStudyRegistry(sites={"cancer-research": {"site1"}}),
        raising=False,
    )
    monkeypatch.setattr(
        job_cmds_module,
        "JobMetaValidator",
        lambda: _FakeJobMetaValidatorWithMeta(
            {
                JobMetaKey.JOB_NAME.value: "study-job",
                JobMetaKey.DEPLOY_MAP.value: {"app1": ["server", "site2"], "app2": ["site3", "site2"]},
            }
        ),
    )

    engine = _FakeEngine()
    conn = _MockConnection(
        app_ctx=engine,
        props={
            ConnProps.FILE_LOCATION: "job.zip",
            ConnProps.ACTIVE_STUDY: "cancer-research",
            ConnProps.USER_NAME: "submitter",
            ConnProps.USER_ORG: "org",
            ConnProps.USER_ROLE: "cert_admin",
        },
    )

    JobCommandModule().submit_job(conn, ["submit_job", "job_folder"])

    assert conn.errors
    assert conn.errors[0][0] == "sites 'site2', 'site3' are not enrolled in study 'cancer-research'"
    assert engine.job_def_manager.created_meta is None
    assert conn.successes == []
