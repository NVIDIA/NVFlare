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

import gc
import io
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import (
    SUBMIT_TOKEN_JOB_DELETED_STATUS,
    FLContextKey,
    ReturnCode,
    ServerCommandKey,
    WorkspaceConstants,
)
from nvflare.apis.job_def import JobMetaKey, RunStatus, SubmitRecordKey, SubmitRecordState
from nvflare.apis.shareable import Shareable
from nvflare.fuel.hci.proto import MetaKey, MetaStatusValue
from nvflare.fuel.hci.server.authz import PreAuthzReturnCode
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.lighter.tool_consts import NVFLARE_SUBMITTER_CRT_FILE
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
        Namespace(u=True, d=True, job_id="12345", m=3, n="hello_", r=False, submit_token=None),
    ),
    (
        ["12345", "-d", "-u", "-n", "hello_", "-m", "3"],
        Namespace(u=True, d=True, job_id="12345", m=3, n="hello_", r=False, submit_token=None),
    ),
    (
        ["-d", "-u", "-n", "hello_", "-m", "3"],
        Namespace(u=True, d=True, job_id=None, m=3, n="hello_", r=False, submit_token=None),
    ),
    (
        ["-u", "-n", "hello_", "-m", "5"],
        Namespace(u=True, d=False, job_id=None, m=5, n="hello_", r=False, submit_token=None),
    ),
    (["-u"], Namespace(u=True, d=False, job_id=None, m=None, n=None, r=False, submit_token=None)),
    (["-r"], Namespace(u=False, d=False, job_id=None, m=None, n=None, r=True, submit_token=None)),
    (["nvflare"], Namespace(u=False, d=False, job_id="nvflare", m=None, n=None, r=False, submit_token=None)),
]


class TestListJobCmdParser:
    @pytest.mark.parametrize("args, expected_args", TEST_CASES)
    def test_parse_args(self, args: list[str], expected_args):
        parser = _create_list_job_cmd_parser()
        parsed_args = parser.parse_args(args)
        assert parsed_args == expected_args


class TestGetJobLogCmdParser:
    def test_parse_args_defaults_to_server(self):
        parser = _create_get_job_log_cmd_parser()
        parsed_args = parser.parse_args(["job-123"])
        assert parsed_args == Namespace(job_id="job-123", target="server", log_file_name="log.txt")

    def test_parse_args_accepts_target(self):
        parser = _create_get_job_log_cmd_parser()
        parsed_args = parser.parse_args(["job-123", "site-1"])
        assert parsed_args == Namespace(job_id="job-123", target="site-1", log_file_name="log.txt")

    def test_parse_args_accepts_internal_log_file_name(self):
        parser = _create_get_job_log_cmd_parser()
        parsed_args = parser.parse_args(["job-123", "site-1", "log.json"])
        assert parsed_args == Namespace(job_id="job-123", target="site-1", log_file_name="log.json")


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


class _FakeJobMetaValidatorFolderOnly:
    def validate(self, folder_name, zip_file_name):
        assert folder_name == "job_folder"
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


class _FakeSubmitTokenJobDefManager:
    def __init__(self):
        self.create_count = 0
        self.new_record_count = 0
        self.created_metas = []
        self.jobs = {}
        self.records = {}

    def create(self, meta, uploaded_content, fl_ctx):
        self.create_count += 1
        result = dict(meta)
        result[JobMetaKey.JOB_ID.value] = result.get(JobMetaKey.JOB_ID.value) or f"job-{self.create_count}"
        self.created_metas.append(result)
        self.jobs[result[JobMetaKey.JOB_ID.value]] = _FakeListedJob(result)
        return result

    def get_job(self, jid, fl_ctx):
        return self.jobs.get(jid)

    def get_all_jobs(self, fl_ctx):
        return list(self.jobs.values())

    def delete(self, jid, fl_ctx):
        self.jobs.pop(jid, None)

    def mark_submit_records_job_deleted(self, job_id, deleted_by, fl_ctx):
        if isinstance(deleted_by, dict):
            deleted_by_info = {
                "name": deleted_by.get("name", ""),
                "org": deleted_by.get("org", ""),
                "role": deleted_by.get("role", ""),
            }
        else:
            deleted_by_info = {
                "name": getattr(deleted_by, "name", ""),
                "org": getattr(deleted_by, "org", ""),
                "role": getattr(deleted_by, "role", ""),
            }
        updated = []
        for record in self.records.values():
            if record.get(SubmitRecordKey.JOB_ID.value) != job_id:
                continue
            if record.get(SubmitRecordKey.STATE.value) == SubmitRecordState.JOB_DELETED.value:
                continue
            record[SubmitRecordKey.STATE.value] = SubmitRecordState.JOB_DELETED.value
            record[SubmitRecordKey.DELETED_TIME.value] = "2026-04-30T10:00:00-07:00"
            record[SubmitRecordKey.DELETED_BY.value] = deleted_by_info
            updated.append(dict(record))
        return updated

    def _record_key(self, study, submitter, submit_token):
        if isinstance(submitter, dict):
            submitter_key = (
                submitter.get("name", ""),
                submitter.get("org", ""),
                submitter.get("role", ""),
            )
        else:
            submitter_key = (
                getattr(submitter, "name", ""),
                getattr(submitter, "org", ""),
                getattr(submitter, "role", ""),
            )
        return study, submitter_key, submit_token

    def get_submit_record(self, study, submitter, submit_token, fl_ctx):
        return self.records.get(self._record_key(study, submitter, submit_token))

    def new_submit_record(
        self,
        study,
        submitter,
        submit_token,
        job_content_hash,
        job_name="",
        job_folder_name="",
        job_id=None,
        state=SubmitRecordState.CREATING.value,
    ):
        self.new_record_count += 1
        if isinstance(submitter, dict):
            submitter_info = submitter
        else:
            submitter_info = {
                "name": getattr(submitter, "name", ""),
                "org": getattr(submitter, "org", ""),
                "role": getattr(submitter, "role", ""),
            }
        return {
            SubmitRecordKey.SCHEMA_VERSION.value: 1,
            SubmitRecordKey.STATE.value: state,
            SubmitRecordKey.SUBMIT_TOKEN.value: submit_token,
            SubmitRecordKey.JOB_ID.value: job_id or f"reserved-job-{self.new_record_count}",
            SubmitRecordKey.STUDY.value: study,
            SubmitRecordKey.SUBMITTER_NAME.value: submitter_info.get("name", ""),
            SubmitRecordKey.SUBMITTER_ORG.value: submitter_info.get("org", ""),
            SubmitRecordKey.SUBMITTER_ROLE.value: submitter_info.get("role", ""),
            SubmitRecordKey.JOB_NAME.value: job_name,
            SubmitRecordKey.JOB_FOLDER_NAME.value: job_folder_name,
            SubmitRecordKey.JOB_CONTENT_HASH.value: job_content_hash,
            SubmitRecordKey.SUBMIT_TIME.value: "2026-04-29T10:00:00-07:00",
        }

    def create_submit_record(self, record, fl_ctx):
        submitter = {
            "name": record.get(SubmitRecordKey.SUBMITTER_NAME.value, ""),
            "org": record.get(SubmitRecordKey.SUBMITTER_ORG.value, ""),
            "role": record.get(SubmitRecordKey.SUBMITTER_ROLE.value, ""),
        }
        key = self._record_key(
            record[SubmitRecordKey.STUDY.value],
            submitter,
            record[SubmitRecordKey.SUBMIT_TOKEN.value],
        )
        if key in self.records:
            raise RuntimeError("submit record already exists")
        self.records[key] = dict(record)
        return True

    def update_submit_record(self, record, fl_ctx):
        submitter = {
            "name": record.get(SubmitRecordKey.SUBMITTER_NAME.value, ""),
            "org": record.get(SubmitRecordKey.SUBMITTER_ORG.value, ""),
            "role": record.get(SubmitRecordKey.SUBMITTER_ROLE.value, ""),
        }
        self.records[
            self._record_key(record[SubmitRecordKey.STUDY.value], submitter, record[SubmitRecordKey.SUBMIT_TOKEN.value])
        ] = dict(record)
        return dict(record)

    def get_job_by_submit_token(self, study, submitter, submit_token, fl_ctx):
        record = self.get_submit_record(study, submitter, submit_token, fl_ctx)
        if not record:
            return None
        return self.get_job(record.get(SubmitRecordKey.JOB_ID.value), fl_ctx)


class _BrokenSubmitTokenJobDefManager(_FakeSubmitTokenJobDefManager):
    def new_submit_record(self, *args, **kwargs):
        record = super().new_submit_record(*args, **kwargs)
        record.pop(SubmitRecordKey.JOB_ID.value, None)
        return record


class _DelayedVisibleSubmitTokenJobDefManager(_FakeSubmitTokenJobDefManager):
    def __init__(self):
        super().__init__()
        self.get_job_count = 0
        self.visible_job_id = None

    def get_job(self, jid, fl_ctx):
        self.get_job_count += 1
        if self.get_job_count >= 2 and jid == self.visible_job_id:
            return _FakeListedJob({JobMetaKey.JOB_ID.value: jid})
        return None


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
        self._clients = []

    def get_workspace(self):
        return self.workspace

    def new_context(self):
        from nvflare.apis.fl_context import FLContext

        return FLContext()

    def get_clients(self):
        return list(self._clients)

    def validate_targets(self, client_names):
        by_name = {c.name: c for c in self._clients}
        clients = []
        invalid = []
        for client_name in client_names:
            client = by_name.get(client_name)
            if client:
                clients.append(client)
            else:
                invalid.append(client_name)
        return clients, invalid


class _FakeClient:
    def __init__(self, name, token):
        self.name = name
        self.token = token


class _FakeStudyRegistry:
    def __init__(self, sites=None):
        self.sites = sites or {}

    def has_study(self, study):
        return study in self.sites

    def get_sites(self, study):
        return self.sites.get(study)

    def get_studies(self):
        return {study: {"site_orgs": {"org": sorted(sites)}} for study, sites in self.sites.items()}


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


def _zip_bytes(files):
    output = io.BytesIO()
    with ZipFile(output, "w") as zip_file:
        for name, content in files.items():
            zip_file.writestr(name, content)
    return output.getvalue()


def _submit_conn(engine, uploaded_content, study="default", user_name="submitter"):
    return _MockConnection(
        app_ctx=engine,
        props={
            ConnProps.FILE_LOCATION: uploaded_content,
            ConnProps.ACTIVE_STUDY: study,
            ConnProps.USER_NAME: user_name,
            ConnProps.USER_ORG: "org",
            ConnProps.USER_ROLE: "role",
        },
    )


def _submitted_job_id(conn):
    assert conn.errors == []
    assert conn.successes
    return conn.successes[-1][1][MetaKey.JOB_ID]


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


def test_submit_job_strips_user_supplied_from_hub_site(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    monkeypatch.setattr(
        job_cmds_module,
        "JobMetaValidator",
        lambda: _FakeJobMetaValidatorWithMeta({JobMetaKey.FROM_HUB_SITE.value: "site-x"}),
    )

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
    assert JobMetaKey.FROM_HUB_SITE.value not in engine.job_def_manager.created_meta


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


def test_server_list_parser_accepts_submit_token():
    parser = _create_list_job_cmd_parser()
    parsed_args = parser.parse_args(["--submit-token", "retry.01:A_b-c"])

    assert parsed_args.submit_token == "retry.01:A_b-c"


@pytest.mark.parametrize("token", ["", "bad token", "bad/token", "x" * 129])
def test_submit_job_rejects_invalid_submit_token(monkeypatch, token):
    monkeypatch.setattr(job_cmds_module, "JobMetaValidator", _FakeJobMetaValidatorFolderOnly)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)

    engine = _FakeEngine()
    conn = _submit_conn(engine, _zip_bytes({"meta.json": "{}"}))

    JobCommandModule().submit_job(conn, ["submit_job", "job_folder", "--submit-token", token])

    assert len(conn.errors) == 1
    assert "submit_token" in conn.errors[0][0]
    assert engine.job_def_manager.created_meta is None


def test_same_submit_token_same_content_returns_same_job(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobMetaValidator", _FakeJobMetaValidatorFolderOnly)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)

    engine = _FakeEngine()
    engine.job_def_manager = _FakeSubmitTokenJobDefManager()
    content = _zip_bytes({"meta.json": "{}", "app/config/config_fed_server.json": "{}"})

    conn1 = _submit_conn(engine, content, study="study-a")
    JobCommandModule().submit_job(conn1, ["submit_job", "job_folder", "--submit-token", "retry-1"])
    conn2 = _submit_conn(engine, content, study="study-a")
    JobCommandModule().submit_job(conn2, ["submit_job", "job_folder", "--submit-token", "retry-1"])

    assert _submitted_job_id(conn1) == _submitted_job_id(conn2)
    assert engine.job_def_manager.create_count == 1
    assert engine.job_def_manager.new_record_count == 1


def test_same_submit_token_after_job_deleted_returns_deleted_status():
    manager = _FakeSubmitTokenJobDefManager()
    submitter = {"name": "admin@nvidia.com", "org": "nvidia", "role": "project_admin"}
    record = manager.new_submit_record(
        study="default",
        submitter=submitter,
        submit_token="retry-1",
        job_content_hash="hash-1",
        job_name="job-name",
        job_folder_name="job_folder",
    )
    record[SubmitRecordKey.STATE.value] = SubmitRecordState.JOB_DELETED.value
    record[SubmitRecordKey.DELETED_TIME.value] = "2026-04-30T10:00:00-07:00"
    manager.records[manager._record_key("default", submitter, "retry-1")] = record
    conn = _MockConnection()

    job_id = JobCommandModule()._handle_submit_token_record_locked(
        conn=conn,
        job_def_manager=manager,
        study="default",
        submitter=submitter,
        submit_token="retry-1",
        job_content_hash="hash-1",
        meta={JobMetaKey.JOB_NAME.value: "job-name"},
        folder_name="job_folder",
        zip_file_name="missing-job.zip",
        fl_ctx=None,
    )

    assert job_id is None
    assert len(conn.errors) == 1
    assert "SUBMIT_TOKEN_JOB_DELETED" in conn.errors[0][0]
    assert conn.errors[0][1][MetaKey.STATUS] == SUBMIT_TOKEN_JOB_DELETED_STATUS
    assert conn.errors[0][1][MetaKey.JOB_ID] == record[SubmitRecordKey.JOB_ID.value]
    assert manager.create_count == 0


def test_same_submit_token_different_content_conflicts(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobMetaValidator", _FakeJobMetaValidatorFolderOnly)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)

    engine = _FakeEngine()
    engine.job_def_manager = _FakeSubmitTokenJobDefManager()

    conn1 = _submit_conn(engine, _zip_bytes({"meta.json": "{}", "app/config/config_fed_server.json": "{}"}))
    JobCommandModule().submit_job(conn1, ["submit_job", "job_folder", "--submit-token", "retry-1"])
    conn2 = _submit_conn(
        engine,
        _zip_bytes({"meta.json": "{}", "app/config/config_fed_server.json": "{}", "app/custom.txt": "changed"}),
    )
    JobCommandModule().submit_job(conn2, ["submit_job", "job_folder", "--submit-token", "retry-1"])

    assert _submitted_job_id(conn1)
    assert len(conn2.errors) == 1
    assert "SUBMIT_TOKEN_CONFLICT" in conn2.errors[0][0]
    assert engine.job_def_manager.create_count == 1


def test_delete_job_marks_submit_record_job_deleted():
    manager = _FakeSubmitTokenJobDefManager()
    submitter = {"name": "submitter", "org": "org", "role": "role"}
    record = manager.new_submit_record(
        study="study-a",
        submitter=submitter,
        submit_token="retry-1",
        job_content_hash="hash-1",
    )
    job_id = record[SubmitRecordKey.JOB_ID.value]
    manager.records[manager._record_key("study-a", submitter, "retry-1")] = record
    manager.jobs[job_id] = _FakeListedJob(
        {JobMetaKey.JOB_ID.value: job_id, JobMetaKey.STATUS: RunStatus.SUBMITTED.value}
    )
    engine = _FakeEngine()
    engine.job_def_manager = manager
    conn = _MockConnection(
        app_ctx=engine,
        props={
            JobCommandModule.JOB: manager.jobs[job_id],
            JobCommandModule.JOB_ID: job_id,
            ConnProps.USER_NAME: "admin@nvidia.com",
            ConnProps.USER_ORG: "nvidia",
            ConnProps.USER_ROLE: "project_admin",
        },
    )

    JobCommandModule().delete_job(conn, ["delete_job", job_id])

    updated = manager.get_submit_record("study-a", submitter, "retry-1", None)
    assert conn.errors == []
    assert updated[SubmitRecordKey.STATE.value] == SubmitRecordState.JOB_DELETED.value
    assert updated[SubmitRecordKey.DELETED_BY.value] == {
        "name": "admin@nvidia.com",
        "org": "nvidia",
        "role": "project_admin",
    }
    assert conn.successes[0][1]["submit_records_marked_deleted"] == 1


def test_same_submit_token_different_study_is_independent(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobMetaValidator", _FakeJobMetaValidatorFolderOnly)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)

    engine = _FakeEngine()
    engine.job_def_manager = _FakeSubmitTokenJobDefManager()
    content_a = _zip_bytes({"meta.json": "{}", "app/config/config_fed_server.json": "study-a"})
    content_b = _zip_bytes({"meta.json": "{}", "app/config/config_fed_server.json": "study-b"})

    conn1 = _submit_conn(engine, content_a, study="study-a")
    JobCommandModule().submit_job(conn1, ["submit_job", "job_folder", "--submit-token", "retry-1"])
    conn2 = _submit_conn(engine, content_b, study="study-b")
    JobCommandModule().submit_job(conn2, ["submit_job", "job_folder", "--submit-token", "retry-1"])

    assert _submitted_job_id(conn1) != _submitted_job_id(conn2)
    assert engine.job_def_manager.create_count == 2
    assert conn2.errors == []


def test_no_submit_token_keeps_duplicate_submit_behavior(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobMetaValidator", _FakeJobMetaValidatorFolderOnly)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)

    engine = _FakeEngine()
    engine.job_def_manager = _FakeSubmitTokenJobDefManager()
    content = _zip_bytes({"meta.json": "{}", "app/config/config_fed_server.json": "{}"})

    conn1 = _submit_conn(engine, content)
    JobCommandModule().submit_job(conn1, ["submit_job", "job_folder"])
    conn2 = _submit_conn(engine, content)
    JobCommandModule().submit_job(conn2, ["submit_job", "job_folder"])

    assert _submitted_job_id(conn1) != _submitted_job_id(conn2)
    assert engine.job_def_manager.create_count == 2


def test_submit_token_is_not_written_to_job_meta(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobMetaValidator", _FakeJobMetaValidatorFolderOnly)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)

    engine = _FakeEngine()
    engine.job_def_manager = _FakeSubmitTokenJobDefManager()
    conn = _submit_conn(engine, _zip_bytes({"meta.json": "{}", "app/config/config_fed_server.json": "{}"}))

    JobCommandModule().submit_job(conn, ["submit_job", "job_folder", "--submit-token", "retry-1"])

    assert _submitted_job_id(conn)
    assert "submit_token" not in engine.job_def_manager.created_metas[0]


def test_submit_token_record_handling_does_not_mutate_caller_meta():
    manager = _FakeSubmitTokenJobDefManager()
    meta = {JobMetaKey.JOB_NAME.value: "job-name"}
    original_meta = dict(meta)

    job_id = JobCommandModule()._handle_submit_token_record_locked(
        conn=_MockConnection(),
        job_def_manager=manager,
        study="default",
        submitter={"name": "admin@nvidia.com", "org": "nvidia", "role": "project_admin"},
        submit_token="retry-1",
        job_content_hash="hash-1",
        meta=meta,
        folder_name="job_folder",
        zip_file_name="job.zip",
        fl_ctx=None,
    )

    assert job_id == "reserved-job-1"
    assert meta == original_meta
    assert manager.created_metas[0][JobMetaKey.JOB_ID.value] == "reserved-job-1"


def test_submit_token_recovery_repairs_record_missing_job_id(tmp_path):
    manager = _FakeSubmitTokenJobDefManager()
    submitter = {"name": "admin@nvidia.com", "org": "nvidia", "role": "project_admin"}
    zip_file = tmp_path / "job.zip"
    zip_file.write_bytes(b"job content")
    existing = manager.new_submit_record(
        study="default",
        submitter=submitter,
        submit_token="retry-1",
        job_content_hash="hash-1",
        job_name="job-name",
        job_folder_name="job_folder",
    )
    existing.pop(SubmitRecordKey.JOB_ID.value)
    manager.records[manager._record_key("default", submitter, "retry-1")] = existing

    job_id = JobCommandModule()._handle_submit_token_record_locked(
        conn=_MockConnection(),
        job_def_manager=manager,
        study="default",
        submitter=submitter,
        submit_token="retry-1",
        job_content_hash="hash-1",
        meta={JobMetaKey.JOB_NAME.value: "job-name"},
        folder_name="job_folder",
        zip_file_name=str(zip_file),
        fl_ctx=None,
    )

    repaired = manager.get_submit_record("default", submitter, "retry-1", None)
    assert job_id == "reserved-job-2"
    assert repaired[SubmitRecordKey.JOB_ID.value] == job_id
    assert repaired[SubmitRecordKey.STATE.value] == SubmitRecordState.CREATED.value
    assert manager.created_metas[0][JobMetaKey.JOB_ID.value] == job_id


def test_submit_token_recovery_rejects_repaired_record_without_job_id(tmp_path):
    manager = _BrokenSubmitTokenJobDefManager()
    submitter = {"name": "admin@nvidia.com", "org": "nvidia", "role": "project_admin"}
    zip_file = tmp_path / "job.zip"
    zip_file.write_bytes(b"job content")
    existing = _FakeSubmitTokenJobDefManager.new_submit_record(
        manager,
        study="default",
        submitter=submitter,
        submit_token="retry-1",
        job_content_hash="hash-1",
        job_name="job-name",
        job_folder_name="job_folder",
    )
    existing.pop(SubmitRecordKey.JOB_ID.value)
    manager.records[manager._record_key("default", submitter, "retry-1")] = existing

    with pytest.raises(RuntimeError, match="missing job_id"):
        JobCommandModule()._handle_submit_token_record_locked(
            conn=_MockConnection(),
            job_def_manager=manager,
            study="default",
            submitter=submitter,
            submit_token="retry-1",
            job_content_hash="hash-1",
            meta={JobMetaKey.JOB_NAME.value: "job-name"},
            folder_name="job_folder",
            zip_file_name=str(zip_file),
            fl_ctx=None,
        )

    assert manager.create_count == 0


def test_submit_token_recovery_rejects_missing_uploaded_content():
    manager = _FakeSubmitTokenJobDefManager()
    submitter = {"name": "admin@nvidia.com", "org": "nvidia", "role": "project_admin"}
    existing = manager.new_submit_record(
        study="default",
        submitter=submitter,
        submit_token="retry-1",
        job_content_hash="hash-1",
        job_name="job-name",
        job_folder_name="job_folder",
    )
    existing.pop(SubmitRecordKey.JOB_ID.value)
    manager.records[manager._record_key("default", submitter, "retry-1")] = existing

    with pytest.raises(RuntimeError, match="uploaded job content is no longer available"):
        JobCommandModule()._handle_submit_token_record_locked(
            conn=_MockConnection(),
            job_def_manager=manager,
            study="default",
            submitter=submitter,
            submit_token="retry-1",
            job_content_hash="hash-1",
            meta={JobMetaKey.JOB_NAME.value: "job-name"},
            folder_name="job_folder",
            zip_file_name="missing-job.zip",
            fl_ctx=None,
        )

    assert manager.create_count == 0


def test_submit_token_recovery_rechecks_existing_job_before_create(tmp_path):
    manager = _DelayedVisibleSubmitTokenJobDefManager()
    submitter = {"name": "admin@nvidia.com", "org": "nvidia", "role": "project_admin"}
    zip_file = tmp_path / "job.zip"
    zip_file.write_bytes(b"job content")
    existing = manager.new_submit_record(
        study="default",
        submitter=submitter,
        submit_token="retry-1",
        job_content_hash="hash-1",
        job_name="job-name",
        job_folder_name="job_folder",
    )
    job_id = existing[SubmitRecordKey.JOB_ID.value]
    manager.visible_job_id = job_id
    manager.records[manager._record_key("default", submitter, "retry-1")] = existing

    recovered_job_id = JobCommandModule()._handle_submit_token_record_locked(
        conn=_MockConnection(),
        job_def_manager=manager,
        study="default",
        submitter=submitter,
        submit_token="retry-1",
        job_content_hash="hash-1",
        meta={JobMetaKey.JOB_NAME.value: "job-name"},
        folder_name="job_folder",
        zip_file_name=str(zip_file),
        fl_ctx=None,
    )

    repaired = manager.get_submit_record("default", submitter, "retry-1", None)
    assert recovered_job_id == job_id
    assert repaired[SubmitRecordKey.STATE.value] == SubmitRecordState.CREATED.value
    assert manager.create_count == 0


def test_submit_token_create_record_false_recovers_existing_record(monkeypatch, tmp_path):
    manager = _FakeSubmitTokenJobDefManager()
    submitter = {"name": "admin@nvidia.com", "org": "nvidia", "role": "project_admin"}
    zip_file = tmp_path / "job.zip"
    zip_file.write_bytes(b"job content")

    def fake_create_submit_record(self, job_def_manager, record, fl_ctx):
        submitter_info = {
            "name": record.get(SubmitRecordKey.SUBMITTER_NAME.value, ""),
            "org": record.get(SubmitRecordKey.SUBMITTER_ORG.value, ""),
            "role": record.get(SubmitRecordKey.SUBMITTER_ROLE.value, ""),
        }
        manager.records[
            manager._record_key(
                record[SubmitRecordKey.STUDY.value], submitter_info, record[SubmitRecordKey.SUBMIT_TOKEN.value]
            )
        ] = dict(record)
        return False

    monkeypatch.setattr(JobCommandModule, "_create_submit_record", fake_create_submit_record)

    job_id = JobCommandModule()._handle_submit_token_record_locked(
        conn=_MockConnection(),
        job_def_manager=manager,
        study="default",
        submitter=submitter,
        submit_token="retry-1",
        job_content_hash="hash-1",
        meta={JobMetaKey.JOB_NAME.value: "job-name"},
        folder_name="job_folder",
        zip_file_name=str(zip_file),
        fl_ctx=None,
    )

    record = manager.get_submit_record("default", submitter, "retry-1", None)
    assert job_id == record[SubmitRecordKey.JOB_ID.value]
    assert record[SubmitRecordKey.STATE.value] == SubmitRecordState.CREATED.value
    assert manager.create_count == 1
    assert manager.created_metas[0][JobMetaKey.JOB_ID.value] == job_id


def test_user_job_meta_submit_token_is_stripped_before_submit_event(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobMetaValidator", _FakeJobMetaValidatorFolderOnly)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)

    engine = _FakeEngine()
    engine.job_def_manager = _FakeSubmitTokenJobDefManager()
    conn = _submit_conn(
        engine,
        _zip_bytes({"meta.json": '{"submit_token": "from-user-meta"}', "app/config/config_fed_server.json": "{}"}),
    )

    JobCommandModule().submit_job(conn, ["submit_job", "job_folder"])

    assert _submitted_job_id(conn)
    assert "submit_token" not in engine.submit_event_meta
    assert "submit_token" not in engine.job_def_manager.created_metas[0]


def test_canonical_hash_ignores_signature_artifacts(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobMetaValidator", _FakeJobMetaValidatorFolderOnly)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)

    engine = _FakeEngine()
    engine.job_def_manager = _FakeSubmitTokenJobDefManager()
    base_files = {"meta.json": "{}", "app/config/config_fed_server.json": "{}"}
    signed_files = dict(base_files)
    signed_files[".__nvfl_sig.json"] = '{"signature": "volatile"}'
    signed_files[NVFLARE_SUBMITTER_CRT_FILE] = "volatile cert"

    conn1 = _submit_conn(engine, _zip_bytes(base_files))
    JobCommandModule().submit_job(conn1, ["submit_job", "job_folder", "--submit-token", "retry-1"])
    conn2 = _submit_conn(engine, _zip_bytes(signed_files))
    JobCommandModule().submit_job(conn2, ["submit_job", "job_folder", "--submit-token", "retry-1"])

    assert _submitted_job_id(conn1) == _submitted_job_id(conn2)
    assert engine.job_def_manager.create_count == 1


def test_list_jobs_by_submit_token_returns_expected_job(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    engine = _FakeListEngine([])
    manager = _FakeSubmitTokenJobDefManager()
    meta = {
        JobMetaKey.JOB_ID.value: "job-1",
        JobMetaKey.JOB_NAME.value: "hello",
        JobMetaKey.STUDY.value: "study-a",
    }
    manager.jobs["job-1"] = _FakeListedJob(meta)
    manager.records[
        (
            "study-a",
            ("submitter", "org", "role"),
            "retry-1",
        )
    ] = {
        "study": "study-a",
        "submitter_name": "submitter",
        "submitter_org": "org",
        "submitter_role": "role",
        "submit_token": "retry-1",
        "job_id": "job-1",
        "job_content_hash": "sha256:abc",
    }
    engine.job_def_manager = manager
    conn = _MockConnection(
        app_ctx=engine,
        props={
            ConnProps.ACTIVE_STUDY: "study-a",
            ConnProps.USER_NAME: "submitter",
            ConnProps.USER_ORG: "org",
            ConnProps.USER_ROLE: "role",
        },
    )

    JobCommandModule().list_jobs(conn, ["list_jobs", "--submit-token", "retry-1"])

    assert conn.errors == []
    assert len(conn.tables) == 1
    assert conn.tables[0].rows[0][1][MetaKey.JOB_ID] == "job-1"


def test_list_jobs_by_submit_token_returns_deleted_status(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    engine = _FakeListEngine([])
    manager = _FakeSubmitTokenJobDefManager()
    manager.records[
        (
            "study-a",
            ("submitter", "org", "role"),
            "retry-1",
        )
    ] = {
        "study": "study-a",
        "submitter_name": "submitter",
        "submitter_org": "org",
        "submitter_role": "role",
        "submit_token": "retry-1",
        "job_id": "job-1",
        "job_content_hash": "sha256:abc",
        "state": "job_deleted",
        "deleted_time": "2026-04-30T10:00:00-07:00",
    }
    engine.job_def_manager = manager
    conn = _MockConnection(
        app_ctx=engine,
        props={
            ConnProps.ACTIVE_STUDY: "study-a",
            ConnProps.USER_NAME: "submitter",
            ConnProps.USER_ORG: "org",
            ConnProps.USER_ROLE: "role",
        },
    )

    JobCommandModule().list_jobs(conn, ["list_jobs", "--submit-token", "retry-1"])

    assert len(conn.errors) == 1
    assert "SUBMIT_TOKEN_JOB_DELETED" in conn.errors[0][0]
    assert conn.errors[0][1][MetaKey.STATUS] == SUBMIT_TOKEN_JOB_DELETED_STATUS
    assert conn.errors[0][1][MetaKey.JOB_ID] == "job-1"
    assert conn.tables == []


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


def test_list_jobs_ignores_duration_parse_failures(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    jobs = [
        _FakeListedJob(
            {
                JobMetaKey.JOB_ID.value: "job-1",
                JobMetaKey.JOB_NAME.value: "broken-duration",
                JobMetaKey.STATUS.value: RunStatus.RUNNING.value,
                JobMetaKey.START_TIME.value: "not-a-timestamp",
            }
        )
    ]
    conn = _MockConnection(app_ctx=_FakeListEngine(jobs), props={ConnProps.ACTIVE_STUDY: "default"})

    JobCommandModule().list_jobs(conn, ["list_jobs"])

    assert conn.errors == []
    assert len(conn.tables) == 1
    assert len(conn.tables[0].rows) == 1
    first_row, _row_meta = conn.tables[0].rows[0]
    assert first_row[0] == "job-1"


def test_list_jobs_shows_execution_exception_status(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    jobs = [
        _FakeListedJob(
            {
                JobMetaKey.JOB_ID.value: "job-1",
                JobMetaKey.JOB_NAME.value: "k8s-pending-timeout",
                JobMetaKey.STATUS.value: RunStatus.FINISHED_EXECUTION_EXCEPTION.value,
            }
        )
    ]
    conn = _MockConnection(app_ctx=_FakeListEngine(jobs), props={ConnProps.ACTIVE_STUDY: "default"})

    JobCommandModule().list_jobs(conn, ["list_jobs"])

    assert conn.errors == []
    assert len(conn.tables) == 1
    assert len(conn.tables[0].rows) == 1
    first_row, row_meta = conn.tables[0].rows[0]
    assert first_row[2] == RunStatus.FINISHED_EXECUTION_EXCEPTION.value
    assert row_meta[MetaKey.STATUS] == RunStatus.FINISHED_EXECUTION_EXCEPTION.value


def test_job_match_tolerates_missing_job_id_and_name():
    assert JobCommandModule._job_match({}, "job", "name", "", "default") is False


def test_get_job_meta_normalizes_legacy_job_study(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    jobs = [_FakeListedJob({JobMetaKey.JOB_ID.value: "legacy-job", JobMetaKey.JOB_NAME.value: "legacy"})]
    conn = _MockConnection(app_ctx=_FakeListEngine(jobs), props={JobCommandModule.JOB_ID: "legacy-job"})

    JobCommandModule().get_job_meta(conn, ["get_job_meta", "legacy-job"])

    assert conn.errors == []
    assert len(conn.dicts) == 1
    assert conn.dicts[0][0][JobMetaKey.STUDY.value] == "default"


def test_get_job_log_client_target_returns_persisted_log(tmp_path, monkeypatch):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", _FakeServerEngine)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    workspace = _FakeWorkspace(tmp_path)
    engine = _FakeServerEngine(workspace)
    engine.job_def_manager.get_client_data.return_value = b"client line1\nclient line2\n"
    conn = _MockConnection(app_ctx=engine, props={JobCommandModule.JOB_ID: "job-1"})

    JobCommandModule().get_job_log(conn, ["get_job_log", "job-1", "site-1"])

    payload, _meta = conn.dicts[0]
    assert payload == {"logs": {"site-1": "client line1\nclient line2\n"}}
    engine.job_def_manager.get_client_data.assert_called_once()


def test_get_job_log_client_target_reads_live_workspace_log(tmp_path, monkeypatch):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", _FakeServerEngine)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    workspace = _FakeWorkspace(tmp_path)
    engine = _FakeServerEngine(workspace)
    client_log = Path(workspace.get_log_root("job-1")) / "site-1" / WorkspaceConstants.LOG_FILE_NAME
    client_log.parent.mkdir(parents=True, exist_ok=True)
    client_log.write_text("live client log\n", encoding="utf-8")
    conn = _MockConnection(app_ctx=engine, props={JobCommandModule.JOB_ID: "job-1"})

    JobCommandModule().get_job_log(conn, ["get_job_log", "job-1", "site-1"])

    payload, _meta = conn.dicts[0]
    assert payload == {"logs": {"site-1": "live client log\n"}}
    engine.job_def_manager.get_storage_component.assert_not_called()
    engine.job_def_manager.get_client_data.assert_not_called()


def test_get_job_log_returns_selected_live_json_log_only(tmp_path, monkeypatch):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", _FakeServerEngine)
    workspace = _FakeWorkspace(tmp_path)
    engine = _FakeServerEngine(workspace)
    log_root = Path(workspace.get_log_root("job-1"))
    (log_root / WorkspaceConstants.LOG_FILE_NAME).write_text("text server log\n", encoding="utf-8")
    (log_root / "log.json").write_text('{"asctime": "2026-04-30 10:00:00", "message": "json server log"}\n')
    conn = _MockConnection(app_ctx=engine, props={JobCommandModule.JOB_ID: "job-1"})

    JobCommandModule().get_job_log(conn, ["get_job_log", "job-1", "server", "log.json"])

    payload, _meta = conn.dicts[0]
    assert payload == {"logs": {"server": '{"asctime": "2026-04-30 10:00:00", "message": "json server log"}\n'}}


def test_get_job_log_client_target_reads_selected_json_log(tmp_path, monkeypatch):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", _FakeServerEngine)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    workspace = _FakeWorkspace(tmp_path)
    engine = _FakeServerEngine(workspace)
    engine.job_def_manager.get_client_data.return_value = None
    client_json = Path(workspace.get_log_root("job-1")) / "site-1" / "log.json"
    client_json.parent.mkdir(parents=True, exist_ok=True)
    client_json.write_text('{"asctime": "2026-04-30 10:00:00", "message": "client json"}\n', encoding="utf-8")
    conn = _MockConnection(app_ctx=engine, props={JobCommandModule.JOB_ID: "job-1"})

    JobCommandModule().get_job_log(conn, ["get_job_log", "job-1", "site-1", "log.json"])

    payload, _meta = conn.dicts[0]
    assert payload["logs"] == {"site-1": '{"asctime": "2026-04-30 10:00:00", "message": "client json"}\n'}
    assert "unavailable" not in payload
    engine.job_def_manager.get_storage_component.assert_not_called()
    engine.job_def_manager.get_client_data.assert_not_called()


def test_get_job_log_client_target_reads_stored_workspace_log(tmp_path, monkeypatch):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", _FakeServerEngine)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    workspace = _FakeWorkspace(tmp_path)
    engine = _FakeServerEngine(workspace)
    engine.job_def_manager.get_storage_component.return_value = _zip_bytes({"site-1/log.txt": "stored client log\n"})
    conn = _MockConnection(app_ctx=engine, props={JobCommandModule.JOB_ID: "job-1"})

    JobCommandModule().get_job_log(conn, ["get_job_log", "job-1", "site-1"])

    payload, _meta = conn.dicts[0]
    assert payload == {"logs": {"site-1": "stored client log\n"}}
    engine.job_def_manager.get_client_data.assert_not_called()


def test_get_job_log_client_target_reads_stored_workspace_json_log(tmp_path, monkeypatch):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", _FakeServerEngine)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    workspace = _FakeWorkspace(tmp_path)
    engine = _FakeServerEngine(workspace)
    engine.job_def_manager.get_client_data.return_value = None
    engine.job_def_manager.get_storage_component.return_value = _zip_bytes(
        {"site-1/log.json": '{"asctime": "2026-04-30 10:00:00", "message": "stored client json"}\n'}
    )
    conn = _MockConnection(app_ctx=engine, props={JobCommandModule.JOB_ID: "job-1"})

    JobCommandModule().get_job_log(conn, ["get_job_log", "job-1", "site-1", "log.json"])

    payload, _meta = conn.dicts[0]
    assert payload["logs"] == {"site-1": '{"asctime": "2026-04-30 10:00:00", "message": "stored client json"}\n'}
    engine.job_def_manager.get_client_data.assert_not_called()


def test_get_job_log_truncates_large_output(tmp_path, monkeypatch):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", _FakeServerEngine)
    workspace = _FakeWorkspace(tmp_path)
    engine = _FakeServerEngine(workspace)
    conn = _MockConnection(app_ctx=engine, props={JobCommandModule.JOB_ID: "job-1"})
    log_file = Path(workspace.get_log_root("job-1")) / WorkspaceConstants.LOG_FILE_NAME
    log_file.write_text("a" * 12 + "\n" + "b" * 12 + "\n", encoding="utf-8")

    monkeypatch.setattr(JobCommandModule, "MAX_RETURNED_JOB_LOG_BYTES", 16)

    JobCommandModule().get_job_log(conn, ["get_job_log", "job-1"])

    payload, _meta = conn.dicts[0]
    assert "truncated after 16 bytes" in payload["logs"]["server"]


def test_get_job_log_reads_server_log_from_stored_workspace_when_live_log_is_gone(tmp_path, monkeypatch):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", _FakeServerEngine)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    workspace = _FakeWorkspace(tmp_path)
    engine = _FakeServerEngine(workspace)
    engine.job_def_manager.get_storage_component.return_value = _zip_bytes(
        {WorkspaceConstants.LOG_FILE_NAME: "stored server log\n"}
    )
    conn = _MockConnection(app_ctx=engine, props={JobCommandModule.JOB_ID: "job-1"})

    JobCommandModule().get_job_log(conn, ["get_job_log", "job-1", "server"])

    payload, _meta = conn.dicts[0]
    assert payload["logs"]["server"] == "stored server log\n"
    engine.job_def_manager.get_storage_component.assert_called_once()


def test_get_job_log_reads_nested_server_log_from_stored_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", _FakeServerEngine)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    workspace = _FakeWorkspace(tmp_path)
    engine = _FakeServerEngine(workspace)
    engine.job_def_manager.get_storage_component.return_value = _zip_bytes(
        {"server/log.txt": "stored server log\n", "site-1/log.txt": "stored client log\n"}
    )
    conn = _MockConnection(app_ctx=engine, props={JobCommandModule.JOB_ID: "job-1"})

    JobCommandModule().get_job_log(conn, ["get_job_log", "job-1", "server"])

    payload, _meta = conn.dicts[0]
    assert payload["logs"]["server"] == "stored server log\n"


def test_find_server_log_member_does_not_use_client_log():
    assert JobCommandModule._find_server_log_member(["site-1/log.txt"]) is None
    assert JobCommandModule._find_server_log_member(["workspace/server/log.txt"]) == "workspace/server/log.txt"


def test_get_job_log_client_target_truncates_large_log(tmp_path, monkeypatch):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", _FakeServerEngine)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    workspace = _FakeWorkspace(tmp_path)
    engine = _FakeServerEngine(workspace)
    engine.job_def_manager.get_client_data.return_value = b"a" * 20
    conn = _MockConnection(app_ctx=engine, props={JobCommandModule.JOB_ID: "job-1"})

    monkeypatch.setattr(JobCommandModule, "MAX_RETURNED_JOB_LOG_BYTES", 16)

    JobCommandModule().get_job_log(conn, ["get_job_log", "job-1", "site-1"])

    payload, _meta = conn.dicts[0]
    assert payload["logs"]["site-1"].startswith("a" * 16)
    assert "truncated after 16 bytes" in payload["logs"]["site-1"]


def test_get_job_log_client_target_returns_structured_error_for_invalid_job_def_manager(tmp_path, monkeypatch):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", _FakeServerEngine)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", type("ExpectedJobDefManager", (), {}))
    workspace = _FakeWorkspace(tmp_path)
    engine = _FakeServerEngine(workspace)
    conn = _MockConnection(app_ctx=engine, props={JobCommandModule.JOB_ID: "job-1"})

    JobCommandModule().get_job_log(conn, ["get_job_log", "job-1", "site-1"])

    assert conn.dicts == []
    assert len(conn.errors) == 1
    message, meta = conn.errors[0]
    assert "job_def_manager in engine is not of type JobDefManagerSpec" in message
    assert meta[MetaKey.STATUS] == MetaStatusValue.INTERNAL_ERROR


def test_get_job_log_all_returns_structured_error_for_invalid_job_def_manager(tmp_path, monkeypatch):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", _FakeServerEngine)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", type("ExpectedJobDefManager", (), {}))
    workspace = _FakeWorkspace(tmp_path)
    engine = _FakeServerEngine(workspace)
    conn = _MockConnection(app_ctx=engine, props={JobCommandModule.JOB_ID: "job-1"})

    JobCommandModule().get_job_log(conn, ["get_job_log", "job-1", "all"])

    assert conn.dicts == []
    assert len(conn.errors) == 1
    message, meta = conn.errors[0]
    assert "job_def_manager in engine is not of type JobDefManagerSpec" in message
    assert meta[MetaKey.STATUS] == MetaStatusValue.INTERNAL_ERROR


def test_get_job_log_all_returns_available_client_logs_and_unavailable_sites(tmp_path, monkeypatch):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", _FakeServerEngine)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    workspace = _FakeWorkspace(tmp_path)
    engine = _FakeServerEngine(workspace)
    log_file = Path(workspace.get_log_root("job-1")) / WorkspaceConstants.LOG_FILE_NAME
    log_file.write_text("server line\n", encoding="utf-8")
    engine.job_def_manager.list_components.return_value = ["LOG_log.txt_site-1", "meta", "workspace"]
    engine.job_def_manager.get_client_data.side_effect = lambda jid, client_name, data_type, fl_ctx: (
        b"client line\n" if client_name == "site-1" else None
    )
    job = _FakeListedJob({JobMetaKey.DEPLOY_MAP.value: {"app": ["server", "site-1", "site-2"]}})
    conn = _MockConnection(app_ctx=engine, props={JobCommandModule.JOB_ID: "job-1", JobCommandModule.JOB: job})

    JobCommandModule().get_job_log(conn, ["get_job_log", "job-1", "all"])

    payload, _meta = conn.dicts[0]
    assert payload["logs"] == {"server": "server line\n", "site-1": "client line\n"}
    assert payload["unavailable"] == {"site-2": "client log stream not available for this job"}


def test_get_job_log_all_returns_workspace_client_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", _FakeServerEngine)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    workspace = _FakeWorkspace(tmp_path)
    engine = _FakeServerEngine(workspace)
    engine.job_def_manager.list_components.return_value = []
    engine.job_def_manager.get_storage_component.return_value = _zip_bytes(
        {
            "log.txt": "server line\n",
            "site-1/log.txt": "site-1 line\n",
            "site-2/log.txt": "site-2 line\n",
        }
    )
    job = _FakeListedJob({JobMetaKey.DEPLOY_MAP.value: {"app": ["server", "site-1", "site-2"]}})
    conn = _MockConnection(app_ctx=engine, props={JobCommandModule.JOB_ID: "job-1", JobCommandModule.JOB: job})

    JobCommandModule().get_job_log(conn, ["get_job_log", "job-1", "all"])

    payload, _meta = conn.dicts[0]
    assert payload == {
        "logs": {
            "server": "server line\n",
            "site-1": "site-1 line\n",
            "site-2": "site-2 line\n",
        }
    }
    engine.job_def_manager.get_storage_component.assert_called_once()


def test_get_job_log_all_reads_workspace_zip_once_for_client_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", _FakeServerEngine)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    workspace = _FakeWorkspace(tmp_path)
    engine = _FakeServerEngine(workspace)
    server_log = Path(workspace.get_log_root("job-1")) / WorkspaceConstants.LOG_FILE_NAME
    server_log.write_text("server line\n", encoding="utf-8")
    engine.job_def_manager.list_components.return_value = []
    engine.job_def_manager.get_storage_component.return_value = _zip_bytes(
        {
            "site-1/log.txt": "site-1 line\n",
            "site-2/log.txt": "site-2 line\n",
            "site-3/log.txt": "site-3 line\n",
        }
    )
    job = _FakeListedJob({JobMetaKey.DEPLOY_MAP.value: {"app": ["server", "site-1", "site-2", "site-3"]}})
    conn = _MockConnection(app_ctx=engine, props={JobCommandModule.JOB_ID: "job-1", JobCommandModule.JOB: job})

    JobCommandModule().get_job_log(conn, ["get_job_log", "job-1", "all"])

    payload, _meta = conn.dicts[0]
    assert payload == {
        "logs": {
            "server": "server line\n",
            "site-1": "site-1 line\n",
            "site-2": "site-2 line\n",
            "site-3": "site-3 line\n",
        }
    }
    engine.job_def_manager.get_storage_component.assert_called_once()
    engine.job_def_manager.get_client_data.assert_not_called()


def test_get_job_log_all_does_not_read_workspace_zip_per_client(tmp_path, monkeypatch):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", _FakeServerEngine)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    workspace = _FakeWorkspace(tmp_path)
    engine = _FakeServerEngine(workspace)
    engine.job_def_manager.list_components.return_value = []
    engine.job_def_manager.get_storage_component.return_value = _zip_bytes(
        {
            "site-1/log.txt": "site-1 line\n",
            "site-2/log.txt": "site-2 line\n",
            "site-3/log.txt": "site-3 line\n",
        }
    )
    job = _FakeListedJob({JobMetaKey.DEPLOY_MAP.value: {"app": ["server", "site-1", "site-2", "site-3"]}})
    conn = _MockConnection(app_ctx=engine, props={JobCommandModule.JOB_ID: "job-1", JobCommandModule.JOB: job})

    with patch.object(
        JobCommandModule,
        "_read_stored_client_job_log",
        side_effect=AssertionError("all-sites log retrieval should not read the workspace ZIP per client"),
    ):
        JobCommandModule().get_job_log(conn, ["get_job_log", "job-1", "all"])

    payload, _meta = conn.dicts[0]
    assert payload == {
        "logs": {
            "site-1": "site-1 line\n",
            "site-2": "site-2 line\n",
            "site-3": "site-3 line\n",
        },
        "unavailable": {"server": "server log not available for this job"},
    }
    engine.job_def_manager.get_storage_component.assert_called_once()


def test_get_job_log_all_marks_missing_server_log_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", _FakeServerEngine)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    workspace = _FakeWorkspace(tmp_path)
    engine = _FakeServerEngine(workspace)
    engine.job_def_manager.list_components.return_value = []
    engine.job_def_manager.get_storage_component.return_value = _zip_bytes({"site-1/log.txt": "site-1 line\n"})
    job = _FakeListedJob({JobMetaKey.DEPLOY_MAP.value: {"app": ["server", "site-1"]}})
    conn = _MockConnection(app_ctx=engine, props={JobCommandModule.JOB_ID: "job-1", JobCommandModule.JOB: job})

    JobCommandModule().get_job_log(conn, ["get_job_log", "job-1", "all"])

    payload, _meta = conn.dicts[0]
    assert payload == {
        "logs": {"site-1": "site-1 line\n"},
        "unavailable": {"server": "server log not available for this job"},
    }


def test_get_job_log_all_includes_empty_server_log_file(tmp_path, monkeypatch):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", _FakeServerEngine)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    workspace = _FakeWorkspace(tmp_path)
    engine = _FakeServerEngine(workspace)
    server_log = Path(workspace.get_log_root("job-1")) / WorkspaceConstants.LOG_FILE_NAME
    server_log.write_text("", encoding="utf-8")
    engine.job_def_manager.list_components.return_value = []
    engine.job_def_manager.get_storage_component.return_value = _zip_bytes({"site-1/log.txt": "site-1 line\n"})
    job = _FakeListedJob({JobMetaKey.DEPLOY_MAP.value: {"app": ["server", "site-1"]}})
    conn = _MockConnection(app_ctx=engine, props={JobCommandModule.JOB_ID: "job-1", JobCommandModule.JOB: job})

    JobCommandModule().get_job_log(conn, ["get_job_log", "job-1", "all"])

    payload, _meta = conn.dicts[0]
    assert payload == {"logs": {"server": "", "site-1": "site-1 line\n"}}


def test_configure_job_log_all_targets_server_and_clients(tmp_path, monkeypatch):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", _FakeServerEngine)
    workspace = _FakeWorkspace(tmp_path)
    engine = _FakeServerEngine(workspace)
    engine.job_def_manager.get_job.return_value = _FakeListedJob({JobMetaKey.STATUS.value: RunStatus.RUNNING.value})
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


def test_configure_job_log_specific_client_target_is_honored(tmp_path, monkeypatch):
    monkeypatch.setattr(cmd_utils_module, "ServerEngineSpec", object)
    monkeypatch.setattr(job_cmds_module, "ServerEngine", _FakeServerEngine)
    job_id = "123e4567-e89b-42d3-a456-426614174000"
    workspace = _FakeWorkspace(tmp_path)
    engine = _FakeServerEngine(workspace)
    engine._clients = [_FakeClient("site-a", "token-a"), _FakeClient("site-b", "token-b")]
    engine.job_def_manager.get_job.return_value = _FakeListedJob(
        {
            JobMetaKey.STATUS.value: RunStatus.RUNNING,
            JobMetaKey.STUDY.value: "default",
        }
    )
    conn = _MockConnection(app_ctx=engine, props={ConnProps.ACTIVE_STUDY: "default"})
    module = JobCommandModule()

    rc = module.authorize_configure_job_log(conn, ["configure_job_log", job_id, "client", "site-a", "DEBUG"])

    assert rc == PreAuthzReturnCode.REQUIRE_AUTHZ
    assert conn.get_prop(JobCommandModule.TARGET_CLIENT_TOKENS) == ["token-a"]
    assert conn.get_prop(JobCommandModule.TARGET_CLIENT_NAMES) == ["site-a"]

    def _send_request_to_clients(conn, message):
        assert conn.get_prop(JobCommandModule.TARGET_CLIENT_TOKENS) == ["token-a"]
        assert conn.get_prop(JobCommandModule.TARGET_CLIENT_NAMES) == ["site-a"]
        return [object()]

    processed = []
    monkeypatch.setattr(module, "send_request_to_clients", _send_request_to_clients)
    monkeypatch.setattr(module, "process_replies_to_table", lambda conn, replies: processed.append(replies))

    module.configure_job_log(conn, ["configure_job_log", job_id, "client", "site-a", "DEBUG"])

    engine.configure_job_log.assert_not_called()
    assert len(processed) == 1


def test_authorize_job_id_hides_jobs_from_other_studies(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    monkeypatch.setattr(job_cmds_module, "StudyRegistryService", _FakeStudyRegistryService, raising=False)
    monkeypatch.setattr(
        _FakeStudyRegistryService,
        "registry",
        _FakeStudyRegistry(sites={"cancer-research": {"site1"}}),
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
            JobMetaKey.SUBMITTER_ROLE.value: "cert_admin",
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
            JobMetaKey.SUBMITTER_ROLE.value: "cert_admin",
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


def test_authorize_job_id_keeps_cert_role_before_authz(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    monkeypatch.setattr(job_cmds_module, "StudyRegistryService", _FakeStudyRegistryService, raising=False)
    monkeypatch.setattr(cmd_utils_module, "StudyRegistryService", _FakeStudyRegistryService, raising=False)
    monkeypatch.setattr(
        _FakeStudyRegistryService,
        "registry",
        _FakeStudyRegistry(sites={"cancer-research": {"site1"}}),
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
            JobMetaKey.SUBMITTER_ROLE.value: "project_admin",
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
    assert conn.get_prop(ConnProps.USER_ROLE) == "project_admin"


def test_abort_job_marks_submitted_job_finished_aborted(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", object)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)

    engine = _FakeEngine()
    engine.job_def_manager = MagicMock()
    engine.job_runner = MagicMock()
    job = _FakeListedJob({JobMetaKey.STATUS.value: RunStatus.SUBMITTED.value})
    job.job_id = "job-123"
    engine.job_def_manager.get_job.return_value = job
    conn = _MockConnection(app_ctx=engine, props={JobCommandModule.JOB_ID: "job-123"})

    JobCommandModule().abort_job(conn, ["abort_job", "job-123"])

    engine.job_def_manager.set_status.assert_called_once()
    engine.job_runner.stop_run.assert_not_called()
    assert any("Aborted the job job-123 before running it." in msg for msg, _meta in conn.strings)


def test_abort_job_handles_missing_status_without_attribute_error(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", object)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)

    engine = _FakeEngine()
    engine.job_def_manager = MagicMock()
    job = _FakeListedJob({})
    job.job_id = "job-123"
    engine.job_def_manager.get_job.return_value = job
    engine.job_runner = MagicMock()
    conn = _MockConnection(app_ctx=engine, props={JobCommandModule.JOB_ID: "job-123"})

    JobCommandModule().abort_job(conn, ["abort_job", "job-123"])

    engine.job_runner.stop_run.assert_called_once()
    assert engine.job_runner.stop_run.call_args[0][0] == "job-123"


def test_submit_job_persists_cert_submitter_role(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    monkeypatch.setattr(job_cmds_module, "StudyRegistryService", _FakeStudyRegistryService, raising=False)
    monkeypatch.setattr(cmd_utils_module, "StudyRegistryService", _FakeStudyRegistryService, raising=False)
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
    assert engine.job_def_manager.created_meta[JobMetaKey.SUBMITTER_ROLE.value] == "cert_admin"


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


def test_download_job_rejects_unfinished_job_before_packaging(monkeypatch, tmp_path):
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    engine = _FakeListEngine(
        [_FakeListedJob({JobMetaKey.JOB_ID.value: "job-1", JobMetaKey.STATUS.value: RunStatus.RUNNING.value})]
    )
    engine.job_def_manager.get_storage_for_download = MagicMock()
    conn = _MockConnection(app_ctx=engine, props={ConnProps.DOWNLOAD_DIR: str(tmp_path)})

    JobCommandModule().download_job(conn, ["download_job", "job-1"])

    assert conn.errors
    assert conn.errors[0][1][MetaKey.STATUS] == MetaStatusValue.JOB_RUNNING
    engine.job_def_manager.get_storage_for_download.assert_not_called()


def test_download_job_packages_all_default_components_for_finished_job(monkeypatch, tmp_path):
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    engine = _FakeListEngine(
        [_FakeListedJob({JobMetaKey.JOB_ID.value: "job-1", JobMetaKey.STATUS.value: RunStatus.FINISHED_ABORTED.value})]
    )
    engine.job_def_manager.get_storage_for_download = MagicMock()
    conn = _MockConnection(app_ctx=engine, props={ConnProps.DOWNLOAD_DIR: str(tmp_path)})
    module = JobCommandModule()
    module.download_folder = MagicMock()

    module.download_job(conn, ["download_job", "job-1"])

    assert conn.errors == []
    assert engine.job_def_manager.get_storage_for_download.call_count == 3
    module.download_folder.assert_called_once()


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


def test_get_job_log_specific_missing_client_returns_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", object)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)

    workspace = _FakeWorkspace(tmp_path)
    engine = _FakeServerEngine(workspace)
    engine.job_def_manager.get_client_data.return_value = None

    conn = _MockConnection(
        app_ctx=engine,
        props={JobCommandModule.JOB_ID: "job-123"},
    )

    JobCommandModule().get_job_log(conn, ["get_job_log", "job-123", "site-2"])

    assert conn.errors == []
    assert conn.dicts[0][0] == {
        "logs": {},
        "unavailable": {"site-2": "client log stream not available for this job"},
    }


def test_get_job_log_missing_server_file_marks_server_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", object)

    workspace = _FakeWorkspace(tmp_path)
    conn = _MockConnection(
        app_ctx=_FakeServerEngine(workspace),
        props={JobCommandModule.JOB_ID: "job-123"},
    )

    JobCommandModule().get_job_log(conn, ["get_job_log", "job-123"])

    assert conn.errors == []
    assert conn.dicts[0][0] == {
        "logs": {},
        "unavailable": {"server": "server log not available for this job"},
    }
    assert conn.successes == []


def test_get_job_log_handles_file_deleted_after_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", object)

    workspace = _FakeWorkspace(tmp_path)
    conn = _MockConnection(
        app_ctx=_FakeServerEngine(workspace),
        props={JobCommandModule.JOB_ID: "job-123"},
    )

    monkeypatch.setattr(job_cmds_module.os.path, "exists", lambda _path: True)

    def _raise_deleted(*_args, **_kwargs):
        raise FileNotFoundError("deleted during read")

    monkeypatch.setattr(JobCommandModule, "_collect_job_log_lines", _raise_deleted)

    JobCommandModule().get_job_log(conn, ["get_job_log", "job-123"])

    assert conn.errors == []
    assert conn.dicts[0][0] == {
        "logs": {},
        "unavailable": {"server": "server log not available for this job"},
    }


def test_get_job_log_mismatched_job_id_sources_returns_error(monkeypatch, tmp_path):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", object)

    workspace = _FakeWorkspace(tmp_path)
    conn = _MockConnection(
        app_ctx=_FakeServerEngine(workspace),
        props={JobCommandModule.JOB_ID: "job-from-prop"},
    )

    JobCommandModule().get_job_log(conn, ["get_job_log", "job-from-arg"])

    assert conn.dicts == []
    assert conn.errors
    assert "job_id mismatch between connection property and parsed argument" in conn.errors[0][0]


def test_get_job_log_accepts_same_job_id_with_different_case(monkeypatch, tmp_path):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", object)

    workspace = _FakeWorkspace(tmp_path)
    log_file = Path(workspace.get_log_root("job-abc123")) / WorkspaceConstants.LOG_FILE_NAME
    log_file.write_text("line1\n", encoding="utf-8")
    conn = _MockConnection(
        app_ctx=_FakeServerEngine(workspace),
        props={JobCommandModule.JOB_ID: "job-abc123"},
    )

    JobCommandModule().get_job_log(conn, ["get_job_log", "JOB-ABC123"])

    assert conn.errors == []
    assert conn.dicts[0][0] == {"logs": {"server": "line1\n"}}


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


def test_get_job_log_rejects_removed_tail_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(job_cmds_module, "ServerEngine", object)
    job_id = "job-123"
    log_root = tmp_path / job_id
    log_root.mkdir(parents=True)
    log_path = log_root / WorkspaceConstants.LOG_FILE_NAME
    log_path.write_text("line1\nline2\nline3\n")

    class _FakeWorkspace:
        def get_log_root(self, _job_id):
            assert _job_id == job_id
            return str(log_root)

    class _FakeEngine:
        job_def_manager = None

        def get_workspace(self):
            return _FakeWorkspace()

    conn = _MockConnection(app_ctx=_FakeEngine(), props={JobCommandModule.JOB_ID: job_id})

    JobCommandModule().get_job_log(conn, ["get_job_log", job_id, "-n", "0"])

    assert len(conn.errors) == 1
    msg, meta = conn.errors[0]
    assert "unrecognized arguments" in msg
    assert meta[MetaKey.STATUS] == "syntax_error"


def test_do_app_command_success_sets_ok_meta(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "ServerEngineInternalSpec", object)
    engine = _FakeEngine()
    engine.run_processes = {"job-123": object()}
    result = Shareable()
    result.set_return_code(ReturnCode.OK)
    result[ServerCommandKey.DATA] = {"answer": 42}
    engine.send_app_command = MagicMock(return_value=result)
    conn = _MockConnection(
        app_ctx=engine,
        props={
            JobCommandModule.JOB_ID: "job-123",
            ConnProps.CMD_PROPS: {"k": "v"},
        },
    )

    JobCommandModule().do_app_command(conn, ["app_command", "job-123", "topic"])

    assert conn.errors == []
    assert conn.dicts[0][0] == {"answer": 42}
    assert conn.dicts[0][1][MetaKey.STATUS] == "ok"


def test_do_app_command_preserves_zero_timeout(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "ServerEngineInternalSpec", object)
    engine = _FakeEngine()
    engine.run_processes = {"job-123": object()}
    result = Shareable()
    result.set_return_code(ReturnCode.OK)
    result[ServerCommandKey.DATA] = {"answer": 42}
    engine.send_app_command = MagicMock(return_value=result)
    conn = _MockConnection(
        app_ctx=engine,
        props={
            JobCommandModule.JOB_ID: "job-123",
            ConnProps.CMD_PROPS: {"k": "v"},
            ConnProps.CMD_TIMEOUT: 0,
        },
    )

    JobCommandModule().do_app_command(conn, ["app_command", "job-123", "topic"])

    engine.send_app_command.assert_called_once_with("job-123", "topic", {"k": "v"}, 0)


def test_do_app_command_usage_error_uses_append_error(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "ServerEngineInternalSpec", object)
    conn = _MockConnection(
        app_ctx=_FakeEngine(),
        props={ConnProps.CMD_ENTRY: type("_CmdEntry", (), {"usage": "app_command job_id topic"})()},
    )

    JobCommandModule().do_app_command(conn, ["app_command", "job-123"])

    assert conn.errors
    assert conn.strings == []
    assert conn.errors[0][0] == "Usage: app_command job_id topic"
    assert conn.errors[0][1][MetaKey.STATUS] == "syntax_error"


def test_submit_token_locks_are_weakly_released():
    submitter = {"name": "admin@nvidia.com", "org": "nvidia", "role": "lead"}
    token = "weak-lock-token"
    key = ("study", submitter["name"], submitter["org"], submitter["role"], token)

    lock = JobCommandModule._submit_token_lock("study", submitter, token)

    assert JobCommandModule._submit_token_locks.get(key) is lock
    del lock
    gc.collect()
    assert key not in JobCommandModule._submit_token_locks
