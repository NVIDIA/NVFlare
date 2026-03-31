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

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.job_def import JobMetaKey
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.hci.proto import MetaKey, MetaStatusValue
from nvflare.private.fed.server import job_cmds as job_cmds_module
from nvflare.private.fed.server.job_cmds import JobCommandModule, _create_list_job_cmd_parser

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


class TestStudyCmdProps:
    @pytest.mark.parametrize(
        "cmd_props, expected_study",
        [
            (None, None),
            ("not-a-dict", None),
            ({}, None),
            ({"study": ""}, ""),
            ({"study": "cancer-research"}, "cancer-research"),
            ({"study": "default"}, "default"),
        ],
    )
    def test_get_requested_study(self, cmd_props, expected_study):
        conn = _MockConnection(cmd_props=cmd_props)

        assert JobCommandModule._get_requested_study(conn) == expected_study
        assert conn.errors == []


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


def test_submit_job_exposes_study_in_submit_event(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobMetaValidator", _FakeJobMetaValidator)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)

    engine = _FakeEngine()
    conn = _MockConnection(
        app_ctx=engine,
        props={
            ConnProps.FILE_LOCATION: "job.zip",
            ConnProps.CMD_PROPS: {"study": "cancer-research"},
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
            ConnProps.CMD_PROPS: {"study": "multiple-sclerosis"},
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
    conn = _MockConnection(app_ctx=_FakeListEngine(jobs), cmd_props={"study": "default"})

    JobCommandModule().list_jobs(conn, ["list_jobs"])

    assert conn.errors == []
    assert len(conn.tables) == 1
    assert len(conn.tables[0].rows) == 1
    assert conn.tables[0].rows[0][1][JobMetaKey.STUDY.value] == "default"
    assert conn.tables[0].rows[0][1][MetaKey.JOB_ID] == "legacy-job"


def test_list_jobs_rejects_invalid_study(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    conn = _MockConnection(app_ctx=_FakeListEngine([]), cmd_props={"study": "Bad Study"})

    JobCommandModule().list_jobs(conn, ["list_jobs"])

    assert conn.errors == []
    assert len(conn.strings) == 1
    assert conn.strings[0][0] == "No jobs found."


def test_submit_job_rejects_invalid_study_when_persisting(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobMetaValidator", _FakeJobMetaValidator)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)

    engine = _FakeEngine()
    conn = _MockConnection(
        app_ctx=engine,
        props={
            ConnProps.FILE_LOCATION: "job.zip",
            ConnProps.CMD_PROPS: {"study": "Bad Study"},
            ConnProps.USER_NAME: "submitter",
            ConnProps.USER_ORG: "org",
            ConnProps.USER_ROLE: "role",
        },
    )

    JobCommandModule().submit_job(conn, ["submit_job", "job_folder"])

    assert len(conn.errors) == 1
    assert conn.errors[0][1][MetaKey.STATUS] == MetaStatusValue.INVALID_JOB_DEFINITION
    assert conn.successes == []


def test_get_job_meta_normalizes_legacy_job_study(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)
    jobs = [_FakeListedJob({JobMetaKey.JOB_ID.value: "legacy-job", JobMetaKey.JOB_NAME.value: "legacy"})]
    conn = _MockConnection(app_ctx=_FakeListEngine(jobs), props={JobCommandModule.JOB_ID: "legacy-job"})

    JobCommandModule().get_job_meta(conn, ["get_job_meta", "legacy-job"])

    assert conn.errors == []
    assert len(conn.dicts) == 1
    assert conn.dicts[0][0][JobMetaKey.STUDY.value] == "default"
