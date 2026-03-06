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

    def get_prop(self, key, default=None):
        return self._props.get(key, default)

    def append_error(self, msg, meta=None):
        self.errors.append((msg, meta))

    def append_string(self, msg, meta=None):
        self.strings.append((msg, meta))

    def append_success(self, msg, meta=None):
        self.successes.append((msg, meta))


class TestProjectCmdProps:
    @pytest.mark.parametrize(
        "cmd_props, expected_meta",
        [
            (None, {}),
            ("not-a-dict", {}),
            ({}, {}),
            ({"project": ""}, {}),
            ({"project": "cancer-research"}, {"project": "cancer-research"}),
            ({"project": "default"}, {"project": "default"}),
        ],
    )
    def test_add_project_to_meta(self, cmd_props, expected_meta):
        conn = _MockConnection(cmd_props=cmd_props)
        meta = {}

        assert JobCommandModule._add_project_to_meta(meta, conn) is True
        assert meta == expected_meta
        assert conn.errors == []

    @pytest.mark.parametrize("project", [123, "Bad Project", " cancer-research ", "../escape"])
    def test_add_project_to_meta_rejects_invalid_values(self, project):
        conn = _MockConnection(cmd_props={"project": project})
        meta = {}

        assert JobCommandModule._add_project_to_meta(meta, conn) is False
        assert meta == {}
        assert len(conn.errors) == 1


class _FakeJobMetaValidator:
    def validate(self, folder_name, zip_file_name):
        assert folder_name == "job_folder"
        assert zip_file_name == "job.zip"
        return True, "", {}


class _FakeJobDefManager:
    def __init__(self):
        self.created_meta = None

    def create(self, meta, uploaded_content, fl_ctx):
        self.created_meta = dict(meta)
        result = dict(meta)
        result[JobMetaKey.JOB_ID.value] = "new-job-id"
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


def test_submit_job_exposes_project_in_submit_event(monkeypatch):
    monkeypatch.setattr(job_cmds_module, "JobMetaValidator", _FakeJobMetaValidator)
    monkeypatch.setattr(job_cmds_module, "JobDefManagerSpec", object)

    engine = _FakeEngine()
    conn = _MockConnection(
        app_ctx=engine,
        props={
            ConnProps.FILE_LOCATION: "job.zip",
            ConnProps.CMD_PROPS: {"project": "cancer-research"},
            ConnProps.USER_NAME: "submitter",
            ConnProps.USER_ORG: "org",
            ConnProps.USER_ROLE: "role",
        },
    )

    JobCommandModule().submit_job(conn, ["submit_job", "job_folder"])

    assert conn.errors == []
    assert len(conn.successes) == 1
    assert engine.submit_event_meta == {JobMetaKey.PROJECT.value: "cancer-research"}
    assert engine.job_def_manager.created_meta[JobMetaKey.PROJECT.value] == "cancer-research"
