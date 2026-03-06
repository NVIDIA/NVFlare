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

from nvflare.fuel.hci.server.constants import ConnProps
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
    def __init__(self, cmd_props=None):
        self._cmd_props = cmd_props

    def get_prop(self, key):
        if key == ConnProps.CMD_PROPS:
            return self._cmd_props
        return None


class TestProjectCmdProps:
    @pytest.mark.parametrize(
        "cmd_props, expected",
        [
            (None, ""),
            ("not-a-dict", ""),
            ({}, ""),
            ({"project": ""}, ""),
            ({"project": "cancer-research"}, "cancer-research"),
            ({"project": "default"}, "default"),
        ],
    )
    def test_get_project_from_cmd_props(self, cmd_props, expected):
        conn = _MockConnection(cmd_props=cmd_props)
        assert JobCommandModule._get_project_from_cmd_props(conn) == expected

    @pytest.mark.parametrize("project", [123, "Bad Project", " cancer-research ", "../escape"])
    def test_get_project_from_cmd_props_rejects_invalid_values(self, project):
        conn = _MockConnection(cmd_props={"project": project})
        with pytest.raises((TypeError, ValueError)):
            JobCommandModule._get_project_from_cmd_props(conn)
