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

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nvflare.apis.job_def import DEFAULT_JOB_STUDY
from nvflare.fuel.hci.client.api_spec import AdminConfigKey
from nvflare.fuel.hci.client.api import CommandInfo
from nvflare.fuel.hci.client.api_status import APIStatus
from nvflare.fuel.hci.client.cli import AdminClient
from nvflare.fuel.hci.tools import admin


def _make_admin_client_for_study(study):
    client = AdminClient.__new__(AdminClient)
    captured = {}

    class _FakeAPI:
        shutdown_received = False
        shutdown_msg = ""

        @staticmethod
        def check_command(line):
            captured["checked_line"] = line
            return CommandInfo.OK

        @staticmethod
        def do_command(line, props=None):
            captured["line"] = line
            captured["props"] = props
            return {"status": APIStatus.SUCCESS, "details": ""}

    client.api = _FakeAPI()
    client._study = study
    client.user_name = ""
    client.debug = False
    client.out_file = None
    client.no_stdout = False
    client.write_string = lambda *_args, **_kwargs: None
    client.write_stdout = lambda *_args, **_kwargs: None
    client.write_error = lambda *_args, **_kwargs: None
    client.print_resp = lambda *_args, **_kwargs: None
    client._set_output_file = lambda *_args, **_kwargs: None
    return client, captured


def test_submit_job_sends_session_study_cmd_props():
    client, captured = _make_admin_client_for_study("cancer-research")

    client._do_default("submit_job hello")

    assert captured["props"] == {"study": "cancer-research"}


def test_list_jobs_sends_default_session_study_cmd_props():
    client, captured = _make_admin_client_for_study(DEFAULT_JOB_STUDY)

    client._do_default("list_jobs")

    assert captured["props"] == {"study": DEFAULT_JOB_STUDY}


def test_list_jobs_merges_existing_cmd_props():
    client, captured = _make_admin_client_for_study("cancer-research")

    with patch(
        "nvflare.fuel.hci.client.cli.parse_command_line",
        return_value=("list_jobs", ["list_jobs"], {"foo": "bar"}),
    ):
        client._do_default("list_jobs")

    assert captured["props"] == {"foo": "bar", "study": "cancer-research"}


def test_clone_job_does_not_send_session_study_cmd_props():
    client, captured = _make_admin_client_for_study("multiple-sclerosis")

    client._do_default("clone_job job-1")

    assert captured["props"] is None


def test_admin_main_passes_launch_study_to_admin_client():
    captured = {}
    fake_conf = SimpleNamespace(
        get_admin_config=lambda: {AdminConfigKey.USERNAME: "admin@nvidia.com"},
        handlers=[],
    )

    class _FakeAdminClient:
        def __init__(self, **kwargs):
            captured["study"] = kwargs["study"]

        @staticmethod
        def run():
            captured["ran"] = True

    with (
        patch(
            "sys.argv",
            ["admin.py", "-m", "/tmp/admin", "-s", "fed_admin.json", "--study", "cancer-research"],
        ),
        patch("os.chdir"),
        patch("nvflare.fuel.hci.tools.admin.Workspace"),
        patch("nvflare.fuel.hci.tools.admin.secure_load_admin_config", return_value=fake_conf),
        patch("nvflare.fuel.hci.tools.admin.AdminClient", _FakeAdminClient),
    ):
        admin.main()

    assert captured == {"study": "cancer-research", "ran": True}


def test_admin_main_exits_non_zero_for_invalid_study():
    with (
        patch("sys.argv", ["admin.py", "-m", "/tmp/admin", "-s", "fed_admin.json", "--study", "Bad Study"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        admin.main()

    assert exc_info.value.code == 1
