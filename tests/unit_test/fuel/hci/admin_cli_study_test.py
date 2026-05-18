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

from nvflare.apis.fl_constant import ConnectionSecurity
from nvflare.apis.job_def import DEFAULT_STUDY
from nvflare.fuel.hci.client.api import CommandInfo, ResultKey
from nvflare.fuel.hci.client.api_spec import AdminConfigKey, UidSource
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


def test_submit_job_relies_on_session_study():
    client, captured = _make_admin_client_for_study("cancer-research")

    client._do_default("submit_job hello")

    assert captured["props"] is None


def test_list_jobs_relies_on_session_study():
    client, captured = _make_admin_client_for_study(DEFAULT_STUDY)

    client._do_default("list_jobs")

    assert captured["props"] is None


def test_list_jobs_preserves_existing_cmd_props():
    client, captured = _make_admin_client_for_study("cancer-research")

    with patch(
        "nvflare.fuel.hci.client.cli.parse_command_line",
        return_value=("list_jobs", ["list_jobs"], {"foo": "bar"}),
    ):
        client._do_default("list_jobs")

    assert captured["props"] == {"foo": "bar"}


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


def test_admin_main_accepts_underscore_study():
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
            ["admin.py", "-m", "/tmp/admin", "-s", "fed_admin.json", "--study", "cancer_research"],
        ),
        patch("os.chdir"),
        patch("nvflare.fuel.hci.tools.admin.Workspace"),
        patch("nvflare.fuel.hci.tools.admin.secure_load_admin_config", return_value=fake_conf),
        patch("nvflare.fuel.hci.tools.admin.AdminClient", _FakeAdminClient),
    ):
        admin.main()

    assert captured == {"study": "cancer_research", "ran": True}


def test_admin_main_exits_non_zero_for_invalid_study():
    with (
        patch("sys.argv", ["admin.py", "-m", "/tmp/admin", "-s", "fed_admin.json", "--study", "Bad Study"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        admin.main()

    assert exc_info.value.code == 1


def test_admin_client_passes_study_to_admin_api(tmp_path):
    captured = {}
    admin_config = {
        AdminConfigKey.CONNECTION_SECURITY: ConnectionSecurity.MTLS,
        AdminConfigKey.CA_CERT: "ca.crt",
        AdminConfigKey.CLIENT_CERT: "client.crt",
        AdminConfigKey.CLIENT_KEY: "client.key",
        AdminConfigKey.UID_SOURCE: UidSource.CERT,
    }

    class _FakeAdminAPI:
        def __init__(self, **kwargs):
            captured["study"] = kwargs["study"]

    with patch("nvflare.fuel.hci.client.cli.AdminAPI", _FakeAdminAPI):
        AdminClient(
            admin_config=admin_config,
            username="admin@nvidia.com",
            handlers=[],
            cli_history_dir=str(tmp_path),
            study="cancer-research",
        )

    assert captured["study"] == "cancer-research"


def test_admin_client_run_reports_login_rejection_and_skips_cmdloop():
    calls = []
    output = []
    client = AdminClient.__new__(AdminClient)

    class _FakeAPI:
        @staticmethod
        def connect(timeout):
            calls.append(("connect", timeout))

        @staticmethod
        def login():
            calls.append(("login",))
            return {
                ResultKey.STATUS: APIStatus.ERROR_AUTHENTICATION,
                ResultKey.DETAILS: "unknown study 'study-a'",
                ResultKey.AUTH_CODE: "AUTH_UNKNOWN_STUDY",
            }

        @staticmethod
        def close():
            calls.append(("close",))

    client.api = _FakeAPI()
    client.login_timeout = 5.0
    client.debug = False
    client.stopped = False
    client.write_string = output.append
    client.cmdloop = lambda *_args, **_kwargs: calls.append(("cmdloop",))

    client.run()

    assert output == ["Login rejected: AUTH_UNKNOWN_STUDY: unknown study 'study-a'"]
    assert calls == [("connect", 5.0), ("login",), ("close",)]
    assert client.stopped is True
