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

from nvflare.apis.job_def import DEFAULT_STUDY
from nvflare.fuel.flare_api.api_spec import AuthenticationError, NoConnection
from nvflare.fuel.flare_api.flare_api import Session, new_secure_session
from nvflare.fuel.hci.client.api import APIStatus, ResultKey
from nvflare.fuel.hci.proto import MetaKey


def _make_session_for_study(study):
    session = Session.__new__(Session)
    session.upload_dir = "/tmp"
    session._study = study
    return session


def test_submit_job_relies_on_session_study():
    session = _make_session_for_study("cancer-research")
    captured = {}

    def _fake_do_command(command, enforce_meta=True, props=None):
        captured["props"] = props
        return {ResultKey.META: {MetaKey.JOB_ID: "job-1"}}

    session._do_command = _fake_do_command
    with patch("os.path.isdir", return_value=True):
        session.submit_job("/tmp/job")

    assert captured["props"] is None


def test_clone_job_does_not_send_study_cmd_props():
    session = _make_session_for_study("multiple-sclerosis")
    captured = {}

    def _fake_do_command(command, enforce_meta=True, props=None):
        captured["props"] = props
        return {ResultKey.META: {MetaKey.JOB_ID: "job-2"}}

    session._do_command = _fake_do_command
    session.clone_job("source-job")

    assert captured["props"] is None


def test_submit_job_with_default_study_uses_session_context_only():
    session = _make_session_for_study(DEFAULT_STUDY)
    captured = {}

    def _fake_do_command(command, enforce_meta=True, props=None):
        captured["props"] = props
        return {ResultKey.META: {MetaKey.JOB_ID: "job-3"}}

    session._do_command = _fake_do_command
    with patch("os.path.isdir", return_value=True):
        session.submit_job("/tmp/job")

    assert captured["props"] is None


def test_list_jobs_relies_on_session_study():
    session = _make_session_for_study(DEFAULT_STUDY)
    captured = {}

    def _fake_do_command(command, enforce_meta=True, props=None):
        captured["props"] = props
        return {ResultKey.META: {MetaKey.JOBS: []}}

    session._do_command = _fake_do_command
    session.list_jobs()

    assert captured["props"] is None


def test_session_rejects_none_study():
    fake_conf = SimpleNamespace(
        get_admin_config=lambda: {"upload_dir": "/tmp", "download_dir": "/tmp"},
        handlers=[],
    )
    with (
        patch("os.path.isdir", return_value=True),
        patch("nvflare.fuel.flare_api.flare_api.secure_load_admin_config", return_value=fake_conf),
        patch("nvflare.fuel.flare_api.flare_api.AdminAPI"),
        pytest.raises(AssertionError, match="study must be str"),
    ):
        Session("admin@nvidia.com", "/tmp/kit", study=None)


def test_session_passes_study_to_admin_api():
    fake_conf = SimpleNamespace(
        get_admin_config=lambda: {"upload_dir": "/tmp", "download_dir": "/tmp"},
        handlers=[],
    )
    captured = {}

    class _FakeAdminAPI:
        def __init__(self, **kwargs):
            captured["study"] = kwargs["study"]

    with (
        patch("os.path.isdir", return_value=True),
        patch("nvflare.fuel.flare_api.flare_api.secure_load_admin_config", return_value=fake_conf),
        patch("nvflare.fuel.flare_api.flare_api.AdminAPI", _FakeAdminAPI),
    ):
        Session("admin@nvidia.com", "/tmp/kit", study="cancer-research")

    assert captured["study"] == "cancer-research"


def test_session_accepts_study_with_underscore():
    fake_conf = SimpleNamespace(
        get_admin_config=lambda: {"upload_dir": "/tmp", "download_dir": "/tmp"},
        handlers=[],
    )
    captured = {}

    class _FakeAdminAPI:
        def __init__(self, **kwargs):
            captured["study"] = kwargs["study"]

    with (
        patch("os.path.isdir", return_value=True),
        patch("nvflare.fuel.flare_api.flare_api.secure_load_admin_config", return_value=fake_conf),
        patch("nvflare.fuel.flare_api.flare_api.AdminAPI", _FakeAdminAPI),
    ):
        Session("admin@nvidia.com", "/tmp/kit", study="cancer_research")

    assert captured["study"] == "cancer_research"


def test_new_secure_session_forwards_study():
    with patch("nvflare.fuel.flare_api.flare_api.new_session") as mock_new_session:
        new_secure_session("admin@nvidia.com", "/tmp/kit", study="cancer-research")
        _, kwargs = mock_new_session.call_args
        assert kwargs["study"] == "cancer-research"


def test_new_secure_session_defaults_study():
    with patch("nvflare.fuel.flare_api.flare_api.new_session") as mock_new_session:
        new_secure_session("admin@nvidia.com", "/tmp/kit")
        _, kwargs = mock_new_session.call_args
        assert kwargs["study"] == DEFAULT_STUDY


def test_try_connect_raises_on_login_failure():
    session = _make_session_for_study(DEFAULT_STUDY)
    session.api = SimpleNamespace(
        closed=False,
        connect=lambda timeout: None,
        login=lambda: {
            ResultKey.STATUS: APIStatus.ERROR_AUTHENTICATION,
            ResultKey.DETAILS: "Incorrect user name or password",
        },
    )

    with pytest.raises(AuthenticationError, match="Incorrect user name or password"):
        session.try_connect(5.0)


def test_try_connect_raises_no_connection_on_server_connection_error():
    session = _make_session_for_study(DEFAULT_STUDY)
    session.api = SimpleNamespace(
        closed=False,
        connect=lambda timeout: None,
        login=lambda: {
            ResultKey.STATUS: APIStatus.ERROR_SERVER_CONNECTION,
            ResultKey.DETAILS: "server unavailable",
        },
    )

    with pytest.raises(NoConnection, match="server unavailable"):
        session.try_connect(5.0)
