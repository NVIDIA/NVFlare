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
from nvflare.fuel.flare_api.flare_api import Session, new_secure_session
from nvflare.fuel.hci.client.api import ResultKey
from nvflare.fuel.hci.proto import MetaKey


def _make_session_for_study(study):
    session = Session.__new__(Session)
    session.upload_dir = "/tmp"
    session._study = study
    return session


def test_submit_job_sends_study_cmd_props():
    session = _make_session_for_study("cancer-research")
    captured = {}

    def _fake_do_command(command, enforce_meta=True, props=None):
        captured["props"] = props
        return {ResultKey.META: {MetaKey.JOB_ID: "job-1"}}

    session._do_command = _fake_do_command
    with patch("os.path.isdir", return_value=True):
        session.submit_job("/tmp/job")

    assert captured["props"] == {"study": "cancer-research"}


def test_clone_job_does_not_send_study_cmd_props():
    session = _make_session_for_study("multiple-sclerosis")
    captured = {}

    def _fake_do_command(command, enforce_meta=True, props=None):
        captured["props"] = props
        return {ResultKey.META: {MetaKey.JOB_ID: "job-2"}}

    session._do_command = _fake_do_command
    session.clone_job("source-job")

    assert captured["props"] is None


def test_submit_job_sends_default_study_cmd_props():
    session = _make_session_for_study(DEFAULT_JOB_STUDY)
    captured = {}

    def _fake_do_command(command, enforce_meta=True, props=None):
        captured["props"] = props
        return {ResultKey.META: {MetaKey.JOB_ID: "job-3"}}

    session._do_command = _fake_do_command
    with patch("os.path.isdir", return_value=True):
        session.submit_job("/tmp/job")

    assert captured["props"] == {"study": DEFAULT_JOB_STUDY}


def test_list_jobs_sends_default_study_cmd_props():
    session = _make_session_for_study(DEFAULT_JOB_STUDY)
    captured = {}

    def _fake_do_command(command, enforce_meta=True, props=None):
        captured["props"] = props
        return {ResultKey.META: {MetaKey.JOBS: []}}

    session._do_command = _fake_do_command
    session.list_jobs()

    assert captured["props"] == {"study": DEFAULT_JOB_STUDY}


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


def test_new_secure_session_forwards_study():
    with patch("nvflare.fuel.flare_api.flare_api.new_session") as mock_new_session:
        new_secure_session("admin@nvidia.com", "/tmp/kit", study="cancer-research")
        _, kwargs = mock_new_session.call_args
        assert kwargs["study"] == "cancer-research"


def test_new_secure_session_defaults_study():
    with patch("nvflare.fuel.flare_api.flare_api.new_session") as mock_new_session:
        new_secure_session("admin@nvidia.com", "/tmp/kit")
        _, kwargs = mock_new_session.call_args
        assert kwargs["study"] == DEFAULT_JOB_STUDY
