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

from unittest.mock import patch

from nvflare.fuel.flare_api.flare_api import Session, new_secure_session
from nvflare.fuel.hci.client.api import ResultKey
from nvflare.fuel.hci.proto import MetaKey


def _make_session_for_project(project):
    session = Session.__new__(Session)
    session.upload_dir = "/tmp"
    session._project = project
    return session


def test_submit_job_sends_project_cmd_props():
    session = _make_session_for_project("cancer-research")
    captured = {}

    def _fake_do_command(command, enforce_meta=True, props=None):
        captured["props"] = props
        return {ResultKey.META: {MetaKey.JOB_ID: "job-1"}}

    session._do_command = _fake_do_command
    with patch("os.path.isdir", return_value=True):
        session.submit_job("/tmp/job")

    assert captured["props"] == {"project": "cancer-research"}


def test_clone_job_sends_project_cmd_props():
    session = _make_session_for_project("multiple-sclerosis")
    captured = {}

    def _fake_do_command(command, enforce_meta=True, props=None):
        captured["props"] = props
        return {ResultKey.META: {MetaKey.JOB_ID: "job-2"}}

    session._do_command = _fake_do_command
    session.clone_job("source-job")

    assert captured["props"] == {"project": "multiple-sclerosis"}


def test_submit_job_without_project_keeps_cmd_props_empty():
    session = _make_session_for_project(None)
    captured = {}

    def _fake_do_command(command, enforce_meta=True, props=None):
        captured["props"] = props
        return {ResultKey.META: {MetaKey.JOB_ID: "job-3"}}

    session._do_command = _fake_do_command
    with patch("os.path.isdir", return_value=True):
        session.submit_job("/tmp/job")

    assert captured["props"] is None


def test_new_secure_session_forwards_project():
    with patch("nvflare.fuel.flare_api.flare_api.new_session") as mock_new_session:
        new_secure_session("admin@nvidia.com", "/tmp/kit", project="cancer-research")
        _, kwargs = mock_new_session.call_args
        assert kwargs["project"] == "cancer-research"
