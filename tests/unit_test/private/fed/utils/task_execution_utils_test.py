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

import io
import json
from zipfile import ZipFile

import pytest

from nvflare.private.fed.utils.task_execution_utils import app_requires_task_runtime, execution_lifetime


@pytest.mark.parametrize(
    "config, required",
    [
        ({}, False),
        ({"execution_lifetime": "job"}, False),
        ({"execution_lifetime": "task"}, True),
        ({"execution_lifetime": "{mode}", "mode": "task"}, True),
        ({"execution_lifetime": "{mode}", "mode": "job"}, True),
        ({"execution_lifetime": "{site_setting}"}, True),
    ],
)
def test_task_runtime_requirement_is_read_without_importing_job_code(config, required):
    config["executors"] = [{"executor": {"path": "must.not.import.Executor"}}]
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr("config/config_fed_client.json", json.dumps(config))
    assert app_requires_task_runtime(stream.getvalue()) is required


def test_lifetime_resolution_uses_runtime_variable_precedence():
    assert execution_lifetime({"execution_lifetime": "{mode}", "mode": "job"}, {"mode": "task"}) == "task"


def test_server_only_application_does_not_require_task_runtime():
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr("config/config_fed_server.json", "{}")
    assert not app_requires_task_runtime(stream.getvalue())
