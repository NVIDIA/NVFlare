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

from tests.integration_test.src.validators.job_log_result_validator import JobLogResultValidator


def test_job_log_validator_finds_each_nested_client_log(tmp_path):
    workspace_root = tmp_path / "workspace"
    clients = [SimpleNamespace(name="site-1"), SimpleNamespace(name="site-2")]
    for client in clients:
        client_log = workspace_root / "app" / client.name / "log.json"
        client_log.parent.mkdir(parents=True)
        client_log.write_text(f'{{"message": "Client {client.name} initialized"}}\n', encoding="utf-8")

    validator = JobLogResultValidator(required_patterns=["Client {client} initialized"])

    assert validator.validate_finished_results({"workspace_root": str(workspace_root)}, clients)


def test_job_log_validator_rejects_missing_client_log(tmp_path):
    workspace_root = tmp_path / "workspace"
    client_log = workspace_root / "site-1" / "log.json"
    client_log.parent.mkdir(parents=True)
    client_log.write_text('{"message": "Client site-1 initialized"}\n', encoding="utf-8")
    clients = [SimpleNamespace(name="site-1"), SimpleNamespace(name="site-2")]
    validator = JobLogResultValidator(required_patterns=["Client {client} initialized"])

    assert not validator.validate_finished_results({"workspace_root": str(workspace_root)}, clients)


def test_job_log_validator_rejects_missing_required_pattern(tmp_path):
    workspace_root = tmp_path / "workspace"
    client_log = workspace_root / "site-1" / "log.json"
    client_log.parent.mkdir(parents=True)
    client_log.write_text('{"message": "some other message"}\n', encoding="utf-8")
    clients = [SimpleNamespace(name="site-1")]
    validator = JobLogResultValidator(required_patterns=["Client {client} initialized"])

    assert not validator.validate_finished_results({"workspace_root": str(workspace_root)}, clients)
