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

import os

import pytest

from nvflare.apis.job_launcher_spec import JobProcessEnv
from nvflare.private.fed.app.client.worker_process import parse_arguments as cj_parse
from nvflare.private.fed.app.server.runner_process import parse_arguments as sj_parse

_ATTR_ENV = {
    "token": JobProcessEnv.AUTH_TOKEN,
    "token_signature": JobProcessEnv.TOKEN_SIGNATURE,
    "ssid": JobProcessEnv.SSID,
}

# fmt: off
_CJ_ARGV = [
    "worker_process.py", "--workspace", "/ws", "--startup", "/ws/startup", "--job_id", "job-1",
    "--client_name", "site-1", "--sp_target", "server:8002", "--sp_scheme", "grpc",
    "--parent_url", "tcp://localhost:8004", "--fed_client", "fed_client.json",
]
_SJ_ARGV = [
    "runner_process.py", "--workspace", "/ws", "--fed_server", "fed_server.json",
    "--app_root", "/ws/job-1/app_server", "--job_id", "job-1", "--root_url", "grpc://server:8002",
    "--host", "server", "--port", "8003", "--parent_url", "tcp://localhost:8004",
]
# fmt: on

_CASES = [
    pytest.param(cj_parse, _CJ_ARGV, ("token", "token_signature", "ssid"), id="cj"),
    pytest.param(sj_parse, _SJ_ARGV, ("token_signature", "ssid"), id="sj"),
]


@pytest.mark.parametrize("parse,base_argv,cred_attrs", _CASES)
class TestJobProcessCredentialParsing:
    def test_env_only_accepted_and_env_removed(self, monkeypatch, parse, base_argv, cred_attrs):
        for attr, env_name in _ATTR_ENV.items():
            monkeypatch.setenv(env_name, f"env-{attr}")
        monkeypatch.setattr("sys.argv", base_argv)

        args = parse()

        for attr in cred_attrs:
            assert getattr(args, attr) == f"env-{attr}"
        # all three env vars removed (SJ pops the unused AUTH_TOKEN too) so children never inherit
        assert not set(_ATTR_ENV.values()) & set(os.environ)

    def test_cli_wins_and_env_removed(self, monkeypatch, parse, base_argv, cred_attrs):
        for attr, env_name in _ATTR_ENV.items():
            monkeypatch.setenv(env_name, f"env-{attr}")
        cli_argv = [t for attr in cred_attrs for t in (f"--{attr}", f"cli-{attr}")]
        monkeypatch.setattr("sys.argv", base_argv + cli_argv)

        args = parse()

        for attr in cred_attrs:
            assert getattr(args, attr) == f"cli-{attr}"
        assert not set(_ATTR_ENV.values()) & set(os.environ)

    def test_missing_both_errors(self, monkeypatch, parse, base_argv, cred_attrs):
        for env_name in _ATTR_ENV.values():
            monkeypatch.delenv(env_name, raising=False)
        monkeypatch.setattr("sys.argv", base_argv)

        with pytest.raises(SystemExit):
            parse()

    def test_empty_env_treated_as_missing(self, monkeypatch, parse, base_argv, cred_attrs):
        for env_name in _ATTR_ENV.values():
            monkeypatch.setenv(env_name, "")
        monkeypatch.setattr("sys.argv", base_argv)

        with pytest.raises(SystemExit):
            parse()
        assert not set(_ATTR_ENV.values()) & set(os.environ)
