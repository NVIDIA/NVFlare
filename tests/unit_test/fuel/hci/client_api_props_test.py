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

from nvflare.apis.job_def import DEFAULT_STUDY
from nvflare.fuel.hci.client.api import AdminAPI
from nvflare.fuel.hci.client.api_spec import CommandContext


def test_do_client_command_preserves_command_props():
    api = AdminAPI.__new__(AdminAPI)
    captured = {}

    def _new_command_context(command, args, ent):
        ctx = CommandContext()
        ctx.set_command(command)
        ctx.set_command_args(args)
        ctx.set_command_entry(ent)
        return ctx

    def _handler(args, ctx):
        captured["props"] = ctx.get_command_props()
        ctx.set_command_result({"status": "ok"})

    api._new_command_context = _new_command_context
    ent = SimpleNamespace(handler=_handler)

    result = api._do_client_command("submit_job hello", ["submit_job", "hello"], ent, props={"study": "study-a"})

    assert result == {"status": "ok"}
    assert captured["props"] == {"study": "study-a"}


def test_user_login_sends_study_header(monkeypatch):
    api = AdminAPI.__new__(AdminAPI)
    api.client_key = "client.key"
    api.client_cert = "client.crt"
    api.user_name = "admin@nvidia.com"
    api.study = "cancer-research"
    api.login_result = None
    captured = {}

    class _FakeIdentityAsserter:
        cert_data = "cert-data"

        def __init__(self, private_key_file, cert_file):
            assert private_key_file == "client.key"
            assert cert_file == "client.crt"

        @staticmethod
        def sign_common_name(nonce=""):
            return "signature"

    monkeypatch.setattr("nvflare.fuel.hci.client.api.IdentityAsserter", _FakeIdentityAsserter)

    def _fake_server_execute(command, reply_processor, headers=None):
        captured["command"] = command
        captured["headers"] = headers
        api.login_result = "OK"

    api.server_execute = _fake_server_execute
    api._after_login = lambda: {"status": "ok"}

    result = api._user_login()

    assert result == {"status": "ok"}
    assert captured["command"] == "_cert_login admin@nvidia.com"
    assert captured["headers"]["study"] == "cancer-research"


def test_user_login_defaults_study_header(monkeypatch):
    api = AdminAPI.__new__(AdminAPI)
    api.client_key = "client.key"
    api.client_cert = "client.crt"
    api.user_name = "admin@nvidia.com"
    api.study = DEFAULT_STUDY
    api.login_result = None
    captured = {}

    class _FakeIdentityAsserter:
        cert_data = "cert-data"

        def __init__(self, private_key_file, cert_file):
            pass

        @staticmethod
        def sign_common_name(nonce=""):
            return "signature"

    monkeypatch.setattr("nvflare.fuel.hci.client.api.IdentityAsserter", _FakeIdentityAsserter)

    def _fake_server_execute(command, reply_processor, headers=None):
        captured["headers"] = headers
        api.login_result = "OK"

    api.server_execute = _fake_server_execute
    api._after_login = lambda: {"status": "ok"}

    api._user_login()

    assert captured["headers"]["study"] == DEFAULT_STUDY
