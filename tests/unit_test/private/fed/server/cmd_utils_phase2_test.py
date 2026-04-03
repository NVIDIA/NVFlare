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

from nvflare.apis.job_def import DEFAULT_STUDY
from nvflare.fuel.hci.server.authz import PreAuthzReturnCode
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.private.fed.server import cmd_utils as cmd_utils_module
from nvflare.private.fed.server.cmd_utils import CommandUtil


class _FakeConnection:
    def __init__(self, props=None, app_ctx=None):
        self._props = dict(props or {})
        self.app_ctx = app_ctx
        self.errors = []

    def get_prop(self, key, default=None):
        return self._props.get(key, default)

    def set_prop(self, key, value):
        self._props[key] = value

    def append_error(self, msg, meta=None):
        self.errors.append((msg, meta))


class _FakeClient:
    def __init__(self, name, token):
        self.name = name
        self.token = token


class _FakeEngine:
    def __init__(self, clients):
        self._clients = list(clients)

    def get_clients(self):
        return list(self._clients)

    def validate_targets(self, client_names):
        by_name = {c.name: c for c in self._clients}
        clients = []
        invalid = []
        for client_name in client_names:
            client = by_name.get(client_name)
            if client:
                clients.append(client)
            else:
                invalid.append(client_name)
        return clients, invalid


class _FakeStudyRegistry:
    def __init__(self, roles=None, studies=None):
        self.roles = roles or {}
        self.studies = studies or {}

    def has_study(self, study):
        return study in self.studies

    def get_role(self, user_name, study):
        return self.roles.get((user_name, study))

    def get_sites(self, study):
        return self.studies.get(study)


class _FakeStudyRegistryService:
    registry = None

    @staticmethod
    def get_registry():
        return _FakeStudyRegistryService.registry


def _install_registry(monkeypatch, registry):
    monkeypatch.setattr(cmd_utils_module, "StudyRegistryService", _FakeStudyRegistryService, raising=False)
    monkeypatch.setattr(cmd_utils_module, "ServerEngineSpec", object)
    _FakeStudyRegistryService.registry = registry


def test_command_authz_required_resolves_study_role_before_authorization(monkeypatch):
    _install_registry(
        monkeypatch,
        _FakeStudyRegistry(
            roles={("admin@nvidia.com", "cancer-research"): "lead"}, studies={"cancer-research": {"site-a"}}
        ),
    )

    conn = _FakeConnection(
        props={
            ConnProps.USER_NAME: "admin@nvidia.com",
            ConnProps.USER_ROLE: "project_admin",
            ConnProps.ACTIVE_STUDY: "cancer-research",
        }
    )

    result = CommandUtil().command_authz_required(conn, ["list_jobs"])

    assert result == PreAuthzReturnCode.REQUIRE_AUTHZ
    assert conn.get_prop(ConnProps.USER_ROLE) == "lead"


def test_authorize_client_operation_resolves_study_role_before_authorization(monkeypatch):
    _install_registry(
        monkeypatch,
        _FakeStudyRegistry(
            roles={("admin@nvidia.com", "cancer-research"): "lead"}, studies={"cancer-research": {"site-a"}}
        ),
    )

    conn = _FakeConnection(
        props={
            ConnProps.USER_NAME: "admin@nvidia.com",
            ConnProps.USER_ROLE: "project_admin",
            ConnProps.ACTIVE_STUDY: "cancer-research",
        },
        app_ctx=_FakeEngine([_FakeClient("site-a", "token-a")]),
    )

    result = CommandUtil().authorize_client_operation(conn, ["report_env", "site-a"])

    assert result == PreAuthzReturnCode.REQUIRE_AUTHZ
    assert conn.get_prop(ConnProps.USER_ROLE) == "lead"


def test_authorize_server_operation_requires_authz_for_client_targets_when_registry_exists(monkeypatch):
    _install_registry(
        monkeypatch,
        _FakeStudyRegistry(
            roles={("admin@nvidia.com", "cancer-research"): "lead"}, studies={"cancer-research": {"site-a"}}
        ),
    )

    conn = _FakeConnection(
        props={
            ConnProps.USER_NAME: "admin@nvidia.com",
            ConnProps.USER_ROLE: "project_admin",
            ConnProps.ACTIVE_STUDY: "cancer-research",
        },
        app_ctx=_FakeEngine([_FakeClient("site-a", "token-a"), _FakeClient("site-b", "token-b")]),
    )

    result = CommandUtil().authorize_server_operation(conn, ["check_status", "client"])

    assert result == PreAuthzReturnCode.REQUIRE_AUTHZ
    assert conn.get_prop(ConnProps.USER_ROLE) == "lead"


def test_authorize_server_operation_preserves_ok_for_client_targets_without_registry(monkeypatch):
    _install_registry(monkeypatch, None)

    conn = _FakeConnection(
        props={
            ConnProps.USER_NAME: "admin@nvidia.com",
            ConnProps.USER_ROLE: "project_admin",
            ConnProps.ACTIVE_STUDY: DEFAULT_STUDY,
        },
        app_ctx=_FakeEngine([_FakeClient("site-a", "token-a")]),
    )

    result = CommandUtil().authorize_server_operation(conn, ["check_status", "client"])

    assert result == PreAuthzReturnCode.OK
    assert conn.get_prop(ConnProps.USER_ROLE) == "project_admin"


def test_validate_command_targets_filters_clients_by_enrolled_study_sites(monkeypatch):
    _install_registry(monkeypatch, _FakeStudyRegistry(studies={"cancer-research": {"site-a"}}))

    conn = _FakeConnection(
        props={ConnProps.ACTIVE_STUDY: "cancer-research"},
        app_ctx=_FakeEngine([_FakeClient("site-a", "token-a"), _FakeClient("site-b", "token-b")]),
    )

    err = CommandUtil().validate_command_targets(conn, ["client", "site-a", "site-b"])

    assert err == ""
    assert conn.get_prop(CommandUtil.TARGET_CLIENT_NAMES) == ["site-a"]
    assert conn.get_prop(CommandUtil.TARGET_CLIENT_TOKENS) == ["token-a"]
    assert conn.get_prop(CommandUtil.TARGET_CLIENTS) == {"token-a": "site-a"}


def test_validate_command_targets_errors_when_named_clients_are_outside_study(monkeypatch):
    _install_registry(monkeypatch, _FakeStudyRegistry(studies={"cancer-research": {"site-a"}}))

    conn = _FakeConnection(
        props={ConnProps.ACTIVE_STUDY: "cancer-research"},
        app_ctx=_FakeEngine([_FakeClient("site-a", "token-a"), _FakeClient("site-b", "token-b")]),
    )

    err = CommandUtil().validate_command_targets(conn, ["client", "site-b"])

    assert err == "site 'site-b' is not enrolled in study 'cancer-research'"


def test_validate_command_targets_keeps_default_sessions_unfiltered_with_registry(monkeypatch):
    _install_registry(monkeypatch, _FakeStudyRegistry(studies={"cancer-research": {"site-a"}}))

    conn = _FakeConnection(
        props={ConnProps.ACTIVE_STUDY: DEFAULT_STUDY},
        app_ctx=_FakeEngine([_FakeClient("site-a", "token-a"), _FakeClient("site-b", "token-b")]),
    )

    err = CommandUtil().validate_command_targets(conn, ["client", "site-a", "site-b"])

    assert err == ""
    assert conn.get_prop(CommandUtil.TARGET_CLIENT_NAMES) == ["site-a", "site-b"]
    assert conn.get_prop(CommandUtil.TARGET_CLIENT_TOKENS) == ["token-a", "token-b"]
    assert conn.get_prop(CommandUtil.TARGET_CLIENTS) == {"token-a": "site-a", "token-b": "site-b"}


def test_validate_command_targets_silently_narrows_all_targets_to_study(monkeypatch):
    _install_registry(monkeypatch, _FakeStudyRegistry(studies={"cancer-research": {"site-a"}}))

    conn = _FakeConnection(
        props={ConnProps.ACTIVE_STUDY: "cancer-research"},
        app_ctx=_FakeEngine([_FakeClient("site-a", "token-a"), _FakeClient("site-b", "token-b")]),
    )

    err = CommandUtil().validate_command_targets(conn, ["all"])

    assert err == ""
    assert conn.get_prop(CommandUtil.TARGET_TYPE) == CommandUtil.TARGET_TYPE_ALL
    assert conn.get_prop(CommandUtil.TARGET_CLIENT_NAMES) == ["site-a"]
    assert conn.get_prop(CommandUtil.TARGET_CLIENT_TOKENS) == ["token-a"]
    assert conn.get_prop(CommandUtil.TARGET_CLIENTS) == {"token-a": "site-a"}
