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

import pytest

from nvflare.app_common.fdn.catalog.admin.catalog_api import CatalogAdminApi
from nvflare.app_common.fdn.catalog.common.payload import ResultStatus, Selection, build_result
from nvflare.app_common.fdn.catalog.common.schema import ReadLimits
from nvflare.app_common.fdn.catalog.server import commands
from nvflare.app_common.fdn.catalog.server.commands import CatalogCommandModule, RequestError, parse_search_request
from nvflare.app_common.fdn.catalog.server.manager import CatalogManager
from nvflare.app_common.fdn.catalog.server.reconcile import Reconciler
from nvflare.app_common.fdn.catalog.server.store import CatalogStore
from nvflare.fuel.flare_api.api_spec import (
    AuthenticationError,
    AuthorizationError,
    CommandError,
    InternalError,
    NoConnection,
    SessionClosed,
)
from nvflare.fuel.hci.client.api_status import APIStatus
from nvflare.fuel.hci.proto import ProtoKey
from nvflare.fuel.hci.server.authz import PreAuthzReturnCode
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.security.study_registry import StudyRegistryService
from tests.unit_test.app_common.fdn.catalog.helpers import make_catalog

STUDY = "study1"
NOW = "2026-06-01T00:00:00Z"


class FakeConn:
    def __init__(self, props):
        self.props, self.replies, self.errors = props, [], []

    def get_prop(self, key, default=None):
        return self.props.get(key, default)

    def append_dict(self, data, meta=None):
        self.replies.append(data)

    def append_error(self, data, meta=None):
        self.errors.append(data)


class FakeSession:
    """Routes do_command through the real command module, shaping the reply like the admin API does."""

    def __init__(self, module, role="project_admin", user="u1"):
        self.module, self.role, self.user = module, role, user

    def do_command(self, command, props=None):
        spec = next(s for s in self.module.get_spec().cmd_specs if s.name == command)
        conn = FakeConn(
            {
                ConnProps.ACTIVE_STUDY: STUDY,
                ConnProps.USER_ROLE: self.role,
                ConnProps.USER_NAME: self.user,
                ConnProps.CERT_STUDIES: [STUDY],  # a session admitted to this Study
                ConnProps.CMD_PROPS: props,
            }
        )
        if spec.authz_func(conn, [command]) != PreAuthzReturnCode.REQUIRE_AUTHZ:
            return {ProtoKey.DATA: [{ProtoKey.TYPE: "error", ProtoKey.DATA: conn.errors[0]}]}
        spec.handler_func(conn, [command])
        return {ProtoKey.DATA: [{ProtoKey.TYPE: ProtoKey.DICT, ProtoKey.DATA: conn.replies[0]}]}


@pytest.fixture
def manager(tmp_path):
    manager = CatalogManager()
    manager._store = CatalogStore(str(tmp_path))
    rec = Reconciler(manager._store, ReadLimits(60, 1000, 1000))
    for client in ["site-a", "site-b"]:
        catalogs = [make_catalog(dataset=f"ds{n}", display_name=f"Data {n}") for n in range(3)]
        rec.apply(STUDY, client, Selection(None), build_result(ResultStatus.OK, catalogs), NOW)
    return manager


@pytest.fixture
def client(manager):
    return CatalogAdminApi(FakeSession(CatalogCommandModule(manager)))


class TestRequestParsing:
    @pytest.mark.parametrize(
        "props",
        [
            {"bogus": 1},
            {"page_size": 0},
            {"page_size": 101},
            {"page_size": True},
            {"text": ""},
            {"text": "x" * 257},
            {"available": "yes"},
            {"schema_version": -1},
            {"cursor": "x" * 1025},
            [],
            "x",
        ],
    )
    def test_bad_search_requests(self, props):
        with pytest.raises(RequestError):
            parse_search_request(props)

    def test_search_request(self):
        query = parse_search_request({"text": "omop", "available": True, "page_size": 5, "schema_version": 2})
        assert (query.text, query.available, query.page_size, query.schema_version) == ("omop", True, 5, 2)
        assert parse_search_request(None).page_size == 50


class TestCommands:
    def test_search_round_trip_returns_typed_entries(self, client):
        page = client.search_page(text="data 1", page_size=10)
        assert page.total == 2 and {e.client for e in page.entries} == {"site-a", "site-b"}
        entry = page.entries[0]
        assert entry.catalog.display_name == "Data 1" and entry.catalog_version == 1 and entry.available

    def test_search_iterator_follows_the_cursor(self, client, monkeypatch):
        search_page = client.search_page
        calls = 0

        def bounded_search_page(*args, **kwargs):
            nonlocal calls
            calls += 1
            assert calls <= 2, "pagination exceeded the two expected pages"
            return search_page(*args, **kwargs)

        monkeypatch.setattr(client, "search_page", bounded_search_page)
        seen = [(e.client, e.dataset) for e in client.search(page_size=4)]
        assert seen == sorted(seen) and len(seen) == 6 and calls == 2

    def test_invalid_request_raises_a_command_error_without_echoing_values(self, client):
        with pytest.raises(CommandError) as e:
            client.search_page(page_size=500)
        assert e.value.error_code == "INVALID_ARGS" and "500" not in str(e.value)

    def test_bad_cursor_is_an_invalid_request(self, client):
        with pytest.raises(CommandError):
            client.search_page(cursor="garbage")

    def test_status(self, client):
        health = {h.client: h for h in client.status()}
        assert set(health) == {"site-a", "site-b"} and health["site-a"].health == "OK"
        assert health["site-a"].last_success_at == NOW and health["site-a"].stale is True  # NOW is long past


class TestNotReady:
    def test_search_and_status_before_the_service_has_started_report_not_ready(self, manager, client):
        manager._store = None
        with pytest.raises(CommandError) as search:
            client.search_page()
        with pytest.raises(CommandError) as status:
            client.status()
        assert search.value.error_code == status.value.error_code == "SYSTEM_NOT_READY"


class TestRegistration:
    def test_standard_server_registry_includes_catalog_commands_once(self):
        from nvflare.private.fed.server.server_cmd_modules import ServerCommandModules

        registered = [m for m in ServerCommandModules.cmd_modules if isinstance(m, CatalogCommandModule)]
        assert len(registered) == 1 and registered[0]._manager is None
        assert {s.name for s in registered[0].get_spec().cmd_specs} == {commands.SEARCH, commands.STATUS}

    def test_handler_resolves_the_catalog_service_from_each_connection(self, tmp_path):
        module = CatalogCommandModule()
        first, second = CatalogManager(), CatalogManager()
        first._store = CatalogStore(str(tmp_path / "first"))
        second._store = CatalogStore(str(tmp_path / "second"))
        for manager in (first, second):
            conn = FakeConn({})
            conn.app_ctx = SimpleNamespace(run_manager=SimpleNamespace(components={"research_catalog": manager}))
            assert module._service(conn) is manager
        # Services do not register commands or change a global command module's selected manager.
        assert module._manager is None

    def test_last_configured_service_preserves_existing_registration_order(self):
        first, second = CatalogManager(), CatalogManager()
        conn = FakeConn({})
        conn.app_ctx = SimpleNamespace(run_manager=SimpleNamespace(components={"first": first, "second": second}))
        assert CatalogCommandModule()._service(conn) is second

    def test_registered_handler_uses_configured_service_for_search_and_status(self, manager):
        module = CatalogCommandModule()
        conn = FakeConn({ConnProps.ACTIVE_STUDY: STUDY})
        conn.app_ctx = SimpleNamespace(run_manager=SimpleNamespace(components={"research_catalog": manager}))
        module.cmd_search(conn, [commands.SEARCH])
        assert len(conn.replies[-1]["entries"]) == 6
        module.cmd_status(conn, [commands.STATUS])
        assert {row["client"] for row in conn.replies[-1]["clients"]} == {"site-a", "site-b"}

    @pytest.mark.parametrize("handler", ["cmd_search", "cmd_status"])
    def test_missing_catalog_service_returns_not_ready(self, handler):
        conn = FakeConn({})
        conn.app_ctx = SimpleNamespace(run_manager=SimpleNamespace(components={}))
        getattr(CatalogCommandModule(), handler)(conn, ["catalog"])
        assert conn.replies[0]["error_code"] == "SYSTEM_NOT_READY"


class TestAuthorization:
    def call(self, manager, role, command, member=False):
        module = CatalogCommandModule(manager)
        registry = SimpleNamespace(has_user=lambda user, study: member)
        original = StudyRegistryService.get_registry
        StudyRegistryService.get_registry = staticmethod(lambda: registry)
        try:
            spec = next(s for s in module.get_spec().cmd_specs if s.name == command)
            conn = FakeConn({ConnProps.ACTIVE_STUDY: STUDY, ConnProps.USER_ROLE: role})
            return spec.authz_func(conn, [command])
        finally:
            StudyRegistryService.get_registry = original

    @pytest.mark.parametrize(
        "role,member,command,allowed",
        [
            ("project_admin", False, commands.SEARCH, False),
            ("project_admin", True, commands.SEARCH, True),
            ("org_admin", False, commands.STATUS, False),
            ("org_admin", True, commands.STATUS, True),
            ("member", True, commands.SEARCH, True),
            ("member", True, commands.STATUS, True),
            ("member", False, commands.SEARCH, False),
        ],
    )
    def test_matrix(self, manager, role, member, command, allowed):
        result = self.call(manager, role, command, member=member)
        assert (result == PreAuthzReturnCode.REQUIRE_AUTHZ) is allowed

    def test_certificate_study_entitlement_counts_as_membership(self, manager):
        module = CatalogCommandModule(manager)
        registry = SimpleNamespace(has_user=lambda user, study: False)
        original = StudyRegistryService.get_registry
        StudyRegistryService.get_registry = staticmethod(lambda: registry)
        try:
            spec = next(s for s in module.get_spec().cmd_specs if s.name == commands.SEARCH)
            props = {ConnProps.ACTIVE_STUDY: STUDY, ConnProps.USER_ROLE: "member", ConnProps.CERT_STUDIES: [STUDY]}
            assert spec.authz_func(FakeConn(props), ["x"]) == PreAuthzReturnCode.REQUIRE_AUTHZ
            props[ConnProps.CERT_STUDIES] = ["other"]
            assert spec.authz_func(FakeConn(props), ["x"]) == PreAuthzReturnCode.ERROR
        finally:
            StudyRegistryService.get_registry = original

    def test_the_site_authorization_policy_decides_the_right(self):
        from nvflare.security.security import COMMAND_CATEGORIES, CommandCategory

        assert COMMAND_CATEGORIES[commands.SEARCH] == CommandCategory.VIEW
        assert COMMAND_CATEGORIES[commands.STATUS] == CommandCategory.VIEW

    def test_the_default_study_has_no_membership_so_the_policy_decides(self):
        module = CatalogCommandModule(SimpleNamespace())
        by_name = {s.name: s for s in module.get_spec().cmd_specs}
        props = {ConnProps.ACTIVE_STUDY: "default", ConnProps.USER_ROLE: "member", ConnProps.USER_NAME: "u"}
        assert by_name[commands.SEARCH].authz_func(FakeConn(props), ["x"]) == PreAuthzReturnCode.REQUIRE_AUTHZ

    def test_a_connection_without_an_active_study_is_in_the_default_study(self):
        module = CatalogCommandModule(SimpleNamespace())
        by_name = {s.name: s for s in module.get_spec().cmd_specs}
        props = {ConnProps.USER_ROLE: "member", ConnProps.USER_NAME: "u"}
        assert by_name[commands.SEARCH].authz_func(FakeConn(props), ["x"]) == PreAuthzReturnCode.REQUIRE_AUTHZ

    def test_no_registry_means_only_admins(self, manager):
        module = CatalogCommandModule(manager)
        spec = module.get_spec().cmd_specs[0]
        conn = FakeConn({ConnProps.ACTIVE_STUDY: STUDY, ConnProps.USER_ROLE: "member", ConnProps.USER_NAME: "u"})
        assert spec.authz_func(conn, ["x"]) == PreAuthzReturnCode.ERROR


class StatusSession:
    """Returns a failed API response with no catalog payload, as the admin client does for a lost connection."""

    def __init__(self, status):
        self.status = status

    def do_command(self, command, props=None):
        return {ProtoKey.STATUS: self.status, ProtoKey.DETAILS: "simulated"}


@pytest.mark.parametrize(
    "status,error",
    [
        (APIStatus.ERROR_SERVER_CONNECTION, NoConnection),
        (APIStatus.ERROR_INACTIVE_SESSION, SessionClosed),
        (APIStatus.ERROR_AUTHENTICATION, AuthenticationError),
        (APIStatus.ERROR_CERT, AuthenticationError),
        (APIStatus.ERROR_AUTHORIZATION, AuthorizationError),
        (APIStatus.ERROR_PROTOCOL, InternalError),
        (APIStatus.ERROR_RUNTIME, InternalError),
    ],
)
def test_a_failed_api_status_raises_the_matching_session_error_not_an_empty_reply(status, error):
    api = CatalogAdminApi(StatusSession(status))
    with pytest.raises(error):
        api.status()
    with pytest.raises(error):
        list(api.search())
