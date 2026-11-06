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
from nvflare.fuel.flare_api.api_spec import CommandError
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
                ConnProps.CMD_PROPS: props,
            }
        )
        if spec.authz_func(conn, [command]) != PreAuthzReturnCode.REQUIRE_AUTHZ:
            return {ProtoKey.DATA: [{ProtoKey.TYPE: "error", ProtoKey.DATA: conn.errors[0]}]}
        spec.handler_func(conn, [command])
        return {ProtoKey.DATA: [{ProtoKey.TYPE: ProtoKey.DICT, ProtoKey.DATA: conn.replies[0]}]}


@pytest.fixture
def manager(tmp_path):
    manager = CatalogManager(register_commands=False)
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


class TestRegistration:
    def test_the_manager_registers_its_commands_once(self, monkeypatch):
        from nvflare.private.fed.server.server_cmd_modules import ServerCommandModules

        modules = []
        monkeypatch.setattr(ServerCommandModules, "cmd_modules", modules)
        first = CatalogManager()
        second = CatalogManager()
        registered = [m for m in modules if isinstance(m, CatalogCommandModule)]
        assert len(registered) == 1 and registered[0]._manager is second and first is not second
        assert {s.name for s in registered[0].get_spec().cmd_specs} == {commands.SEARCH, commands.STATUS}


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
            ("project_admin", False, commands.SEARCH, True),
            ("org_admin", False, commands.STATUS, True),
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

    def test_no_registry_means_only_admins(self, manager):
        module = CatalogCommandModule(manager)
        spec = module.get_spec().cmd_specs[0]
        conn = FakeConn({ConnProps.ACTIVE_STUDY: STUDY, ConnProps.USER_ROLE: "member", ConnProps.USER_NAME: "u"})
        assert spec.authz_func(conn, ["x"]) == PreAuthzReturnCode.ERROR
