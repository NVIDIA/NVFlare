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

"""Admin commands for catalog search and status."""

from dataclasses import fields

from nvflare.apis.job_def import DEFAULT_STUDY
from nvflare.app_common.fdn.catalog.server.manager import CatalogManager, NotReady
from nvflare.app_common.fdn.catalog.server.search import MAX_PAGE_SIZE, SearchQuery, entry_to_dict
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import MetaStatusValue, make_meta
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.fuel.hci.server.authz import PreAuthzReturnCode
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.security.study_registry import StudyRegistryService

SEARCH = "fdn_catalog_search"
STATUS = "fdn_catalog_status"

_MAX_TEXT = 256
_MAX_CURSOR = 1024


class RequestError(ValueError):
    """A malformed request. The message never repeats request values."""


def _text(value, name: str, limit: int = _MAX_TEXT) -> str:
    if not isinstance(value, str) or not 0 < len(value) <= limit:
        raise RequestError(f"{name} must be a non-empty string of at most {limit} characters")
    return value


def _integer(value, name: str, low: int, high: int | None = None) -> int:
    if type(value) is not int or value < low or (high is not None and value > high):
        raise RequestError(f"{name} must be an integer of at least {low}" + (f" and at most {high}" if high else ""))
    return value


def _request(props, allowed: set[str]) -> dict:
    props = {} if props is None else props
    if not isinstance(props, dict) or set(props) - allowed:
        raise RequestError("request has unsupported fields")
    return props


def parse_search_request(props) -> SearchQuery:
    allowed = {f.name for f in fields(SearchQuery)}
    props = _request(props, allowed)
    values = {}
    for name, value in props.items():
        if name == "cursor":
            values[name] = _text(value, name, _MAX_CURSOR)
        elif name == "page_size":
            values[name] = _integer(value, name, 1, MAX_PAGE_SIZE)
        elif name == "schema_version":
            values[name] = _integer(value, name, 0)
        elif name == "available":
            if not isinstance(value, bool):
                raise RequestError("available must be a boolean")
            values[name] = value
        else:
            values[name] = _text(value, name)
    return SearchQuery(**values)


class CatalogCommandModule(CommandModule):
    def __init__(self, manager: CatalogManager | None = None):
        self._manager = manager

    def _service(self, conn: Connection) -> CatalogManager:
        if self._manager is not None:
            return self._manager
        engine = conn.app_ctx
        runtime = getattr(engine, "run_manager", None)
        components = getattr(runtime, "components", {})
        # Resolve by type, not a deployment-specific component ID. As before,
        # the last configured catalog manager supplies the command service.
        for component in reversed(list(components.values())):
            if isinstance(component, CatalogManager):
                return component
        raise NotReady("the catalog service is not configured")

    def get_spec(self):
        def spec(name, description, handler):
            return CommandSpec(
                name=name,
                description=description,
                usage=name,
                handler_func=handler,
                authz_func=self._authorizer(),
                visible=False,
            )

        return CommandModuleSpec(
            name="fdn_catalog",
            cmd_specs=[
                spec(SEARCH, "search the federated data catalog", self.cmd_search),
                spec(STATUS, "show catalog read health per site", self.cmd_status),
            ],
        )

    def _authorizer(self):
        def authorize(conn: Connection, args: list[str]):
            study = conn.get_prop(ConnProps.ACTIVE_STUDY, DEFAULT_STUDY)
            role = conn.get_prop(ConnProps.USER_ROLE, "")
            registry = StudyRegistryService.get_registry()
            # The default Study has no membership (login does not check it); the authorization policy decides.
            member = (
                study == DEFAULT_STUDY
                or study in (conn.get_prop(ConnProps.CERT_STUDIES) or ())
                or (registry is not None and registry.has_user(conn.get_prop(ConnProps.USER_NAME, ""), study))
            )
            if member:
                # Study membership is checked here; the site's authorization policy then decides the right.
                return PreAuthzReturnCode.REQUIRE_AUTHZ
            conn.append_error(f"NOT_AUTHORIZED for {role}", meta=make_meta(MetaStatusValue.NOT_AUTHORIZED))
            return PreAuthzReturnCode.ERROR

        return authorize

    @staticmethod
    def _reply(conn: Connection, payload: dict):
        conn.append_dict(payload, meta=make_meta(MetaStatusValue.OK))

    def _error(self, conn: Connection, error_code: str, message: str):
        self._reply(conn, {"error_code": error_code, "message": message})

    def cmd_search(self, conn: Connection, args: list[str]):
        try:
            query = parse_search_request(conn.get_prop(ConnProps.CMD_PROPS))
            page = self._service(conn).search(conn.get_prop(ConnProps.ACTIVE_STUDY, DEFAULT_STUDY), query)
        except (RequestError, ValueError) as e:
            self._error(conn, "INVALID_ARGS", str(e))
            return
        except NotReady as e:
            self._error(conn, "SYSTEM_NOT_READY", str(e))
            return
        self._reply(
            conn,
            {
                "entries": [entry_to_dict(e) for e in page.entries],
                "next_cursor": page.next_cursor,
                "facets": page.facets,
                "total": page.total,
            },
        )

    def cmd_status(self, conn: Connection, args: list[str]):
        try:
            clients = self._service(conn).health(conn.get_prop(ConnProps.ACTIVE_STUDY, DEFAULT_STUDY))
        except NotReady as e:
            self._error(conn, "SYSTEM_NOT_READY", str(e))
            return
        self._reply(conn, {"clients": clients})
