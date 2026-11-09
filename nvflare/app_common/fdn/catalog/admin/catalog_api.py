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

"""The admin-side API of the catalog commands. It owns no connection or login; the Session carries the Study."""

from collections.abc import Iterator
from dataclasses import dataclass

from nvflare.app_common.fdn.catalog.server.commands import SEARCH, STATUS
from nvflare.app_common.fdn.catalog.server.search import CatalogEntry, SearchPage, entry_from_dict
from nvflare.fuel.flare_api.api_spec import (
    AuthenticationError,
    AuthorizationError,
    CommandError,
    InternalError,
    NoConnection,
    SessionClosed,
)
from nvflare.fuel.hci.client.api_status import APIStatus
from nvflare.fuel.hci.proto import MetaKey, MetaStatusValue, ProtoKey


def _raise_for_api_status(reply: dict, command: str) -> None:
    """Raise the session exception that matches a failed API status, as Session's own commands do.

    ``Session.do_command`` returns the raw reply, so a lost connection or an expired session arrives as a status
    with no catalog payload, and must not be mistaken for an empty or internal reply."""
    status = reply.get(ProtoKey.STATUS) if isinstance(reply, dict) else None
    if status is None or status == APIStatus.SUCCESS:
        return
    details = reply.get(ProtoKey.DETAILS, "")
    text = status.value if hasattr(status, "value") else str(status)
    if status in (APIStatus.ERROR_CERT, APIStatus.ERROR_AUTHENTICATION):
        raise AuthenticationError(details or f"user not authenticated: {text}", auth_code=reply.get("auth_code"))
    if status == APIStatus.ERROR_AUTHORIZATION:
        raise AuthorizationError(f"user not authorized for the action '{command}'")
    if status == APIStatus.ERROR_INACTIVE_SESSION:
        raise SessionClosed("the session is closed on server")
    if status == APIStatus.ERROR_SERVER_CONNECTION:
        raise NoConnection(f"cannot connect to server: {text}")
    if status == APIStatus.ERROR_SYNTAX:
        raise CommandError(
            error_code="CATALOG_UNAVAILABLE",
            message=f"the server does not recognize the command '{command}'",
            hint="Use a server that includes the catalog commands and has a CatalogManager configured.",
        )
    raise InternalError(f"{text}: {details}" if details else text)


_REJECTION_HINTS = {
    "SYSTEM_NOT_READY": "Configure a CatalogManager on the server, or retry after it has finished starting.",
}


@dataclass(frozen=True)
class ClientHealth:
    client: str
    health: str
    reason_code: str | None
    failures: int
    last_attempt_at: str | None
    last_success_at: str | None
    stale: bool


class CatalogAdminApi:
    def __init__(self, session):
        self._session = session

    def _call(self, command: str, request: dict) -> dict:
        reply = self._session.do_command(command, props=request)
        _raise_for_api_status(reply, command)
        payload = {}
        for item in reply.get(ProtoKey.DATA, []) if isinstance(reply, dict) else []:
            if isinstance(item, dict) and item.get(ProtoKey.TYPE) == ProtoKey.DICT:
                payload = item.get(ProtoKey.DATA, {})
                break
        if not isinstance(payload, dict):
            raise InternalError("catalog reply must be a dict")
        meta = reply.get(ProtoKey.META) if isinstance(reply, dict) else None
        status = (meta or {}).get(MetaKey.STATUS, MetaStatusValue.OK)
        if status != MetaStatusValue.OK:
            # the server refused the command (for example not authorized) before the handler ran
            raise CommandError(error_code=str(status).upper(), message=(meta or {}).get(MetaKey.INFO, ""))
        if not payload:
            raise InternalError("empty catalog reply")
        if payload.get("error_code"):
            code = payload["error_code"]
            raise CommandError(
                error_code=code,
                message=payload.get("message", ""),
                hint=_REJECTION_HINTS.get(code, ""),
                exit_code=4 if code == "INVALID_ARGS" else 1,
            )
        return payload

    def search_page(self, cursor: str | None = None, page_size: int = 50, **filters) -> SearchPage:
        request = {k: v for k, v in {**filters, "cursor": cursor, "page_size": page_size}.items() if v is not None}
        reply = self._call(SEARCH, request)
        try:
            return SearchPage(
                entries=tuple(entry_from_dict(e) for e in reply["entries"]),
                next_cursor=reply["next_cursor"],
                total=reply["total"],
                facets=reply["facets"],
            )
        except (KeyError, TypeError, ValueError) as e:
            raise InternalError(f"malformed catalog search reply: {e!r}") from e

    def search(self, **filters) -> Iterator[CatalogEntry]:
        cursor = None
        while True:
            page = self.search_page(cursor=cursor, **filters)
            yield from page.entries
            cursor = page.next_cursor
            if cursor is None:
                return

    def status(self) -> list[ClientHealth]:
        reply = self._call(STATUS, {})
        try:
            return [
                ClientHealth(
                    c["client"],
                    c["health"],
                    c["reason_code"],
                    c["failures"],
                    c["last_attempt_at"],
                    c["last_success_at"],
                    c["stale"],
                )
                for c in reply["clients"]
            ]
        except (KeyError, TypeError, ValueError) as e:
            raise InternalError(f"malformed catalog status reply: {e!r}") from e
