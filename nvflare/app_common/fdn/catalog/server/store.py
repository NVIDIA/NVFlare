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

"""The server's catalog state: one JSON file per (study, client).

No database. The client file is the unit of atomicity. All state is also held in memory, and every mutation
happens under `lock`, which callers hold across a read-modify-commit sequence.
"""

import copy
import json
import logging
import os
import re
import threading
from dataclasses import replace
from pathlib import Path

from nvflare.app_common.fdn.catalog.common.schema import CatalogInfo
from nvflare.app_common.fdn.catalog.server.fileio import write_atomically
from nvflare.app_common.fdn.catalog.server.search import CatalogEntry, SearchPage, SearchQuery, run_search

FORMAT = 2
HEALTH_UNKNOWN = "UNKNOWN"
STORE_DIR = "fdn"  # under the server workspace; only the manager reads and writes it
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def new_client_state() -> dict:
    return {
        "format": FORMAT,
        "client": {
            "health": HEALTH_UNKNOWN,
            "reason_code": None,
            "failures": 0,
            "last_attempt_at": None,
            "last_success_at": None,
        },
        "datasets": {},
    }


def _is(value, *types) -> bool:
    return type(value) in types  # exact types: a bool is not a count


# the type of every persisted field; None is allowed where it is listed
_CLIENT_FIELDS = {
    "health": (str,),
    "reason_code": (str, type(None)),
    "failures": (int,),
    "last_attempt_at": (str, type(None)),
    "last_success_at": (str, type(None)),
}
_DATASET_FIELDS = {
    "metadata": (dict,),
    "metadata_digest": (str,),
    "catalog_version": (int,),
    "schema_version": (int,),
    "observed_at": (str, type(None)),
    "published_at": (str, type(None)),
    "available": (bool,),
    "last_checked_at": (str, type(None)),
    "withheld": (bool,),
}


def _valid_row(name, row) -> bool:
    if not isinstance(name, str) or not isinstance(row, dict) or set(row) != set(_DATASET_FIELDS):
        return False
    if not all(_is(row[k], *types) for k, types in _DATASET_FIELDS.items()):
        return False
    try:  # the metadata must be something every reader can build a catalog from
        CatalogInfo.from_wire({"dataset": name, "metadata": row["metadata"], "observed_at": row["observed_at"]})
    except Exception:
        return False
    return True


def _valid_state(state) -> bool:
    """The persisted structure every reader relies on, field by field; anything else is rebuilt, not trusted, so one
    damaged client file cannot break a search of the Study."""
    try:
        return (
            state["format"] == FORMAT
            and set(state["client"]) == set(_CLIENT_FIELDS)
            and all(_is(state["client"][k], *types) for k, types in _CLIENT_FIELDS.items())
            and isinstance(state.get("applied_jobs", []), list)
            and all(isinstance(job, str) and job for job in state.get("applied_jobs", []))
            and len(set(state.get("applied_jobs", []))) == len(state.get("applied_jobs", []))
            and isinstance(state["datasets"], dict)
            and all(_valid_row(n, r) for n, r in state["datasets"].items())
        )
    except (KeyError, TypeError):
        return False


def store_root(workspace_root: str) -> Path:
    return Path(workspace_root) / STORE_DIR / "catalog"


_KEY_FILE = "manifest.key"
_KEY_BYTES = 32


def manifest_key(workspace_root: str) -> bytes | None:
    """The server's master key for signing manifests, created on first use. It never leaves the manager: a job gets
    only the key derived for it. A missing or damaged key file is replaced whole, never left half-written, so a crash
    cannot disable the catalog; jobs in flight then fail their signature check once. None if it cannot be written."""
    path = Path(workspace_root) / STORE_DIR / _KEY_FILE
    try:
        key = path.read_bytes()
        if len(key) == _KEY_BYTES:
            return key
        _logger.warning("the catalog manifest key was damaged and is replaced")
    except FileNotFoundError:
        pass
    except OSError:
        return None
    key = os.urandom(_KEY_BYTES)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(key)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except OSError:
        return None
    return key


def _check_name(value: str) -> str:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        raise ValueError("invalid study or client name")
    return value


_logger = logging.getLogger(__name__)


class CatalogStore:
    def __init__(self, workspace_root: str):
        self._root = store_root(workspace_root)
        self._root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self._clients: dict[tuple[str, str], dict] = {}
        self._entry_cache: dict[tuple[str, str], tuple[dict, tuple[CatalogEntry, ...]]] = {}
        self._load()

    # ---- loading -------------------------------------------------------------------------------------------

    def _load(self) -> None:
        for study_dir in sorted(p for p in self._root.iterdir() if p.is_dir() and _SAFE_NAME.fullmatch(p.name)):
            study = study_dir.name
            for tmp in study_dir.glob("*.tmp"):
                try:
                    tmp.unlink()
                except OSError as error:  # a leftover partial write must not keep the service from starting
                    _logger.warning(f"could not remove {tmp.name}: {type(error).__name__}")
            for path in sorted(study_dir.glob("*.json")):
                client = path.stem
                if not _SAFE_NAME.fullmatch(client):
                    continue
                try:
                    state = json.loads(path.read_text(encoding="utf-8"))
                    if not _valid_state(state):
                        raise ValueError
                except (OSError, ValueError, KeyError, TypeError):
                    kept = self._keep_aside(path)
                    _logger.error(
                        f"catalog state for a client was unreadable and starts empty: {client}; the file is kept as {kept}"
                    )
                    state = new_client_state()
                self._clients[(study, client)] = state

    @staticmethod
    def _keep_aside(path: Path) -> str:
        """The next commit replaces the client's file; keep the damaged one for whoever has to find out why."""
        target = path.with_name(path.name + ".damaged")
        try:
            os.replace(path, target)
        except OSError:
            return "in place (it could not be moved)"
        return target.name

    # ---- state access --------------------------------------------------------------------------------------

    def client_state(self, study: str, client: str) -> dict:
        """A copy of the client's state, safe to modify and hand back to commit()."""
        with self.lock:
            return copy.deepcopy(self._clients.get((_check_name(study), _check_name(client))) or new_client_state())

    def indexed(self, study: str) -> dict[str, list[str]]:
        """The dataset names held for each client of the Study."""
        with self.lock:
            return {c: sorted(self._clients[(s, c)]["datasets"]) for s, c in self._clients if s == study}

    def commit(self, study: str, client: str, state: dict) -> None:
        """Atomically replace the client's file, then the in-memory copy. A failed replace changes neither."""
        study_dir = self._root / _check_name(study)
        client = _check_name(client)
        with self.lock:
            study_dir.mkdir(exist_ok=True)
            write_atomically(str(study_dir / f"{client}.json"), state)
            self._clients[(study, client)] = copy.deepcopy(state)
            self._entry_cache.pop((study, client), None)

    def commit_application(
        self,
        study: str,
        client: str,
        state: dict,
        application_id: str | None = None,
        pending_applications: set[str] | None = None,
    ) -> None:
        """Commit the update and its delivery receipt together; retain receipts only for outstanding results."""
        with self.lock:
            if application_id is not None:
                receipts = set(self.client_state(study, client).get("applied_jobs", []))
                if application_id in receipts:
                    return
                outstanding = receipts if pending_applications is None else pending_applications
                state["applied_jobs"] = sorted((receipts & outstanding) | {application_id})
            self.commit(study, client, state)

    # ---- results -------------------------------------------------------------------------------------------

    def record_no_result(
        self,
        study: str,
        client: str,
        reason: str,
        now: str,
        application_id: str | None = None,
        pending_applications: set[str] | None = None,
    ) -> None:
        """A job that produced nothing for the client. Availability and check times are untouched."""
        with self.lock:
            state = self.client_state(study, client)
            c = state["client"]
            c.update(health="FAILED", reason_code=reason, failures=c["failures"] + 1)
            c.update(last_attempt_at=now)
            self.commit_application(study, client, state, application_id, pending_applications)

    # ---- queries -------------------------------------------------------------------------------------------

    def _entries(self, study: str) -> list[CatalogEntry]:
        with self.lock:
            snapshot = [
                (key, state, self._entry_cache.get(key)) for key, state in self._clients.items() if key[0] == study
            ]
        entries = []
        for key, state, cached in snapshot:
            if cached is not None and cached[0] is state:
                entries.extend(cached[1])
                continue
            client_entries = self._client_entries(key[1], state)
            with self.lock:
                if self._clients.get(key) is state:
                    self._entry_cache[key] = (state, client_entries)
            entries.extend(client_entries)
        return entries

    @staticmethod
    def _client_entries(client: str, state: dict) -> tuple[CatalogEntry, ...]:
        # Committed states are replaced whole and never mutated; this snapshot is safe outside the lock.
        entries = []
        withheld = {n for n, r in state["datasets"].items() if r["withheld"]}
        for name, row in state["datasets"].items():
            if row["withheld"]:
                continue
            catalog = None
            if row["available"]:
                catalog = CatalogInfo.from_wire(
                    {"dataset": name, "metadata": row["metadata"], "observed_at": row["observed_at"]}
                )
                if any(x.dataset in withheld for x in catalog.lineage):
                    lineage = tuple(x for x in catalog.lineage if x.dataset not in withheld)
                    catalog = replace(catalog, lineage=lineage)
            entries.append(
                CatalogEntry(
                    client=client,
                    dataset=name,
                    available=row["available"],
                    last_checked_at=row["last_checked_at"],
                    catalog=catalog,
                    catalog_version=row["catalog_version"] if row["available"] else None,
                    schema_version=row["schema_version"] if row["available"] else None,
                    published_at=row["published_at"] if row["available"] else None,
                )
            )
        return tuple(entries)

    def search(self, study: str, query: SearchQuery) -> SearchPage:
        return run_search(study, self._entries(study), query)

    def health(self, study: str) -> list[dict]:
        with self.lock:
            return [
                {"client": client, **state["client"]}
                for (s, client), state in sorted(self._clients.items())
                if s == study
            ]
