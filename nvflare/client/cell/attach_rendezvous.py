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

"""Shared-filesystem discovery for a CJ-owned Client API Attach listener."""

import json
import os
import shutil
import stat
import tempfile
import time
import uuid
from typing import Optional

from nvflare.apis.fl_constant import ConnectionSecurity
from nvflare.client.cell.attach import make_attach_trainer_fqcn, validate_attach_id, validate_attach_profile
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.drivers.file_driver import SCHEME as SHARED_FILE_SCHEME

ATTACH_COMM_CONFIG = "client_api_attach"
ATTACH_ENDPOINT_SCHEMA_VERSION = 1
ATTACH_RENDEZVOUS_SUBDIR = ".nvflare/client_api_attach"
ATTACH_ENDPOINT_FILE = "endpoint.json"
ATTACH_OWNER_FILE = "owner.json"
ATTACH_LEASE_FILE = "lease"
ATTACH_RENDEZVOUS_DIR_MODE = 0o770
ATTACH_RENDEZVOUS_FILE_MODE = 0o660
ATTACH_RENDEZVOUS_LEASE_INTERVAL = 5.0
ATTACH_RENDEZVOUS_LEASE_TIMEOUT = 30.0
_WAIT_INTERVAL = 0.2


class AttachEndpointKey:
    SCHEMA_VERSION = "schema_version"
    INSTANCE_ID = "instance_id"
    ATTACH_ID = "attach_id"
    SITE_NAME = "site_name"
    CJ_FQCN = "cj_fqcn"
    TRAINER_FQCN = "trainer_fqcn"
    CONNECT_URL = "connect_url"
    CONNECTION_SECURITY = "connection_security"
    LEASE_TIMEOUT = "lease_timeout"


def _validate_site_name(site_name: str) -> str:
    if not isinstance(site_name, str) or not site_name or len(FQCN.split(site_name)) != 1 or FQCN.validate(site_name):
        raise ValueError(f"site_name must be one valid FQCN segment, but got {site_name!r}")
    return site_name


def _validate_root_dir(root_dir: str) -> str:
    if not isinstance(root_dir, str) or not root_dir:
        raise ValueError("attach rendezvous_dir must be a non-empty string")
    if not os.path.isabs(root_dir):
        raise ValueError(f"attach rendezvous_dir must be absolute, but got {root_dir!r}")
    return os.path.abspath(root_dir)


def attach_claim_dir(root_dir: str, site_name: str, attach_id: str) -> str:
    """Return the deterministic claim directory for one site/attach ID."""
    root_dir = _validate_root_dir(root_dir)
    site_name = _validate_site_name(site_name)
    attach_id = validate_attach_id(attach_id)
    return os.path.join(root_dir, ATTACH_RENDEZVOUS_SUBDIR, site_name, f"{attach_id}.claim")


def _atomic_write_json(path: str, data: dict, file_mode: int) -> None:
    parent = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=parent, prefix=".attach-", suffix=".tmp")
    fd_owned = True
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, file_mode)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fd_owned = False
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
    except BaseException:
        if fd_owned:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
        raise


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _ensure_private_dir(path: str, mode: int) -> None:
    try:
        os.mkdir(path, mode)
        os.chmod(path, mode)
    except FileExistsError:
        pass
    value = os.lstat(path)
    if not stat.S_ISDIR(value.st_mode):
        raise RuntimeError(f"attach rendezvous path is not a real directory: {path}")
    if stat.S_IMODE(value.st_mode) & 0o007:
        raise RuntimeError(f"attach rendezvous directory grants access to other users: {path}")


def _ensure_rendezvous_parent(root_dir: str, site_name: str) -> str:
    root_value = os.lstat(root_dir)
    if not stat.S_ISDIR(root_value.st_mode):
        raise RuntimeError(f"shared-file root is not a real directory: {root_dir}")
    if stat.S_IMODE(root_value.st_mode) & stat.S_IWOTH:
        raise RuntimeError(f"shared-file root is world-writable: {root_dir}")

    current = root_dir
    for segment in (".nvflare", "client_api_attach", site_name):
        current = os.path.join(current, segment)
        _ensure_private_dir(current, ATTACH_RENDEZVOUS_DIR_MODE)
    return current


def _claim_is_stale(claim_dir: str) -> bool:
    try:
        value = os.lstat(claim_dir)
        if not stat.S_ISDIR(value.st_mode):
            raise RuntimeError(f"attach claim path is not a real directory: {claim_dir}")
        lease_mtime = os.stat(os.path.join(claim_dir, ATTACH_LEASE_FILE)).st_mtime
    except FileNotFoundError:
        try:
            lease_mtime = os.lstat(claim_dir).st_mtime
        except FileNotFoundError:
            return True
    return time.time() - lease_mtime > ATTACH_RENDEZVOUS_LEASE_TIMEOUT


def _remove_owned_tree(path: str) -> None:
    """Remove one exact, internally minted claim/tombstone directory."""
    shutil.rmtree(path, ignore_errors=True)


class AttachEndpointPublisher:
    """Own one attach-ID claim and publish a live listener endpoint within it."""

    def __init__(self, root_dir: str, site_name: str, attach_id: str):
        self.root_dir = _validate_root_dir(root_dir)
        self.site_name = _validate_site_name(site_name)
        self.attach_id = validate_attach_id(attach_id)
        self.instance_id = uuid.uuid4().hex
        self.claim_dir = attach_claim_dir(self.root_dir, self.site_name, self.attach_id)
        self._closed = False
        self._claim()

    def _claim(self) -> None:
        parent = _ensure_rendezvous_parent(self.root_dir, self.site_name)
        for _ in range(3):
            try:
                os.mkdir(self.claim_dir, ATTACH_RENDEZVOUS_DIR_MODE)
                os.chmod(self.claim_dir, ATTACH_RENDEZVOUS_DIR_MODE)
                break
            except FileExistsError:
                if not _claim_is_stale(self.claim_dir):
                    raise RuntimeError(
                        f"attach_id {self.attach_id!r} is already claimed by a live CJ at site {self.site_name!r}"
                    )
                tombstone = os.path.join(parent, f".{self.attach_id}.stale-{uuid.uuid4().hex}")
                try:
                    os.rename(self.claim_dir, tombstone)
                except FileNotFoundError:
                    continue
                _remove_owned_tree(tombstone)
        else:
            raise RuntimeError(f"could not claim attach_id {self.attach_id!r}")

        try:
            _atomic_write_json(
                os.path.join(self.claim_dir, ATTACH_OWNER_FILE),
                {AttachEndpointKey.INSTANCE_ID: self.instance_id},
                ATTACH_RENDEZVOUS_FILE_MODE,
            )
            self.touch()
        except BaseException:
            _remove_owned_tree(self.claim_dir)
            raise

    def publish(self, cj_fqcn: str, trainer_fqcn: str, connect_url: str, connection_security: str) -> None:
        record = {
            AttachEndpointKey.SCHEMA_VERSION: ATTACH_ENDPOINT_SCHEMA_VERSION,
            AttachEndpointKey.INSTANCE_ID: self.instance_id,
            AttachEndpointKey.ATTACH_ID: self.attach_id,
            AttachEndpointKey.SITE_NAME: self.site_name,
            AttachEndpointKey.CJ_FQCN: cj_fqcn,
            AttachEndpointKey.TRAINER_FQCN: trainer_fqcn,
            AttachEndpointKey.CONNECT_URL: connect_url,
            AttachEndpointKey.CONNECTION_SECURITY: connection_security,
            AttachEndpointKey.LEASE_TIMEOUT: ATTACH_RENDEZVOUS_LEASE_TIMEOUT,
        }
        _validate_endpoint_record(record, self.site_name, self.attach_id)
        _atomic_write_json(
            os.path.join(self.claim_dir, ATTACH_ENDPOINT_FILE),
            record,
            ATTACH_RENDEZVOUS_FILE_MODE,
        )
        self.touch()

    def touch(self) -> None:
        if self._closed:
            return
        owner = _read_json(os.path.join(self.claim_dir, ATTACH_OWNER_FILE))
        if not owner or owner.get(AttachEndpointKey.INSTANCE_ID) != self.instance_id:
            raise RuntimeError(f"lost ownership of attach rendezvous claim {self.claim_dir}")
        lease_path = os.path.join(self.claim_dir, ATTACH_LEASE_FILE)
        try:
            os.utime(lease_path, None)
        except FileNotFoundError:
            with open(lease_path, "ab"):
                pass
            os.chmod(lease_path, ATTACH_RENDEZVOUS_FILE_MODE)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        owner = _read_json(os.path.join(self.claim_dir, ATTACH_OWNER_FILE))
        if not owner or owner.get(AttachEndpointKey.INSTANCE_ID) != self.instance_id:
            return
        tombstone = f"{self.claim_dir}.closing-{self.instance_id}"
        try:
            os.rename(self.claim_dir, tombstone)
        except FileNotFoundError:
            return
        _remove_owned_tree(tombstone)


def _claim_is_live(claim_dir: str, record: dict) -> bool:
    timeout = record.get(AttachEndpointKey.LEASE_TIMEOUT)
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        return False
    try:
        mtime = os.stat(os.path.join(claim_dir, ATTACH_LEASE_FILE)).st_mtime
    except OSError:
        return False
    return time.time() - mtime <= timeout


def _validate_endpoint_record(record: dict, site_name: str, attach_id: str) -> dict:
    if record.get(AttachEndpointKey.SCHEMA_VERSION) != ATTACH_ENDPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported attach endpoint record schema")
    expected = {
        AttachEndpointKey.SITE_NAME: site_name,
        AttachEndpointKey.ATTACH_ID: attach_id,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"attach endpoint record {key!r} mismatch: expected {value!r}")
    for key in (
        AttachEndpointKey.INSTANCE_ID,
        AttachEndpointKey.CJ_FQCN,
        AttachEndpointKey.CONNECT_URL,
    ):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"attach endpoint record requires non-empty {key!r}")
    cj_path = FQCN.split(record[AttachEndpointKey.CJ_FQCN])
    if len(cj_path) != 2 or cj_path[0] != site_name or FQCN.validate(record[AttachEndpointKey.CJ_FQCN]):
        raise ValueError("attach endpoint record has invalid CJ FQCN")
    expected_trainer_fqcn = make_attach_trainer_fqcn(record[AttachEndpointKey.CJ_FQCN], attach_id)
    if record.get(AttachEndpointKey.TRAINER_FQCN) != expected_trainer_fqcn:
        raise ValueError(
            f"attach endpoint record {AttachEndpointKey.TRAINER_FQCN!r} mismatch: "
            f"expected {expected_trainer_fqcn!r}"
        )
    security = validate_attach_profile(
        record[AttachEndpointKey.CONNECT_URL],
        record.get(AttachEndpointKey.CONNECTION_SECURITY),
    )
    if security != ConnectionSecurity.CLEAR:
        raise ValueError("shared-file rendezvous may publish only a clear non-network endpoint")
    if not record[AttachEndpointKey.CONNECT_URL].startswith(f"{SHARED_FILE_SCHEME}://"):
        raise ValueError("shared-file rendezvous published a non-shared-file endpoint")
    return dict(record)


def wait_for_attach_endpoint(
    root_dir: str,
    site_name: str,
    attach_id: str,
    timeout: Optional[float],
) -> dict:
    """Wait for a live endpoint record and return its validated contents."""
    claim_dir = attach_claim_dir(root_dir, site_name, attach_id)
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        record = _read_json(os.path.join(claim_dir, ATTACH_ENDPOINT_FILE))
        if record and _claim_is_live(claim_dir, record):
            return _validate_endpoint_record(record, site_name, attach_id)
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(
                f"no live attach endpoint for site={site_name!r} attach_id={attach_id!r} "
                f"within job_wait_timeout={timeout}s"
            )
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        time.sleep(_WAIT_INTERVAL if remaining is None else min(_WAIT_INTERVAL, remaining))
