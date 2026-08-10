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

import errno
import json
import os
import shutil
import stat
import tempfile
import time
import uuid
from typing import Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - shared-file Attach is a POSIX/HPC transport
    fcntl = None

from nvflare.apis.fl_constant import ConnectionSecurity
from nvflare.client.cell.attach import make_attach_trainer_fqcn, validate_attach_id, validate_attach_profile
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.file_driver import CONNS_DIR as SHARED_FILE_CONNS_DIR
from nvflare.fuel.f3.drivers.file_driver import LISTENER_PREFIX as SHARED_FILE_LISTENER_PREFIX
from nvflare.fuel.f3.drivers.file_driver import OWNER_MARKER as SHARED_FILE_OWNER_MARKER
from nvflare.fuel.f3.drivers.file_driver import SCHEME as SHARED_FILE_SCHEME
from nvflare.fuel.f3.drivers.file_driver import parse_file_url

ATTACH_COMM_CONFIG = "client_api_attach"
ATTACH_ENDPOINT_SCHEMA_VERSION = 1
ATTACH_RENDEZVOUS_SUBDIR = ".nvflare/client_api_attach"
ATTACH_ENDPOINT_FILE = "endpoint.json"
ATTACH_OWNER_FILE = "owner.json"
ATTACH_CLAIM_LOCK_SUFFIX = ".lock"
ATTACH_RENDEZVOUS_DIR_MODE = 0o770
ATTACH_RENDEZVOUS_FILE_MODE = 0o660
_WAIT_INTERVAL = 0.2


class AttachEndpointOwnershipError(RuntimeError):
    """The publisher's process-held claim no longer matches its claim directory."""


class AttachRendezvousCancelled(RuntimeError):
    """A trainer stopped while waiting for a shared-file endpoint."""


class AttachEndpointKey:
    SCHEMA_VERSION = "schema_version"
    INSTANCE_ID = "instance_id"
    ATTACH_ID = "attach_id"
    SITE_NAME = "site_name"
    CJ_FQCN = "cj_fqcn"
    TRAINER_FQCN = "trainer_fqcn"
    CONNECT_URL = "connect_url"
    CONNECTION_SECURITY = "connection_security"


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


def _read_json_strict(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise AttachEndpointOwnershipError(f"attach rendezvous file does not contain an object: {path}")
    return data


def _validate_real_dir(path: str, *, private: bool, label: str) -> None:
    value = os.lstat(path)
    if not stat.S_ISDIR(value.st_mode):
        raise RuntimeError(f"{label} is not a real directory: {path}")
    mode = stat.S_IMODE(value.st_mode)
    if private and mode & 0o007:
        raise RuntimeError(f"{label} grants access to other users: {path}")
    if not private and mode & stat.S_IWOTH:
        raise RuntimeError(f"{label} is world-writable: {path}")


def _validate_private_file(path: str, label: str) -> None:
    value = os.lstat(path)
    if not stat.S_ISREG(value.st_mode):
        raise RuntimeError(f"{label} is not a regular file: {path}")
    if stat.S_IMODE(value.st_mode) & 0o007:
        raise RuntimeError(f"{label} grants access to other users: {path}")


def _ensure_private_dir(path: str, mode: int) -> None:
    try:
        os.mkdir(path, mode)
        os.chmod(path, mode)
    except FileExistsError:
        pass
    _validate_real_dir(path, private=True, label="attach rendezvous path")


def _ensure_rendezvous_parent(root_dir: str, site_name: str) -> str:
    _validate_real_dir(root_dir, private=False, label="shared-file root")
    current = root_dir
    for segment in (".nvflare", "client_api_attach", site_name):
        current = os.path.join(current, segment)
        _ensure_private_dir(current, ATTACH_RENDEZVOUS_DIR_MODE)
    return current


def _claim_lock_path(parent: str, attach_id: str) -> str:
    return os.path.join(parent, f".{attach_id}{ATTACH_CLAIM_LOCK_SUFFIX}")


def _acquire_claim_lock(parent: str, attach_id: str) -> int:
    """Acquire a process-held cross-node claim lock or reject an existing live owner."""
    if fcntl is None:
        raise RuntimeError("shared-file Attach requires POSIX advisory lock support")
    path = _claim_lock_path(parent, attach_id)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, ATTACH_RENDEZVOUS_FILE_MODE)
    try:
        os.fchmod(fd, ATTACH_RENDEZVOUS_FILE_MODE)
        value = os.fstat(fd)
        if not stat.S_ISREG(value.st_mode):
            raise RuntimeError(f"attach claim lock is not a regular file: {path}")
        if stat.S_IMODE(value.st_mode) & 0o007:
            raise RuntimeError(f"attach claim lock grants access to other users: {path}")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EAGAIN):
                raise AttachEndpointOwnershipError(f"attach_id {attach_id!r} is already claimed by a live CJ") from e
            raise RuntimeError(f"shared-file Attach requires working cross-node advisory locks for {path}: {e}") from e
        return fd
    except BaseException:
        os.close(fd)
        raise


def _release_claim_lock(fd: Optional[int]) -> None:
    if fd is None:
        return
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


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
        self._claim_lock_fd: Optional[int] = None
        self._claim()

    def _claim(self) -> None:
        parent = _ensure_rendezvous_parent(self.root_dir, self.site_name)
        self._claim_lock_fd = _acquire_claim_lock(parent, self.attach_id)
        try:
            if os.path.lexists(self.claim_dir):
                _validate_real_dir(self.claim_dir, private=True, label="attach claim path")
                tombstone = os.path.join(parent, f".{self.attach_id}.stale-{uuid.uuid4().hex}")
                os.rename(self.claim_dir, tombstone)
                _remove_owned_tree(tombstone)
            os.mkdir(self.claim_dir, ATTACH_RENDEZVOUS_DIR_MODE)
            os.chmod(self.claim_dir, ATTACH_RENDEZVOUS_DIR_MODE)
            _atomic_write_json(
                os.path.join(self.claim_dir, ATTACH_OWNER_FILE),
                {AttachEndpointKey.INSTANCE_ID: self.instance_id},
                ATTACH_RENDEZVOUS_FILE_MODE,
            )
        except BaseException:
            _remove_owned_tree(self.claim_dir)
            _release_claim_lock(self._claim_lock_fd)
            self._claim_lock_fd = None
            raise

    def _assert_owner(self) -> None:
        try:
            owner = _read_json_strict(os.path.join(self.claim_dir, ATTACH_OWNER_FILE))
        except json.JSONDecodeError as e:
            raise AttachEndpointOwnershipError(f"attach rendezvous owner record is corrupt: {self.claim_dir}") from e
        if owner.get(AttachEndpointKey.INSTANCE_ID) != self.instance_id:
            raise AttachEndpointOwnershipError(f"lost ownership of attach rendezvous claim {self.claim_dir}")

    def publish(self, cj_fqcn: str, trainer_fqcn: str, connect_url: str, connection_security: str) -> None:
        if self._closed:
            raise AttachEndpointOwnershipError("cannot publish a closed attach rendezvous claim")
        record = {
            AttachEndpointKey.SCHEMA_VERSION: ATTACH_ENDPOINT_SCHEMA_VERSION,
            AttachEndpointKey.INSTANCE_ID: self.instance_id,
            AttachEndpointKey.ATTACH_ID: self.attach_id,
            AttachEndpointKey.SITE_NAME: self.site_name,
            AttachEndpointKey.CJ_FQCN: cj_fqcn,
            AttachEndpointKey.TRAINER_FQCN: trainer_fqcn,
            AttachEndpointKey.CONNECT_URL: connect_url,
            AttachEndpointKey.CONNECTION_SECURITY: connection_security,
        }
        _validate_endpoint_record(record, self.site_name, self.attach_id)
        validate_shared_file_listener(self.root_dir, connect_url)
        self._assert_owner()
        _atomic_write_json(
            os.path.join(self.claim_dir, ATTACH_ENDPOINT_FILE),
            record,
            ATTACH_RENDEZVOUS_FILE_MODE,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            owner = _read_json(os.path.join(self.claim_dir, ATTACH_OWNER_FILE))
            if owner and owner.get(AttachEndpointKey.INSTANCE_ID) == self.instance_id:
                tombstone = f"{self.claim_dir}.closing-{self.instance_id}"
                try:
                    os.rename(self.claim_dir, tombstone)
                except FileNotFoundError:
                    pass
                else:
                    _remove_owned_tree(tombstone)
        finally:
            _release_claim_lock(self._claim_lock_fd)
            self._claim_lock_fd = None


def _existing_rendezvous_tree_is_safe(root_dir: str, site_name: str, attach_id: str) -> bool:
    """Validate every existing discovery component; return False while trainer-first paths are absent."""
    site_dir = os.path.join(root_dir, ATTACH_RENDEZVOUS_SUBDIR, site_name)
    paths = (
        (root_dir, False, "shared-file root"),
        (os.path.join(root_dir, ".nvflare"), True, "attach rendezvous path"),
        (os.path.join(root_dir, ".nvflare", "client_api_attach"), True, "attach rendezvous path"),
        (site_dir, True, "attach rendezvous path"),
        (attach_claim_dir(root_dir, site_name, attach_id), True, "attach claim path"),
    )
    for path, private, label in paths:
        try:
            _validate_real_dir(path, private=private, label=label)
        except FileNotFoundError:
            return False
    try:
        _validate_private_file(_claim_lock_path(site_dir, attach_id), "attach claim lock")
    except FileNotFoundError:
        return False
    return True


def validate_shared_file_listener(root_dir: str, connect_url: str) -> str:
    """Validate a FileDriver listener as a concrete endpoint inside the configured trust root."""
    root_dir = _validate_root_dir(root_dir)
    _validate_real_dir(root_dir, private=False, label="shared-file root")
    try:
        listener_dir = os.path.abspath(parse_file_url(connect_url))
    except CommError as e:
        raise ValueError(str(e)) from e
    if os.path.dirname(listener_dir) != root_dir or not os.path.basename(listener_dir).startswith(
        SHARED_FILE_LISTENER_PREFIX
    ):
        raise RuntimeError(f"shared-file Attach listener must be an immediate child of configured root {root_dir}")
    _validate_real_dir(listener_dir, private=True, label="shared-file listener")
    _validate_real_dir(
        os.path.join(listener_dir, SHARED_FILE_CONNS_DIR),
        private=True,
        label="shared-file listener connection path",
    )
    _validate_private_file(os.path.join(listener_dir, SHARED_FILE_OWNER_MARKER), "shared-file listener owner marker")
    return listener_dir


def _claim_has_live_publisher(root_dir: str, site_name: str, attach_id: str) -> bool:
    """Return whether a CJ still owns the stable, process-held attach lock."""
    if fcntl is None:
        raise RuntimeError("shared-file Attach requires POSIX advisory lock support")
    site_dir = os.path.join(root_dir, ATTACH_RENDEZVOUS_SUBDIR, site_name)
    path = _claim_lock_path(site_dir, attach_id)
    flags = os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        return False
    except OSError as e:
        if e.errno in (errno.EACCES, errno.ELOOP, errno.EPERM):
            raise RuntimeError(f"cannot safely open attach claim lock {path}: {e}") from e
        raise
    try:
        value = os.fstat(fd)
        if not stat.S_ISREG(value.st_mode):
            raise RuntimeError(f"attach claim lock is not a regular file: {path}")
        if stat.S_IMODE(value.st_mode) & 0o007:
            raise RuntimeError(f"attach claim lock grants access to other users: {path}")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EAGAIN):
                return True
            if e.errno in (errno.ENOSYS, errno.EOPNOTSUPP):
                raise RuntimeError(
                    f"shared-file Attach requires working cross-node advisory locks for {path}: {e}"
                ) from e
            raise
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


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
    cj_fqcn = record[AttachEndpointKey.CJ_FQCN]
    cp_fqcn = FQCN.get_parent(cj_fqcn)
    if (
        not cp_fqcn
        or FQCN.split(cp_fqcn)[-1] != site_name
        or len(FQCN.split(cj_fqcn)) != len(FQCN.split(cp_fqcn)) + 1
        or FQCN.validate(cj_fqcn)
    ):
        raise ValueError("attach endpoint record has invalid CJ FQCN")
    expected_trainer_fqcn = make_attach_trainer_fqcn(cp_fqcn, attach_id)
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


def _read_valid_endpoint(root_dir: str, site_name: str, attach_id: str) -> Optional[dict]:
    if not _existing_rendezvous_tree_is_safe(root_dir, site_name, attach_id):
        return None
    claim_dir = attach_claim_dir(root_dir, site_name, attach_id)
    owner_path = os.path.join(claim_dir, ATTACH_OWNER_FILE)
    endpoint_path = os.path.join(claim_dir, ATTACH_ENDPOINT_FILE)
    try:
        _validate_private_file(owner_path, "attach rendezvous owner")
        _validate_private_file(endpoint_path, "attach rendezvous endpoint")
    except FileNotFoundError:
        return None
    owner = _read_json(owner_path)
    record = _read_json(endpoint_path)
    if not owner or not record:
        return None
    record = _validate_endpoint_record(record, site_name, attach_id)
    instance_id = record[AttachEndpointKey.INSTANCE_ID]
    if owner.get(AttachEndpointKey.INSTANCE_ID) != instance_id:
        return None
    validate_shared_file_listener(root_dir, record[AttachEndpointKey.CONNECT_URL])
    return record if _claim_has_live_publisher(root_dir, site_name, attach_id) else None


def wait_for_attach_endpoint(
    root_dir: str,
    site_name: str,
    attach_id: str,
    timeout: Optional[float],
    stop_event=None,
) -> dict:
    """Wait for an endpoint whose publisher still holds the cross-node claim lock."""
    root_dir = _validate_root_dir(root_dir)
    site_name = _validate_site_name(site_name)
    attach_id = validate_attach_id(attach_id)
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        if stop_event is not None and stop_event.is_set():
            raise AttachRendezvousCancelled("attach rendezvous wait was stopped")
        try:
            record = _read_valid_endpoint(root_dir, site_name, attach_id)
        except OSError:
            # Network filesystems can transiently fail metadata reads. The
            # caller's local deadline still bounds this retry loop.
            record = None
        now = time.monotonic()
        if record:
            return record
        if deadline is not None and now >= deadline:
            raise TimeoutError(
                f"no live attach endpoint for site={site_name!r} attach_id={attach_id!r} "
                f"within job_wait_timeout={timeout}s"
            )
        remaining = None if deadline is None else max(0.0, deadline - now)
        wait_time = _WAIT_INTERVAL if remaining is None else min(_WAIT_INTERVAL, remaining)
        if stop_event is not None:
            stop_event.wait(wait_time)
        else:
            time.sleep(wait_time)
