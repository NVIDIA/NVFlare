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

import datetime
import hashlib
import importlib
import json
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from cryptography.hazmat.primitives import serialization

from nvflare.fuel.sec.admin_cert import validate_admin_leaf_cert
from nvflare.lighter.utils import load_crt, load_crt_chain, load_private_key_file, verify_cert_chain

EPHEMERAL_ADMIN_CERT_PROVIDER_CONFIG_KEY = "provider_config"
EPHEMERAL_ADMIN_CERT_PROVIDER_KEY = "provider"
EPHEMERAL_ADMIN_CERT_CACHE_DIR = "ephemeral_admin_certs"
EPHEMERAL_ADMIN_CERT_CLIENT_CERT = "client.crt"
EPHEMERAL_ADMIN_CERT_CLIENT_KEY = "client.key"
EPHEMERAL_ADMIN_CERT_CACHE_LOCK = ".lock"
BUILTIN_EPHEMERAL_ADMIN_CERT_PROVIDERS = {
    "step_ca": "nvflare.fuel.sec.step_ca_admin_cert:obtain_step_ca_admin_cert_files",
}


class EphemeralAdminCertError(ValueError):
    """Raised when short-lived admin certificate acquisition fails."""


@dataclass
class EphemeralAdminCertFiles:
    client_key: str
    client_cert: str
    expires_at: float = 0.0
    temp_dir: Optional[tempfile.TemporaryDirectory] = field(default=None, repr=False)

    def needs_renewal(self, renewal_window: float = 60.0, now: Optional[float] = None) -> bool:
        if not self.expires_at:
            return True
        now = time.time() if now is None else now
        return self.expires_at - now <= max(0.0, renewal_window)

    def cleanup(self):
        if self.temp_dir:
            self.temp_dir.cleanup()
            self.temp_dir = None


def obtain_ephemeral_admin_cert_files(config: Mapping, root_ca_file: str) -> EphemeralAdminCertFiles:
    if not isinstance(config, Mapping):
        raise EphemeralAdminCertError(f"ephemeral_admin_cert must be a mapping but got {type(config)}")
    if not root_ca_file:
        raise EphemeralAdminCertError("root_ca_file is required")

    provider = config.get(EPHEMERAL_ADMIN_CERT_PROVIDER_KEY)
    if not provider:
        raise EphemeralAdminCertError(f"ephemeral_admin_cert.{EPHEMERAL_ADMIN_CERT_PROVIDER_KEY} is required")

    provider_config = config.get(EPHEMERAL_ADMIN_CERT_PROVIDER_CONFIG_KEY) or {}
    if not isinstance(provider_config, Mapping):
        raise EphemeralAdminCertError(
            f"ephemeral_admin_cert.{EPHEMERAL_ADMIN_CERT_PROVIDER_CONFIG_KEY} must be a mapping"
        )

    provider_name = str(provider)
    provider_config = dict(provider_config)
    renewal_window = _renewal_window(config)
    cache_dir = _cache_base_dir() / _cache_key(
        provider=provider_name, provider_config=provider_config, root_ca_file=root_ca_file
    )
    with _cache_lock(cache_dir):
        cached_files = _load_cached_ephemeral_admin_cert_files(cache_dir, root_ca_file, renewal_window)
        if cached_files:
            return cached_files

        provider_func = _load_provider(provider_name)
        files = provider_func(config=provider_config, root_ca_file=root_ca_file)
        if not isinstance(files, EphemeralAdminCertFiles):
            raise EphemeralAdminCertError(f"ephemeral admin cert provider returned {type(files)}")

        try:
            cert = validate_ephemeral_admin_cert_files(files.client_cert, files.client_key, root_ca_file)
        except Exception:
            files.cleanup()
            raise
        files.expires_at = cert_time(cert, "not_valid_after").timestamp()
        return _store_ephemeral_admin_cert_files(files, cache_dir, root_ca_file)


def validate_ephemeral_admin_cert_files(
    cert_path: str,
    key_path: str,
    root_ca_file: str,
):
    cert_chain = load_crt_chain(cert_path)
    cert = cert_chain[0]
    root_ca_cert = load_crt(root_ca_file)
    try:
        verify_cert_chain(leaf_cert=cert, intermediate_certs=cert_chain[1:], root_ca_cert=root_ca_cert)
    except Exception as ex:
        raise EphemeralAdminCertError("ephemeral admin certificate does not chain to the configured CA") from ex

    private_key = load_private_key_file(key_path)
    if _public_key_pem(cert.public_key()) != _public_key_pem(private_key.public_key()):
        raise EphemeralAdminCertError("ephemeral admin certificate is for a different private key")

    try:
        validate_admin_leaf_cert(cert)
    except Exception as ex:
        raise EphemeralAdminCertError(str(ex)) from ex
    return cert


def cert_time(cert, field_name: str) -> datetime.datetime:
    value = getattr(cert, f"{field_name}_utc", None)
    if value is not None:
        return value
    return getattr(cert, field_name).replace(tzinfo=datetime.timezone.utc)


def _load_cached_ephemeral_admin_cert_files(
    cache_dir: Path,
    root_ca_file: str,
    renewal_window: float,
) -> Optional[EphemeralAdminCertFiles]:
    issuance_dirs = sorted(
        (path for path in cache_dir.iterdir() if path.is_dir() and not path.name.startswith(".")),
        key=lambda path: path.name,
        reverse=True,
    )
    for issuance_dir in issuance_dirs:
        cert_path = issuance_dir / EPHEMERAL_ADMIN_CERT_CLIENT_CERT
        key_path = issuance_dir / EPHEMERAL_ADMIN_CERT_CLIENT_KEY
        if not cert_path.is_file() or not key_path.is_file():
            continue
        try:
            cert = validate_ephemeral_admin_cert_files(str(cert_path), str(key_path), root_ca_file)
            files = EphemeralAdminCertFiles(
                client_key=str(key_path),
                client_cert=str(cert_path),
                expires_at=cert_time(cert, "not_valid_after").timestamp(),
            )
            if not files.needs_renewal(renewal_window=renewal_window):
                return files
        except Exception:
            continue
    return None


def _store_ephemeral_admin_cert_files(
    files: EphemeralAdminCertFiles,
    cache_dir: Path,
    root_ca_file: str,
) -> EphemeralAdminCertFiles:
    _ensure_private_dir(cache_dir)
    temp_dir = Path(tempfile.mkdtemp(prefix=".new-", dir=cache_dir))
    issuance_dir = cache_dir / str(time.time_ns())
    cert_path = temp_dir / EPHEMERAL_ADMIN_CERT_CLIENT_CERT
    key_path = temp_dir / EPHEMERAL_ADMIN_CERT_CLIENT_KEY

    try:
        _copy_file_private(files.client_cert, cert_path)
        _copy_file_private(files.client_key, key_path)
        os.replace(temp_dir, issuance_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        files.cleanup()
        raise

    files.cleanup()
    _remove_stale_cache_entries(cache_dir, root_ca_file, keep=issuance_dir)
    return EphemeralAdminCertFiles(
        client_key=str(issuance_dir / EPHEMERAL_ADMIN_CERT_CLIENT_KEY),
        client_cert=str(issuance_dir / EPHEMERAL_ADMIN_CERT_CLIENT_CERT),
        expires_at=files.expires_at,
    )


def _remove_stale_cache_entries(cache_dir: Path, root_ca_file: str, keep: Path):
    now = time.time()
    for issuance_dir in cache_dir.iterdir():
        if not issuance_dir.is_dir() or issuance_dir == keep or issuance_dir.name.startswith("."):
            continue
        cert_path = issuance_dir / EPHEMERAL_ADMIN_CERT_CLIENT_CERT
        key_path = issuance_dir / EPHEMERAL_ADMIN_CERT_CLIENT_KEY
        try:
            cert = validate_ephemeral_admin_cert_files(str(cert_path), str(key_path), root_ca_file)
            expired = cert_time(cert, "not_valid_after").timestamp() <= now
        except Exception:
            expired = True
        if expired:
            shutil.rmtree(issuance_dir, ignore_errors=True)


@contextmanager
def _cache_lock(cache_dir: Path):
    try:
        import fcntl
    except ImportError as ex:
        raise EphemeralAdminCertError("ephemeral admin certificate caching requires POSIX file locking") from ex

    _ensure_private_dir(cache_dir)
    lock_path = cache_dir / EPHEMERAL_ADMIN_CERT_CACHE_LOCK
    with open(lock_path, "a+b") as lock_file:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _cache_base_dir() -> Path:
    cache_dir = Path.home() / ".nvflare" / EPHEMERAL_ADMIN_CERT_CACHE_DIR
    _ensure_private_dir(cache_dir)
    return cache_dir


def _ensure_private_dir(path: Path):
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def _copy_file_private(src: str, dst: Path):
    fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with open(src, "rb") as in_file:
            with os.fdopen(fd, "wb") as out_file:
                fd = None
                shutil.copyfileobj(in_file, out_file)
    finally:
        if fd is not None:
            os.close(fd)


def _cache_key(provider: str, provider_config: Mapping, root_ca_file: str) -> str:
    with open(root_ca_file, "rb") as f:
        root_ca_hash = hashlib.sha256(f.read()).hexdigest()

    cache_material = {
        "version": 1,
        "root_ca_sha256": root_ca_hash,
        "provider": provider,
        "provider_config": provider_config,
    }
    encoded = json.dumps(cache_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _renewal_window(config: Mapping) -> float:
    renewal_window = config.get("renewal_window", 60.0)
    try:
        renewal_window = float(renewal_window)
    except (TypeError, ValueError) as ex:
        raise EphemeralAdminCertError("ephemeral_admin_cert.renewal_window must be a number") from ex
    if renewal_window <= 0.0:
        raise EphemeralAdminCertError("ephemeral_admin_cert.renewal_window must be greater than zero")
    return renewal_window


def _load_provider(provider: str):
    provider_path = BUILTIN_EPHEMERAL_ADMIN_CERT_PROVIDERS.get(provider, provider)
    if ":" not in provider_path:
        raise EphemeralAdminCertError(
            f"ephemeral admin cert provider '{provider}' must be a built-in provider name or module:function path"
        )

    module_name, func_name = provider_path.split(":", 1)
    if not module_name or not func_name:
        raise EphemeralAdminCertError(
            f"ephemeral admin cert provider '{provider}' must be a built-in provider name or module:function path"
        )

    try:
        module = importlib.import_module(module_name)
        provider_func = getattr(module, func_name)
    except Exception as ex:
        raise EphemeralAdminCertError(f"cannot load ephemeral admin cert provider '{provider}': {ex}") from ex
    if not callable(provider_func):
        raise EphemeralAdminCertError(f"ephemeral admin cert provider '{provider}' is not callable")
    return provider_func


def _public_key_pem(public_key) -> bytes:
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
