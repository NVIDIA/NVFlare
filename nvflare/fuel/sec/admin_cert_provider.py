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
import math
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from nvflare.fuel.sec.admin_cert import validate_admin_leaf_cert
from nvflare.lighter.utils import load_crt, load_crt_chain, load_private_key_file, verify_cert_chain

PROVIDER_CONFIG_KEY = "provider_config"
PROVIDER_KEY = "provider"
ADMIN_CERT_CACHE_DIR = "admin_certificates"
ADMIN_CERT_CLIENT_CERT = "client.crt"
ADMIN_CERT_CLIENT_KEY = "client.key"
ADMIN_CERT_CACHE_LOCK = ".lock"
DEFAULT_ADMIN_CERT_RENEWAL_WINDOW = 12 * 60 * 60
BUILTIN_ADMIN_CERT_PROVIDERS = {
    "step_ca": "nvflare.fuel.sec.step_ca_admin_cert:obtain_step_ca_admin_cert_files",
}


class AdminCertProviderError(ValueError):
    """Raised when admin certificate acquisition fails."""


class AdminCertProviderRequestError(AdminCertProviderError):
    """A provider request failed and may succeed on retry; local validation errors must not use this type."""


@dataclass
class AdminCertFiles:
    client_key: str
    client_cert: str
    expires_at: float = 0.0
    temp_dir: Optional[tempfile.TemporaryDirectory] = field(default=None, repr=False)

    def needs_renewal(
        self, renewal_window: float = DEFAULT_ADMIN_CERT_RENEWAL_WINDOW, now: Optional[float] = None
    ) -> bool:
        if not self.expires_at:
            return True
        now = time.time() if now is None else now
        return self.expires_at - now <= max(0.0, renewal_window)

    def cleanup(self):
        if self.temp_dir:
            self.temp_dir.cleanup()
            self.temp_dir = None


def obtain_admin_cert_files(config: Mapping, root_ca_file: str) -> AdminCertFiles:
    config = validate_admin_cert_provider_config(config)
    if not root_ca_file:
        raise AdminCertProviderError("root_ca_file is required")

    provider = config[PROVIDER_KEY]
    provider_config = config[PROVIDER_CONFIG_KEY]
    renewal_window = get_admin_cert_renewal_window(config)
    cache_dir = _cache_base_dir() / _cache_key(
        provider=provider, provider_config=provider_config, root_ca_file=root_ca_file
    )
    with _cache_lock(cache_dir):
        for staging_dir in cache_dir.glob(".new-*"):
            if staging_dir.is_dir():
                shutil.rmtree(staging_dir, ignore_errors=True)

        cached_files = _load_cached_admin_cert_files(cache_dir, root_ca_file, renewal_window)
        if cached_files:
            return cached_files

        provider_func = _load_provider(provider)
        try:
            files = provider_func(config=provider_config, root_ca_file=root_ca_file)
        except (AdminCertProviderError, OSError) as ex:
            if provider in BUILTIN_ADMIN_CERT_PROVIDERS or provider in BUILTIN_ADMIN_CERT_PROVIDERS.values():
                raise
            # Legacy custom providers use these types for acquisition failures.
            # Keep retries at the provider boundary, after local setup succeeds.
            raise AdminCertProviderRequestError(str(ex)) from ex
        if not isinstance(files, AdminCertFiles):
            raise AdminCertProviderError(f"admin certificate provider returned {type(files)}")

        try:
            cert = validate_admin_cert_files(files.client_cert, files.client_key, root_ca_file)
            files.expires_at = cert_time(cert, "not_valid_after").timestamp()
            if files.needs_renewal(renewal_window=renewal_window):
                raise AdminCertProviderError("issued admin certificate must remain valid beyond the renewal window")
        except Exception:
            files.cleanup()
            raise
        return _store_admin_cert_files(files, cache_dir, root_ca_file)


def validate_admin_cert_files(
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
        raise AdminCertProviderError(f"admin certificate chain validation failed: {ex}") from ex

    try:
        private_key = load_private_key_file(key_path)
    except Exception as ex:
        raise AdminCertProviderError("admin private key must be an unencrypted PEM private key") from ex

    public_key = cert.public_key()
    if not isinstance(private_key, rsa.RSAPrivateKey) or not isinstance(public_key, rsa.RSAPublicKey):
        raise AdminCertProviderError("admin certificate and private key must use RSA")
    if _public_key_pem(public_key) != _public_key_pem(private_key.public_key()):
        raise AdminCertProviderError("admin certificate is for a different private key")

    try:
        validate_admin_leaf_cert(cert)
    except Exception as ex:
        raise AdminCertProviderError(str(ex)) from ex
    for oid, field_name in (
        (NameOID.ORGANIZATION_NAME, "organizationName"),
        (NameOID.UNSTRUCTURED_NAME, "unstructuredName"),
    ):
        attrs = cert.subject.get_attributes_for_oid(oid)
        if not attrs or not attrs[0].value:
            raise AdminCertProviderError(f"admin certificate missing subject {field_name}")
    return cert


def cert_time(cert, field_name: str) -> datetime.datetime:
    value = getattr(cert, f"{field_name}_utc", None)
    if value is not None:
        return value
    return getattr(cert, field_name).replace(tzinfo=datetime.timezone.utc)


def _load_cached_admin_cert_files(
    cache_dir: Path,
    root_ca_file: str,
    renewal_window: float,
) -> Optional[AdminCertFiles]:
    issuance_dirs = sorted(
        (path for path in cache_dir.iterdir() if path.is_dir() and not path.name.startswith(".")),
        key=lambda path: path.name,
        reverse=True,
    )
    for issuance_dir in issuance_dirs:
        cert_path = issuance_dir / ADMIN_CERT_CLIENT_CERT
        key_path = issuance_dir / ADMIN_CERT_CLIENT_KEY
        if not cert_path.is_file() or not key_path.is_file():
            continue
        try:
            cert = validate_admin_cert_files(str(cert_path), str(key_path), root_ca_file)
            files = AdminCertFiles(
                client_key=str(key_path),
                client_cert=str(cert_path),
                expires_at=cert_time(cert, "not_valid_after").timestamp(),
            )
            if not files.needs_renewal(renewal_window=renewal_window):
                return files
        except Exception:
            continue
    return None


def _store_admin_cert_files(
    files: AdminCertFiles,
    cache_dir: Path,
    root_ca_file: str,
) -> AdminCertFiles:
    _ensure_private_dir(cache_dir)
    temp_dir = Path(tempfile.mkdtemp(prefix=".new-", dir=cache_dir))
    issuance_dir = cache_dir / str(time.time_ns())
    cert_path = temp_dir / ADMIN_CERT_CLIENT_CERT
    key_path = temp_dir / ADMIN_CERT_CLIENT_KEY

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
    return AdminCertFiles(
        client_key=str(issuance_dir / ADMIN_CERT_CLIENT_KEY),
        client_cert=str(issuance_dir / ADMIN_CERT_CLIENT_CERT),
        expires_at=files.expires_at,
    )


def _remove_stale_cache_entries(cache_dir: Path, root_ca_file: str, keep: Path):
    now = time.time()
    for issuance_dir in cache_dir.iterdir():
        if not issuance_dir.is_dir() or issuance_dir == keep or issuance_dir.name.startswith("."):
            continue
        cert_path = issuance_dir / ADMIN_CERT_CLIENT_CERT
        key_path = issuance_dir / ADMIN_CERT_CLIENT_KEY
        try:
            cert = validate_admin_cert_files(str(cert_path), str(key_path), root_ca_file)
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
        raise AdminCertProviderError("admin certificate caching requires POSIX file locking") from ex

    _ensure_private_dir(cache_dir)
    lock_path = cache_dir / ADMIN_CERT_CACHE_LOCK
    with open(lock_path, "a+b") as lock_file:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _cache_base_dir() -> Path:
    cache_dir = Path.home() / ".nvflare" / ADMIN_CERT_CACHE_DIR
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


def get_admin_cert_renewal_window(config: Mapping) -> float:
    renewal_window = config.get("renewal_window", DEFAULT_ADMIN_CERT_RENEWAL_WINDOW)
    if isinstance(renewal_window, bool):
        raise AdminCertProviderError("admin_cert_provider.renewal_window must be a finite number")
    try:
        renewal_window = float(renewal_window)
    except (TypeError, ValueError) as ex:
        raise AdminCertProviderError("admin_cert_provider.renewal_window must be a finite number") from ex
    if not math.isfinite(renewal_window):
        raise AdminCertProviderError("admin_cert_provider.renewal_window must be a finite number")
    if renewal_window <= 0.0:
        raise AdminCertProviderError("admin_cert_provider.renewal_window must be greater than zero")
    return renewal_window


def validate_admin_cert_provider_config(config: Mapping) -> dict:
    if not isinstance(config, Mapping):
        raise AdminCertProviderError(f"admin_cert_provider must be a mapping but got {type(config)}")

    result = dict(config)
    provider = result.get(PROVIDER_KEY)
    if not isinstance(provider, str) or not provider:
        raise AdminCertProviderError(f"admin_cert_provider.{PROVIDER_KEY} is required")
    _validate_provider_name(provider)

    provider_config = result.get(PROVIDER_CONFIG_KEY) or {}
    if not isinstance(provider_config, Mapping):
        raise AdminCertProviderError(f"admin_cert_provider.{PROVIDER_CONFIG_KEY} must be a mapping")
    result[PROVIDER_CONFIG_KEY] = dict(provider_config)
    get_admin_cert_renewal_window(result)
    return result


def _validate_provider_name(provider: str):
    provider_path = BUILTIN_ADMIN_CERT_PROVIDERS.get(provider, provider)
    if ":" not in provider_path:
        raise AdminCertProviderError(
            f"admin certificate provider '{provider}' must be a built-in provider name or module:function path"
        )
    module_name, func_name = provider_path.split(":", 1)
    if not module_name or not func_name:
        raise AdminCertProviderError(
            f"admin certificate provider '{provider}' must be a built-in provider name or module:function path"
        )


def _load_provider(provider: str):
    provider_path = BUILTIN_ADMIN_CERT_PROVIDERS.get(provider, provider)
    _validate_provider_name(provider)
    module_name, func_name = provider_path.split(":", 1)

    try:
        module = importlib.import_module(module_name)
        provider_func = getattr(module, func_name)
    except Exception as ex:
        raise AdminCertProviderError(f"cannot load admin certificate provider '{provider}': {ex}") from ex
    if not callable(provider_func):
        raise AdminCertProviderError(f"admin certificate provider '{provider}' is not callable")
    return provider_func


def _public_key_pem(public_key) -> bytes:
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
