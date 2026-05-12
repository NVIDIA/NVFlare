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

"""Shared startup kit registry helpers for the NVFlare CLI."""

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from pyhocon import ConfigFactory as CF
from pyhocon import ConfigTree, HOCONConverter

from nvflare.fuel.utils.config_factory import ConfigFactory
from nvflare.tool.job.job_client_const import CONFIG_CONF

try:
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
    from cryptography.x509.oid import NameOID
except ImportError:
    x509 = None
    default_backend = None
    NameOID = None

CONFIG_VERSION = "version"
CURRENT_CONFIG_VERSION = 2
STARTUP_KITS_ACTIVE_KEY = "startup_kits.active"
STARTUP_KITS_ENTRIES_KEY = "startup_kits.entries"
NVFLARE_STARTUP_KIT_DIR = "NVFLARE_STARTUP_KIT_DIR"

ADMIN_STARTUP_KIT_REQUIRED_FILES = (
    os.path.join("startup", "fed_admin.json"),
    os.path.join("startup", "client.crt"),
    os.path.join("startup", "rootCA.pem"),
)
SITE_STARTUP_KIT_REQUIRED_FILES = (os.path.join("startup", "fed_client.json"),)
SERVER_STARTUP_KIT_REQUIRED_FILES = (os.path.join("startup", "fed_server.json"),)
STARTUP_KIT_KIND_ADMIN = "admin"
STARTUP_KIT_KIND_SITE = "site"
STARTUP_KIT_KIND_SERVER = "server"
STARTUP_KIT_CERT_EXPIRING_SOON_DAYS = 30
_HOCON_SIMPLE_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class StartupKitConfigError(ValueError):
    """Configuration or validation error for startup kit resolution."""

    def __init__(self, message: str, hint: str = None):
        super().__init__(message)
        self.hint = hint


def get_cli_config_path() -> Path:
    return Path.home() / ".nvflare" / CONFIG_CONF


def _empty_v2_config() -> ConfigTree:
    config = CF.parse_string("{}")
    config.put(CONFIG_VERSION, CURRENT_CONFIG_VERSION)
    config.put(STARTUP_KITS_ENTRIES_KEY, ConfigTree())
    return config


def _ensure_v2_config(config: ConfigTree) -> ConfigTree:
    config.put(CONFIG_VERSION, CURRENT_CONFIG_VERSION)
    if not isinstance(config.get(STARTUP_KITS_ENTRIES_KEY, None), ConfigTree):
        config.put(STARTUP_KITS_ENTRIES_KEY, ConfigTree())
    return config


def _remove_legacy_startup_kit_keys(config: ConfigTree) -> ConfigTree:
    for key in (
        "startup_kit.path",
        "startup_kit",
        "poc.startup_kit",
        "prod.startup_kit",
    ):
        try:
            config.pop(key, None)
        except Exception:
            pass
    return config


def load_cli_config() -> ConfigTree:
    """Load ~/.nvflare/config.conf or return an empty version-2 config."""
    config_path = get_cli_config_path()
    if not config_path.is_file():
        return _empty_v2_config()

    try:
        config = CF.parse_file(str(config_path))
    except Exception as e:
        raise StartupKitConfigError(
            f"cannot parse {config_path}",
            hint="Fix the config file, or move it aside and run nvflare poc prepare.",
        ) from e

    return _ensure_v2_config(config)


def save_cli_config(config: ConfigTree) -> None:
    """Atomically write ~/.nvflare/config.conf as config schema version 2."""
    config_path = get_cli_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = _remove_legacy_startup_kit_keys(_ensure_v2_config(config))

    config_text = HOCONConverter.to_hocon(config=config, level=1)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=str(config_path.parent), delete=False) as outfile:
            temp_path = outfile.name
            outfile.write(f"{config_text}\n")
            outfile.flush()
            os.fsync(outfile.fileno())
        os.replace(temp_path, config_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def _hocon_path_key(key: str) -> str:
    """Return a ConfigTree.put path segment that preserves the registry ID literally."""
    if _HOCON_SIMPLE_KEY_PATTERN.match(key):
        return key
    # ConfigTree.put treats dots as path separators. Quote emails and other complex IDs so
    # values like lead@nvidia.com remain one registry key instead of nested HOCON paths.
    return json.dumps(key)


def _decode_hocon_key(key: str) -> str:
    if len(key) >= 2 and key[0] == '"' and key[-1] == '"':
        try:
            return json.loads(key)
        except Exception:
            return key[1:-1]
    return key


def _find_raw_entry_key(entries: ConfigTree, kit_id: str) -> str | None:
    for raw_key in entries.keys():
        if _decode_hocon_key(raw_key) == kit_id:
            return raw_key
    return None


def _get_entries_tree(config: ConfigTree) -> ConfigTree:
    _ensure_v2_config(config)
    entries = config.get(STARTUP_KITS_ENTRIES_KEY, None)
    if isinstance(entries, ConfigTree):
        return entries
    config.put(STARTUP_KITS_ENTRIES_KEY, ConfigTree())
    return config.get(STARTUP_KITS_ENTRIES_KEY)


def _normalize_kit_id(kit_id: str) -> str:
    kit_id = kit_id.strip() if kit_id is not None else ""
    if not kit_id:
        raise StartupKitConfigError("startup kit id cannot be empty", hint="Choose a non-empty local ID.")
    return kit_id


def get_startup_kit_entries(config: ConfigTree) -> dict[str, str]:
    """Return the startup kit registry as an ID-to-path mapping."""
    entries = config.get(STARTUP_KITS_ENTRIES_KEY, None)
    if not isinstance(entries, ConfigTree):
        return {}

    result = {}
    for raw_key, value in entries.items():
        if isinstance(value, str):
            result[_decode_hocon_key(raw_key)] = value
    return result


def get_active_startup_kit_id(config: ConfigTree) -> str | None:
    active = config.get_string(STARTUP_KITS_ACTIVE_KEY, None) if config else None
    return active.strip() if active and active.strip() else None


def resolve_startup_kit_dir_by_id(kit_id: str) -> str:
    """Resolve a registered startup-kit ID to a validated admin user dir without changing active config."""
    kit_id = _normalize_kit_id(kit_id)
    config = load_cli_config()
    entries = get_startup_kit_entries(config)
    if kit_id not in entries:
        raise StartupKitConfigError(
            f"startup kit id '{kit_id}' is not registered",
            hint="Run nvflare config list.",
        )
    return _validate_registered_path(kit_id, entries[kit_id])


def _as_existing_dir(path: str) -> Path:
    if not path or not str(path).strip():
        raise StartupKitConfigError("startup kit directory is not specified")

    expanded = Path(path).expanduser()
    if not expanded.exists():
        raise StartupKitConfigError(f"startup kit path does not exist: {path}")
    if not expanded.is_dir():
        raise StartupKitConfigError(
            f"path is not a valid startup kit: {path}",
            hint="Use the participant startup directory produced by provisioning.",
        )
    return expanded


def validate_admin_startup_kit(path: str) -> str:
    """Return normalized admin user dir. Raise StartupKitConfigError on invalid kit."""
    kind, normalized_path = classify_startup_kit(path)
    if kind != STARTUP_KIT_KIND_ADMIN:
        raise StartupKitConfigError(
            f"path is not an admin startup kit: {path}",
            hint="Use an admin/user startup directory for commands that connect to FLARE.",
        )
    return normalized_path


def _has_required_files(startup_kit_dir: Path, required_files) -> bool:
    return all((startup_kit_dir / rel_path).is_file() for rel_path in required_files)


def classify_startup_kit(path: str) -> tuple[str, str]:
    """Return (kind, normalized participant dir) for a generated startup kit."""
    startup_path = _as_existing_dir(path)
    startup_kit_dir = startup_path.parent if startup_path.name == "startup" else startup_path

    for kind, required_files in (
        (STARTUP_KIT_KIND_ADMIN, ADMIN_STARTUP_KIT_REQUIRED_FILES),
        (STARTUP_KIT_KIND_SITE, SITE_STARTUP_KIT_REQUIRED_FILES),
        (STARTUP_KIT_KIND_SERVER, SERVER_STARTUP_KIT_REQUIRED_FILES),
    ):
        if _has_required_files(startup_kit_dir, required_files):
            return kind, str(startup_kit_dir.resolve())

    raise StartupKitConfigError(
        f"path is not a valid startup kit: {path}",
        hint="Use the participant startup directory produced by provisioning.",
    )


def validate_startup_kit(path: str) -> str:
    """Return normalized participant dir for any generated startup kit."""
    _, normalized_path = classify_startup_kit(path)
    return normalized_path


def _validate_registered_path(kit_id: str, path: str) -> str:
    if not path:
        raise StartupKitConfigError(
            f"startup kit id '{kit_id}' is not registered",
            hint="Run nvflare config list.",
        )

    try:
        return validate_admin_startup_kit(path)
    except StartupKitConfigError as e:
        path_obj = Path(path).expanduser()
        if not path_obj.exists():
            raise StartupKitConfigError(
                f"startup kit path for '{kit_id}' does not exist: {path}",
                hint="Restore the startup kit, remove the registration, or activate another kit.",
            ) from e
        raise StartupKitConfigError(
            f"registered path for '{kit_id}' is not a valid startup kit for admin use",
            hint=f"Run nvflare config use <admin-id>, or replace it with nvflare config add {kit_id} <startup-kit-dir> --force.",
        ) from e


def add_startup_kit_entry(config: ConfigTree, kit_id: str, path: str, force: bool = False) -> ConfigTree:
    """Register ID -> path for an admin/user startup kit. Never changes active."""
    kit_id = _normalize_kit_id(kit_id)
    normalized_path = validate_admin_startup_kit(path)
    entries = _get_entries_tree(config)

    raw_existing_key = _find_raw_entry_key(entries, kit_id)
    if raw_existing_key is not None and not force:
        raise StartupKitConfigError(
            f"startup kit id '{kit_id}' already exists",
            hint="Use --force to replace this local registration.",
        )
    if raw_existing_key is not None:
        entries.pop(raw_existing_key, None)

    config.put(f"{STARTUP_KITS_ENTRIES_KEY}.{_hocon_path_key(kit_id)}", normalized_path)
    return _ensure_v2_config(config)


def set_active_startup_kit(config: ConfigTree, kit_id: str) -> ConfigTree:
    """Validate ID and path, then set startup_kits.active."""
    kit_id = _normalize_kit_id(kit_id)
    entries = get_startup_kit_entries(config)
    if kit_id not in entries:
        raise StartupKitConfigError(
            f"startup kit id '{kit_id}' is not registered",
            hint="Run nvflare config list.",
        )

    _validate_registered_path(kit_id, entries[kit_id])
    config.put(STARTUP_KITS_ACTIVE_KEY, kit_id)
    return _ensure_v2_config(config)


def remove_startup_kit_entry(config: ConfigTree, kit_id: str) -> ConfigTree:
    kit_id = _normalize_kit_id(kit_id)
    entries = _get_entries_tree(config)
    raw_key = _find_raw_entry_key(entries, kit_id)
    if raw_key is None:
        raise StartupKitConfigError(f"startup kit id '{kit_id}' is not registered")

    entries.pop(raw_key, None)
    return clear_active_if(config, {kit_id})


def clear_active_if(config: ConfigTree, removed_ids: set[str]) -> ConfigTree:
    """Clear startup_kits.active when it points to a removed ID."""
    active = get_active_startup_kit_id(config)
    if active in removed_ids:
        try:
            config.pop(STARTUP_KITS_ACTIVE_KEY, None)
        except Exception:
            pass
    return _ensure_v2_config(config)


def _canonical_path(path: str) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _absolute_path(path: str) -> Path:
    return Path(path).expanduser().absolute()


def _path_match_candidates(path: str) -> tuple[Path, ...]:
    """Return real-path and spelling-preserving variants for workspace containment checks."""
    canonical_path = _canonical_path(path)
    absolute_path = _absolute_path(path)
    if canonical_path == absolute_path:
        return (canonical_path,)
    return canonical_path, absolute_path


def _is_relative_to(path: Path, base: Path) -> bool:
    return path == base or path.is_relative_to(base)


def remove_entries_under_workspace(config: ConfigTree, workspace: str) -> tuple[ConfigTree, set[str]]:
    """Remove entries whose canonical or lexical paths are under the workspace path."""
    workspace_paths = _path_match_candidates(workspace)
    entries = _get_entries_tree(config)
    removed = set()

    for raw_key, path in list(entries.items()):
        if not isinstance(path, str):
            continue
        path_candidates = _path_match_candidates(path)
        if any(
            _is_relative_to(path_candidate, workspace_path)
            for path_candidate in path_candidates
            for workspace_path in workspace_paths
        ):
            removed.add(_decode_hocon_key(raw_key))
            entries.pop(raw_key, None)

    clear_active_if(config, removed)
    return config, removed


def _finding(code: str, severity: str, message: str, hint: str = None) -> dict[str, str]:
    result = {"code": code, "severity": severity, "message": message}
    if hint:
        result["hint"] = hint
    return result


def _empty_startup_kit_metadata(kind: str = None) -> dict:
    return {
        "identity": None,
        "cert_role": None,
        "role": None,
        "org": None,
        "project": None,
        "kind": kind,
        "certificate": None,
        "findings": [],
    }


def _first_cert_subject_value(cert, oid) -> str | None:
    attrs = cert.subject.get_attributes_for_oid(oid)
    return attrs[0].value if attrs else None


def _first_cert_issuer_value(cert, oid) -> str | None:
    attrs = cert.issuer.get_attributes_for_oid(oid)
    return attrs[0].value if attrs else None


def _cert_not_valid_after(cert):
    expires_at = getattr(cert, "not_valid_after_utc", None)
    if expires_at is None:
        expires_at = cert.not_valid_after.replace(tzinfo=timezone.utc)
    elif expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at.astimezone(timezone.utc)


def _format_utc_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _certificate_expiration_metadata(cert, cert_path: str) -> tuple[dict, list]:
    findings = []
    expires_at = _cert_not_valid_after(cert)
    now = datetime.now(timezone.utc)
    seconds_remaining = (expires_at - now).total_seconds()
    days_remaining = int(seconds_remaining // 86400)
    if seconds_remaining < 0:
        status = "expired"
        findings.append(
            _finding(
                "STARTUP_KIT_CERT_EXPIRED",
                "error",
                f"Startup kit certificate expired at {_format_utc_timestamp(expires_at)}.",
                "Request or select a renewed startup kit.",
            )
        )
    elif days_remaining <= STARTUP_KIT_CERT_EXPIRING_SOON_DAYS:
        status = "expiring_soon"
        findings.append(
            _finding(
                "STARTUP_KIT_CERT_EXPIRING_SOON",
                "warning",
                f"Startup kit certificate expires at {_format_utc_timestamp(expires_at)}.",
                "Plan to renew or replace this startup kit before it expires.",
            )
        )
    else:
        status = "ok"

    return (
        {
            "path": cert_path,
            "expires_at": _format_utc_timestamp(expires_at),
            "days_remaining": days_remaining,
            "status": status,
        },
        findings,
    )


def _inspect_admin_cert_metadata(startup_dir: str, metadata: dict) -> None:
    cert_path = os.path.join(startup_dir, "client.crt")
    if not os.path.isfile(cert_path):
        metadata["findings"].append(
            _finding(
                "STARTUP_KIT_CERT_MISSING",
                "warning",
                "Startup kit certificate file is missing.",
                "Replace this startup kit or re-run provisioning.",
            )
        )
        return

    if not (x509 and default_backend and NameOID):
        metadata["certificate"] = {"path": cert_path, "status": "unknown"}
        metadata["findings"].append(
            _finding(
                "STARTUP_KIT_CERT_UNREADABLE",
                "warning",
                "Startup kit certificate could not be inspected because cryptography is unavailable.",
            )
        )
        return

    try:
        with open(cert_path, "rb") as f:
            cert = x509.load_pem_x509_certificate(f.read(), default_backend())
    except Exception:
        metadata["certificate"] = {"path": cert_path, "status": "unreadable"}
        metadata["findings"].append(
            _finding(
                "STARTUP_KIT_CERT_UNREADABLE",
                "warning",
                "Startup kit certificate could not be read.",
                "Replace this startup kit if the certificate file is corrupted.",
            )
        )
        return

    metadata["cert_role"] = _first_cert_subject_value(cert, NameOID.UNSTRUCTURED_NAME)
    metadata["role"] = metadata["cert_role"]
    metadata["org"] = _first_cert_subject_value(cert, NameOID.ORGANIZATION_NAME)
    metadata["project"] = _first_cert_issuer_value(cert, NameOID.COMMON_NAME)
    cn = _first_cert_subject_value(cert, NameOID.COMMON_NAME)
    if cn and not metadata["identity"]:
        metadata["identity"] = cn
    metadata["certificate"], cert_findings = _certificate_expiration_metadata(cert, cert_path)
    metadata["findings"].extend(cert_findings)


def inspect_startup_kit_metadata(path: str) -> dict:
    """Best-effort metadata inspection for display."""
    metadata = _empty_startup_kit_metadata()
    try:
        kind, startup_kit_dir = classify_startup_kit(path)
        metadata["kind"] = kind
    except StartupKitConfigError:
        return metadata

    startup_dir = os.path.join(startup_kit_dir, "startup")
    if kind == STARTUP_KIT_KIND_ADMIN:
        try:
            fed_admin_config = ConfigFactory.load_config("fed_admin.json", [startup_dir])
            if fed_admin_config:
                config_dict = fed_admin_config.to_dict()
                metadata["identity"] = config_dict.get("admin", {}).get("username")
        except Exception:
            pass

        _inspect_admin_cert_metadata(startup_dir, metadata)

    if not metadata["identity"]:
        metadata["identity"] = os.path.basename(startup_kit_dir)

    return metadata


def get_startup_kit_status(
    path: str,
) -> tuple[str, str | None, dict]:
    """Return (status, normalized_path, metadata) without raising for stale entries."""
    path_obj = Path(path).expanduser() if path else Path("")
    if not path or not path_obj.exists():
        metadata = _empty_startup_kit_metadata()
        metadata["findings"].append(
            _finding(
                "STARTUP_KIT_PATH_MISSING",
                "warning",
                f"Registered startup kit path does not exist: {path}",
                "Remove the stale registration or replace it with a valid startup kit.",
            )
        )
        return "missing", None, metadata

    try:
        _, normalized_path = classify_startup_kit(path)
    except StartupKitConfigError:
        metadata = _empty_startup_kit_metadata()
        metadata["findings"].append(
            _finding(
                "STARTUP_KIT_INVALID",
                "warning",
                f"Registered path is not a valid startup kit: {path}",
                "Remove the stale registration or replace it with a valid startup kit.",
            )
        )
        return "invalid", None, metadata

    return "ok", normalized_path, inspect_startup_kit_metadata(normalized_path)


def resolve_startup_kit_dir() -> str:
    """Resolve env var or active config to a validated admin user dir."""
    env_startup_kit_dir = os.getenv(NVFLARE_STARTUP_KIT_DIR)
    if env_startup_kit_dir is not None and env_startup_kit_dir.strip():
        try:
            return validate_admin_startup_kit(env_startup_kit_dir)
        except StartupKitConfigError as e:
            env_path = Path(env_startup_kit_dir).expanduser()
            if not env_path.exists():
                raise StartupKitConfigError(
                    f"{NVFLARE_STARTUP_KIT_DIR} points to a missing path\nPath: {env_startup_kit_dir}",
                    hint=f"Unset {NVFLARE_STARTUP_KIT_DIR}, or set it to a valid startup kit directory.",
                ) from e
            raise StartupKitConfigError(
                f"{NVFLARE_STARTUP_KIT_DIR} does not point to a valid startup kit for admin use\nPath: {env_startup_kit_dir}",
                hint=f"Unset {NVFLARE_STARTUP_KIT_DIR}, or set it to a valid admin startup kit directory.",
            ) from e

    config = load_cli_config()
    active = get_active_startup_kit_id(config)
    if not active:
        raise StartupKitConfigError(
            "no active startup kit is configured",
            hint=(
                "Run nvflare poc prepare, run nvflare config add <id> <startup-kit-dir> then "
                "nvflare config use <id>, pass --kit-id <id> or --startup-kit <path>, or set "
                f"{NVFLARE_STARTUP_KIT_DIR}."
            ),
        )

    entries = get_startup_kit_entries(config)
    if active not in entries:
        raise StartupKitConfigError(
            f"active startup kit '{active}' is not registered",
            hint="Run nvflare config list, then nvflare config use <id>.",
        )

    path = entries[active]
    try:
        return validate_admin_startup_kit(path)
    except StartupKitConfigError as e:
        path_obj = Path(path).expanduser()
        if not path_obj.exists():
            raise StartupKitConfigError(
                f"active startup kit '{active}' points to a missing path\nPath: {path}",
                hint=f"Run nvflare config use <id> or nvflare config remove {active}.",
            ) from e
        raise StartupKitConfigError(
            f"active startup kit '{active}' is not a valid startup kit for admin use\nPath: {path}",
            hint=f"Run nvflare config use <id> or nvflare config remove {active}.",
        ) from e


def resolve_admin_user_and_dir_from_startup_kit(
    startup_kit_dir: str,
) -> tuple[str, str]:
    """Resolve admin username and normalized admin user dir from a startup kit path."""
    admin_user_dir = validate_admin_startup_kit(startup_kit_dir)
    startup_dir = os.path.join(admin_user_dir, "startup")
    fed_admin_config = ConfigFactory.load_config("fed_admin.json", [startup_dir])
    if not fed_admin_config:
        raise StartupKitConfigError(
            f"Unable to locate fed_admin configuration from startup kit location {startup_kit_dir}"
        )

    config_dict = fed_admin_config.to_dict()
    username = config_dict.get("admin", {}).get("username")
    if not username:
        metadata = inspect_startup_kit_metadata(admin_user_dir)
        username = metadata.get("identity")
    if not username:
        raise StartupKitConfigError(f"Unable to resolve admin username from startup kit location {startup_kit_dir}")
    return username, admin_user_dir
