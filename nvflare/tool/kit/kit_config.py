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
import tempfile
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from pyhocon import ConfigFactory as CF
from pyhocon import ConfigTree, HOCONConverter

from nvflare.fuel.utils.config_factory import ConfigFactory
from nvflare.tool.job.job_client_const import CONFIG_CONF

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
    for key in ("startup_kit.path", "startup_kit", "poc.startup_kit", "prod.startup_kit"):
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
    _remove_legacy_startup_kit_keys(_ensure_v2_config(config))

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


def _quote_hocon_key(key: str) -> str:
    return json.dumps(key)


def _decode_hocon_key(key: str) -> str:
    if len(key) >= 2 and key[0] == '"' and key[-1] == '"':
        try:
            return json.loads(key)
        except Exception:
            return key[1:-1]
    return key


def _find_raw_entry_key(entries: ConfigTree, kit_id: str) -> Optional[str]:
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


def get_startup_kit_entries(config: ConfigTree) -> Dict[str, str]:
    """Return the startup kit registry as an ID-to-path mapping."""
    entries = config.get(STARTUP_KITS_ENTRIES_KEY, None)
    if not isinstance(entries, ConfigTree):
        return {}

    result = {}
    for raw_key, value in entries.items():
        if isinstance(value, str):
            result[_decode_hocon_key(raw_key)] = value
    return result


def get_active_startup_kit_id(config: ConfigTree) -> Optional[str]:
    active = config.get_string(STARTUP_KITS_ACTIVE_KEY, None) if config else None
    return active.strip() if active and active.strip() else None


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


def classify_startup_kit(path: str) -> Tuple[str, str]:
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
            hint="Run nvflare kit list.",
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
            hint=f"Run nvflare kit use <admin-id>, or replace it with nvflare kit add {kit_id} <startup-kit-dir> --force.",
        ) from e


def add_startup_kit_entry(config: ConfigTree, kit_id: str, path: str, force: bool = False) -> ConfigTree:
    """Register ID -> path for any startup kit. Never changes active."""
    kit_id = _normalize_kit_id(kit_id)
    normalized_path = validate_startup_kit(path)
    entries = _get_entries_tree(config)

    raw_existing_key = _find_raw_entry_key(entries, kit_id)
    if raw_existing_key is not None and not force:
        raise StartupKitConfigError(
            f"startup kit id '{kit_id}' already exists",
            hint="Use --force to replace this local registration.",
        )
    if raw_existing_key is not None:
        entries.pop(raw_existing_key, None)

    config.put(f"{STARTUP_KITS_ENTRIES_KEY}.{_quote_hocon_key(kit_id)}", normalized_path)
    return _ensure_v2_config(config)


def set_active_startup_kit(config: ConfigTree, kit_id: str) -> ConfigTree:
    """Validate ID and path, then set startup_kits.active."""
    kit_id = _normalize_kit_id(kit_id)
    entries = get_startup_kit_entries(config)
    if kit_id not in entries:
        raise StartupKitConfigError(f"startup kit id '{kit_id}' is not registered", hint="Run nvflare kit list.")

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


def clear_active_if(config: ConfigTree, removed_ids: Set[str]) -> ConfigTree:
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


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        return path == base or path.is_relative_to(base)
    except AttributeError:
        try:
            return os.path.commonpath([str(path), str(base)]) == str(base)
        except ValueError:
            return False


def remove_entries_under_workspace(config: ConfigTree, workspace: str) -> Tuple[ConfigTree, Set[str]]:
    """Remove entries whose canonical paths are under canonical workspace path."""
    workspace_path = _canonical_path(workspace)
    entries = _get_entries_tree(config)
    removed = set()

    for raw_key, path in list(entries.items()):
        if not isinstance(path, str):
            continue
        if _is_relative_to(_canonical_path(path), workspace_path):
            removed.add(_decode_hocon_key(raw_key))
            entries.pop(raw_key, None)

    clear_active_if(config, removed)
    return config, removed


def inspect_startup_kit_metadata(path: str) -> Dict[str, Optional[str]]:
    """Best-effort metadata inspection for display."""
    metadata = {"identity": None, "cert_role": None, "kind": None}
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

        try:
            from cryptography import x509
            from cryptography.hazmat.backends import default_backend
            from cryptography.x509.oid import NameOID

            cert_path = os.path.join(startup_dir, "client.crt")
            with open(cert_path, "rb") as f:
                cert = x509.load_pem_x509_certificate(f.read(), default_backend())
            role_attrs = cert.subject.get_attributes_for_oid(NameOID.UNSTRUCTURED_NAME)
            if role_attrs:
                metadata["cert_role"] = role_attrs[0].value
            cn_attrs = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
            if cn_attrs and not metadata["identity"]:
                metadata["identity"] = cn_attrs[0].value
        except Exception:
            pass

    if not metadata["identity"]:
        metadata["identity"] = os.path.basename(startup_kit_dir)

    return metadata


def get_startup_kit_status(path: str) -> Tuple[str, Optional[str], Dict[str, Optional[str]]]:
    """Return (status, normalized_path, metadata) without raising for stale entries."""
    path_obj = Path(path).expanduser() if path else Path("")
    if not path or not path_obj.exists():
        return "missing", None, {"identity": None, "cert_role": None}

    try:
        _, normalized_path = classify_startup_kit(path)
    except StartupKitConfigError:
        return "invalid", None, {"identity": None, "cert_role": None}

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
            hint="Run nvflare poc prepare, or run nvflare kit add <id> <startup-kit-dir> then nvflare kit use <id>.",
        )

    entries = get_startup_kit_entries(config)
    if active not in entries:
        raise StartupKitConfigError(
            f"active startup kit '{active}' is not registered",
            hint="Run nvflare kit list, then nvflare kit use <id>.",
        )

    path = entries[active]
    try:
        return validate_admin_startup_kit(path)
    except StartupKitConfigError as e:
        path_obj = Path(path).expanduser()
        if not path_obj.exists():
            raise StartupKitConfigError(
                f"active startup kit '{active}' points to a missing path\nPath: {path}",
                hint=f"Run nvflare kit use <id> or nvflare kit remove {active}.",
            ) from e
        raise StartupKitConfigError(
            f"active startup kit '{active}' is not a valid startup kit for admin use\nPath: {path}",
            hint=f"Run nvflare kit use <id> or nvflare kit remove {active}.",
        ) from e


def resolve_admin_user_and_dir_from_startup_kit(startup_kit_dir: str) -> Tuple[str, str]:
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
