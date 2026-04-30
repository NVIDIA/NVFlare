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
"""CLI-scoped session helpers."""

import os

from nvflare.fuel.flare_api.flare_api import Session, new_secure_session
from nvflare.tool.kit.kit_config import (
    NVFLARE_STARTUP_KIT_DIR,
    StartupKitConfigError,
    get_active_startup_kit_id,
    load_cli_config,
    resolve_admin_user_and_dir_from_startup_kit,
    resolve_startup_kit_dir,
    resolve_startup_kit_dir_by_id,
)


def add_startup_kit_selection_args(parser) -> None:
    """Add non-mutating startup-kit selectors to an online command parser."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--kit-id",
        dest="kit_id",
        default=None,
        help="use this registered startup-kit ID for this command only",
    )
    group.add_argument(
        "--startup-kit",
        dest="startup_kit",
        default=None,
        help="use this admin startup-kit directory for this command only",
    )


def _get_arg_value(args, name: str, default=None):
    if args is None:
        return default
    try:
        return vars(args).get(name, default)
    except TypeError:
        return getattr(args, name, default)


def _startup_kit_selectors_for_args(args=None):
    kit_id = _get_arg_value(args, "kit_id")
    startup_kit = _get_arg_value(args, "startup_kit")
    if kit_id and startup_kit:
        raise StartupKitConfigError(
            "--kit-id and --startup-kit are mutually exclusive",
            hint="Use only one startup-kit selector for a command.",
        )
    return kit_id, startup_kit


def resolve_startup_kit_dir_for_args(args=None) -> str:
    """Resolve per-command startup-kit selectors, falling back to env/active config."""
    kit_id, startup_kit = _startup_kit_selectors_for_args(args)
    if kit_id:
        return resolve_startup_kit_dir_by_id(kit_id)
    if startup_kit:
        _username, admin_user_dir = resolve_admin_user_and_dir_from_startup_kit(startup_kit)
        return admin_user_dir
    return resolve_startup_kit_dir()


def resolve_startup_kit_info_for_args(args=None) -> dict:
    """Resolve startup-kit selection metadata for machine-readable command output."""
    kit_id, startup_kit = _startup_kit_selectors_for_args(args)
    if kit_id:
        return {
            "source": "kit_id",
            "id": kit_id,
            "path": resolve_startup_kit_dir_by_id(kit_id),
        }
    if startup_kit:
        _username, admin_user_dir = resolve_admin_user_and_dir_from_startup_kit(startup_kit)
        return {
            "source": "startup_kit",
            "id": None,
            "path": admin_user_dir,
        }

    env_startup_kit_dir = os.getenv(NVFLARE_STARTUP_KIT_DIR)
    if env_startup_kit_dir is not None and env_startup_kit_dir.strip():
        return {
            "source": "env",
            "id": None,
            "path": resolve_startup_kit_dir(),
        }

    config = load_cli_config()
    active_id = get_active_startup_kit_id(config)
    return {
        "source": "active",
        "id": active_id,
        "path": resolve_startup_kit_dir(),
    }


def resolve_admin_user_and_dir_for_args(args=None):
    """Resolve the admin identity and startup-kit directory for a command invocation."""
    kit_id, startup_kit = _startup_kit_selectors_for_args(args)
    if startup_kit:
        return resolve_admin_user_and_dir_from_startup_kit(startup_kit)
    startup_dir = resolve_startup_kit_dir_by_id(kit_id) if kit_id else resolve_startup_kit_dir()
    return resolve_admin_user_and_dir_from_startup_kit(startup_dir)


def new_cli_session(
    username: str,
    startup_kit_location: str,
    timeout: float,
    study: str = "default",
    secure_mode: bool = True,
    debug: bool = False,
) -> Session:
    """Compatibility wrapper for CLI callers around the shared secure session factory."""
    return new_secure_session(
        username=username,
        startup_kit_location=startup_kit_location,
        debug=debug,
        study=study,
        timeout=timeout,
        auto_login_max_tries=1,
    )


def new_active_cli_session(timeout: float, study: str = "default", debug: bool = False) -> Session:
    """Create a CLI session using NVFLARE_STARTUP_KIT_DIR or the active registered startup kit."""
    startup_dir = resolve_startup_kit_dir()
    username, admin_user_dir = resolve_admin_user_and_dir_from_startup_kit(startup_dir)
    return new_cli_session(
        username=username,
        startup_kit_location=admin_user_dir,
        timeout=timeout,
        study=study,
        debug=debug,
    )


def new_cli_session_for_args(args=None, timeout: float = 5.0, study: str = "default", debug: bool = False) -> Session:
    """Create a CLI session using per-command selectors, env var, or the active registered startup kit."""
    username, admin_user_dir = resolve_admin_user_and_dir_for_args(args)
    return new_cli_session(
        username=username,
        startup_kit_location=admin_user_dir,
        timeout=timeout,
        study=study,
        debug=debug,
    )
