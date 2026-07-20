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

"""Workspace staging for ``nvflare deploy slurm stage``."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import stat
import tempfile
import uuid
from pathlib import Path

from nvflare.app_opt.job_launcher.slurm.config import CONTROL_DIR, DEPLOYMENT_FILE, SCHEMA_VERSION
from nvflare.tool.cli_output import output_ok
from nvflare.tool.deploy.deploy_common import SLURM_STAGE_MANIFEST, SLURM_START_SH, _fail, _paths_overlap

KIT_DIR = "kit"
NEXT_KIT_DIR = "kit.next"


class SlurmStageError(RuntimeError):
    """Raised when a prepared kit cannot be staged safely."""


def stage_slurm_deployment(args) -> None:
    """Stage a prepared Slurm kit into its configured runtime workspace."""

    try:
        prepared = _resolve_prepared_kit(args)
        prepared, workspace, identity = stage_prepared_kit(prepared)
    except (OSError, SlurmStageError) as ex:
        _fail(
            "SLURM_STAGE_FAILED",
            f"Failed to stage Slurm deployment: {ex}",
            "Fix the prepared kit or workspace, then run the stage command again.",
        )

    start_script = prepared / "startup" / SLURM_START_SH
    output_ok(
        {
            "status": "staged",
            "prepared_kit": str(prepared),
            "workspace_path": str(workspace),
            "deployment_uuid": identity["deployment_uuid"],
            "next_step": "Start the server/client parent with the start_command.",
            "start_command": shlex.quote(str(start_script)),
        }
    )


def stage_prepared_kit(prepared_root: str | os.PathLike) -> tuple[Path, Path, dict]:
    """Install a prepared kit and return the canonical kit, workspace, and identity."""

    prepared, workspace, site = _read_prepared_kit(Path(prepared_root))
    _create_or_validate_workspace(workspace)
    identity = _establish_identity(workspace, site)
    _validate_runtime_links(workspace)
    _replace_kit(prepared, workspace)
    _ensure_runtime_links(workspace)
    return prepared, workspace, identity


def _resolve_prepared_kit(args) -> Path:
    positional_kit = getattr(args, "kit", None)
    flag_kit = getattr(args, "kit_flag", None)
    if positional_kit and flag_kit:
        _fail("INVALID_ARGS", "Specify the prepared startup kit only once.", "Use either positional kit or --kit.")
    kit_arg = positional_kit or flag_kit
    if not kit_arg:
        _fail(
            "INVALID_ARGS",
            "Missing prepared startup kit directory.",
            "Run nvflare deploy slurm stage <prepared-kit-dir>.",
        )
    return Path(kit_arg).expanduser()


def _read_prepared_kit(prepared_root: Path) -> tuple[Path, Path, str]:
    try:
        prepared = prepared_root.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as ex:
        raise SlurmStageError(f"cannot resolve prepared kit '{prepared_root}': {ex}") from ex
    if not prepared.is_dir():
        raise SlurmStageError(f"prepared kit must be a directory: {prepared}")

    manifest_path = prepared / "local" / SLURM_STAGE_MANIFEST
    manifest = _read_json(manifest_path)
    expected_fields = {"schema_version", "site", "prepared_path", "workspace_path"}
    if set(manifest) != expected_fields or manifest.get("schema_version") != SCHEMA_VERSION:
        raise SlurmStageError(f"invalid Slurm staging manifest: {manifest_path}")
    site = manifest.get("site")
    if not isinstance(site, str) or not site:
        raise SlurmStageError(f"invalid site in Slurm staging manifest: {manifest_path}")
    if manifest.get("prepared_path") != str(prepared):
        raise SlurmStageError("staging manifest does not identify this canonical prepared output")
    workspace_text = manifest.get("workspace_path")
    if not isinstance(workspace_text, str) or not Path(workspace_text).is_absolute():
        raise SlurmStageError("slurm_launcher workspace_path must be absolute")
    workspace = Path(os.path.normpath(workspace_text))
    try:
        if workspace.resolve(strict=False) != workspace:
            raise SlurmStageError(f"slurm_launcher workspace_path must be canonical: {workspace}")
    except (OSError, RuntimeError, ValueError) as ex:
        raise SlurmStageError(f"cannot resolve slurm_launcher workspace_path: {workspace}") from ex
    if _paths_overlap(prepared, workspace):
        raise SlurmStageError("prepared kit and runtime workspace must not overlap")
    return prepared, workspace, site


def _read_json(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise SlurmStageError(f"expected a regular JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as ex:
        raise SlurmStageError(f"cannot read JSON file '{path}': {ex}") from ex
    if not isinstance(value, dict):
        raise SlurmStageError(f"JSON file must contain an object: {path}")
    return value


def _create_or_validate_workspace(workspace: Path) -> None:
    workspace.mkdir(parents=True, mode=0o700, exist_ok=True)
    info = workspace.lstat()
    if workspace.is_symlink() or not stat.S_ISDIR(info.st_mode):
        raise SlurmStageError(f"workspace must be a real directory: {workspace}")
    if info.st_uid != os.geteuid():
        raise SlurmStageError(f"workspace must be owned by current uid {os.geteuid()}: {workspace}")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise SlurmStageError(f"workspace must not grant group/world permissions: {workspace}")


def _establish_identity(workspace: Path, site: str) -> dict:
    control = workspace / CONTROL_DIR
    control.mkdir(mode=0o700, exist_ok=True)
    if control.is_symlink() or not control.is_dir():
        raise SlurmStageError(f"Slurm control path must be a real directory: {control}")
    identity_path = control / DEPLOYMENT_FILE
    if identity_path.exists():
        identity = _read_json(identity_path)
        if (
            set(identity) != {"schema_version", "deployment_uuid", "site"}
            or type(identity.get("schema_version")) is not int
            or identity["schema_version"] != SCHEMA_VERSION
        ):
            raise SlurmStageError(f"invalid workspace deployment identity: {identity_path}")
        if identity.get("site") != site:
            raise SlurmStageError("prepared kit site does not match the workspace identity")
        try:
            parsed_uuid = uuid.UUID(identity["deployment_uuid"])
        except (KeyError, TypeError, ValueError, AttributeError) as ex:
            raise SlurmStageError(f"invalid workspace deployment identity: {identity_path}") from ex
        if parsed_uuid.version != 4 or str(parsed_uuid) != identity["deployment_uuid"]:
            raise SlurmStageError(f"invalid workspace deployment identity: {identity_path}")
        return identity

    for path in control.glob(f".{DEPLOYMENT_FILE}.*"):
        path.unlink()
    if set(os.listdir(workspace)) - {CONTROL_DIR} or os.listdir(control):
        raise SlurmStageError("workspace contains deployment data but has no identity")
    identity = {
        "schema_version": SCHEMA_VERSION,
        "deployment_uuid": str(uuid.uuid4()),
        "site": site,
    }
    fd, temp_name = tempfile.mkstemp(prefix=f".{DEPLOYMENT_FILE}.", dir=control)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(identity, stream)
            stream.write("\n")
        os.replace(temp_name, identity_path)
    finally:
        Path(temp_name).unlink(missing_ok=True)
    return identity


def _validate_runtime_links(workspace: Path) -> None:
    for name in ("startup", "local"):
        path = workspace / name
        if os.path.lexists(path) and (not path.is_symlink() or os.readlink(path) != f"{KIT_DIR}/{name}"):
            raise SlurmStageError(f"runtime {name} must be the relative symlink '{KIT_DIR}/{name}'")


def _replace_kit(prepared: Path, workspace: Path) -> None:
    current = workspace / KIT_DIR
    next_kit = workspace / NEXT_KIT_DIR
    for path in (current, next_kit):
        if os.path.lexists(path) and (path.is_symlink() or not path.is_dir()):
            raise SlurmStageError(f"runtime kit path must be a real directory: {path}")
    if next_kit.exists():
        shutil.rmtree(next_kit)
    shutil.copytree(prepared, next_kit)
    if current.exists():
        shutil.rmtree(current)
    os.replace(next_kit, current)


def _ensure_runtime_links(workspace: Path) -> None:
    for name in ("startup", "local"):
        path = workspace / name
        if not os.path.lexists(path):
            os.symlink(f"{KIT_DIR}/{name}", path)
