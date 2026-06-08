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

import os
import shutil
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Iterable, Optional

_SKILL_PACKAGES = {
    "nvflare-autofl": "nvflare.agent.skills.autofl",
}


def available_skills():
    """Return bundled agent skills that can be installed."""

    return [{"name": name, "package": package} for name, package in sorted(_SKILL_PACKAGES.items())]


def install_skills(
    target_dir: Optional[str] = None,
    skill_names: Optional[Iterable[str]] = None,
    dry_run: bool = False,
):
    """Install bundled NVFlare agent skills into ``target_dir``.

    The default call remains safe for existing ``poc prepare`` users: if no
    target directory is supplied and ``NVFLARE_SKILLS_DIR`` is unset, the
    installer records a skip and performs no writes.  This function intentionally
    returns a summary dict and does not raise, because callers treat skills as an
    optional enhancement.

    Args:
        target_dir: Destination directory for skill subdirectories.
        skill_names: Optional subset of bundled skill names.
        dry_run: If True, report planned actions without writing files.

    Returns:
        Summary with ``installed``, ``skipped``, ``backed_up``, and ``errors``.
    """

    selected_names = list(skill_names) if skill_names is not None else sorted(_SKILL_PACKAGES)
    resolved_target = target_dir or os.environ.get("NVFLARE_SKILLS_DIR")
    result = {
        "target_dir": resolved_target,
        "installed": [],
        "skipped": [],
        "backed_up": [],
        "errors": [],
        "dry_run": dry_run,
    }

    if not resolved_target:
        for name in selected_names:
            result["skipped"].append({"name": name, "reason": "no target_dir or NVFLARE_SKILLS_DIR configured"})
        return result

    target = Path(resolved_target).expanduser()
    for name in selected_names:
        package = _SKILL_PACKAGES.get(name)
        if not package:
            result["errors"].append({"name": name, "reason": "unknown bundled skill"})
            continue

        destination = target / name
        try:
            source = resources.files(package)
            if _destination_matches_source(source, destination):
                result["skipped"].append({"name": name, "path": str(destination), "reason": "already current"})
                continue

            backup_path = None
            if destination.exists():
                backup_path = _backup_path(target, name)
                if not dry_run:
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(destination), str(backup_path))
                result["backed_up"].append({"name": name, "path": str(backup_path)})

            if not dry_run:
                destination.mkdir(parents=True, exist_ok=True)
                _copy_skill_resources(source, destination)
            result["installed"].append({"name": name, "path": str(destination)})
        except Exception as e:
            result["errors"].append({"name": name, "reason": str(e)})
    return result


def _backup_path(target: Path, skill_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    return target / ".bak" / timestamp / skill_name


def _destination_matches_source(source, destination: Path) -> bool:
    source_skill = source / "SKILL.md"
    destination_skill = destination / "SKILL.md"
    if not destination_skill.exists() or not source_skill.is_file():
        return False
    return destination_skill.read_bytes() == source_skill.read_bytes()


def _copy_skill_resources(source, destination: Path) -> None:
    for child in source.iterdir():
        if child.name == "__pycache__" or child.name.endswith((".py", ".pyc")):
            continue
        target = destination / child.name
        if child.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            _copy_skill_resources(child, target)
        elif child.is_file():
            target.write_bytes(child.read_bytes())
