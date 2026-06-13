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

"""Container-side skill visibility setup for benchmark runs."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from ..agents.base import SkillExposureResult, SkillExposureSpec
from ..common import write_json

EXPOSURE_ACTION_TIMEOUT_SECONDS = 300


def discover_bundled_skills_root() -> str | None:
    value = os.environ.get("SDK_PACKAGED_SKILLS_ROOT", "")
    return value or None


def copy_optional_metadata_files(source_dir: Path, result_dir: Path, names: tuple[str, ...]) -> dict[str, Any]:
    copied = []
    missing = []
    for name in names:
        if Path(name).name != name:
            raise ValueError(f"metadata file name must not contain path separators: {name}")
        source = source_dir / name
        if source.is_file():
            shutil.copy2(source, result_dir / name)
            copied.append({"source": str(source), "target": str(result_dir / name)})
        else:
            missing.append(str(source))
    payload = {"copied": copied, "missing": missing}
    if missing:
        write_json(result_dir / "skills_metadata_missing.json", payload)
    return payload


def copy_metadata_paths(paths: list[Path], result_dir: Path, container_home: Path | None) -> list[dict[str, str]]:
    copied = []
    missing = []
    for source in paths:
        if container_home is None or not is_within_path(source, container_home):
            write_json(
                result_dir / "skills_state.json",
                {
                    "status": "error",
                    "reason": "metadata_file_outside_container_home",
                    "metadata_file": str(source),
                    "container_home": str(container_home) if container_home else None,
                },
            )
            raise SystemExit(2)
        if source.is_file():
            target = result_dir / source.name
            shutil.copy2(source, target)
            copied.append({"source": str(source), "target": str(target)})
        else:
            missing.append(str(source))
    if missing:
        write_json(result_dir / "skills_metadata_missing.json", {"copied": copied, "missing": missing})
    return copied


def remove_directory_contents(root: Path) -> list[str]:
    disabled = []
    root.mkdir(parents=True, exist_ok=True)
    for child in list(root.iterdir()):
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)
        disabled.append(str(child))
    return disabled


def resolved(path: Path) -> Path:
    return path.resolve(strict=False)


def is_within_path(path: Path, root: Path) -> bool:
    try:
        resolved(path).relative_to(resolved(root))
    except ValueError:
        return False
    return True


def validate_skill_root_scope(spec: SkillExposureSpec, result_dir: Path) -> None:
    if spec.skill_root is None:
        return
    if spec.container_home is None:
        write_json(
            result_dir / "skills_state.json",
            {
                "status": "error",
                "reason": "container_home_required_for_skill_root",
                "mechanism_type": spec.mechanism_type,
                "skill_root": str(spec.skill_root),
            },
        )
        raise SystemExit(2)
    if is_within_path(spec.skill_root, spec.container_home):
        return
    write_json(
        result_dir / "skills_state.json",
        {
            "status": "error",
            "reason": "skill_root_outside_container_home",
            "mechanism_type": spec.mechanism_type,
            "skill_root": str(spec.skill_root),
            "container_home": str(spec.container_home),
        },
    )
    raise SystemExit(2)


def skill_root_has_installed_skills(spec: SkillExposureSpec, result_dir: Path) -> bool:
    if spec.skill_root is None:
        return True
    try:
        return spec.skill_root.is_dir() and any(path.is_dir() for path in spec.skill_root.iterdir())
    except OSError as exc:
        write_json(
            result_dir / "skills_state.json",
            {
                "status": "error",
                "reason": "skill_root_unreadable",
                "mechanism_type": spec.mechanism_type,
                "skill_root": str(spec.skill_root),
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise SystemExit(2) from exc


def timeout_output_text(exc: subprocess.TimeoutExpired) -> str:
    output = exc.output if exc.output is not None else getattr(exc, "stdout", None)
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    if isinstance(output, str):
        return output
    return ""


def run_exposure_action(
    action: list[str],
    result_dir: Path,
    name: str,
    env: dict[str, str],
    *,
    timeout_seconds: int = EXPOSURE_ACTION_TIMEOUT_SECONDS,
) -> tuple[str, str | None]:
    if not action:
        return "skipped", None
    output_path = result_dir / f"skills_{name}_output.txt"
    try:
        result = subprocess.run(
            action,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, **env},
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        output_path.write_text(timeout_output_text(exc), encoding="utf-8")
        write_json(
            result_dir / "skills_state.json",
            {
                "status": "error",
                "reason": "action_timeout",
                "action_name": name,
                "timeout_seconds": timeout_seconds,
                "mechanism_type": "action",
                "command": action,
                "output_ref": str(output_path),
            },
        )
        raise SystemExit(2) from exc
    output_path.write_text(result.stdout or "", encoding="utf-8")
    if result.returncode != 0:
        write_json(
            result_dir / "skills_state.json",
            {
                "status": "error",
                "reason": f"{name}_action_failed",
                "exit_code": result.returncode,
                "mechanism_type": "action",
                "command": action,
                "output_ref": str(output_path),
            },
        )
        raise SystemExit(2)
    return "passed", str(output_path)


def apply_skill_exposure(
    *,
    spec: SkillExposureSpec,
    skills_enabled: bool,
    result_dir: Path,
    sdk_image_kind: str,
    bundled_skills_root: Callable[[], str | None] = discover_bundled_skills_root,
) -> SkillExposureResult:
    validate_skill_root_scope(spec, result_dir)
    if spec.mechanism_type == "none":
        result = SkillExposureResult(status="skipped", mechanism_type=spec.mechanism_type)
        write_json(
            result_dir / "skills_state.json",
            {
                "status": result.status,
                "skills_enabled": skills_enabled,
                "mechanism_type": spec.mechanism_type,
                "source": sdk_image_kind,
            },
        )
        return result

    if skills_enabled:
        setup_status, setup_output_ref = run_exposure_action(
            spec.setup_action, result_dir, "setup_action", spec.environment
        )
        if not skill_root_has_installed_skills(spec, result_dir):
            write_json(
                result_dir / "skills_state.json",
                {
                    "status": "error",
                    "reason": f"preinstalled skills are missing from {spec.skill_root}",
                    "mechanism_type": spec.mechanism_type,
                },
            )
            raise SystemExit(2)
        metadata_files = copy_metadata_paths(spec.metadata_files, result_dir, spec.container_home)
        probe_status, probe_output_ref = run_exposure_action(
            spec.probe_action, result_dir, "probe_action", spec.environment
        )
        write_json(
            result_dir / "skills_state.json",
            {
                "status": "prepared",
                "source": sdk_image_kind,
                "skills_enabled": True,
                "mechanism_type": spec.mechanism_type,
                "setup_status": setup_status,
                "setup_output_ref": setup_output_ref,
                "probe_status": probe_status,
                "probe_output_ref": probe_output_ref,
            },
        )
        return SkillExposureResult(
            status="prepared",
            mechanism_type=spec.mechanism_type,
            installed_paths=[str(spec.skill_root)] if spec.skill_root else [],
            launch_args=list(spec.launch_args),
            environment=dict(spec.environment),
            probe_status=probe_status,
            probe_output_ref=probe_output_ref,
            metadata_files=metadata_files,
        )

    disable_status, disable_output_ref = run_exposure_action(
        spec.disable_action, result_dir, "disable_action", spec.environment
    )
    disabled_paths = remove_directory_contents(spec.skill_root) if spec.skill_root else []
    parser_warnings = []
    if not spec.skill_root:
        parser_warnings.append("no skill_root configured for disabled skill exposure")
    bundled_root = bundled_skills_root() if spec.disable_packaged_source else None
    removed_packaged_source = False
    if bundled_root:
        path = Path(bundled_root)
        if not is_within_path(path, Path("/workspace")):
            write_json(
                result_dir / "skills_state.json",
                {
                    "status": "error",
                    "reason": "bundled_skill_source_outside_workspace",
                    "mechanism_type": spec.mechanism_type,
                    "bundled_skill_source_path": str(path),
                },
            )
            raise SystemExit(2)
        if path.is_dir():
            shutil.rmtree(path)
            removed_packaged_source = True
            disabled_paths.append(str(path))

    write_json(
        result_dir / "skills_state.json",
        {
            "status": "disabled",
            "source": sdk_image_kind,
            "skills_enabled": False,
            "image_kind": sdk_image_kind,
            "mechanism_type": spec.mechanism_type,
            "disabled_paths": disabled_paths,
            "disable_status": disable_status,
            "disable_output_ref": disable_output_ref,
            "note": parser_warnings[0] if parser_warnings else "",
            "packaged_skill_source_removed_during_agent": removed_packaged_source,
            "packaged_skill_source_path": bundled_root,
            "reporting_note": (
                "Wrapper-side reports run from the skills image so benchmark contracts are available "
                "outside the measured agent container."
            ),
        },
    )
    write_json(
        result_dir / "skills_list.json",
        {"status": "skipped", "installed": [], "reason": "skills intentionally removed for baseline run"},
    )
    return SkillExposureResult(
        status="disabled",
        mechanism_type=spec.mechanism_type,
        disabled_paths=disabled_paths,
        parser_warnings=parser_warnings,
    )
