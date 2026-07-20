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

"""Runtime-neutral orchestration for ``nvflare deploy``."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Protocol

import yaml

from nvflare.tool.cli_output import output_ok
from nvflare.tool.deploy import deploy_common, docker_deploy, k8s_deploy, slurm_deploy
from nvflare.tool.deploy.k8s_stage import reject_replacing_staged_k8_output, stage_k8_deployment, unstage_k8_deployment
from nvflare.tool.deploy.slurm_stage import stage_slurm_deployment

__all__ = ["prepare_deployment", "stage_k8_deployment", "stage_slurm_deployment", "unstage_k8_deployment"]


class _DeployBackend(Protocol):
    def validate_config(self, config: dict[str, Any]) -> None:
        pass

    def prepare(self, kit_info: deploy_common.KitInfo, final_output: Path, config: dict[str, Any]) -> dict[str, Any]:
        pass


def _get_backend(runtime: str) -> _DeployBackend:
    if runtime == deploy_common.RUNTIME_DOCKER:
        return docker_deploy
    if runtime == deploy_common.RUNTIME_K8S:
        return k8s_deploy
    if runtime == deploy_common.RUNTIME_SLURM:
        return slurm_deploy
    deploy_common._fail(
        "INVALID_CONFIG",
        f"Unsupported runtime: {runtime}",
        "Use runtime: docker, runtime: k8s, or runtime: slurm.",
    )
    raise AssertionError("unreachable")


def prepare_deployment(args) -> None:
    kit, output_arg, config_path = _resolve_prepare_inputs(args)

    config = _load_config(config_path)
    runtime = config["runtime"]
    backend = _get_backend(runtime)
    backend.validate_config(config)
    output = _resolve_output_path(kit, output_arg, runtime)
    kit_info = deploy_common.validate_kit(kit)
    if kit_info.role == deploy_common.ROLE_ADMIN:
        deploy_common._fail(
            "UNSUPPORTED_KIT",
            "Admin startup kits are not supported by 'nvflare deploy prepare'.",
            "Use a server or client startup kit.",
        )
    if output == kit or _is_path_relative_to(kit, output):
        deploy_common._fail(
            "INVALID_ARGS",
            "--output must not be the same as --kit or contain it.",
            "Choose a prepared-kit directory outside the kit, or use the default <kit>/prepared/<runtime>.",
        )
    if output.exists() and not output.is_dir():
        deploy_common._fail(
            "INVALID_ARGS", f"Output path exists and is not a directory: {output}", "Choose a directory path."
        )
    reject_replacing_staged_k8_output(output)

    parent_dir = output.parent
    try:
        parent_dir.mkdir(parents=True, exist_ok=True)
    except Exception as ex:
        deploy_common._fail(
            "INVALID_ARGS", f"Cannot create output parent directory: {parent_dir}", f"Check the path: {ex}"
        )

    temp_parent = None if _is_path_relative_to(output, kit) else str(parent_dir)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output.name}.prepare-", dir=temp_parent))
    prepared_dir = temp_dir / output.name
    try:
        shutil.copytree(kit, prepared_dir, ignore=_ignore_output_path(kit, output))
        prepared_info = deploy_common.KitInfo(
            kit_dir=prepared_dir,
            role=kit_info.role,
            name=kit_info.name,
            org=kit_info.org,
            fed_learn_port=kit_info.fed_learn_port,
            admin_port=kit_info.admin_port,
        )
        result = backend.prepare(prepared_info, output, config)

        if output.exists():
            shutil.rmtree(output)
        shutil.move(str(prepared_dir), str(output))
    except SystemExit:
        raise
    except Exception as ex:
        deploy_common._fail(
            "DEPLOY_PREPARE_FAILED", f"Failed to prepare deployment: {ex}", "Check the kit and runtime config."
        )
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    output_ok(result)


def _resolve_prepare_inputs(args) -> tuple[Path, str | None, Path]:
    positional_kit = getattr(args, "kit", None)
    flag_kit = getattr(args, "kit_flag", None)
    if positional_kit and flag_kit:
        deploy_common._fail("INVALID_ARGS", "Specify the startup kit only once.", "Use either positional kit or --kit.")
    kit_arg = positional_kit or flag_kit
    if not kit_arg:
        deploy_common._fail(
            "INVALID_ARGS", "Missing startup kit directory.", "Run nvflare deploy prepare <startup-kit-dir>."
        )

    kit = Path(kit_arg).expanduser().resolve()
    output_arg = getattr(args, "output", None)
    config_arg = getattr(args, "config", None)
    config_path = Path(config_arg).expanduser().resolve() if config_arg else kit / "config.yaml"
    return kit, output_arg, config_path


def _resolve_output_path(kit: Path, output_arg: str | None, runtime: str) -> Path:
    if output_arg:
        return Path(output_arg).expanduser().resolve()
    return kit / "prepared" / runtime


def _is_path_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _ignore_output_path(kit: Path, output: Path):
    if not _is_path_relative_to(output, kit):
        return None

    excluded = output.resolve()
    prepared_root = (kit / "prepared").resolve()

    def _ignore(dir_name: str, names: list[str]) -> list[str]:
        current = Path(dir_name).resolve()
        ignored = []
        for name in names:
            path = (current / name).resolve()
            if path == excluded or path == prepared_root:
                ignored.append(name)
        return ignored

    return _ignore


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        deploy_common._fail(
            "CONFIG_NOT_FOUND", f"Config file not found: {config_path}", "Provide a YAML runtime config file."
        )
    try:
        with config_path.open("rt", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as ex:
        deploy_common._fail("INVALID_CONFIG", f"Failed to parse config file: {ex}", "Ensure the file is valid YAML.")
    if not isinstance(config, dict):
        deploy_common._fail(
            "INVALID_CONFIG",
            "Runtime config must be a YAML mapping.",
            "Add runtime: docker, runtime: k8s, or runtime: slurm.",
        )
    runtime = config.get("runtime")
    if runtime not in {deploy_common.RUNTIME_DOCKER, deploy_common.RUNTIME_K8S, deploy_common.RUNTIME_SLURM}:
        deploy_common._fail(
            "INVALID_CONFIG",
            "Config must contain runtime: docker, runtime: k8s, or runtime: slurm.",
            "Set a supported runtime.",
        )
    return config
