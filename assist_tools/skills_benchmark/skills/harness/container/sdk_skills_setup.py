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

"""Apply configured SDK skills setup inside the benchmark image."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

SDK_SKILLS_SOURCE = Path("/tmp/sdk_skills")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def agent_home() -> Path:
    return Path(env("BENCHMARK_AGENT_HOME", "/workspace/agent-home"))


def skills_target() -> Path:
    return agent_home() / "skills"


def run_command(command: str, output_path: Path) -> None:
    if not command:
        raise SystemExit("skills.setup.type=command requires SKILLS_INSTALL_COMMAND")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        subprocess.run(["/bin/bash", "-c", command], check=True, stdout=stream)


def visible_files(path: Path) -> list[str]:
    if not path.is_dir():
        return []
    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        directory = Path(dirpath)
        dirnames[:] = [name for name in dirnames if not (directory / name).is_symlink()]
        for name in filenames:
            item = directory / name
            if item.is_symlink() or not item.is_file():
                continue
            files.append(str(item.relative_to(path)))
    return sorted(files)


def copy_skills_folder() -> dict:
    target = skills_target()
    if not SDK_SKILLS_SOURCE.is_dir():
        raise SystemExit(f"Staged SDK skills folder is missing: {SDK_SKILLS_SOURCE}")
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SDK_SKILLS_SOURCE, target, dirs_exist_ok=True)
    files = visible_files(target)
    return {
        "status": "success",
        "agent": env("BENCHMARK_DOCKER_AGENT", "unknown"),
        "mechanism": "copy",
        "source": str(SDK_SKILLS_SOURCE),
        "target": str(target),
        "file_count": len(files),
    }


def write_generic_list(output_path: Path, *, reason: str | None = None) -> None:
    target = skills_target()
    payload = {
        "status": "success" if reason is None else "skipped",
        "agent": env("BENCHMARK_DOCKER_AGENT", "unknown"),
        "mechanism": env("SKILLS_SETUP_TYPE", "none"),
        "target": str(target),
        "installed": visible_files(target),
    }
    if reason:
        payload["reason"] = reason
    write_json(output_path, payload)


def write_setup_metadata(home: Path) -> None:
    payload = {
        "agent": env("BENCHMARK_DOCKER_AGENT", "unknown"),
        "command": env("SKILLS_INSTALL_COMMAND"),
        "expected_source": env("SKILLS_INSTALL_EXPECTED_SOURCE"),
        "network_isolation_enforced": False,
        "setup_type": env("SKILLS_SETUP_TYPE", "none"),
        "skills_source": str(SDK_SKILLS_SOURCE),
        "target": str(skills_target()),
    }
    write_json(home / "sdk_skills_build_metadata.json", payload)


def main() -> int:
    home = agent_home()
    home.mkdir(parents=True, exist_ok=True)
    install_output = home / env("SKILLS_INSTALL_OUTPUT", "skills_build_install.json")
    list_output = home / env("SKILLS_LIST_OUTPUT", "skills_list.json")
    setup_type = env("SKILLS_SETUP_TYPE", "none")
    write_setup_metadata(home)

    if setup_type == "command":
        run_command(env("SKILLS_INSTALL_COMMAND"), install_output)
        list_command = env("SKILLS_LIST_COMMAND")
        if list_command:
            run_command(list_command, list_output)
        else:
            write_generic_list(list_output, reason="SKILLS_LIST_COMMAND is empty")
    elif setup_type == "copy":
        write_json(install_output, copy_skills_folder())
        list_command = env("SKILLS_LIST_COMMAND")
        if list_command:
            run_command(list_command, list_output)
        else:
            write_generic_list(list_output)
    elif setup_type == "none":
        write_json(
            install_output,
            {
                "status": "skipped",
                "agent": env("BENCHMARK_DOCKER_AGENT", "unknown"),
                "mechanism": "none",
                "reason": "SDK profile skills.setup.type is none",
            },
        )
        write_generic_list(list_output, reason="SDK profile skills.setup.type is none")
    else:
        raise SystemExit(f"Unsupported SKILLS_SETUP_TYPE={setup_type!r}")

    print(list_output.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
