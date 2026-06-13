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

"""Write benchmark Docker build metadata into the configured agent home."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def output(argv: list[str]) -> str:
    return subprocess.check_output(argv, text=True).strip()


def shell_output(command: str) -> str:
    return subprocess.check_output(["/bin/bash", "-lc", command], text=True).strip()


def main() -> None:
    version_cmd = os.environ.get("AGENT_VERSION_COMMAND") or ""
    version_output = shell_output(version_cmd) if version_cmd else ""
    home = Path(os.environ.get("BENCHMARK_AGENT_HOME", "/workspace/agent-home"))
    home.mkdir(parents=True, exist_ok=True)
    metadata = {
        "agent": os.environ.get("BENCHMARK_DOCKER_AGENT", "agent"),
        "agent_cli": os.environ.get("AGENT_CLI_NAME", "agent"),
        "agent_cli_install_command": os.environ.get("AGENT_INSTALL_COMMAND"),
        "agent_cli_version_command": version_cmd,
        "agent_cli_version_output": version_output,
        "apt_package_policy": (
            "Debian apt packages are intentionally not exact-version pinned; rebuild comparability is tracked "
            "through the pinned base image/build args and recorded package/tool versions."
        ),
        "node_image": os.environ.get("NODE_IMAGE"),
        "node_version": output(["node", "--version"]),
        "sdk_import_name": os.environ.get("SDK_IMPORT_NAME"),
        "sdk_package_name": os.environ.get("SDK_PACKAGE_NAME"),
        "skills_install_expected_source": os.environ.get("SKILLS_INSTALL_EXPECTED_SOURCE"),
        "skills_install_network_proof": "not_enforced_by_dockerfile",
        "tree_version": output(["tree", "--version"]).splitlines()[0],
        "uv_image": os.environ.get("UV_IMAGE"),
        "uv_version": output(["uv", "--version"]),
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
    }
    (home / "build_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
