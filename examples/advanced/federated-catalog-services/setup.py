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

"""Prepare this deployment using the standard provision and admin-kit CLIs."""

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
MANAGER = "nvflare.app_common.fdn.catalog.server.manager.CatalogManager"


def cli(*args, step, verbose=False):
    """Run an nvflare command. Show one line for the step; show the command's own output only on failure."""
    command = [sys.executable, "-m", "nvflare.cli", *map(str, args)]
    if verbose:
        print(f"+ {shlex.join(command)}", flush=True)
        subprocess.run(command, check=True)
        return
    print(f"{step} ...", end=" ", flush=True)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        print("failed", flush=True)
        sys.stderr.write(f"+ {shlex.join(command)}\n{result.stdout}{result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, command)
    print("done", flush=True)


def write_config(path, content):
    temporary = path.with_name(path.name + ".tmp")
    temporary.touch(mode=0o600, exist_ok=True)
    temporary.chmod(0o600)
    temporary.write_text(content)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT / "workspace", help="provisioning workspace")
    parser.add_argument(
        "--reprovision", action="store_true", help="create a new kit generation instead of reusing kits"
    )
    parser.add_argument("--verbose", action="store_true", help="show the output of every nvflare command")
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    if "\n" in str(workspace) or "\r" in str(workspace):
        raise ValueError("Workspace paths cannot contain newlines")
    project_file = ROOT / "project.yml"
    project = yaml.safe_load(project_file.read_text())
    project_root = workspace / project["name"]

    def generations():
        return sorted(
            (p for p in project_root.glob("prod_*") if p.is_dir() and re.fullmatch(r"prod_\d+", p.name)),
            key=lambda p: int(p.name.removeprefix("prod_")),
        )

    if args.reprovision or not generations():
        cli(
            "provision",
            "-p",
            project_file,
            "-w",
            workspace,
            "--force",
            step="Creating startup kits",
            verbose=args.verbose,
        )
    kits = generations()[-1]
    for participant in project["participants"]:
        if not (kits / participant["name"] / "startup").is_dir():
            raise ValueError(f"Incomplete kit generation {kits}; rerun with --reprovision")
    server = next(p["name"] for p in project["participants"] if p["type"] == "server")
    admin = next(p["name"] for p in project["participants"] if p.get("role") == "project_admin")
    resources = kits / server / "local" / "resources.json"
    source = resources if resources.exists() else resources.with_suffix(".json.default")
    config = json.loads(source.read_text())
    components = config.setdefault("components", [])
    existing = next((c for c in components if c.get("id") == "catalog_manager"), None)
    if existing is not None and existing.get("path") != MANAGER:
        raise ValueError(f"catalog_manager in {resources} already has a different implementation")
    if existing is None:
        components.append({"id": "catalog_manager", "path": MANAGER})
    write_config(resources, json.dumps(config, indent=2) + "\n")
    cli("config", "add", admin, kits / admin, "--force", step=f"Registering admin kit {admin}", verbose=args.verbose)
    cli("config", "use", admin, step="Selecting it as the active admin", verbose=args.verbose)

    # Keep other Compose settings (for example POSTGRES_PASSWORD) when rerun.
    env_file = ROOT / ".env"
    lines = env_file.read_text().splitlines() if env_file.exists() else []
    lines = [line for line in lines if not re.match(r"^\s*(?:export\s+)?CATALOG_KITS\s*=", line)]
    value = str(kits).replace("'", "\\'")
    lines.append(f"CATALOG_KITS='{value}'")
    write_config(env_file, "\n".join(lines) + "\n")
    print(f"\nReady. Kits: {kits}\nNext: docker compose up -d --build --wait")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
