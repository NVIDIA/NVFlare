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

"""Set up the catalog demo with the standard NVFLARE commands, and show each one.

Every step prints the command it runs, so you can repeat it by hand or
use it in your own deployment:

  1. nvflare provision -p project.yml -w <workspace>   create startup kits for the server, sites and admin
  2. nvflare config add <admin> <kit> / config use     register the admin kit, so `nvflare` commands can connect
  3. install config/server/resources.json            configure server services, including CatalogManager
  4. CATALOG_KITS in .env                              tells `docker compose` where the kits are

Rerunning setup reuses existing kits and installs the example deployment configuration again.
"""

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
STEPS = 4


def shown(path):
    """A path as the user would type it: relative to where they are when that is shorter."""
    path = Path(path)
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def step(number, title, why):
    print(f"\nStep {number}/{STEPS}  {title}\n  {why}", flush=True)


def run_command(*args):
    """Print and run exactly the same CLI command a user can run in their terminal."""
    command = [str(arg) for arg in args]
    print(f"  $ {shlex.join(command)}", flush=True)
    subprocess.run(command, check=True)


def write_private(path, content):
    temporary = path.with_name(path.name + ".tmp")
    temporary.touch(mode=0o600, exist_ok=True)
    temporary.chmod(0o600)
    temporary.write_text(content)
    temporary.replace(path)


def main():
    example_root = ROOT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--workspace", type=Path, default=example_root / "workspace", help="where the startup kits are written"
    )
    parser.add_argument("--reprovision", action="store_true", help="create new startup kits even if some exist")
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    if "\n" in str(workspace) or "\r" in str(workspace):
        raise ValueError("Workspace paths cannot contain newlines")
    project_file = example_root / "project.yml"
    with project_file.open() as stream:
        project = yaml.safe_load(stream)
    project_root = workspace / project["name"]

    def generations():
        return sorted(
            (p for p in project_root.glob("prod_*") if p.is_dir() and re.fullmatch(r"prod_\d+", p.name)),
            key=lambda p: int(p.name.removeprefix("prod_")),
        )

    participants = project["participants"]
    server = next(p["name"] for p in participants if p["type"] == "server")
    admin = next(p["name"] for p in participants if p.get("role") == "project_admin")
    sites = [p["name"] for p in participants if p["type"] == "client"]

    # 1. Startup kits: one directory per participant with its certificates, configuration and start scripts.
    step(
        1,
        f"Create startup kits for the server, {', '.join(sites)} and the admin",
        "`nvflare provision` reads project.yml (participants, ports, host names) and writes one kit for each.",
    )
    if args.reprovision or not generations():
        run_command("nvflare", "provision", "-p", project_file, "-w", workspace, "--force")
    else:
        print("  Kits already exist; reusing them. Add --reprovision to create new ones.", flush=True)
    kits = generations()[-1]
    for participant in participants:
        if not (kits / participant["name"] / "startup").is_dir():
            raise ValueError(f"Incomplete kit generation {kits}; rerun with --reprovision")
    print(f"  Kits: {shown(kits)}", flush=True)

    # 2. The admin kit: the identity `nvflare job`, `nvflare catalog` and `nvflare system` commands connect with.
    step(
        2,
        f"Register the admin kit as {admin}",
        "The admin kit stays on your computer; it is never mounted into a container. `nvflare` commands use the active one.",
    )
    run_command("nvflare", "config", "add", admin, kits / admin, "--force")
    run_command("nvflare", "config", "use", admin)

    # 3. CatalogManager is already declared in config/server/resources.json alongside
    #    the standard server services. Load that deployment file and install it in the kit.
    #    This is equivalent to copying the file to <server kit>/local/resources.json.
    source = example_root / "config" / "server" / "resources.json"
    resources = kits / server / "local" / "resources.json"
    step(3, "Configure the server resources", f"Installs {shown(source)} as {shown(resources)}.")
    config = json.loads(source.read_text())
    write_private(resources, json.dumps(config, indent=2) + "\n")
    for site in sites:
        source = example_root / "config" / site / "study_runtime.yaml"
        target = kits / site / "local" / "study_runtime.yaml"
        if source.exists():
            run_command("cp", source, target)

    # 4. Docker Compose reads .env from the example directory automatically.
    #    Write the same one-line file a user would create after provisioning.
    env_file = example_root / ".env"
    step(4, "Configure Docker Compose", f"Writes the startup kit location to {shown(env_file)}.")
    value = str(kits).replace("'", "\\'")
    content = f"CATALOG_KITS='{value}'\n"
    write_private(env_file, content)
    print(content, end="", flush=True)

    run_command("nvflare", "config", "list")
    print("\nNext: docker compose up -d --build --wait")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
