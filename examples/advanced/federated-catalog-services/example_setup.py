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

Nothing here is specific to this example. Every step prints the command it runs, so you can repeat it by hand or
use it in your own deployment:

  1. nvflare provision -p project.yml -w <workspace>   create startup kits for the server, sites and admin
  2. nvflare config add <admin> <kit> / config use     register the admin kit, so `nvflare` commands can connect
  3. add the CatalogManager to the server's            the one deployment setting the catalog needs
     local/resources.json
  4. CATALOG_KITS in .env                              tells `docker compose` where the kits are

Run it again at any time: existing kits, other server components and other .env settings are kept.
"""

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

from nvflare.apis.fl_constant import WorkspaceConstants
from nvflare.lighter.utils import load_yaml

ROOT = Path(__file__).resolve().parent
COMPONENT = {"id": "catalog_manager", "path": "nvflare.app_common.fdn.catalog.server.manager.CatalogManager"}
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


def nvflare(*args, verbose=False):
    """Show the command as a user would type it (paths shortened), then run it with absolute paths, using this
    Python so it is the same installation."""
    typed = [shown(a) if isinstance(a, Path) else str(a) for a in args]
    print(f"  $ {shlex.join(['nvflare', *typed])}", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "nvflare.cli", *map(str, args)], capture_output=not verbose, text=True, check=False
    )
    if result.returncode:  # the command's own output is shown only when it fails, or with --verbose
        sys.stderr.write(f"{result.stdout or ''}{result.stderr or ''}")
        raise subprocess.CalledProcessError(result.returncode, ["nvflare", *typed])


def write_private(path, content):
    temporary = path.with_name(path.name + ".tmp")
    temporary.touch(mode=0o600, exist_ok=True)
    temporary.chmod(0o600)
    temporary.write_text(content)
    temporary.replace(path)


def main(example_root=ROOT):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--workspace", type=Path, default=example_root / "workspace", help="where the startup kits are written"
    )
    parser.add_argument("--reprovision", action="store_true", help="create new startup kits even if some exist")
    parser.add_argument("--verbose", action="store_true", help="show the full output of every command")
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    if "\n" in str(workspace) or "\r" in str(workspace):
        raise ValueError("Workspace paths cannot contain newlines")
    project_file = example_root / "project.yml"
    project = load_yaml(str(project_file))
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
        nvflare("provision", "-p", project_file, "-w", workspace, "--force", verbose=args.verbose)
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
    nvflare("config", "add", admin, kits / admin, "--force", verbose=args.verbose)
    nvflare("config", "use", admin, verbose=args.verbose)

    # 3. The one catalog-specific deployment setting: the server runs a CatalogManager component. Any server component
    #    is declared the same way, in the `components` list of the server's local/resources.json.
    local = kits / server / WorkspaceConstants.SITE_FOLDER_NAME
    resources = local / WorkspaceConstants.RESOURCES_CONFIG
    source = resources if resources.exists() else local / WorkspaceConstants.DEFAULT_RESOURCES_CONFIG
    step(
        3,
        "Enable the catalog on the server",
        f"Adds the CatalogManager to {shown(resources)}; the other settings and components are kept.",
    )
    config = json.loads(source.read_text())
    components = config.setdefault("components", [])
    existing = next((c for c in components if c.get("id") == COMPONENT["id"]), None)
    if existing is not None and existing.get("path") != COMPONENT["path"]:
        raise ValueError(f"{COMPONENT['id']} in {resources} already has a different implementation")
    if existing is None:
        components.append(COMPONENT)
    write_private(resources, json.dumps(config, indent=2) + "\n")
    print(f"  Component: {json.dumps(COMPONENT)}", flush=True)
    for site in sites:
        source = example_root / "config" / site / "study_runtime.yaml"
        target = kits / site / WorkspaceConstants.SITE_FOLDER_NAME / "study_runtime.yaml"
        if source.exists() and not target.exists():
            write_private(target, source.read_text())
            print(f"  Site runtime: {shown(source)} -> {shown(target)}", flush=True)

    # 4. Docker Compose mounts each participant's own kit, so it needs to know where they are. Other settings in
    #    .env (for example POSTGRES_PASSWORD) are kept.
    step(
        4,
        "Tell Docker Compose where the kits are",
        "Each container mounts only its own kit; the admin kit is not mounted.",
    )
    env_file = example_root / ".env"
    lines = env_file.read_text().splitlines() if env_file.exists() else []
    lines = [line for line in lines if not re.match(r"^\s*(?:export\s+)?CATALOG_KITS\s*=", line)]
    value = str(kits).replace("'", "\\'")
    lines.append(f"CATALOG_KITS='{value}'")
    write_private(env_file, "\n".join(lines) + "\n")
    print(f"  CATALOG_KITS={shown(kits)}  (in {shown(env_file)})", flush=True)

    print(
        "\nDone. To see what is registered:  nvflare config list\n"
        "Next:                             docker compose up -d --build --wait"
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
