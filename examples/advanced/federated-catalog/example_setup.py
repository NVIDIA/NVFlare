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
  3. write local/resources.json from the JSON below  configure server services, including CatalogManager
  4. CATALOG_KITS in .env                              tells `docker compose` where the kits are

Run it again at any time: existing kits, other server components and other .env settings are kept.
"""

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
# The complete server local/resources.json, as an operator would author it.
# Omitting class_allow_list uses FLARE's built-in default policy; enforcement stays enabled.
SERVER_RESOURCES_JSON = """
{
  "format_version": 2,
  "class_list_enforcement_mode": "enforce",
  "servers": [
    {
      "admin_storage": "transfer",
      "max_num_clients": 100,
      "heart_beat_timeout": 600,
      "admin_timeout": 10.0
    }
  ],
  "snapshot_persistor": {
    "path": "nvflare.app_common.state_persistors.storage_state_persistor.StorageStatePersistor",
    "args": {
      "uri_root": "/",
      "storage": {
        "path": "nvflare.app_common.storages.filesystem_storage.FilesystemStorage",
        "args": {
          "root_dir": "/tmp/nvflare/snapshot-storage",
          "uri_root": "/"
        }
      }
    }
  },
  "components": [
    {
      "id": "job_scheduler",
      "path": "nvflare.app_common.job_schedulers.job_scheduler.DefaultJobScheduler",
      "args": {
        "max_jobs": 4
      }
    },
    {
      "id": "job_manager",
      "path": "nvflare.apis.impl.job_def_manager.SimpleJobDefManager",
      "args": {
        "uri_root": "/tmp/nvflare/jobs-storage",
        "job_store_id": "job_store"
      }
    },
    {
      "id": "job_store",
      "path": "nvflare.app_common.storages.filesystem_storage.FilesystemStorage"
    },
    {
      "id": "process_launcher",
      "path": "nvflare.app_common.job_launcher.server_process_launcher.ServerProcessJobLauncher"
    },
    {
      "id": "log_receiver",
      "path": "nvflare.app_common.logging.job_log_receiver.JobLogReceiver"
    },
    {
      "id": "catalog_manager",
      "path": "nvflare.app_common.fdn.catalog.server.manager.CatalogManager"
    }
  ]
}
"""
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

    # 3. Write the same complete JSON file an operator would put in the server startup kit.
    resources = kits / server / "local" / "resources.json"
    step(
        3,
        "Configure the server resources",
        f"Writes the SERVER_RESOURCES_JSON template to {shown(resources)} for a new kit.",
    )
    config = json.loads(SERVER_RESOURCES_JSON)
    if resources.exists():
        existing = json.loads(resources.read_text())
        if not any(
            component.get("id") == "catalog_manager"
            and component.get("path") == "nvflare.app_common.fdn.catalog.server.manager.CatalogManager"
            for component in existing.get("components", [])
        ):
            raise ValueError(
                f"{resources} already exists without CatalogManager. Use SERVER_RESOURCES_JSON as a reference "
                "to enable it in your existing configuration; setup will not overwrite your resources."
            )
        print(f"  Keeping existing {shown(resources)}", flush=True)
    else:
        write_private(resources, json.dumps(config, indent=2) + "\n")
        print(SERVER_RESOURCES_JSON, flush=True)
    for site in sites:
        source = example_root / "config" / site / "study_runtime.yaml"
        target = kits / site / "local" / "study_runtime.yaml"
        if source.exists() and not target.exists():
            run_command("cp", source, target)

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

    run_command("nvflare", "config", "list")
    print("\nNext: docker compose up -d --build --wait")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
