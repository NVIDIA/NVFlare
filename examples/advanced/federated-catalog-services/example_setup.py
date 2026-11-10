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
  3. .env and .sdk/                                select startup kits and the matching container SDK

Rerunning setup reuses existing kits and writes .env again. Kits created for a different SDK are refused;
add --reprovision to replace them.
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path

import yaml
from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parent
STEPS = 3
KIT_SDK_FILE = ".example-sdk"  # beside the kits: the container SDK they were created for


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


def installed_sdk_package():
    """Build the containers from the same SDK revision that selects the downloaded example."""
    sdk_dir = ROOT / ".sdk"
    sdk_dir.mkdir(exist_ok=True)
    (sdk_dir / ".keep").touch()
    override = os.environ.get("NVFLARE_PACKAGE")
    if override:
        if "\n" in override or "\r" in override:
            raise ValueError(
                "NVFLARE_PACKAGE contains a line break. Put the wheel path or package requirement on one line, "
                "for example: export NVFLARE_PACKAGE=/tmp/nvflare-sdk-wheels/<wheel>.whl[catalog]"
            )
        wheel = re.fullmatch(r"(.+\.whl)(\[[^\[\]]+\])?", override)
        if wheel:
            source = Path(wheel[1]).expanduser().resolve(strict=True)
            target = sdk_dir / source.name
            if source != target.resolve():
                shutil.copyfile(source, target)
            return f"/opt/sdk/{source.name}{wheel[2] or '[catalog]'}"
        return override
    from nvflare import _version

    version = _version.get_versions()
    revision = version.get("full-revisionid")
    if version.get("dirty") or not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(
            "Cannot identify a clean installed NVFlare revision. Use a clean installation or set NVFLARE_PACKAGE "
            "to a compatible SDK wheel/package for the container build."
        )
    return f"nvflare[catalog] @ https://github.com/NVIDIA/NVFlare/archive/{revision}.tar.gz"


def installed_sdk_base_version():
    """Supply Versioneer's source-archive fallback without fixing the example to a release number."""
    from nvflare import _version

    try:
        return Version(_version.get_versions()["version"]).base_version
    except InvalidVersion:
        # Local archive/backup Git tags can make Versioneer's display string non-PEP-440.
        for name in ("nvflare", "nvflare-nightly"):
            try:
                return Version(distribution_version(name)).base_version
            except PackageNotFoundError:
                continue
        raise


def main():
    example_root = ROOT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--workspace", type=Path, default=example_root / "workspace", help="where the startup kits are written"
    )
    parser.add_argument("--reprovision", action="store_true", help="create new startup kits even if some exist")
    args = parser.parse_args()
    sdk_package = installed_sdk_package()
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
        kits = generations()[-1]
        (kits / KIT_SDK_FILE).write_text(sdk_package)
    else:
        kits = generations()[-1]
        recorded = (kits / KIT_SDK_FILE).read_text() if (kits / KIT_SDK_FILE).is_file() else None
        if recorded != sdk_package:
            # Kits carry the server and site components of the SDK that created them; mixing SDKs fails jobs.
            raise ValueError(
                f"The startup kits in {shown(kits)} were created for a different NVFlare SDK "
                f"({recorded or 'unknown'}); this setup selects {sdk_package}. "
                "Rerun with --reprovision to create matching kits."
            )
        print("  Kits already exist for this SDK; reusing them. Add --reprovision to create new ones.", flush=True)
    for participant in participants:
        if not (kits / participant["name"] / "startup").is_dir():
            raise ValueError(f"Incomplete kit generation {kits}; rerun with --reprovision")
    print(f"  Kits: {shown(kits)}", flush=True)

    # 2. The admin kit: the identity `nvflare job`, `nvflare system` and `get_catalog` connect with.
    step(
        2,
        f"Register the admin kit as {admin}",
        "The admin kit stays on your computer; it is never mounted into a container. `nvflare` commands use the active one.",
    )
    run_command("nvflare", "config", "add", admin, kits / admin, "--force")
    run_command("nvflare", "config", "use", admin)

    # 3. Docker Compose reads .env from the example directory automatically.
    #    Pin the container SDK to the revision selected by the installed NVFlare.
    env_file = example_root / ".env"
    step(3, "Configure Docker Compose", f"Writes the startup kit location and SDK selection to {shown(env_file)}.")
    values = {
        "CATALOG_KITS": str(kits),
        "CATALOG_SDK_PACKAGE": sdk_package,
        "CATALOG_SDK_BASE_VERSION": installed_sdk_base_version(),
    }
    content = "".join(
        f"{key}={json.dumps(value.replace('$', '$$'), ensure_ascii=False)}\n" for key, value in values.items()
    )
    write_private(env_file, content)
    print(content, end="", flush=True)

    run_command("nvflare", "config", "list")
    print("\nNext: docker compose up -d --build --wait")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
