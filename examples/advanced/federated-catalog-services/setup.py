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

"""Provision this example's POC and start/stop its catalog services."""
import argparse
import json
import secrets
import subprocess
import time

import requests
from common import PIN, ROOT, WORKSPACE, prod_dir, write_json


def compose(*args):
    subprocess.run(["docker", "compose", *args], cwd=ROOT, check=True, timeout=300)


def setup():
    if WORKSPACE.exists():
        raise RuntimeError("Workspace already exists; use the existing setup or choose a fresh example directory")
    env_file = ROOT / ".env"
    if not env_file.exists():
        env_file.write_text("POSTGRES_PASSWORD=" + secrets.token_urlsafe(24) + "\n")
        env_file.chmod(0o600)
    (ROOT / "data/generated").mkdir(parents=True, exist_ok=True)
    assets = ROOT / "data/playground"
    assets.mkdir(parents=True, exist_ok=True)
    for name, source in [("sanctuary.py", "marimo-playground/notebooks/sanctuary.py"), ("LICENSE", "LICENSE")]:
        response = requests.get(
            f"https://raw.githubusercontent.com/open-lakehouse/unitycatalog-playground/{PIN}/{source}", timeout=30
        )
        response.raise_for_status()
        (assets / name).write_bytes(response.content)
    compose("up", "-d", "--wait")
    deadline = time.monotonic() + 90
    while True:
        try:
            if requests.get("http://127.0.0.1:18080/api/2.1/unity-catalog/catalogs", timeout=2).status_code == 200:
                break
        except requests.RequestException:
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("Unity Catalog did not become ready; inspect docker compose logs")
        time.sleep(1)
    from nvflare.tool.poc.poc_commands import prepare_poc_provision

    (WORKSPACE / "jobs").mkdir(parents=True, exist_ok=True)
    prepare_poc_provision(
        clients=["site-1", "site-2"],
        number_of_clients=2,
        workspace=str(WORKSPACE),
        docker_image=None,
        examples_dir=str(WORKSPACE / "jobs"),
    )
    server = prod_dir() / "server/local"
    resources_file = server / "resources.json"
    resources = json.loads(
        (resources_file if resources_file.exists() else server / "resources.json.default").read_text()
    )
    resources["components"].append(
        {"id": "catalog_manager", "path": "nvflare.app_common.fdn.catalog.server.manager.CatalogManager"}
    )
    allow = resources.get("class_allow_list", [])
    manager = "nvflare.app_common.fdn.catalog.server.manager.CatalogManager"
    if isinstance(allow, list) and manager not in allow:
        allow.append(manager)
    write_json(resources_file, resources)
    print("Setup complete. Next: python prepare_data.py, then python job.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["setup", "stop"], nargs="?", default="setup")
    args = parser.parse_args()
    if args.action == "setup":
        setup()
    else:
        from nvflare.tool.poc.poc_commands import _stop_poc

        try:
            if WORKSPACE.exists():
                _stop_poc(str(WORKSPACE), excluded=["admin@nvidia.com"], services_list=[])
        finally:
            compose("down")


if __name__ == "__main__":
    main()
