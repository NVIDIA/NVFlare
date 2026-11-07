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

"""One-shot Compose initialization: distribute only each participant's own startup kit."""
import json
import os
import shutil
from pathlib import Path

import yaml
from common import ROOT, write_json

from nvflare.lighter.constants import CtxKey, ProvisionMode
from nvflare.lighter.prov_utils import prepare_builders
from nvflare.lighter.provision import prepare_project
from nvflare.lighter.provisioner import Provisioner
from nvflare.lighter.utils import update_storage_locations


def main():
    config = yaml.safe_load((ROOT / "project.yml").read_text())
    ctx = Provisioner("/provision", prepare_builders(config)).provision(prepare_project(config), mode=ProvisionMode.POC)
    if ctx.get(CtxKey.BUILD_ERROR):
        raise RuntimeError("Provisioning failed")
    generated = Path(ctx.get(CtxKey.CURRENT_PROD_DIR))
    for name, destination in [
        ("nvflare-poc-server", "/kits/server"),
        ("site-1", "/kits/site-1"),
        ("site-2", "/kits/site-2"),
        ("admin@nvidia.com", "/admin-kit"),
    ]:
        target = Path(destination)
        # Persist runtime state and identity across ordinary Compose restarts.
        if destination == "/admin-kit" or not (target / "startup").exists():
            shutil.copytree(generated / name, target, dirs_exist_ok=True)
    server = Path("/kits/server/local")
    resources_file = server / "resources.json"
    update_storage_locations(
        local_dir=str(server),
        workspace="/workspace",
        default_resource_name="resources.json" if resources_file.exists() else "resources.json.default",
    )
    resources = json.loads(
        (resources_file if resources_file.exists() else server / "resources.json.default").read_text()
    )
    manager = "nvflare.app_common.fdn.catalog.server.manager.CatalogManager"
    if not any(c.get("id") == "catalog_manager" for c in resources["components"]):
        resources["components"].append({"id": "catalog_manager", "path": manager})
    if manager not in resources["class_allow_list"]:
        resources["class_allow_list"].append(manager)
    write_json(resources_file, resources)
    path = Path("/admin-kit/startup/fed_admin.json")
    admin = json.loads(path.read_text())
    admin["admin"]["host"] = "localhost"
    write_json(path, admin)
    owner = Path("/example-host").stat()
    for exported in [Path("/admin-kit"), *Path("/admin-kit").rglob("*")]:
        if not exported.is_symlink():
            os.chown(exported, owner.st_uid, owner.st_gid)


if __name__ == "__main__":
    main()
