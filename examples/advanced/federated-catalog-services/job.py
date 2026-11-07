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

"""Run a catalog-only job on a provisioned two-site POC and save its federated discovery view."""
import json
import time
from contextlib import closing

from common import ROOT, prod_dir

from nvflare.app_common.fdn.catalog.admin.catalog_api import CatalogAdminApi
from nvflare.app_common.fdn.catalog.server.search import entry_to_dict
from nvflare.fuel.flare_api.flare_api import new_session
from nvflare.recipe import ProdEnv
from nvflare.recipe.catalog import CatalogRecipe


def main():
    admin = str(prod_dir() / "admin@nvidia.com")
    # ProdEnv submits to the provisioned POC, exactly as to separately deployed sites.
    env = ProdEnv(startup_kit_location=admin, login_timeout=60)
    recipe = CatalogRecipe(sites=["site-1", "site-2"])
    run = recipe.execute(env)
    deadline = time.monotonic() + 300
    while not run.get_status().startswith("FINISHED"):
        if time.monotonic() >= deadline:
            raise TimeoutError("Catalog job did not finish within five minutes")
        time.sleep(1)
    if run.get_status() != "FINISHED:COMPLETED":
        raise RuntimeError(f"Catalog job failed: {run.get_status()}")
    with closing(new_session("admin@nvidia.com", admin, timeout=30)) as session:
        api = CatalogAdminApi(session)
        deadline = time.monotonic() + 30
        while True:
            entries = [entry_to_dict(entry) for entry in api.search()]
            if len(entries) == 2:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Server did not publish both catalogs; inspect POC server/site logs")
            time.sleep(0.5)
    output = {"entries": entries, "total": len(entries)}
    (ROOT / "discovery.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
