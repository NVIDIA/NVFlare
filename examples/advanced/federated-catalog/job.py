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

"""Run catalog discovery with SimEnv and inspect the persisted, policy-filtered metadata."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from prepare_data import DEFAULT_DATA, SITES
from sim_components import SimulationCatalogController, SimulationCatalogExecutor

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.job_def import RunStatus
from nvflare.app_common.fdn.catalog.common.payload import MANIFEST_FILE, Selection, parse_manifest
from nvflare.app_common.fdn.catalog.common.schema import DEFAULT_LIMITS, ReadLimits
from nvflare.app_common.fdn.catalog.server.reconcile import Reconciler
from nvflare.app_common.fdn.catalog.server.search import SearchQuery, entry_to_dict
from nvflare.app_common.fdn.catalog.server.store import CatalogStore
from nvflare.job_config.api import FedJob
from nvflare.recipe import SimEnv
from nvflare.recipe.spec import Recipe

DEFAULT_WORKSPACE = "/tmp/nvflare/federated-catalog/simulation"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA)
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--store-root", default="/tmp/nvflare/federated-catalog/store")
    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()
    for site in SITES:
        if not (data_root / site / "local/fdn/default.json").is_file():
            parser.error("Run python prepare_data.py first (using the same --data-root).")
    store_root = str(Path(args.store_root).resolve())
    job = FedJob(name="catalog_read", min_clients=2)
    job.to_server(SimulationCatalogController(store_root))
    job.to_clients(SimulationCatalogExecutor(str(data_root)), tasks=["catalog.read"])
    recipe = Recipe(job)
    env = SimEnv(clients=SITES, num_threads=2, workspace_root=args.workspace)
    run = recipe.execute(env)
    if run.get_status() != RunStatus.FINISHED_COMPLETED.value:
        raise RuntimeError(f"Catalog simulation failed: {run.get_status()}")

    # In simulation the driver takes the place of the production parent process.
    # Never use this unsigned local handoff in a deployed federation.
    manifests = list((Path(env.workspace_root) / "catalog_read" / "server").rglob(MANIFEST_FILE))
    if len(manifests) != 1:
        raise RuntimeError(f"Expected one catalog manifest, found {len(manifests)}")
    entries = parse_manifest(json.loads(manifests[0].read_text())["manifest"])
    store = CatalogStore(store_root)
    reconciler = Reconciler(store, ReadLimits(**DEFAULT_LIMITS))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for site in SITES:
        entry = entries[site]
        if entry.rc != ReturnCode.OK or entry.result is None or entry.result.get("status") != "OK":
            raise RuntimeError(f"Catalog read failed for {site}: {entry}")
        reconciler.apply("default", site, Selection(None), entry.result, now)
    # Reopen to demonstrate that discovery survives a new store instance.
    store = CatalogStore(store_root)
    page = store.search("default", SearchQuery(page_size=100))
    expected = {
        (site, "clinical")
        for site in SITES
        if json.loads((data_root / site / "local/fdn/default.json").read_text())["datasets"]["clinical"]["share"]
    }
    if {(e.client, e.dataset) for e in page.entries} != expected:
        raise RuntimeError("Discovery does not match the current site publication policies")
    for entry in page.entries:
        catalog = entry.catalog
        if catalog is None or catalog.owner is not None:
            raise RuntimeError("Expected available metadata with owner excluded")
        if {c.name for t in catalog.tables for c in t.columns} != {"age", "bmi"}:
            raise RuntimeError("Expected only the policy-approved age and bmi columns")
    print("Verified site selection, dataset withholding and column/owner exclusions.")
    output = {"entries": [entry_to_dict(e) for e in page.entries], "total": page.total}
    result = Path(store_root) / "discovery.json"
    result.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    print(f"Persisted discovery result: {result}")


if __name__ == "__main__":
    main()
