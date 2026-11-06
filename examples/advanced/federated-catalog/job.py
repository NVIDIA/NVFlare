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
from pathlib import Path

from client import ParquetCatalogReader
from prepare_data import DEFAULT_DATA, SITES

from nvflare.recipe import SimEnv
from nvflare.recipe.catalog import CatalogRecipe

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
    recipe = CatalogRecipe(
        sites=SITES,
        catalog_reader=ParquetCatalogReader(),
        site_config_root=str(data_root),
        output_dir=store_root,
    )
    recipe.execute(SimEnv(clients=SITES, num_threads=2, workspace_root=args.workspace))
    output = recipe.catalog_results
    expected = {
        (site, "clinical")
        for site in SITES
        if json.loads((data_root / site / "local/fdn/default.json").read_text())["datasets"]["clinical"]["share"]
    }
    if {(e["client"], e["dataset"]) for e in output["entries"]} != expected:
        raise RuntimeError("Discovery does not match the current site publication policies")
    for entry in output["entries"]:
        metadata = entry.get("metadata")
        if metadata is None or metadata["owner"] is not None:
            raise RuntimeError("Expected available metadata with owner excluded")
        if {c["name"] for t in metadata["tables"] for c in t["columns"]} != {"age", "bmi"}:
            raise RuntimeError("Expected only the policy-approved age and bmi columns")
    print("Verified site selection, dataset withholding and column/owner exclusions.")
    result = Path(store_root) / "discovery.json"
    print(json.dumps(output, indent=2))
    print(f"Persisted discovery result: {result}")


if __name__ == "__main__":
    main()
