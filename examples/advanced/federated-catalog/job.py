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

"""Read every site's catalog in SimEnv, print a summary and save catalog_results.json."""

import argparse
from pathlib import Path

from constants import DEFAULT_DATA, DEFAULT_RESULTS, SITES

from nvflare.recipe import SimEnv
from nvflare.recipe.catalog import get_catalog

DEFAULT_WORKSPACE = "/tmp/nvflare/federated-catalog/simulation"


def print_catalog(results):
    """One line per site, then one per published dataset. Unavailable and skipped datasets carry no metadata."""
    print(f"\nCatalog of Study {results['study']} ({len(results['datasets'])} datasets)")
    for site, info in results["sites"].items():
        extra = "".join(f", {key}: {', '.join(info[key])}" for key in ("unavailable", "skipped") if info[key])
        print(f"  {site}: {info['status']}{extra}")
    for entry in results["datasets"]:
        metadata = entry["metadata"]
        tables = ", ".join(f"{t['name']} ({len(t['columns'])} columns)" for t in metadata["tables"])
        print(f"  {entry['site']} / {entry['dataset']} [{metadata['provenance']['source']}]: {tables}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA)
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--output", default=DEFAULT_RESULTS, help="where to save catalog_results.json")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results = get_catalog(
        SimEnv(clients=SITES, num_threads=2, workspace_root=args.workspace),
        site_config_root=str(Path(args.data_root).resolve()),
        output_path=args.output,
    )
    print_catalog(results)
    print(f"Full metadata: {args.output}")


if __name__ == "__main__":
    main()
