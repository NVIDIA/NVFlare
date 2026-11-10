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

"""Read the catalog of the running federation's sites, print a summary and save catalog_results.json."""

import argparse

from nvflare.recipe import ProdEnv
from nvflare.recipe.catalog import CatalogReadError, get_catalog
from nvflare.tool.kit.kit_config import (
    StartupKitConfigError,
    resolve_admin_user_and_dir_from_startup_kit,
    resolve_startup_kit_dir,
)


def print_catalog(results):
    """One line per site, then one per published dataset. Unavailable and skipped datasets carry no metadata."""
    print(f"\nCatalog of Study {results['study']}, job {results['job_id']}")
    for site, info in results["sites"].items():
        extra = "".join(f", {key}: {', '.join(info[key])}" for key in ("unavailable", "skipped") if info[key])
        reason = f" ({info['reason_code']})" if info["reason_code"] else ""
        print(f"  {site}: {info['status']}{reason}{extra}")
    for entry in results["datasets"]:
        metadata = entry["metadata"]
        tables = ", ".join(f"{t['name']} ({len(t['columns'])} columns)" for t in metadata["tables"])
        print(f"  {entry['site']} / {entry['dataset']} [{metadata['provenance']['source']}]: {tables}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="catalog_results.json", help="where to save catalog_results.json")
    args = parser.parse_args()
    # The admin startup kit that example_setup.py registered with `nvflare config use`.
    try:
        username, kit_dir = resolve_admin_user_and_dir_from_startup_kit(resolve_startup_kit_dir())
    except StartupKitConfigError as error:
        print(f"No usable admin startup kit: {error}")
        print("Run `python example_setup.py` first; it registers the admin kit with `nvflare config use`.")
        raise SystemExit(1)
    env = ProdEnv(startup_kit_location=kit_dir, username=username, login_timeout=60)
    try:
        results = get_catalog(env, sites=["site-1", "site-2"], output_path=args.output)
    except CatalogReadError as error:
        print(f"Catalog read failed: {error}")
        if error.job_id:
            print(f"Inspect the job with: nvflare job logs {error.job_id} --site all --tail 100")
        raise SystemExit(1)
    print_catalog(results)
    print(f"Full metadata: {args.output}")


if __name__ == "__main__":
    main()
