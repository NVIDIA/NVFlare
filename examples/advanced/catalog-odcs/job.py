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

"""Read a federated catalog of ODCS contracts from two simulated sites.

site-1 provides a contract file its data steward maintains (``sites/site-1``). site-2 has its contract generated from
a Parquet file by the datacontract library (``pip install "datacontract-cli[parquet]"``).

    python job.py

The catalog is saved under ``<workdir>/catalog``; ``datacontract catalog --files "<workdir>/catalog/*.yaml"`` renders
it as an HTML catalog.
"""

import argparse
import glob
import json
import os
import shutil
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from nvflare.app_opt.catalog.classic import get_catalog
from nvflare.recipe import SimEnv

SITE_1_FILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sites", "site-1")


def prepare_sites(config_root: str):
    """Create each simulated site's configuration folder under ``config_root``.

    site-1 gets the files in ``sites/site-1``. site-2 gets a Parquet file and a catalog.yaml pointing at it; they are
    created here because the configuration names the data file by its absolute path.
    """
    shutil.copytree(SITE_1_FILES, os.path.join(config_root, "site-1"))

    site_2 = os.path.join(config_root, "site-2")
    os.makedirs(site_2)
    person = os.path.join(site_2, "person.parquet")
    pq.write_table(
        pa.table({"person_id": [1, 2], "year_of_birth": [1970, 1985], "race_concept_id": [8527, 8516]}), person
    )
    datasets = {"omop": {"import": {"format": "parquet", "source": person}}}
    with open(os.path.join(site_2, "catalog.yaml"), "w") as f:
        yaml.safe_dump({"studies": {"default": {"datasets": datasets}}}, f)


def compare(catalog_dir: str, table: str | None = None, field: str = "physicalType") -> dict:
    """Cross-site comparison as ``{row: {"site.dataset": value}}``.

    Without ``table``: a row per table, holding its column count at each source. With ``table``: a row per column of
    that table, holding ``field`` at each source. A source missing from a row does not publish it.
    ``pandas.DataFrame.from_dict(compare(catalog_dir), orient="index")`` gives the same as a table.
    """
    rows = {}
    for path in sorted(glob.glob(os.path.join(catalog_dir, "*.odcs.yaml"))):
        source = os.path.basename(path).removesuffix(".odcs.yaml")
        with open(path) as f:
            contract = yaml.safe_load(f)
        for schema_object in contract.get("schema", []):
            properties = schema_object.get("properties") or []
            if table is None:
                rows.setdefault(schema_object["name"], {})[source] = len(properties)
            elif schema_object["name"] == table:
                for column in properties:
                    rows.setdefault(column["name"], {})[source] = column.get(field)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=None)
    args = parser.parse_args()

    workdir = args.workdir or tempfile.mkdtemp(prefix="catalog-odcs-")
    config_root = os.path.join(workdir, "sites")
    prepare_sites(config_root)

    env = SimEnv(clients=["site-1", "site-2"], workspace_root=os.path.join(workdir, "simulation"))
    catalog_dir = get_catalog(env, output_dir=os.path.join(workdir, "catalog"), config_root=config_root)

    print("catalog:", catalog_dir, sorted(os.listdir(catalog_dir)))
    with open(os.path.join(catalog_dir, "sites.json")) as f:
        print("sites:", f.read())
    print("tables:", json.dumps(compare(catalog_dir), indent=2))
    print("person:", json.dumps(compare(catalog_dir, table="person"), indent=2))


if __name__ == "__main__":
    main()
