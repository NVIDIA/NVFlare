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

"""Initialize one site's demo source and install its catalog configuration at container startup."""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from common import ROOT, WORKSPACE, write_json


def prepare(site):
    spec = importlib.util.spec_from_file_location("playground_sanctuary", ROOT / "data/playground/sanctuary.py")
    sanctuary = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sanctuary
    spec.loader.exec_module(sanctuary)
    # Same four-column projection as the Playground's unity.sanctuary.pets table.
    rows = sanctuary.rows(
        sanctuary.roster(50 if site == "site-1" else 70, seed=2026 if site == "site-1" else 2027), sanctuary.PET_COLUMNS
    )
    if site == "site-1":
        schema = pa.schema(
            [
                pa.field("uuid", pa.string(), nullable=False),
                pa.field("name", pa.string(), nullable=False),
                pa.field("age", pa.int32(), nullable=False),
                pa.field("adopted", pa.bool_(), nullable=False),
            ]
        )
        if not Path("/data/pets.parquet").exists():
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), Path("/data/pets.parquet"))
        api = os.environ["UNITY_CATALOG_URL"]
        deadline = time.monotonic() + 90
        while True:
            try:
                if requests.get(f"{api}/catalogs", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("Unity Catalog did not become ready")
            time.sleep(1)

        def create(path, body):
            if path in {"catalogs", "schemas", "tables"}:
                name = (
                    body["name"]
                    if path == "catalogs"
                    else body["catalog_name"]
                    + "."
                    + (body["schema_name"] + "." if path == "tables" else "")
                    + body["name"]
                )
                existing = requests.get(f"{api}/{path}/{name}", timeout=10)
                if existing.status_code == 200:
                    return
                if existing.status_code != 404:
                    existing.raise_for_status()
            response = requests.post(f"{api}/{path}", json=body, timeout=10)
            response.raise_for_status()

        create("catalogs", {"name": "flare_demo"})
        create("schemas", {"catalog_name": "flare_demo", "name": "sanctuary"})
        columns = []
        for position, (name, kind, type_name) in enumerate(
            [
                ("uuid", "string", "STRING"),
                ("name", "string", "STRING"),
                ("age", "integer", "INT"),
                ("adopted", "boolean", "BOOLEAN"),
            ]
        ):
            columns.append(
                {
                    "name": name,
                    "type_text": kind,
                    "type_name": type_name,
                    "position": position,
                    "nullable": False,
                    "type_json": json.dumps({"name": name, "type": kind, "nullable": False, "metadata": {}}),
                }
            )
        create(
            "tables",
            {
                "catalog_name": "flare_demo",
                "schema_name": "sanctuary",
                "name": "pets",
                "table_type": "EXTERNAL",
                "data_source_format": "PARQUET",
                "storage_location": "file:///home/unitycatalog/demo-data/pets.parquet",
                "columns": columns,
            },
        )
    else:
        with psycopg.connect(
            host="postgres",
            port=5432,
            dbname="sanctuary",
            user="postgres",
            password=os.environ["POSTGRES_PASSWORD"],
            autocommit=True,
        ) as connection:
            connection.execute("CREATE SCHEMA IF NOT EXISTS sanctuary")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS sanctuary.pets (uuid TEXT PRIMARY KEY, name TEXT NOT NULL, age INTEGER NOT NULL, adopted BOOLEAN NOT NULL)"
            )
            with connection.cursor() as cursor:
                if not connection.execute("SELECT 1 FROM sanctuary.pets LIMIT 1").fetchone():
                    cursor.executemany(
                        "INSERT INTO sanctuary.pets VALUES (%s, %s, %s, %s)",
                        [tuple(row[k] for k in sanctuary.PET_COLUMNS) for row in rows],
                    )
            if not connection.execute("SELECT 1 FROM pg_roles WHERE rolname = 'catalog_reader'").fetchone():
                connection.execute("CREATE ROLE catalog_reader LOGIN PASSWORD 'local-demo-reader'")
            connection.execute("GRANT CONNECT ON DATABASE sanctuary TO catalog_reader")
            connection.execute("GRANT USAGE ON SCHEMA sanctuary TO catalog_reader")
    config = ROOT / "config" / site
    local = WORKSPACE / "local/fdn"
    write_json(local / "catalog_site.json", json.loads((config / "catalog_site.json").read_text()))
    write_json(local / "default/catalog_policy.json", json.loads((config / "catalog_policy.json").read_text()))
    print(f"Prepared catalog source for {site}", flush=True)
