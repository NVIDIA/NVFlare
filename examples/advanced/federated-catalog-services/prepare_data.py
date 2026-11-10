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


"""Populate one site's source service with synthetic OMOP-shaped demo data."""

import argparse
import os
import time
from pathlib import Path

import omop_data
import pyarrow as pa
import pyarrow.parquet as pq
import requests

_ARROW = {
    "int": pa.int32(),
    "date": pa.date32(),
    "datetime": pa.timestamp("us"),
    "str": pa.string(),
    "float": pa.float64(),
}
# Unity Catalog column type: (type_text and the Spark JSON type, type_name)
_UC_TYPES = {
    "int": ("integer", "INT"),
    "date": ("date", "DATE"),
    "datetime": ("timestamp", "TIMESTAMP"),
    "str": ("string", "STRING"),
    "float": ("double", "DOUBLE"),
}
UC_CATALOG, UC_SCHEMA = "flare_demo", "omop"


def wait_for_unity_catalog(api):
    deadline = time.monotonic() + 90
    while True:
        try:
            if requests.get(f"{api}/catalogs", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError("Unity Catalog did not become ready")
        time.sleep(1)


def create_in_unity_catalog(api, path, body):
    """Keep matching registrations; repair changed columns only on this demo's external Parquet tables."""
    name = body["name"]
    if path != "catalogs":
        name = f"{body['catalog_name']}.{body['schema_name'] + '.' if path == 'tables' else ''}{body['name']}"
    url = f"{api}/{path}/{name}"
    existing = requests.get(url, timeout=10)
    if existing.status_code == 200:
        if path != "tables":
            return
        table = existing.json()
        for key in ("table_type", "data_source_format", "storage_location"):
            if table.get(key) != body[key]:
                raise ValueError(f"Refusing to replace {name}: existing {key} differs from the demo declaration")
        fields = ("name", "type_text", "type_name", "position", "nullable")
        expected = [{key: col[key] for key in fields} for col in body["columns"]]
        actual = [{key: col.get(key) for key in fields} for col in table.get("columns", [])]
        if actual == expected:
            return
        # Deleting this external registration leaves the separately mounted Parquet files in place.
        requests.delete(url, timeout=10).raise_for_status()
    elif existing.status_code != 404:
        existing.raise_for_status()
    requests.post(f"{api}/{path}", json=body, timeout=10).raise_for_status()


def prepare_unity_catalog(site, data_root=Path("/data")):
    """Parquet files on the shared volume, registered as external tables of one Unity Catalog schema."""
    data = omop_data.generate(site)
    for table, columns in omop_data.schema(site).items():
        target = Path(data_root) / f"{table}.parquet"
        arrow_schema = pa.schema([pa.field(n, _ARROW[k], nullable=not r) for n, k, r in columns])
        arrow_table = pa.table(data[table], schema=arrow_schema)
        try:
            complete = pq.read_metadata(target).num_rows == arrow_table.num_rows and pq.read_schema(target).equals(
                arrow_schema
            )
        except (OSError, pa.ArrowException):
            complete = False
        if not complete:
            temporary = target.with_suffix(".parquet.tmp")
            try:
                pq.write_table(arrow_table, temporary)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    api = os.environ["UNITY_CATALOG_URL"]
    wait_for_unity_catalog(api)
    create_in_unity_catalog(api, "catalogs", {"name": UC_CATALOG})
    create_in_unity_catalog(api, "schemas", {"catalog_name": UC_CATALOG, "name": UC_SCHEMA})
    for table, columns in omop_data.schema(site).items():
        definitions = []
        for position, (name, kind, required) in enumerate(columns):
            text, type_name = _UC_TYPES[kind]
            definitions.append(
                {
                    "name": name,
                    "type_text": text,
                    "type_name": type_name,
                    "position": position,
                    "nullable": not required,
                    "type_json": f'{{"name": "{name}", "type": "{text}", "nullable": {str(not required).lower()}, "metadata": {{}}}}',
                }
            )
        create_in_unity_catalog(
            api,
            "tables",
            {
                "catalog_name": UC_CATALOG,
                "schema_name": UC_SCHEMA,
                "name": table,
                "table_type": "EXTERNAL",
                "data_source_format": "PARQUET",
                "storage_location": f"file:///home/unitycatalog/demo-data/{table}.parquet",
                "columns": definitions,
            },
        )


def prepare_postgres(site):
    """Seed/repair the demo schema and role in one transaction using the example's PostgreSQL driver."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import URL

    engine = create_engine(
        URL.create(
            "postgresql+pg8000",
            username="postgres",
            password=os.environ["POSTGRES_PASSWORD"],
            host="postgres",
            port=5432,
            database="omop",
        ),
        connect_args={"timeout": 5},
        hide_parameters=True,
    )
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE SCHEMA IF NOT EXISTS omop"))
            data = omop_data.generate(site)
            # Check every table/column and row count, including volumes left by older interrupted seeds.
            complete = True
            for table, columns in omop_data.schema(site).items():
                existing = connection.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema = 'omop' AND table_name = :table ORDER BY ordinal_position"
                    ),
                    {"table": table},
                ).fetchall()
                if [row[0] for row in existing] != [name for name, _kind, _required in columns]:
                    complete = False
                    break
                expected_rows = len(next(iter(data[table].values())))
                if connection.execute(text(f"SELECT COUNT(*) FROM omop.{table}")).fetchone()[0] != expected_rows:
                    complete = False
                    break
            if not complete:
                statements = omop_data.create_statements(site, "omop.")
                for table, _statement in reversed(statements):
                    connection.execute(text(f"DROP TABLE IF EXISTS omop.{table}"))
                for table, statement in statements:
                    connection.execute(text(statement))
                    columns = data[table]
                    values = ", ".join(f":{column}" for column in columns)
                    rows = [dict(zip(columns, row)) for row in omop_data.insert_rows(data, table)]
                    if rows:
                        connection.execute(text(f"INSERT INTO omop.{table} VALUES ({values})"), rows)
            if not connection.execute(text("SELECT 1 FROM pg_roles WHERE rolname = 'catalog_reader'")).fetchone():
                connection.execute(text("CREATE ROLE catalog_reader LOGIN PASSWORD 'local-demo-reader'"))
            connection.execute(text("GRANT CONNECT ON DATABASE omop TO catalog_reader"))
            connection.execute(text("GRANT USAGE ON SCHEMA omop TO catalog_reader"))
    finally:
        engine.dispose()


def prepare(site):
    if site == "site-1":
        prepare_unity_catalog(site)
    else:
        prepare_postgres(site)
    version, tables, people = omop_data.SITE_OMOP[site]
    print(f"Prepared {people} synthetic patients (CDM {version}: {', '.join(tables)}) for {site}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", choices=["site-1", "site-2"])
    prepare(parser.parse_args().site)
