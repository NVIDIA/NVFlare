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

"""Prepare deterministic synthetic OMOP 5.4-shaped subsets at each demo site."""

import argparse
import json
import shutil
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from constants import DEFAULT_DATA, SITES


def make_tables(site_index, people):
    """Two linked tables with shared columns and one site-specific condition field."""
    persons = [site_index * 1000 + i + 1 for i in range(people)]
    conditions = [site_index * 10000 + i + 1 for i in range(people * 2)]
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(people * 2)]
    tables = {
        "person": {
            "person_id": persons,
            "gender_concept_id": [0] * people,
            "year_of_birth": [1960 + i for i in range(people)],
            "person_source_value": [f"synthetic-{p}" for p in persons],
        },
        "condition_occurrence": {
            "condition_occurrence_id": conditions,
            "person_id": [p for p in persons for _ in range(2)],
            "condition_concept_id": [0] * len(conditions),
            "condition_start_date": dates,
            "condition_type_concept_id": [0] * len(conditions),
            "condition_source_value": ["demo-condition"] * len(conditions),
        },
    }
    if site_index == 1:
        tables["condition_occurrence"]["condition_end_date"] = dates
    else:
        tables["condition_occurrence"]["condition_status_concept_id"] = [0] * len(conditions)
    return tables


def table_declarations():
    return {
        "person": {"primary_key": ["person_id"], "relationships": []},
        "condition_occurrence": {
            "primary_key": ["condition_occurrence_id"],
            "relationships": [{"columns": ["person_id"], "target_table": "person", "target_columns": ["person_id"]}],
        },
    }


def write_sqlite(path, tables):
    """Build the RDBMS variation with real primary and foreign-key declarations."""
    declarations = table_declarations()
    connection = sqlite3.connect(path)
    try:
        for name, columns in tables.items():
            fields = []
            for field in pa.table(columns).schema:
                sql_type = (
                    "INTEGER"
                    if pa.types.is_integer(field.type)
                    else ("REAL" if pa.types.is_floating(field.type) else "TEXT")
                )
                fields.append(f'"{field.name}" {sql_type}')
            definition = declarations[name]
            fields.append("PRIMARY KEY (" + ", ".join(definition["primary_key"]) + ")")
            for relation in definition["relationships"]:
                fields.append(
                    "FOREIGN KEY ("
                    + ", ".join(relation["columns"])
                    + ") REFERENCES "
                    + relation["target_table"]
                    + "("
                    + ", ".join(relation["target_columns"])
                    + ")"
                )
            connection.execute(f'DROP TABLE IF EXISTS "{name}"')
            connection.execute(f'CREATE TABLE "{name}" (' + ", ".join(fields) + ")")
            rows = zip(*columns.values())
            connection.executemany(
                f'INSERT INTO "{name}" VALUES (' + ",".join("?" for _ in columns) + ")",
                [tuple(v.isoformat() if isinstance(v, date) else v for v in row) for row in rows],
            )
        connection.commit()
    finally:
        connection.close()


def prepare_data(data_root, site=None):
    root = Path(data_root).resolve()
    example_root = Path(__file__).resolve().parent
    for index, name in enumerate(SITES, start=1):
        if site is not None and name != site:
            continue
        directory = root if site is not None else root / name
        directory.mkdir(parents=True, exist_ok=True)
        declarations = example_root / "data" / "metadata" / name / "catalog.json"
        config = example_root / "config" / name / "catalog_site.json"
        storage_format = json.loads(config.read_text())["default_connector"]
        tables = make_tables(index, people=8 if index == 1 else 12)
        if storage_format == "parquet":
            source = directory / "omop"
            if not source.exists():
                source.mkdir()
                for table_name, columns in tables.items():
                    pq.write_table(pa.table(columns), source / f"{table_name}.parquet")
                shutil.copyfile(declarations, source / "catalog.json")
        elif storage_format == "sql":
            source = directory / "omop.sqlite"
            if not source.exists():
                write_sqlite(source, tables)
                shutil.copyfile(declarations, source.with_suffix(".catalog.json"))
        else:
            raise ValueError(f"Synthetic data preparation does not support {storage_format!r}")
        if name == "site-1":
            for table_name in ("lab_results", "visits"):
                csv_path = directory / f"{table_name}.csv"
                if not csv_path.exists():
                    shutil.copyfile(example_root / "data" / name / f"{table_name}.csv", csv_path)
        print(f"{name}: {storage_format}, omop dataset with person and condition_occurrence")
        if name == "site-1":
            print("site-1: csv, lab_results and visits datasets (headers with inferred types)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA)
    parser.add_argument("--site", choices=SITES, help="prepare only this site's data directly under --data-root")
    args = parser.parse_args()
    prepare_data(args.data_root, args.site)
