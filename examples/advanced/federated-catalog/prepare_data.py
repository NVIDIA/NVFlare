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

"""Prepare deterministic synthetic OMOP 5.4-shaped subsets and each site's folder for the simulation.

Each site gets its own folder under the data root, laid out like a deployed site:

    <data-root>/<site>/data/                the site's dataset files
    <data-root>/<site>/local/datasets.yaml  the site's dataset configuration, installed from config/<site>/
"""

import argparse
import shutil
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from constants import DEFAULT_DATA, SITES

PEOPLE = {"site-1": 8, "site-2": 12, "site-3": 10}
# One condition_occurrence column that only this site has.
SITE_COLUMN = {
    "site-1": ("condition_end_date", None),
    "site-2": ("condition_status_concept_id", 0),
    "site-3": ("stop_reason", "resolved"),
}


def make_tables(site_index, people, site):
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
    column, value = SITE_COLUMN[site]
    tables["condition_occurrence"][column] = dates if value is None else [value] * len(conditions)
    return tables


def table_keys():
    return {
        "person": {"primary_key": ["person_id"], "relationships": []},
        "condition_occurrence": {
            "primary_key": ["condition_occurrence_id"],
            "relationships": [{"columns": ["person_id"], "target_table": "person", "target_columns": ["person_id"]}],
        },
    }


def _sql_type(field, dialect):
    if pa.types.is_integer(field.type):
        return "INTEGER"
    if pa.types.is_date(field.type):
        return "DATE" if dialect == "duckdb" else "TEXT"
    if pa.types.is_floating(field.type):
        return "DOUBLE" if dialect == "duckdb" else "REAL"
    return "VARCHAR" if dialect == "duckdb" else "TEXT"


def write_database(connection, tables, dialect):
    """Build tables with real primary and foreign-key constraints, then load the rows."""
    declarations = table_keys()
    for name, columns in tables.items():
        fields = [f'"{field.name}" {_sql_type(field, dialect)}' for field in pa.table(columns).schema]
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
            [tuple(v.isoformat() if isinstance(v, date) and dialect == "sql" else v for v in row) for row in rows],
        )


def write_sqlite(path, tables):
    connection = sqlite3.connect(path)
    try:
        write_database(connection, tables, "sql")
        connection.commit()
    finally:
        connection.close()


def write_duckdb(path, tables):
    import duckdb

    connection = duckdb.connect(str(path))
    try:
        write_database(connection, tables, "duckdb")
        connection.execute("CHECKPOINT")
    finally:
        connection.close()


def _remove(path):
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


PLACEHOLDER = "file://${SITE_DATA}/"
MARKER = ".federated-catalog-example"  # marks a site folder this script created, and may therefore replace


def install_config(config_file, site_root):
    """Install the site's checked-in datasets.yaml with each file uri pointing into the site's data folder.

    Each uri is built with Path.as_uri(), which percent-encodes characters such as '#' and '%' in the data root, and
    the file is written with a YAML serializer, so no path text is ever parsed as YAML."""
    config = yaml.safe_load(config_file.read_text())
    for dataset in config["studies"]["default"]["datasets"].values():
        uri = dataset["uri"]
        if uri.startswith(PLACEHOLDER):
            dataset["uri"] = (site_root / "data" / uri.removeprefix(PLACEHOLDER)).as_uri()
    target = site_root / "local" / "datasets.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, sort_keys=False))
    return config


def local_path(uri):
    """The local path of a file uri written by install_config."""
    return Path(unquote(urlsplit(uri).path))


def claim_site_folder(site_root, force=False):
    """Refuse to replace a site folder this script did not create, such as a real or POC site workspace, unless
    force is set."""
    existing = [p for p in (site_root / "data", site_root / "local") if p.exists()]
    if existing and not force and not (site_root / MARKER).is_file():
        raise SystemExit(
            f"{site_root} already has {', '.join(p.name for p in existing)} that this example did not create. "
            "Choose another --data-root, or add --force to replace them."
        )
    site_root.mkdir(parents=True, exist_ok=True)
    (site_root / MARKER).write_text("Created by examples/advanced/federated-catalog/prepare_data.py\n")


def prepare_data(data_root, force=False):
    root = Path(data_root).resolve()
    example_root = Path(__file__).resolve().parent
    for index, site in enumerate(SITES, start=1):
        site_root = root / site
        claim_site_folder(site_root, force)
        # Regenerate every file, removing what earlier runs of this example left behind.
        for old in (site_root / "data", site_root / "local"):
            _remove(old)
        (site_root / "data").mkdir(parents=True)
        config = install_config(example_root / "config" / site / "datasets.yaml", site_root)
        datasets = config["studies"]["default"]["datasets"]
        omop = datasets["omop"]
        source = local_path(omop["uri"])
        tables = make_tables(index, PEOPLE[site], site)
        if omop["type"] == "parquet":
            source.mkdir()
            for name, columns in tables.items():
                pq.write_table(pa.table(columns), source / f"{name}.parquet")
        elif omop["type"] == "sql":
            write_sqlite(source, tables)
        elif omop["type"] == "duckdb":
            write_duckdb(source, tables)
        else:
            raise ValueError(f"Synthetic data preparation does not support {omop['type']!r}")
        for name, dataset in datasets.items():
            if dataset["type"] == "csv":
                shutil.copyfile(example_root / "data" / site / f"{name}.csv", local_path(dataset["uri"]))
        print(f"{site}: " + ", ".join(f"{name} ({dataset['type']})" for name, dataset in datasets.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=DEFAULT_DATA)
    parser.add_argument(
        "--force", action="store_true", help="replace site data/ and local/ folders this example did not create"
    )
    args = parser.parse_args()
    prepare_data(args.data_root, args.force)
