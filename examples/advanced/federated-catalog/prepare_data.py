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

"""Prepare deterministic synthetic OMOP 5.4-shaped subsets and site publication policies."""

import argparse
import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_DATA = "/tmp/nvflare/federated-catalog/data"
SITES = ["site-1", "site-2"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def make_tables(site_index, people):
    """Small schema subsets, not a complete or vocabulary-validated OMOP database.

    Concept IDs are deliberately 0 (unmapped); no licensed vocabularies are bundled.
    Local person/visit IDs are coherent across the generated clinical tables.
    """
    persons = [site_index * 10000 + i + 1 for i in range(people)]
    visits = [site_index * 100000 + i + 1 for i in range(people * 2)]
    visit_people = [p for p in persons for _ in range(2)]
    dates = [date(2024, 1, 1) + timedelta(days=i % 180) for i in range(people * 2)]
    zeros = [0] * people
    event_zeros = [0] * len(visits)
    tables = {
        "person": {
            "person_id": persons,
            "gender_concept_id": zeros,
            "year_of_birth": [1940 + i % 60 for i in range(people)],
            "month_of_birth": [1 + i % 12 for i in range(people)],
            "day_of_birth": [1] * people,
            "race_concept_id": zeros,
            "ethnicity_concept_id": zeros,
            "person_source_value": [f"synthetic-source-{p}" for p in persons],
        },
        "observation_period": {
            "observation_period_id": persons,
            "person_id": persons,
            "observation_period_start_date": [date(2024, 1, 1)] * people,
            "observation_period_end_date": [date(2024, 12, 31)] * people,
            "period_type_concept_id": zeros,
        },
        "visit_occurrence": {
            "visit_occurrence_id": visits,
            "person_id": visit_people,
            "visit_concept_id": event_zeros,
            "visit_start_date": dates,
            "visit_end_date": dates,
            "visit_type_concept_id": event_zeros,
            "visit_source_value": ["synthetic-outpatient"] * len(visits),
        },
        "condition_occurrence": {
            "condition_occurrence_id": visits,
            "person_id": visit_people,
            "condition_concept_id": event_zeros,
            "condition_start_date": dates,
            "condition_type_concept_id": event_zeros,
            "visit_occurrence_id": visits,
            "condition_source_value": ["demo-condition-A" if i % 2 else "demo-condition-B" for i in range(len(visits))],
        },
        "measurement": {
            "measurement_id": visits,
            "person_id": visit_people,
            "measurement_concept_id": event_zeros,
            "measurement_date": dates,
            "measurement_type_concept_id": event_zeros,
            "value_as_number": [80.0 + i % 60 for i in range(len(visits))],
            "unit_concept_id": event_zeros,
            "visit_occurrence_id": visits,
            "measurement_source_value": ["demo-lab"] * len(visits),
        },
    }
    if site_index == 1:
        tables["condition_occurrence"]["condition_end_date"] = dates
    else:
        tables["condition_occurrence"]["condition_status_concept_id"] = event_zeros
        del tables["condition_occurrence"]["visit_occurrence_id"]
    if site_index == 1:
        tables["drug_exposure"] = {
            "drug_exposure_id": visits,
            "person_id": visit_people,
            "drug_concept_id": event_zeros,
            "drug_exposure_start_date": dates,
            "drug_exposure_end_date": dates,
            "drug_type_concept_id": event_zeros,
            "quantity": [1.0] * len(visits),
            "visit_occurrence_id": visits,
            "drug_source_value": ["demo-drug"] * len(visits),
        }
    return tables


def table_declarations(tables):
    """Site-declared keys/relationships supplement the observed Parquet schemas."""
    result = {}
    for name, columns in tables.items():
        primary = "person_id" if name == "person" else f"{name}_id"
        relationships = []
        if name != "person":
            relationships.append({"columns": ["person_id"], "target_table": "person", "target_columns": ["person_id"]})
        if name not in ("person", "visit_occurrence") and "visit_occurrence_id" in columns:
            relationships.append(
                {
                    "columns": ["visit_occurrence_id"],
                    "target_table": "visit_occurrence",
                    "target_columns": ["visit_occurrence_id"],
                }
            )
        result[name] = {"primary_key": [primary], "relationships": relationships}
    return result


def write_sqlite(path, tables):
    """Build the RDBMS variation with real primary and foreign-key declarations."""
    declarations = table_declarations(tables)
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


def prepare_data(data_root, site_2_format="parquet"):
    root = Path(data_root).resolve()
    for index, site in enumerate(SITES, start=1):
        people = 64 if index == 1 else 96
        omop = root / site / "datasets/omop"
        internal = root / site / "datasets/internal"
        omop.mkdir(parents=True, exist_ok=True)
        internal.mkdir(parents=True, exist_ok=True)
        tables = make_tables(index, people)
        storage_format = site_2_format if site == "site-2" else "parquet"
        if storage_format == "parquet":
            for name, columns in tables.items():
                pq.write_table(pa.table(columns), omop / f"{name}.parquet")
        pq.write_table(
            pa.table({"source_person_id": ["synthetic-private-id"], "person_id": [index * 10000 + 1]}),
            internal / "identity_crosswalk.parquet",
        )
        metadata = {
            "display_name": f"Synthetic OMOP clinical extract at {site}",
            "description": "OMOP 5.4-shaped subset for metadata discovery; not a validated CDM database",
            "owner": "private-site-contact",
            "modalities": ["tabular"],
            "operations": ["statistics"],
            "source_version": "synthetic-2024-v1",
            "semantic_profile": {
                "standard": "OMOP",
                "version": "5.4",
                "mapping_version": f"demo-etl-{site}-v1",
                "declaration_ref": f"synthetic-omop-subset:{site}",
            },
            "tables": table_declarations(tables),
        }
        if storage_format == "parquet":
            write_json(omop / "catalog.json", metadata)
        source = omop
        if storage_format == "sqlite":
            source = root / site / "datasets/omop.sqlite"
            write_sqlite(source, tables)
        exclusions = {name: [c for c in columns if c.endswith("_source_value")] for name, columns in tables.items()}
        exclusions["person"] += ["month_of_birth", "day_of_birth"]
        local = root / site / "local"
        write_json(
            local / "study_runtime.yaml",
            {
                "format_version": 2,
                "studies": {
                    "default": {
                        "datasets": {
                            "omop": {"source": str(source), "mode": "ro"},
                            "internal": {"source": str(internal), "mode": "ro"},
                        }
                    }
                },
            },
        )
        write_json(
            local / "fdn/catalog_site.json",
            {
                "format_version": 1,
                "installed_operations": ["statistics"],
                "studies": {
                    "default": {
                        "omop": {
                            "connector": storage_format,
                            "options": {
                                "storage_format": storage_format,
                                "metadata": {k: v for k, v in metadata.items() if k != "tables"},
                            },
                        }
                    }
                },
            },
        )
        write_json(
            local / "fdn/default.json",
            {
                "format_version": 1,
                "datasets": {
                    "omop": {
                        "share": True,
                        "exclude_columns": exclusions,
                        "exclude_metadata": ["owner"],
                        "exclude_tables": ["measurement"] if index == 2 else [],
                        "operations": ["statistics"],
                    },
                    "internal": {"share": False},
                },
            },
        )
        print(
            f"{site}: {storage_format}, {people} synthetic persons, {len(tables)} local tables; policy at {local / 'fdn/default.json'}"
        )
    print(f"Prepared OMOP-shaped data in {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA)
    parser.add_argument("--site-2-format", choices=["parquet", "sqlite"], default="parquet")
    args = parser.parse_args()
    prepare_data(args.data_root, args.site_2_format)
