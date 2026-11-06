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

"""Optionally check a completed demo's discovery output against its site policies."""

import argparse
import json
from pathlib import Path

from constants import DEFAULT_DATA, SITES


def verify_result(data_root, store_root):
    data_root = Path(data_root)
    output = json.loads((Path(store_root) / "discovery.json").read_text())
    expected = {
        (site, dataset)
        for site in SITES
        for dataset, rule in json.loads((data_root / site / "local/fdn/default.json").read_text())["datasets"].items()
        if rule["share"]
    }
    if {(e["client"], e["dataset"]) for e in output["entries"]} != expected:
        raise RuntimeError("Discovery does not match the current site publication policies")
    for entry in output["entries"]:
        metadata = entry.get("metadata")
        if metadata is None or metadata["owner"] is not None:
            raise RuntimeError("Expected available metadata with owner excluded")
        tables = {t["name"]: t for t in metadata["tables"]}
        expected_tables = {"person", "observation_period", "visit_occurrence", "condition_occurrence"}
        if entry["client"] == "site-1":
            expected_tables.add("drug_exposure")
        if entry["dataset"] not in ("oncology", "registry") and not (
            entry["client"] == "site-2" and entry["dataset"] == "omop"
        ):
            expected_tables.add("measurement")
        if set(tables) != expected_tables:
            raise RuntimeError(f"Unexpected published tables at {entry['client']}: {sorted(tables)}")
        profile = metadata["semantic_profile"]
        if profile["standard"] != "OMOP" or profile["version"] != "5.4":
            raise RuntimeError("Expected the site-declared OMOP 5.4 profile")
        for name, table in tables.items():
            columns = {c["name"] for c in table["columns"]}
            if any(c.endswith("_source_value") for c in columns):
                raise RuntimeError("Source-value column metadata was not excluded")
            if name == "person" and columns & {"month_of_birth", "day_of_birth"}:
                raise RuntimeError("Fine-grained birth fields were not excluded")
            if not set(table["primary_key"]) <= columns or not table["primary_key"]:
                raise RuntimeError("Expected a declared primary key among published columns")
            for relation in table["relationships"]:
                target = tables[relation["target_table"]]
                if not set(relation["columns"]) <= columns or not set(relation["target_columns"]) <= {
                    c["name"] for c in target["columns"]
                }:
                    raise RuntimeError("Relationship references missing columns")
        condition_columns = {c["name"] for c in tables["condition_occurrence"]["columns"]}
        common = {
            "condition_occurrence_id",
            "person_id",
            "condition_concept_id",
            "condition_start_date",
            "condition_type_concept_id",
        }
        if not common <= condition_columns:
            raise RuntimeError("Expected overlapping condition schema")
        extra = (
            {"condition_end_date", "visit_occurrence_id"}
            if entry["client"] == "site-1"
            else {"condition_status_concept_id"}
        )
        if condition_columns - common != extra:
            raise RuntimeError("Unexpected site-specific condition schema")
        config = json.loads((data_root / entry["client"] / "local/fdn/catalog_site.json").read_text())
        binding = config.get("studies", {}).get("default", {}).get(entry["dataset"], {})
        backend = binding.get("connector", config.get("default_connector", "parquet"))
        if metadata["provenance"]["source"] != backend:
            raise RuntimeError("Catalog provenance does not match the site's storage format")
        if not tables["condition_occurrence"]["relationships"]:
            raise RuntimeError("Expected declared person/visit relationships")
    print(
        f"Verified {len(output['entries'])} catalog entries: OMOP profile, table coverage, relationships and policy exclusions."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA)
    parser.add_argument("--store-root", default="/tmp/nvflare/federated-catalog/store")
    args = parser.parse_args()
    verify_result(args.data_root, args.store_root)
