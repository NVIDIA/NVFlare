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

"""Find sites publishing the tables required by a proposed analysis, using metadata only."""

import argparse

from nvflare.app_common.fdn.catalog.server.search import SearchQuery
from nvflare.app_common.fdn.catalog.server.store import CatalogStore

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", default="/tmp/nvflare/federated-catalog/store")
    parser.add_argument("--require-tables", nargs="+", default=["person", "condition_occurrence", "measurement"])
    parser.add_argument("--table", default="condition_occurrence", help="Compare column coverage for this table")
    parser.add_argument("--dataset", help="Limit discovery and comparison to this dataset name")
    parser.add_argument("--show-schema", action="store_true", help="Print all published table schemas")
    args = parser.parse_args()
    required = set(args.require_tables)
    store = CatalogStore(args.store_root)
    page = store.search(
        "default", SearchQuery(standard="OMOP", standard_version="5.4", dataset=args.dataset, page_size=100)
    )
    print(
        "Site/dataset                  Published tables                                              Metadata candidate"
    )
    schemas = {}
    for entry in page.entries:
        if not entry.available or entry.catalog is None:
            continue
        tables = {t.name for t in entry.catalog.tables}
        for table in entry.catalog.tables:
            if table.name == args.table:
                schemas[f"{entry.client}/{entry.dataset}"] = {column.name: column.type for column in table.columns}
        missing = required - tables
        status = "yes" if not missing else "missing: " + ", ".join(sorted(missing))
        print(f"{entry.client}/{entry.dataset:20} {', '.join(sorted(tables)):65} {status}")
    if args.show_schema:
        for entry in page.entries:
            if not entry.available or entry.catalog is None:
                continue
            print(f"\n{entry.client}/{entry.dataset} ({entry.catalog.provenance.source})")
            for table in entry.catalog.tables:
                print(f"  TABLE {table.name}")
                for column in table.columns:
                    nullable = "NULL" if column.nullable else "NOT NULL" if column.nullable is False else "UNKNOWN"
                    print(f"    {column.name}: {column.type} {nullable}")
                print(f"    PRIMARY KEY: {', '.join(table.primary_key) or '(undeclared)'}")
                for relation in table.relationships:
                    print(
                        f"    FOREIGN KEY ({', '.join(relation.columns)}) -> "
                        f"{relation.target_table}({', '.join(relation.target_columns)})"
                    )
    if schemas:
        common = set.intersection(*(set(columns) for columns in schemas.values()))
        print(f"\n{args.table}: common columns: {', '.join(sorted(common))}")
        for site, columns in schemas.items():
            print(f"{site} additional columns: {', '.join(sorted(set(columns) - common)) or '(none)'}")
        type_differences = {
            name: {site: columns[name] for site, columns in schemas.items()}
            for name in common
            if len({columns[name] for columns in schemas.values()}) > 1
        }
        for name, types in sorted(type_differences.items()):
            print(f"{name} storage types: {types}")
    print("Metadata compatibility only; no patient counts, cohort matching or clinical validation was performed.")
