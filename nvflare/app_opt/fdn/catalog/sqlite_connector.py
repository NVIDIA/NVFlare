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

"""Read-only SQLite schema connector using the database schema without reading application rows."""

import sqlite3
import time

from nvflare.app_common.fdn.catalog.common.schema import PROFILE_KEYS, CatalogInfo
from nvflare.app_common.fdn.catalog.site.connector import CatalogConnector, ConnectorError
from nvflare.app_opt.fdn.catalog.declarations import read_declarations


def quote_identifier(name):
    return '"' + name.replace('"', '""') + '"'


class SQLiteConnector(CatalogConnector):
    source = "sqlite"

    def read_catalog(self, dataset, limits, options=None):
        if options:
            raise ConnectorError("built-in sqlite reader accepts no options")
        path = self._datasets.get(dataset)
        if path is None or not path.is_file():
            raise ConnectorError("database is unavailable")
        deadline = time.monotonic() + limits.max_duration_s
        calls = 0
        try:
            connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1)
        except sqlite3.Error:
            raise ConnectorError("database is unavailable") from None
        try:

            def query(sql):
                nonlocal calls
                calls += 1
                if calls > limits.max_calls or time.monotonic() >= deadline:
                    raise ConnectorError("schema read budget exceeded", budget=True)
                rows = connection.execute(sql).fetchmany(limits.max_records + 1)
                if len(rows) > limits.max_records:
                    raise ConnectorError("schema record budget exceeded", budget=True)
                return rows

            tables = []
            records = 0
            names = query(
                "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            for (name,) in names:
                fields = query(f"PRAGMA table_info({quote_identifier(name)})")
                keys = sorted((f[5], f[1]) for f in fields if f[5])
                grouped = {}
                for fk in query(f"PRAGMA foreign_key_list({quote_identifier(name)})"):
                    grouped.setdefault(fk[0], []).append(fk)
                relationships = []
                for group in grouped.values():
                    group.sort(key=lambda fk: fk[1])
                    relationships.append(
                        {
                            "columns": [f[3] for f in group],
                            "target_table": group[0][2],
                            "target_columns": [f[4] for f in group],
                        }
                    )
                tables.append(
                    {
                        "name": name,
                        "columns": [{"name": f[1], "type": f[2], "nullable": not bool(f[3] or f[5])} for f in fields],
                        "primary_key": [name for _, name in keys],
                        "relationships": relationships,
                    }
                )
                records += 1 + len(fields) + len(relationships)
                if records > limits.max_records:
                    raise ConnectorError("schema record budget exceeded", budget=True)
            if time.monotonic() >= deadline:
                raise ConnectorError("schema read budget exceeded", budget=True)
            metadata = {
                "display_name": dataset,
                "description": None,
                "owner": None,
                "modalities": ["tabular"],
                "operations": [],
                "source_version": None,
            }
            declarations = read_declarations(path.with_suffix(".catalog.json"))
            metadata.update({k: v for k, v in declarations.items() if k not in {"tables", "observed_at", "provenance"}})
            profile = metadata.get("semantic_profile", {})
            metadata["semantic_profile"] = {key: profile.get(key) for key in PROFILE_KEYS}
            metadata.update(
                tables=tables,
                resources=[],
                lineage=metadata.get("lineage", []),
                provenance={"source": self.source, "declaration_ref": None, "validation_ref": None},
            )
            catalog = CatalogInfo.from_wire(
                {"dataset": dataset, "metadata": metadata, "observed_at": declarations.get("observed_at")}
            )
            if catalog.record_count() > limits.max_records:
                raise ConnectorError("schema record budget exceeded", budget=True)
            return catalog
        except sqlite3.Error:
            raise ConnectorError("database schema is unreadable") from None
        finally:
            connection.close()
