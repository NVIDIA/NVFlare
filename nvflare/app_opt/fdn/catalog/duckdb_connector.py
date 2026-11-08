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


"""Read the metadata of a mounted DuckDB database file, without reading application rows."""

from nvflare.app_common.fdn.catalog.site.connector import CatalogConnector, ConnectorError
from nvflare.app_opt.fdn.catalog._metadata import ReadBudget, build_catalog, selection
from nvflare.app_opt.fdn.catalog.declarations import read_declarations

_SCHEMA = "SELECT 1 FROM information_schema.schemata WHERE catalog_name = current_database() AND schema_name = ?"
_TABLES = (
    "SELECT table_name FROM information_schema.tables "
    "WHERE table_catalog = current_database() AND table_schema = ? AND table_type = 'BASE TABLE'"
)
_COLUMNS = (
    "SELECT table_name, column_name, data_type, is_nullable FROM information_schema.columns "
    "WHERE table_catalog = current_database() AND table_schema = ? ORDER BY table_name, ordinal_position"
)
_CONSTRAINTS = (
    "SELECT table_name, constraint_type, constraint_column_names, referenced_table, referenced_column_names "
    "FROM duckdb_constraints() "
    "WHERE database_name = current_database() AND schema_name = ? AND constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY') "
    "ORDER BY table_name, constraint_index"
)


class DuckDBConnector(CatalogConnector):
    """Use a mounted DuckDB database file. Column types, nullability, primary keys and foreign keys come from
    DuckDB's own catalog views; optional descriptive and semantic declarations stay in ``<database>.catalog.json``.

    The file is opened read-only with external file access disabled, and only four fixed metadata queries run.
    ``schema`` (default ``main``) and ``tables`` are the only options. The ``duckdb`` package is an optional
    dependency of the site.
    """

    source = "duckdb"

    def read_catalog(self, dataset, limits, options=None):
        options = selection(options, {"schema", "tables"})
        budget = ReadBudget(limits)
        path = self._datasets.get(dataset)
        if path is None or not path.is_file():
            raise ConnectorError("database is unavailable")
        connection = None
        try:
            import duckdb

            budget.remaining()
            connection = duckdb.connect(str(path.resolve()), read_only=True, config={"enable_external_access": False})
            schema = options.get("schema", "main")

            def run(sql):
                budget.request()
                return connection.execute(sql, [schema]).fetchall()

            if not run(_SCHEMA):
                raise ConnectorError("database schema is unavailable")
            available = sorted(row[0] for row in run(_TABLES))
            names = options.get("tables")
            if names is None:
                names = available
            elif set(names) - set(available):
                raise ConnectorError("database table is unavailable")
            budget.records(1 + len(names))
            columns = {name: [] for name in names}
            for table, column, data_type, nullable in run(_COLUMNS):
                if table in columns:
                    columns[table].append({"name": column, "type": data_type, "nullable": nullable == "YES"})
            keys = {name: [] for name in names}
            relationships = {name: [] for name in names}
            for table, kind, constrained, referred, referred_columns in run(_CONSTRAINTS):
                if table not in keys:
                    continue
                if kind == "PRIMARY KEY":
                    keys[table] = list(constrained)
                # The model only represents relationships within this dataset's selected tables.
                elif referred in keys:
                    relationships[table].append(
                        {
                            "columns": list(constrained),
                            "target_table": referred,
                            "target_columns": list(referred_columns),
                        }
                    )
            tables = [
                {
                    "name": name,
                    "columns": columns[name],
                    "primary_key": keys[name],
                    "relationships": relationships[name],
                }
                for name in sorted(names)
            ]
            budget.records(1 + sum(1 + len(t["columns"]) + len(t["relationships"]) for t in tables))
            declarations = read_declarations(path.with_suffix(".catalog.json"))
            catalog = build_catalog(dataset, self.source, tables, limits, declarations)
            budget.remaining()
            return catalog
        except ConnectorError:
            raise
        except Exception:
            # Database exceptions may contain file paths and SQL text.
            raise ConnectorError("database metadata is unavailable") from None
        finally:
            if connection is not None:
                connection.close()
