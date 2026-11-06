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

"""Reflect relational database metadata through SQLAlchemy, without reading application rows."""

import os
from pathlib import Path
from urllib.parse import unquote, urlsplit

from nvflare.app_common.fdn.catalog.site.connector import CatalogConnector, ConnectorError
from nvflare.app_opt.fdn.catalog._metadata import ReadBudget, build_catalog, selection
from nvflare.app_opt.fdn.catalog.declarations import read_declarations


class SQLConnector(CatalogConnector):
    """Use a site environment variable containing a SQLAlchemy URL, or a mounted SQLite file.

    ``url_env`` names the variable, never its value. ``connect_args`` contains
    non-secret driver settings, including driver-specific connection/query timeouts.
    SQLAlchemy and the selected database driver are optional dependencies.
    """

    source = "sql"

    def __init__(self, datasets, url_env=None, connect_args=None):
        super().__init__(datasets)
        self.url_env = url_env
        self.connect_args = dict(connect_args or {})

    def _url(self, dataset):
        from sqlalchemy.engine import URL, make_url

        if self.url_env is not None:
            value = os.environ.get(self.url_env)
            if not value:
                raise ConnectorError("database connection is not configured")
            url = make_url(value)
            if url.get_backend_name() != "sqlite":
                return url
            database = url.database
            if not database or database == ":memory:":
                raise ConnectorError("an existing database is required")
            path = Path(unquote(urlsplit(database).path) if database.startswith("file:") else database)
        else:
            path = self._datasets.get(dataset)
        if path is None or not path.is_file():
            raise ConnectorError("database is unavailable")
        # SQLAlchemy passes URI options to sqlite3. mode=ro prevents creating or
        # changing a database even when the configured URL omitted read-only mode.
        return URL.create("sqlite+pysqlite", database=path.resolve().as_uri(), query={"mode": "ro", "uri": "true"})

    def read_catalog(self, dataset, limits, options=None):
        from sqlalchemy import create_engine, event, inspect
        from sqlalchemy.pool import NullPool

        options = selection(options, {"schema", "tables"})
        budget = ReadBudget(limits)
        engine = None
        try:
            url = self._url(dataset)
            budget.remaining()
            engine = create_engine(url, connect_args=self.connect_args, poolclass=NullPool, hide_parameters=True)
            # Count actual reflection statements, including additional statements
            # emitted internally by an Inspector method. No global event handler.
            event.listen(engine, "before_cursor_execute", lambda *args: budget.request())
            with engine.connect() as connection:
                budget.remaining()
                inspector = inspect(connection)
                schema = options.get("schema")
                if schema is not None and not inspector.has_schema(schema):
                    raise ConnectorError("database schema is unavailable")
                names = options.get("tables")
                if names is None:
                    names = inspector.get_table_names(schema=schema)
                budget.records(1 + len(names))
                tables = []
                records = 1
                effective_schema = schema or inspector.default_schema_name
                for name in sorted(names):
                    budget.remaining()
                    columns = inspector.get_columns(name, schema=schema)
                    budget.records(records + 1 + len(columns))
                    primary_key = inspector.get_pk_constraint(name, schema=schema).get("constrained_columns", [])
                    relationships = []
                    for foreign_key in inspector.get_foreign_keys(name, schema=schema):
                        # The v1 model only represents relationships within this
                        # dataset; do not name an unselected or external table.
                        if foreign_key["referred_schema"] not in (None, effective_schema):
                            continue
                        if foreign_key["referred_table"] not in names:
                            continue
                        relationships.append(
                            {
                                "columns": foreign_key["constrained_columns"],
                                "target_table": foreign_key["referred_table"],
                                "target_columns": foreign_key["referred_columns"],
                            }
                        )
                    tables.append(
                        {
                            "name": name,
                            "columns": [
                                {
                                    "name": column["name"],
                                    "type": column["type"].compile(dialect=engine.dialect),
                                    "nullable": column.get("nullable"),
                                }
                                for column in columns
                            ],
                            "primary_key": primary_key,
                            "relationships": relationships,
                        }
                    )
                    records += 1 + len(columns) + len(relationships)
                    budget.records(records)
            path = self._datasets.get(dataset) if self.url_env is None else None
            declarations = read_declarations(path.with_suffix(".catalog.json")) if path is not None else {}
            catalog = build_catalog(dataset, self.source, tables, limits, declarations)
            budget.remaining()
            return catalog
        except ConnectorError:
            raise
        except Exception:
            # Database exceptions may contain URLs, credentials and SQL text.
            raise ConnectorError("database metadata is unavailable") from None
        finally:
            if engine is not None:
                engine.dispose()
