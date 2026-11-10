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

from collections.abc import Mapping
from pathlib import Path
from urllib.parse import unquote, urlsplit

from nvflare.app_common.fdn.catalog.site.connector import CatalogConnector, ConnectorError
from nvflare.app_opt.fdn.catalog._metadata import ReadBudget, build_catalog, selection
from nvflare.app_opt.fdn.catalog.declarations import read_declarations


def _type_name(column_type, dialect) -> str:
    """The column's type as the database names it; a type SQLAlchemy cannot render is published as UNKNOWN."""
    from sqlalchemy.exc import CompileError

    try:
        return column_type.compile(dialect=dialect)
    except CompileError:
        return "UNKNOWN"


def _sqlite_url(path: Path):
    from sqlalchemy.engine import URL

    if not path.is_file():
        raise ConnectorError("database is unavailable")
    # SQLAlchemy passes URI options to sqlite3. mode=ro prevents creating or changing a database.
    return URL.create("sqlite+pysqlite", database=path.resolve().as_uri(), query={"mode": "ro", "uri": "true"})


class SQLConnector(CatalogConnector):
    """Reads a relational database's metadata from the dataset's configuration.

    - ``uri``: a ``file://`` URI is a SQLite database file, opened read-only. Any other URI is a SQLAlchemy URL such
      as ``postgresql+psycopg://host:5432/db``, without credentials; a query string can carry driver settings.
    - secrets ``username`` and ``password``, when configured, complete the URL.
    - options: ``schema`` and ``tables`` select what is read; ``connect_args`` holds non-secret driver settings, such
      as connection or query timeouts.

    SQLAlchemy and the database's driver are optional dependencies.
    """

    source = "sql"

    @staticmethod
    def _url(dataset, secret):
        from sqlalchemy.engine import make_url

        if dataset.path is not None:
            return _sqlite_url(Path(dataset.path)), Path(dataset.path)
        url = make_url(dataset.uri)
        if url.get_backend_name() == "sqlite":
            database = url.database
            if not database or database == ":memory:":
                raise ConnectorError("an existing database is required")
            path = Path(unquote(urlsplit(database).path) if database.startswith("file:") else database)
            return _sqlite_url(path), path
        if "username" in dataset.secret_keys:
            url = url.set(username=secret("username"))
        if "password" in dataset.secret_keys:
            url = url.set(password=secret("password"))
        return url, None

    def read_catalog(self, dataset, secret, limits):
        from sqlalchemy import create_engine, event, inspect
        from sqlalchemy.pool import NullPool

        options = selection(dataset.options, {"schema", "tables", "connect_args"})
        connect_args = options.get("connect_args", {})
        if not isinstance(connect_args, Mapping):
            raise ConnectorError("connect_args must be a mapping")
        budget = ReadBudget(limits)
        engine = None
        try:
            url, path = self._url(dataset, secret)
            budget.remaining()
            engine = create_engine(url, connect_args=dict(connect_args), poolclass=NullPool, hide_parameters=True)
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
                        # The model only represents relationships within this
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
                                    "type": _type_name(column["type"], engine.dialect),
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
            declarations = read_declarations(path.with_suffix(".catalog.json")) if path is not None else {}
            catalog = build_catalog(dataset.name, self.source, tables, limits, declarations)
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
