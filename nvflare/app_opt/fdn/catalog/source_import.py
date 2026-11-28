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

"""Adapt site-owned source settings to Data Contract CLI's ODCS importers."""

import sqlite3
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import get_args
from urllib.parse import unquote, urlsplit

from nvflare.app_common.fdn.catalog.site.source_error import CatalogSourceError
from nvflare.app_opt.fdn.catalog._metadata import local_path, selection
from nvflare.app_opt.fdn.catalog._parquet_files import table_files


def _structure(element):
    """Keep structural ODCS fields at every level, including keys on nested properties."""
    result = {
        key: element[key]
        for key in ("id", "name", "physicalType", "logicalType", "required", "primaryKey", "primaryKeyPosition")
        if key in element
    }
    # The Parquet importer stores DECIMAL parameters as numeric ODCS custom properties.
    if result.get("physicalType") == "DECIMAL":
        params = {p["property"]: p.get("value") for p in element.get("customProperties", [])}
        precision, scale = params.get("precision"), params.get("scale")
        if type(precision) is int and type(scale) is int:
            result["physicalType"] = f"DECIMAL({precision},{scale})"
    if "relationships" in element:
        result["relationships"] = [
            {key: rel[key] for key in ("type", "from", "to") if key in rel} for rel in element["relationships"]
        ]
    if "properties" in element:
        result["properties"] = [_structure(prop) for prop in element["properties"]]
    if "items" in element:
        result["items"] = _structure(element["items"])
    if "map" in element:
        result["map"] = {key: _structure(element["map"][key]) for key in ("key", "value")}
    return result


def _import_objects(budget, source_format, source=None, /, **kwargs):
    from datacontract.data_contract import DataContract

    budget.request()
    model = DataContract.import_from_source(format=source_format, source=source, **kwargs)
    return [
        _structure(table.model_dump(mode="json", by_alias=True, exclude_none=True)) for table in model.schema_ or []
    ]


def _postgres(dataset, secret, budget):
    url = urlsplit(dataset.uri)
    if (
        url.scheme not in {"postgresql", "postgresql+psycopg"}
        or not url.hostname
        or not url.path.strip("/")
        or url.query
        or url.fragment
        or url.username is not None
    ):
        raise CatalogSourceError("expected a PostgreSQL host and database for the postgres importer")
    options = selection(dataset.options, {"schema", "tables"})
    objects = _import_objects(
        budget,
        "postgres",
        url.hostname,
        port=url.port,
        database=unquote(url.path[1:]),
        schema=options.get("schema", "public"),
        postgres_table=options.get("tables"),
        config={
            "DATACONTRACT_POSTGRES_USERNAME": secret("username"),
            "DATACONTRACT_POSTGRES_PASSWORD": secret("password"),
        },
    )

    if "tables" in options and {obj["name"] for obj in objects} != set(options["tables"]):
        raise CatalogSourceError("requested PostgreSQL table unavailable")
    return objects


def _unity(dataset, secret, budget):
    from databricks.sdk import WorkspaceClient

    url = urlsplit(dataset.uri)
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username is not None
        or url.query
        or url.fragment
        or url.path.rstrip("/") not in {"", "/api/2.1/unity-catalog"}
    ):
        raise CatalogSourceError("invalid Unity Catalog API root")
    options = selection(dataset.options, {"catalog", "schema", "tables"}, required=("catalog", "schema"))
    host = f"{url.scheme}://{url.netloc}"
    if "token" in dataset.secret_keys:
        if url.scheme != "https" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise CatalogSourceError("Unity Catalog bearer-token authentication requires HTTPS")
        token = secret("token")
        if not token:
            raise CatalogSourceError("missing Unity Catalog token")
    else:
        # The SDK requires an authentication method even for a local OSS server with auth disabled.
        token = "unused-for-unauthenticated-unity-catalog"
    names = options.get("tables")
    if names is None:
        client = WorkspaceClient(host=host, token=token, auth_type="pat")
        budget.request()
        names = []
        for table in client.tables.list(catalog_name=options["catalog"], schema_name=options["schema"]):
            names.append(table.name)
            budget.records(1 + len(names))
    objects = []
    for name in names:
        parts = [options["catalog"], options["schema"], name]
        if any(not p or "." in p for p in parts):
            raise CatalogSourceError("invalid Unity Catalog qualified table name")
        imported = _import_objects(
            budget,
            "unity",
            unity_table_full_name=[".".join(parts)],
            config={
                # The pinned importer gives a profile precedence over an explicit host and token.
                "DATACONTRACT_DATABRICKS_PROFILE": "",
                "DATACONTRACT_DATABRICKS_SERVER_HOSTNAME": host,
                "DATACONTRACT_DATABRICKS_TOKEN": token,
            },
        )
        if len(imported) != 1 or imported[0]["name"] != name:
            raise CatalogSourceError("Unity Catalog returned an unexpected table")
        objects.extend(imported)
    return objects


def _parquet(dataset, budget):
    from pyarrow import parquet

    selection(dataset.options, ())
    objects = []
    for name, parts in table_files(local_path(dataset), budget.deadline, budget.request):
        if len(parts) > 1:
            # Compare the native footer types before the upstream importer simplifies them.
            budget.request()
            schema = parquet.read_schema(parts[0])
            for path in parts[1:]:
                budget.request()
                if not schema.equals(parquet.read_schema(path), check_metadata=False):
                    raise CatalogSourceError("inconsistent Parquet part schemas")
        (table,) = _import_objects(budget, "parquet", str(parts[0]))
        table["name"] = name
        objects.append(table)
    return objects


def _database_file(dataset, budget):
    """The upstream SQL importer reads DDL. Extract definitions from the read-only file catalog."""
    options = selection(dataset.options, {"schema", "tables"})
    schema = options.get("schema", "main")
    path = local_path(dataset).resolve(strict=True)
    budget.request()
    if dataset.type == "duckdb":
        import duckdb

        with closing(duckdb.connect(str(path), read_only=True, config={"enable_external_access": False})) as db:
            if not db.execute(
                "SELECT 1 FROM information_schema.schemata WHERE catalog_name=current_database() AND schema_name=?",
                [schema],
            ).fetchone():
                raise CatalogSourceError("requested DuckDB schema unavailable")
            budget.request()
            columns = db.execute(
                "SELECT table_name, column_name FROM duckdb_columns() WHERE database_name=current_database() AND schema_name=? ORDER BY table_name, column_index",
                [schema],
            ).fetchall()
            budget.request()
            rows = db.execute(
                "SELECT table_name, sql FROM duckdb_tables() WHERE database_name=current_database() AND schema_name = ? ORDER BY table_name",
                [schema],
            ).fetchall()
        dialect = "duckdb"
    else:
        if schema != "main":
            raise CatalogSourceError("SQLite file catalogs support the main schema")
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
            rows = db.execute(
                "SELECT name, sql FROM sqlite_schema WHERE type='table' AND name NOT GLOB 'sqlite_*' ORDER BY name"
            ).fetchall()
            names = options.get("tables", list(dict(rows)))
            columns = []
            # Limit each IN list to fit SQLite's host parameter limit on older builds.
            for offset in range(0, len(names), 500):
                batch = names[offset : offset + 500]
                placeholders = ",".join("?" for _ in batch)
                budget.request()
                columns.extend(
                    db.execute(
                        "SELECT s.name, c.name FROM sqlite_schema s CROSS JOIN pragma_table_xinfo(s.name, 'main') c "
                        f"WHERE s.type='table' AND s.name IN ({placeholders}) AND c.hidden != 1 ORDER BY s.name, c.cid",
                        batch,
                    )
                )
        dialect = "sqlite"
    definitions = dict(rows)
    names = options.get("tables", list(definitions))
    if set(names) - definitions.keys():
        raise CatalogSourceError("requested table definition unavailable")
    budget.records(1 + len(names))
    if not names:
        return []
    with TemporaryDirectory() as tmp:
        ddl = Path(tmp) / "schema.sql"
        ddl.write_text("\n".join(definitions[name].rstrip(";") + ";" for name in names), encoding="utf-8")
        objects = _import_objects(budget, "sql", str(ddl), dialect=dialect)
    if {obj["name"] for obj in objects} != set(names):
        raise CatalogSourceError("SQL importer did not return all requested tables")
    expected_columns = {name: [] for name in names}
    for table_name, column_name in columns:
        if table_name in expected_columns:
            expected_columns[table_name].append(column_name)
    for obj in objects:
        if [p["name"] for p in obj.get("properties", [])] != expected_columns[obj["name"]]:
            raise CatalogSourceError("SQL importer returned an incomplete table schema")
    # The DDL of a selected table can reference a table outside the publication scope.
    targets = {f"{obj['name']}.{col['name']}" for obj in objects for col in obj.get("properties", [])}
    for obj in objects:
        for col in obj.get("properties", []):
            if "relationships" in col:
                col["relationships"] = [
                    rel
                    for rel in col["relationships"]
                    if all(ref in targets for ref in (rel["to"] if isinstance(rel["to"], list) else [rel["to"]]))
                ]
    return objects


def import_schema(dataset, secret, budget):
    if dataset.type == "parquet":
        return _parquet(dataset, budget)
    if dataset.type == "unity-catalog":
        return _unity(dataset, secret, budget)
    if dataset.type in {"duckdb", "sql"} and dataset.path is not None:
        return _database_file(dataset, budget)
    if dataset.type == "sql":
        return _postgres(dataset, secret, budget)
    # An explicit site setting selects a metadata importer from the upstream registry.
    # Other dataset types in this shared Study configuration need not support discovery.
    args = dict(dataset.options)
    source_form = args.pop("metadata_source", None)
    url = urlsplit(dataset.uri)
    if source_form not in ("host", "uri") or dataset.path is not None or not url.hostname or url.username is not None:
        raise CatalogSourceError("expected a configured network metadata importer")
    if source_form == "host" and (url.path not in ("", "/") or url.port is not None or url.query or url.fragment):
        raise CatalogSourceError("host metadata sources take a bare host; put other importer arguments in options")
    if {"format", "source"} & args.keys():
        raise CatalogSourceError("reserved metadata importer argument")
    settings = args.pop("config", {})
    if not isinstance(settings, Mapping) or set(settings).intersection(dataset.secret_keys):
        raise CatalogSourceError("invalid or duplicate importer settings")
    if settings:
        from datacontract.config import Config
        from datacontract.config.settings import env_name
        from pydantic import SecretStr

        for name, field in Config.model_fields.items():
            if env_name(name, field) in settings and (
                field.annotation is SecretStr or SecretStr in get_args(field.annotation)
            ):
                raise CatalogSourceError("credential setting in options.config; put it in secrets")
    config = {**settings, **{key: secret(key) for key in dataset.secret_keys}}
    source = url.hostname if source_form == "host" else dataset.uri
    return _import_objects(budget, dataset.type, source, config=config, **args)
