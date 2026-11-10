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

import json
import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import pytest

from nvflare.app_common.fdn.catalog.common.schema import DEFAULT_LIMITS, ReadLimits
from nvflare.app_common.fdn.catalog.site.connector import ConnectorError
from nvflare.app_opt.fdn.catalog import _metadata
from nvflare.app_opt.fdn.catalog.sql_connector import SQLConnector
from tests.unit_test.app_opt.fdn.catalog.datasets import read as read_dataset

NETWORK = {
    "uri": "postgresql+psycopg://private-host/clinical",
    "secrets": {"username": "STUDY_DB_USER", "password": "STUDY_DB_PASSWORD"},
}

sqlalchemy = pytest.importorskip("sqlalchemy")
LIMITS = ReadLimits(**DEFAULT_LIMITS)


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "source #1.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE person (person_id INTEGER PRIMARY KEY, year_of_birth INTEGER);
            CREATE TABLE condition_occurrence (condition_occurrence_id INTEGER PRIMARY KEY,
                person_id INTEGER REFERENCES person(person_id), condition_start_date TEXT);
            INSERT INTO person VALUES (1, 1950);
            INSERT INTO condition_occurrence VALUES (1, 1, 'private-source-row');
            CREATE TABLE "quoted""table" (id INTEGER PRIMARY KEY);
            CREATE TABLE composite (a INTEGER, b INTEGER, PRIMARY KEY (b, a));
            CREATE VIEW private_view AS SELECT * FROM person;
        """
        )
    return path


@pytest.fixture
def connections(monkeypatch):
    opened, statements, engines = [], [], []
    create_engine = sqlalchemy.create_engine

    def create(*args, **kwargs):
        engine = create_engine(*args, **kwargs)
        engines.append(engine)
        sqlalchemy.event.listen(engine, "connect", lambda connection, _: opened.append(connection))
        sqlalchemy.event.listen(engine, "before_cursor_execute", lambda c, cur, sql, *args: statements.append(sql))
        return engine

    monkeypatch.setattr(sqlalchemy, "create_engine", create)
    yield opened, statements
    for engine in engines:
        engine.dispose()
    for connection in opened:
        connection.close()


def assert_closed(opened):
    assert opened
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")


def test_reflects_columns_ordered_keys_and_relationships_without_source_rows(database, connections):
    result = read_dataset(SQLConnector(), LIMITS, name="omop", type_="sql", path=database)
    tables = {table.name: table for table in result.tables}
    assert set(tables) == {"person", "condition_occurrence", 'quoted"table', "composite"}
    assert tables["person"].primary_key == ("person_id",)
    assert tables["composite"].primary_key == ("b", "a")
    relationship = tables["condition_occurrence"].relationships[0]
    assert relationship.columns == ("person_id",) and relationship.target_table == "person"
    assert relationship.target_columns == ("person_id",)
    assert tables["person"].columns[1].type == "INTEGER"
    assert tables["person"].columns[1].nullable
    opened, statements = connections
    assert statements
    assert all(sql.startswith("PRAGMA ") or "sqlite_master" in sql for sql in statements)
    assert "private-source-row" not in json.dumps(result.to_wire())
    assert result.provenance.source == "sql"
    assert_closed(opened)


def test_a_column_type_sqlalchemy_cannot_render_is_published_as_unknown():
    from sqlalchemy import Integer
    from sqlalchemy.dialects import sqlite
    from sqlalchemy.types import NullType

    from nvflare.app_opt.fdn.catalog.sql_connector import _type_name

    assert _type_name(Integer(), sqlite.dialect()) == "INTEGER"
    assert _type_name(NullType(), sqlite.dialect()) == "UNKNOWN"


def test_mounted_sqlite_is_opened_read_only(database, monkeypatch):
    create_engine = sqlalchemy.create_engine

    def create(*args, **kwargs):
        engine = create_engine(*args, **kwargs)

        def check(connection, _):
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                connection.execute("CREATE TABLE forbidden (id INTEGER)")

        sqlalchemy.event.listen(engine, "connect", check)
        return engine

    monkeypatch.setattr(sqlalchemy, "create_engine", create)
    read_dataset(SQLConnector(), LIMITS, name="omop", type_="sql", path=database)


def test_a_sqlalchemy_uri_names_the_database(database, connections):
    result = read_dataset(
        SQLConnector(), LIMITS, name="omop", type_="sql", uri=f"sqlite:///{database}", options={"schema": "main"}
    )
    assert {table.name for table in result.tables} == {"person", "condition_occurrence", 'quoted"table', "composite"}
    assert_closed(connections[0])


def test_table_selection_does_not_expose_unselected_relationship_targets(database, connections):
    result = read_dataset(
        SQLConnector(), LIMITS, name="omop", type_="sql", path=database, options={"tables": ["condition_occurrence"]}
    )
    assert len(result.tables) == 1 and result.tables[0].name == "condition_occurrence"
    assert result.tables[0].relationships == ()
    assert_closed(connections[0])


def test_missing_schema_is_not_published_as_an_empty_catalog(database, connections):
    with pytest.raises(ConnectorError, match="schema is unavailable"):
        read_dataset(SQLConnector(), LIMITS, name="omop", type_="sql", path=database, options={"schema": "missing"})
    assert_closed(connections[0])


@pytest.mark.parametrize("overrides", [{"max_calls": 1}, {"max_records": 1}, {"max_records": 8}])
def test_budgets_discard_incomplete_metadata_and_close_connections(database, connections, overrides):
    with pytest.raises(ConnectorError) as error:
        read_dataset(SQLConnector(), replace(LIMITS, **overrides), name="omop", type_="sql", path=database)
    assert error.value.budget
    assert_closed(connections[0])


def test_missing_database_is_not_created(tmp_path):
    path = tmp_path / "missing.sqlite"
    with pytest.raises(ConnectorError):
        read_dataset(SQLConnector(), LIMITS, name="omop", type_="sql", path=path)
    assert not path.exists()


def test_an_unset_credential_makes_the_database_unavailable(monkeypatch):
    monkeypatch.delenv("STUDY_DB_PASSWORD", raising=False)
    with pytest.raises(ConnectorError) as error:
        read_dataset(
            SQLConnector(),
            LIMITS,
            name="omop",
            type_="sql",
            uri="postgresql+psycopg://private-host/clinical",
            secrets={"password": "STUDY_DB_PASSWORD"},
        )
    assert not error.value.budget


def test_expired_deadline_stops_before_schema_io(database, connections, monkeypatch):
    ticks = iter([0.0, 100.0])
    monkeypatch.setattr(_metadata, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    with pytest.raises(ConnectorError) as error:
        read_dataset(SQLConnector(), LIMITS, name="omop", type_="sql", path=database)
    assert error.value.budget and connections == ([], [])


def test_sidecar_supplies_semantics_without_replacing_observed_structure(database, connections):
    database.with_suffix(".catalog.json").write_text(
        json.dumps(
            {
                "semantic_profile": {"standard": "OMOP", "version": "5.4"},
                "tables": {"invented": {"primary_key": ["not-real"]}},
            }
        )
    )
    result = read_dataset(SQLConnector(), LIMITS, name="omop", type_="sql", path=database)
    assert result.semantic_profile.standard == "OMOP" and result.semantic_profile.version == "5.4"
    assert "invented" not in {table.name for table in result.tables}
    assert_closed(connections[0])


def test_invalid_declaration_fails_without_leaving_connections_open(database, connections):
    database.with_suffix(".catalog.json").write_text('{"unsupported": true}')
    with pytest.raises(ConnectorError, match="unsupported content"):
        read_dataset(SQLConnector(), LIMITS, name="omop", type_="sql", path=database)
    assert_closed(connections[0])


@pytest.mark.parametrize("options", [{"metadata": {}}, {"tables": []}, {"tables": ["person", "person"]}, {"schema": 3}])
def test_invalid_options_are_rejected_before_connecting(database, connections, options):
    with pytest.raises(ConnectorError):
        read_dataset(SQLConnector(), LIMITS, name="omop", type_="sql", path=database, options=options)
    assert connections == ([], [])


def test_database_errors_do_not_expose_url_credentials_or_driver_text(monkeypatch):
    monkeypatch.setenv("STUDY_DB_USER", "reader")
    monkeypatch.setenv("STUDY_DB_PASSWORD", "secret")

    def fail(*args, **kwargs):
        raise RuntimeError("reader:secret@private-host SELECT private_table")

    monkeypatch.setattr(sqlalchemy, "create_engine", fail)
    with pytest.raises(ConnectorError) as error:
        read_dataset(SQLConnector(), LIMITS, name="omop", type_="sql", **NETWORK)
    assert str(error.value) == "database metadata is unavailable" and not error.value.budget


def test_sqlalchemy_receives_the_dialect_credentials_and_driver_settings(monkeypatch):
    monkeypatch.setenv("STUDY_DB_USER", "reader")
    monkeypatch.setenv("STUDY_DB_PASSWORD", "secret")
    captured = []

    def create(url, **kwargs):
        captured.append((url, kwargs))
        raise RuntimeError("stop before opening a real network connection")

    monkeypatch.setattr(sqlalchemy, "create_engine", create)
    with pytest.raises(ConnectorError):
        read_dataset(
            SQLConnector(),
            LIMITS,
            name="omop",
            type_="sql",
            options={"connect_args": {"connect_timeout": 5}},
            **NETWORK,
        )
    url, kwargs = captured[0]
    assert url.drivername == "postgresql+psycopg" and url.database == "clinical" and url.host == "private-host"
    assert (url.username, url.password) == ("reader", "secret")
    assert kwargs["connect_args"] == {"connect_timeout": 5} and kwargs["hide_parameters"]
