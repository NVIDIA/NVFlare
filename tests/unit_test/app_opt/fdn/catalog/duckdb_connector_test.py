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
from dataclasses import replace

import pytest

from nvflare.app_common.fdn.catalog.common.schema import DEFAULT_LIMITS, ReadLimits
from nvflare.app_common.fdn.catalog.site.connector import ConnectorError
from nvflare.app_opt.fdn.catalog import _metadata
from nvflare.app_opt.fdn.catalog.duckdb_connector import DuckDBConnector
from tests.unit_test.app_opt.fdn.catalog.datasets import read as read_dataset

duckdb = pytest.importorskip("duckdb")
LIMITS = ReadLimits(**DEFAULT_LIMITS)


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "source #1.duckdb"
    connection = duckdb.connect(str(path))
    try:
        connection.execute(
            """
            CREATE TABLE person (person_id INTEGER PRIMARY KEY, year_of_birth INTEGER NOT NULL, gender VARCHAR);
            CREATE TABLE condition_occurrence (condition_occurrence_id INTEGER PRIMARY KEY, person_id INTEGER,
                condition_start_date DATE, cost DECIMAL(10, 2), tags VARCHAR[],
                FOREIGN KEY (person_id) REFERENCES person (person_id));
            INSERT INTO person VALUES (1, 1950, 'private-source-row');
            CREATE TABLE composite (a INTEGER, b INTEGER, PRIMARY KEY (b, a));
            CREATE TABLE pair_link (a INTEGER, b INTEGER, FOREIGN KEY (b, a) REFERENCES composite (b, a));
            CREATE SCHEMA other;
            CREATE TABLE other.extra (id INTEGER PRIMARY KEY);
            CREATE VIEW private_view AS SELECT * FROM person;
            """
        )
    finally:
        connection.close()
    return path


def test_reads_columns_types_keys_and_relationships_without_source_rows(database):
    result = read_dataset(DuckDBConnector(), LIMITS, name="omop", type_="duckdb", path=database)
    tables = {table.name: table for table in result.tables}
    assert set(tables) == {"person", "condition_occurrence", "composite", "pair_link"}
    assert [(c.name, c.type, c.nullable) for c in tables["person"].columns] == [
        ("person_id", "INTEGER", False),
        ("year_of_birth", "INTEGER", False),
        ("gender", "VARCHAR", True),
    ]
    condition = tables["condition_occurrence"]
    assert [c.type for c in condition.columns][2:] == ["DATE", "DECIMAL(10,2)", "VARCHAR[]"]
    assert tables["person"].primary_key == ("person_id",)
    assert tables["composite"].primary_key == ("b", "a")
    relationship = condition.relationships[0]
    assert (relationship.columns, relationship.target_table) == (("person_id",), "person")
    assert relationship.target_columns == ("person_id",)
    assert tables["pair_link"].relationships[0].columns == ("b", "a")
    assert result.provenance.source == "duckdb"
    assert "private-source-row" not in json.dumps(result.to_wire())


def test_views_and_other_schemas_are_not_published(database):
    names = {t.name for t in read_dataset(DuckDBConnector(), LIMITS, name="omop", type_="duckdb", path=database).tables}
    assert "private_view" not in names and "extra" not in names
    other = read_dataset(
        DuckDBConnector(), LIMITS, name="omop", type_="duckdb", path=database, options={"schema": "other"}
    )
    assert [t.name for t in other.tables] == ["extra"]


def test_table_selection_does_not_expose_unselected_relationship_targets(database):
    result = read_dataset(
        DuckDBConnector(),
        LIMITS,
        name="omop",
        type_="duckdb",
        path=database,
        options={"tables": ["condition_occurrence"]},
    )
    assert [t.name for t in result.tables] == ["condition_occurrence"]
    assert result.tables[0].relationships == ()


def test_the_database_is_opened_read_only_without_external_access(database, monkeypatch):
    seen = []
    connect = duckdb.connect

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return connect(*args, **kwargs)

    monkeypatch.setattr(duckdb, "connect", spy)
    read_dataset(DuckDBConnector(), LIMITS, name="omop", type_="duckdb", path=database)
    assert seen == [{"read_only": True, "config": {"enable_external_access": False}}]
    with pytest.raises(duckdb.Error):
        connection = duckdb.connect(str(database), read_only=True)
        try:
            connection.execute("CREATE TABLE forbidden (id INTEGER)")
        finally:
            connection.close()


def test_the_connection_is_closed_on_success_and_failure(database, monkeypatch):
    opened = []
    connect = duckdb.connect

    def spy(*args, **kwargs):
        opened.append(connect(*args, **kwargs))
        return opened[-1]

    monkeypatch.setattr(duckdb, "connect", spy)
    read_dataset(DuckDBConnector(), LIMITS, name="omop", type_="duckdb", path=database)
    with pytest.raises(ConnectorError):
        read_dataset(
            DuckDBConnector(), LIMITS, name="omop", type_="duckdb", path=database, options={"tables": ["missing"]}
        )
    assert len(opened) == 2
    for connection in opened:
        with pytest.raises(duckdb.ConnectionException):
            connection.execute("SELECT 1")


@pytest.mark.parametrize("options", [{"metadata": {}}, {"tables": []}, {"tables": ["person", "person"]}, {"schema": 3}])
def test_invalid_options_are_rejected(database, options):
    with pytest.raises(ConnectorError):
        read_dataset(DuckDBConnector(), LIMITS, name="omop", type_="duckdb", path=database, options=options)


def test_missing_schema_or_table_is_unavailable(database):
    for options in ({"schema": "missing"}, {"tables": ["missing"]}):
        with pytest.raises(ConnectorError):
            read_dataset(DuckDBConnector(), LIMITS, name="omop", type_="duckdb", path=database, options=options)


def test_missing_unmounted_or_unreadable_files_are_unavailable(tmp_path):
    bad = tmp_path / "bad.duckdb"
    bad.write_text("not a database")
    for where in ({"uri": "https://example.com/db"}, {"path": tmp_path / "absent.duckdb"}, {"path": bad}):
        with pytest.raises(ConnectorError) as error:
            read_dataset(DuckDBConnector(), LIMITS, name="omop", type_="duckdb", **where)
        assert not error.value.budget
    assert "bad.duckdb" not in str(error.value) and "not a database" not in str(error.value)


@pytest.mark.parametrize("overrides", [{"max_calls": 2}, {"max_records": 3}])
def test_budgets_stop_the_read(database, overrides):
    with pytest.raises(ConnectorError) as error:
        read_dataset(DuckDBConnector(), replace(LIMITS, **overrides), name="omop", type_="duckdb", path=database)
    assert error.value.budget


def test_an_exhausted_duration_budget_stops_the_read(database, monkeypatch):
    clock = iter([0, 0, 1e9, 1e9, 1e9, 1e9])
    monkeypatch.setattr(_metadata.time, "monotonic", lambda: next(clock))
    with pytest.raises(ConnectorError) as error:
        read_dataset(DuckDBConnector(), LIMITS, name="omop", type_="duckdb", path=database)
    assert error.value.budget
