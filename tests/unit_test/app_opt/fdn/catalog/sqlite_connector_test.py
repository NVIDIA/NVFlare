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

import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import pytest

from nvflare.app_common.fdn.catalog.common.schema import DEFAULT_LIMITS, ReadLimits
from nvflare.app_common.fdn.catalog.site.connector import ConnectorError
from nvflare.app_opt.fdn.catalog import sqlite_connector as subject


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "source.sqlite"
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE person (person_id INTEGER PRIMARY KEY, year_of_birth INTEGER);
            CREATE TABLE condition_occurrence (condition_occurrence_id INTEGER PRIMARY KEY,
                person_id INTEGER REFERENCES person(person_id), condition_start_date TEXT);
            INSERT INTO person VALUES (1, 1950);
            INSERT INTO condition_occurrence VALUES (1, 1, '2024-01-01');
            CREATE TABLE "quoted""table" (id INTEGER PRIMARY KEY);
        """
        )
        connection.commit()
    finally:
        connection.close()
    return path


@pytest.fixture
def connections(monkeypatch):
    opened, statements = [], []

    def connect(*args, **kwargs):
        assert args[0].endswith("?mode=ro") and kwargs["uri"]
        connection = sqlite3.connect(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        opened.append(connection)
        return connection

    monkeypatch.setattr(subject, "sqlite3", SimpleNamespace(connect=connect, Error=sqlite3.Error))
    yield opened, statements
    for connection in opened:
        # Even assertion failures leave no live database handles behind.
        connection.close()


def assert_closed(opened):
    assert opened
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")


def test_reads_schema_keys_and_relationships_without_clinical_rows(database, connections):
    result = subject.SQLiteConnector({"omop": database}).read_catalog("omop", ReadLimits(**DEFAULT_LIMITS))
    tables = {table.name: table for table in result.tables}
    assert set(tables) == {"person", "condition_occurrence", 'quoted"table'}
    assert tables["person"].primary_key == ("person_id",)
    relationship = tables["condition_occurrence"].relationships[0]
    assert relationship.columns == ("person_id",) and relationship.target_table == "person"
    assert relationship.target_columns == ("person_id",)
    opened, statements = connections
    assert all(sql.startswith("PRAGMA ") or "FROM sqlite_schema" in sql for sql in statements)
    assert result.provenance.source == "sqlite"
    assert_closed(opened)


@pytest.mark.parametrize("overrides", [{"max_calls": 1}, {"max_records": 1}])
def test_budgets_fail_with_cleanup(database, connections, overrides):
    with pytest.raises(ConnectorError) as error:
        subject.SQLiteConnector({"omop": database}).read_catalog(
            "omop", replace(ReadLimits(**DEFAULT_LIMITS), **overrides)
        )
    assert error.value.budget
    assert_closed(connections[0])


def test_missing_database_is_not_created(tmp_path):
    path = tmp_path / "missing.sqlite"
    with pytest.raises(ConnectorError):
        subject.SQLiteConnector({"omop": path}).read_catalog("omop", ReadLimits(**DEFAULT_LIMITS))
    assert not path.exists()


def test_expired_deadline_stops_before_query_and_closes_connection(database, connections, monkeypatch):
    clock = iter([0.0, 100.0])
    monkeypatch.setattr(subject, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    with pytest.raises(ConnectorError) as error:
        subject.SQLiteConnector({"omop": database}).read_catalog("omop", ReadLimits(**DEFAULT_LIMITS))
    assert error.value.budget and connections[1] == []
    assert_closed(connections[0])
