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

from nvflare.app_common.fdn.catalog.common.limits import DEFAULT_LIMITS, ReadLimits
from nvflare.app_opt.fdn.catalog.odcs import read_odcs
from tests.unit_test.app_opt.fdn.catalog.datasets import dataset_config

LIMITS = ReadLimits(**DEFAULT_LIMITS)


def test_csv_projects_only_original_header_names(tmp_path):
    pytest.importorskip("datacontract")
    path = tmp_path / "site's private-file.csv"
    dataset, secret = dataset_config(path=path, type_="csv")
    prefix = '\ufeff"First, Name","a""b",Case,case\n'
    docs = []
    for suffix in ("", "private-marker,99,x,y\n", "0,1,2,3\n"):
        path.write_text(prefix + suffix, encoding="utf-8")
        docs.append(read_odcs(dataset, secret, LIMITS, site="site-1"))
    assert docs[0] == docs[1] == docs[2]
    assert docs[0]["schema"] == [
        {"name": "ds", "properties": [{"name": name} for name in ("First, Name", 'a"b', "Case", "case")]}
    ]
    assert "private" not in json.dumps(docs[0])
    assert docs[0]["id"] != read_odcs(dataset, secret, LIMITS, site="site-2")["id"]


@pytest.mark.parametrize("newline", ["\r", "\r\n", "\n"])
def test_csv_header_preserves_quoted_line_endings_and_ignores_row_encoding(tmp_path, newline):
    pytest.importorskip("datacontract")
    path = tmp_path / "visits.csv"
    path.write_bytes(f'id,"visit{newline}date"{newline}'.encode() + b"1,\xff")
    dataset, secret = dataset_config(path=path, type_="csv")
    contract = read_odcs(dataset, secret, LIMITS, site="one")
    assert contract["schema"][0]["properties"] == [{"name": "id"}, {"name": f"visit{newline}date"}]


@pytest.mark.parametrize("content", ["", "id,id\n", 'a,""\n', '"unterminated\n'])
def test_invalid_csv_header_is_refused(tmp_path, content):
    import csv

    from nvflare.app_common.fdn.catalog.site.source_error import CatalogSourceError

    pytest.importorskip("datacontract")
    path = tmp_path / "header.csv"
    path.write_text(content)
    dataset, secret = dataset_config(path=path, type_="csv")
    with pytest.raises((CatalogSourceError, csv.Error)):
        read_odcs(dataset, secret, LIMITS, site="site")


def test_provided_contract_wins_even_when_missing_or_invalid(tmp_path, monkeypatch):
    pytest.importorskip("ruamel.yaml")
    pytest.importorskip("jsonschema")
    from jsonschema import ValidationError

    from nvflare.app_opt.fdn.catalog import odcs

    def no_live_discovery(*args):
        pytest.fail("a supplied contract must never trigger SQL access")

    monkeypatch.setattr(odcs, "import_schema", no_live_discovery)
    path = tmp_path / "approved.yaml"
    dataset, secret = dataset_config(type_="sql", uri="postgresql://example.org/study")
    dataset = replace(dataset, usage={"catalog": {"contract_uri": path.as_uri()}})
    with pytest.raises(ValueError):
        read_odcs(dataset, secret, LIMITS, site="site")
    contract = {
        "apiVersion": "v3.2.0",
        "kind": "DataContract",
        "id": "steward-id",
        "version": "5",
        "description": {"purpose": "Approved by the steward"},
    }
    path.write_text(json.dumps(contract))
    assert read_odcs(dataset, secret, LIMITS, site="site") == contract
    path.write_text(json.dumps({**contract, "unsupported": True}))
    with pytest.raises(ValidationError):
        read_odcs(dataset, secret, LIMITS, site="site")


def test_sqlite_selection_imports_ddl_without_disclosing_external_foreign_keys(tmp_path):
    import sqlite3

    pytest.importorskip("datacontract")
    path = tmp_path / "source.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE private_target (id INTEGER PRIMARY KEY)")
        db.execute("CREATE TABLE visits (id INTEGER PRIMARY KEY, subject INTEGER REFERENCES private_target(id))")
    dataset, secret = dataset_config(type_="sql", path=path, options={"tables": ["visits"]})
    contract = read_odcs(dataset, secret, LIMITS, site="one")
    assert [table["name"] for table in contract["schema"]] == ["visits"]
    assert [col["name"] for col in contract["schema"][0]["properties"]] == ["id", "subject"]
    assert "private_target" not in json.dumps(contract)


def test_postgres_import_connects_with_the_dataset_credential(monkeypatch):
    pytest.importorskip("datacontract")
    psycopg = pytest.importorskip("psycopg")
    from unittest.mock import Mock

    # Stop at the database boundary, after the real upstream importer has selected its driver.
    connect = Mock(side_effect=RuntimeError("test connection reached"))
    monkeypatch.setattr(psycopg, "connect", connect)
    monkeypatch.setenv("SITE_DB_USER", "study_reader")
    monkeypatch.setenv("SITE_DB_PASSWORD", "test-password")
    dataset, secret = dataset_config(
        type_="sql",
        uri="postgresql://study.example:5544/clinical",
        options={"schema": "study"},
        secrets={"username": "SITE_DB_USER", "password": "SITE_DB_PASSWORD"},
    )
    with pytest.raises(RuntimeError, match="test connection reached"):
        read_odcs(dataset, secret, LIMITS, site="one")
    connect.assert_called_once_with(
        host="study.example", port=5544, dbname="clinical", user="study_reader", password="test-password"
    )


def test_partition_parts_must_have_matching_imported_schemas(tmp_path):
    pytest.importorskip("datacontract")
    arrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    from nvflare.app_common.fdn.catalog.site.source_error import CatalogSourceError

    parts = tmp_path / "visits"
    parts.mkdir()
    parquet.write_table(arrow.table({"id": [1]}), parts / "a.parquet")
    parquet.write_table(arrow.table({"id": ["private"]}), parts / "b.parquet")
    dataset, secret = dataset_config(path=tmp_path)
    with pytest.raises(CatalogSourceError, match="inconsistent"):
        read_odcs(dataset, secret, LIMITS, site="one")
