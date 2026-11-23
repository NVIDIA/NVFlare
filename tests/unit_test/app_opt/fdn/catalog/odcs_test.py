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

    from nvflare.app_common.fdn.catalog.site.connector import ConnectorError

    pytest.importorskip("datacontract")
    path = tmp_path / "header.csv"
    path.write_text(content)
    dataset, secret = dataset_config(path=path, type_="csv")
    with pytest.raises((ConnectorError, csv.Error)):
        read_odcs(dataset, secret, LIMITS, site="site")


def test_provided_contract_wins_even_when_missing_or_invalid(tmp_path, monkeypatch):
    pytest.importorskip("ruamel.yaml")
    pytest.importorskip("jsonschema")
    from jsonschema import ValidationError

    from nvflare.app_opt.fdn.catalog import sql_connector

    def no_live_discovery(*args):
        pytest.fail("a supplied contract must never trigger SQL access")

    monkeypatch.setattr(sql_connector, "read_tables", no_live_discovery)
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
