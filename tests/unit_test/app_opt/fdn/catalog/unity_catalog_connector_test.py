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

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import requests

from nvflare.app_common.fdn.catalog.common.schema import DEFAULT_LIMITS, ReadLimits
from nvflare.app_common.fdn.catalog.site.connector import ConnectorError
from nvflare.app_opt.fdn.catalog import _metadata
from nvflare.app_opt.fdn.catalog import unity_catalog_connector as subject

LIMITS = ReadLimits(**DEFAULT_LIMITS)
OPTIONS = {"catalog": "unity", "schema": "clinical"}


def table(name="person"):
    return {
        "name": name,
        "catalog_name": "unity",
        "schema_name": "clinical",
        "columns": [{"name": "person_id", "type_text": "bigint", "nullable": False}],
        "storage_location": "s3://private-bucket/private-data",
        "owner": "private-owner@example.com",
        "properties": {"password": "private-secret"},
        "view_definition": "SELECT private_columns FROM private_table",
    }


@pytest.fixture
def transport(monkeypatch):
    """The HTTP boundary is controlled; no live server, sockets or workers."""
    monkeypatch.setenv("UNITY_CATALOG_URL", "http://localhost:8080/api/2.1/unity-catalog")
    responses, calls, closed = [], [], []

    class Response:
        def __init__(self, data, status_code=200):
            self.data = data
            self.status_code = status_code

        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append("response")

        def json(self):
            return self.data

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append("session")

        def get(self, url, **kwargs):
            calls.append((url, copy.deepcopy(kwargs)))
            data = responses.pop(0)
            if isinstance(data, Exception):
                raise data
            return data if isinstance(data, Response) else Response(data)

    monkeypatch.setattr(subject.requests, "Session", Session)
    return SimpleNamespace(responses=responses, calls=calls, closed=closed, response=Response)


def test_reads_all_pages_and_only_publishes_structural_metadata(transport):
    transport.responses.extend(
        [{"tables": [table("visits")], "next_page_token": "page-2"}, {"tables": [table("person")]}]
    )
    catalog = subject.UnityCatalogConnector({}).read_catalog("omop", LIMITS, OPTIONS)
    assert [t.name for t in catalog.tables] == ["person", "visits"]
    assert catalog.tables[0].columns[0].type == "bigint" and catalog.tables[0].columns[0].nullable is False
    assert catalog.tables[0].primary_key == () and catalog.tables[0].relationships == ()
    assert catalog.provenance.source == "unity-catalog"
    wire = json.dumps(catalog.to_wire())
    assert all(word not in wire for word in ["private-", "password", "view_definition", "s3://"])
    assert transport.calls[0][1]["params"] == {"catalog_name": "unity", "schema_name": "clinical", "max_results": 50}
    assert transport.calls[1][1]["params"]["page_token"] == "page-2"
    assert all(call[1]["allow_redirects"] is False and 0 < call[1]["timeout"] <= 10 for call in transport.calls)
    assert transport.closed == ["response", "response", "session"]


def test_explicit_table_selection_fetches_only_selected_metadata(transport, monkeypatch):
    monkeypatch.setenv("STUDY_UC_TOKEN", "site-token")
    transport.responses.append(table("quoted/name"))
    catalog = subject.UnityCatalogConnector({}, token_env="STUDY_UC_TOKEN").read_catalog(
        "omop", LIMITS, {**OPTIONS, "tables": ["quoted/name"]}
    )
    assert catalog.tables[0].name == "quoted/name"
    assert len(transport.calls) == 1 and transport.calls[0][0].endswith("/tables/unity.clinical.quoted%2Fname")
    assert transport.calls[0][1]["headers"]["Authorization"] == "Bearer site-token"
    assert "site-token" not in json.dumps(catalog.to_wire())


def test_list_without_columns_gets_complete_table_details(transport):
    brief = table()
    del brief["columns"]
    transport.responses.extend([{"tables": [brief]}, table()])
    catalog = subject.UnityCatalogConnector({}).read_catalog("omop", LIMITS, OPTIONS)
    assert catalog.tables[0].columns[0].name == "person_id"
    assert transport.calls[1][0].endswith("/tables/unity.clinical.person")


def test_missing_column_details_fail_without_returning_partial_metadata(transport):
    brief = table()
    del brief["columns"]
    transport.responses.extend([{"tables": [brief]}, brief])
    with pytest.raises(ConnectorError, match="column inventory is missing"):
        subject.UnityCatalogConnector({}).read_catalog("omop", LIMITS, OPTIONS)
    assert len(transport.calls) == 2 and transport.closed[-1] == "session"


@pytest.mark.parametrize("overrides", [{"max_calls": 1}, {"max_records": 2}])
def test_budgets_never_return_a_partial_schema(transport, overrides):
    transport.responses.extend([{"tables": [table()], "next_page_token": "page-2"}, {"tables": [table("visits")]}])
    with pytest.raises(ConnectorError) as error:
        subject.UnityCatalogConnector({}).read_catalog("omop", replace(LIMITS, **overrides), OPTIONS)
    assert error.value.budget and transport.closed[-1] == "session"


def test_expired_deadline_performs_no_http_request(transport, monkeypatch):
    ticks = iter([0.0, 100.0])
    monkeypatch.setattr(_metadata, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    with pytest.raises(ConnectorError) as error:
        subject.UnityCatalogConnector({}).read_catalog("omop", LIMITS, OPTIONS)
    assert error.value.budget and not transport.calls and transport.closed == ["session"]


def test_server_failures_are_sanitized_and_connections_are_closed(transport):
    transport.responses.extend(
        [{"tables": [table()], "next_page_token": "p2"}, transport.response({"secret": "x"}, 403)]
    )
    with pytest.raises(ConnectorError) as error:
        subject.UnityCatalogConnector({}).read_catalog("omop", LIMITS, OPTIONS)
    assert str(error.value) == "Unity Catalog metadata is unavailable" and not error.value.budget
    assert transport.closed == ["response", "response", "session"]


def test_transport_failure_does_not_expose_endpoint_token_or_exception(transport):
    transport.responses.append(requests.ConnectionError("private-url Bearer private-token"))
    with pytest.raises(ConnectorError) as error:
        subject.UnityCatalogConnector({}).read_catalog("omop", LIMITS, OPTIONS)
    assert str(error.value) == "Unity Catalog metadata is unavailable" and transport.closed == ["session"]


def test_request_timeout_is_a_budget_stop_not_evidence_that_a_dataset_is_gone(transport):
    transport.responses.append(requests.Timeout("private-url Bearer private-token"))
    with pytest.raises(ConnectorError) as error:
        subject.UnityCatalogConnector({}).read_catalog("omop", LIMITS, OPTIONS)
    assert error.value.budget and str(error.value) == "Unity Catalog metadata request timed out"
    assert transport.closed == ["session"]


def test_repeated_page_token_fails_instead_of_looping(transport):
    transport.responses.extend([{"tables": [], "next_page_token": "same"}] * 2)
    with pytest.raises(ConnectorError, match="pagination is inconsistent"):
        subject.UnityCatalogConnector({}).read_catalog("omop", LIMITS, OPTIONS)
    assert len(transport.calls) == 2


@pytest.mark.parametrize("change", [{"schema_name": "another-study"}, {"name": "wrong"}, {"columns": None}])
def test_inconsistent_table_metadata_fails_closed(transport, change):
    transport.responses.append({**table(), **change})
    with pytest.raises(ConnectorError):
        subject.UnityCatalogConnector({}).read_catalog("omop", LIMITS, {**OPTIONS, "tables": ["person"]})
    assert transport.closed[-1] == "session"


def test_duplicate_tables_across_pages_fail_closed(transport):
    transport.responses.extend([{"tables": [table()], "next_page_token": "p2"}, {"tables": [table()]}])
    with pytest.raises(ConnectorError, match="inventory is inconsistent"):
        subject.UnityCatalogConnector({}).read_catalog("omop", LIMITS, OPTIONS)


@pytest.mark.parametrize(
    "options", [{}, {"schema": "clinical"}, {**OPTIONS, "tables": []}, {**OPTIONS, "token": "secret"}]
)
def test_invalid_selection_is_rejected_before_io(transport, options):
    with pytest.raises(ConnectorError):
        subject.UnityCatalogConnector({}).read_catalog("omop", LIMITS, options)
    assert transport.calls == []


def test_configured_but_missing_credentials_never_fall_back_to_anonymous(transport, monkeypatch):
    monkeypatch.delenv("MISSING_UC_TOKEN", raising=False)
    with pytest.raises(ConnectorError, match="credentials are not configured"):
        subject.UnityCatalogConnector({}, token_env="MISSING_UC_TOKEN").read_catalog("omop", LIMITS, OPTIONS)
    assert transport.calls == []


@pytest.mark.parametrize("host", ["uc.example.com", "localhost.example.com", "10.0.0.1", "0.0.0.0", "[::]"])
def test_bearer_tokens_never_reach_non_loopback_http_endpoints(transport, monkeypatch, host):
    monkeypatch.setenv("UNITY_CATALOG_URL", f"http://{host}:8080/api/2.1/unity-catalog")
    monkeypatch.setenv("UC_TOKEN", "synthetic-token")
    with pytest.raises(ConnectorError, match="requires HTTPS"):
        subject.UnityCatalogConnector({}, token_env="UC_TOKEN").read_catalog("omop", LIMITS, OPTIONS)
    assert transport.calls == []


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://uc.example.com",
        "http://localhost",
        "http://localhost.",
        "http://127.0.0.1",
        "http://127.12.0.1",
        "http://[::1]",
    ],
)
def test_bearer_tokens_use_https_or_explicit_local_loopback(transport, monkeypatch, endpoint):
    monkeypatch.setenv("UNITY_CATALOG_URL", endpoint + "/api/2.1/unity-catalog")
    monkeypatch.setenv("UC_TOKEN", "synthetic-token")
    transport.responses.append({"tables": [table()]})
    subject.UnityCatalogConnector({}, token_env="UC_TOKEN").read_catalog("omop", LIMITS, OPTIONS)
    assert len(transport.calls) == 1
    assert transport.calls[0][1]["headers"]["Authorization"] == "Bearer synthetic-token"
    assert transport.calls[0][1]["allow_redirects"] is False


def test_databricks_workspace_uses_same_rest_contract_and_follows_empty_pages(transport, monkeypatch):
    monkeypatch.setenv("UNITY_CATALOG_URL", "https://workspace.cloud.databricks.com/api/2.1/unity-catalog")
    monkeypatch.setenv("DATABRICKS_TOKEN", "site-managed-token")
    delta_table = table("events")
    delta_table.update(table_type="MANAGED", data_source_format="DELTA", table_id="provider-table-id")
    delta_table["columns"][0]["type_text"] = "array<struct<code:string,value:double>>"
    transport.responses.extend(
        [
            {"tables": [], "next_page_token": "continue-after-empty-page"},
            {"tables": [delta_table]},
        ]
    )
    result = subject.UnityCatalogConnector({}, token_env="DATABRICKS_TOKEN").read_catalog("events", LIMITS, OPTIONS)
    assert result.tables[0].columns[0].type == "array<struct<code:string,value:double>>"
    assert len(transport.calls) == 2
    assert transport.calls[1][1]["params"]["page_token"] == "continue-after-empty-page"
    assert all(
        call[0].startswith("https://workspace.cloud.databricks.com/api/2.1/unity-catalog/tables")
        for call in transport.calls
    )
    assert all(call[1]["headers"]["Authorization"] == "Bearer site-managed-token" for call in transport.calls)
    assert "site-managed-token" not in json.dumps(result.to_wire())
