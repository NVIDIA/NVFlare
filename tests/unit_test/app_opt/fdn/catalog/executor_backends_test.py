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

import pytest
import yaml

from nvflare.apis.dataset_config_spec import DATASET_CONFIG_ENV
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.fdn.catalog.common.payload import (
    CATALOG_JOB_META,
    CATALOG_READ_KEY,
    TASK_NAME,
    ResultStatus,
    build_task_payload,
    parse_result,
)
from nvflare.app_common.fdn.catalog.common.schema import DEFAULT_LIMITS, ReadLimits
from nvflare.app_common.fdn.catalog.site.executor import CatalogExecutor


@pytest.mark.parametrize("managed", [False, True], ids=["unity-catalog-oss", "databricks"])
def test_one_executor_reads_file_sql_and_unity_catalog_sources(tmp_path, monkeypatch, managed):
    """Real file/SQL metadata and a controlled REST boundary, configured in one datasets.yaml and read through the
    dataset configuration API, share the same task and catalog rules."""
    import sqlite3

    arrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("sqlalchemy")
    from nvflare.app_opt.fdn.catalog import unity_catalog_connector as uc_module

    root = tmp_path / "data" / "study1"
    root.mkdir(parents=True)
    parquet.write_table(arrow.table({"id": [1], "secret": ["private-row"]}), root / "parquet")
    (root / "csv").write_text("id,secret\n1,private-row\n")
    with sqlite3.connect(root / "sql") as connection:
        connection.execute("CREATE TABLE patients (id INTEGER PRIMARY KEY, secret TEXT)")
    endpoint = "https://workspace.cloud.databricks.com" if managed else "http://localhost:8080"
    monkeypatch.setenv("SITE_CATALOG_TOKEN", "site-only-token")
    calls = []

    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def json(self):
            return {
                "tables": [
                    {
                        "name": "patients",
                        "catalog_name": "health",
                        "schema_name": "ehr",
                        "columns": [
                            {"name": "id", "type_text": "bigint", "nullable": False},
                            {"name": "secret", "type_text": "string", "nullable": True},
                        ],
                        "storage_location": "s3://site-private-data",
                    }
                ]
            }

    class Session(Response):
        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            return Response()

    monkeypatch.setattr(uc_module.requests, "Session", Session)

    def dataset(type_, uri, table, **extra):
        return {
            "type": type_,
            "uri": uri,
            **extra,
            "usage": {"catalog": {"share": True, "exclude_columns": {table: ["secret"]}}},
        }

    catalog_entry = dataset(
        "unity-catalog",
        endpoint + "/api/2.1/unity-catalog",
        "patients",
        options={"catalog": "health", "schema": "ehr"},
    )
    if managed:
        catalog_entry["secrets"] = {"token": "SITE_CATALOG_TOKEN"}
    config = {
        "format_version": 1,
        "studies": {
            "study1": {
                "datasets": {
                    "parquet": dataset("parquet", (root / "parquet").as_uri(), "parquet"),
                    "csv": dataset("csv", (root / "csv").as_uri(), "csv"),
                    "sql": dataset("sql", (root / "sql").as_uri(), "patients", options={"tables": ["patients"]}),
                    "catalog": catalog_entry,
                }
            }
        },
    }
    config_file = tmp_path / "datasets.yaml"
    config_file.write_text(yaml.safe_dump(config))
    monkeypatch.setenv(DATASET_CONFIG_ENV, str(config_file))

    ctx = FLContext()
    ctx.set_prop(FLContextKey.JOB_META, {"study": "study1", CATALOG_JOB_META: True}, private=True, sticky=False)
    data = Shareable()
    data[CATALOG_READ_KEY] = build_task_payload(None)
    result = CatalogExecutor().execute(TASK_NAME, data, ctx, Signal())
    parsed = parse_result(result[CATALOG_READ_KEY], None, ReadLimits(**DEFAULT_LIMITS))
    assert parsed.status == ResultStatus.OK and not parsed.unavailable and not parsed.skipped
    assert {c.dataset: c.provenance.source for c in parsed.catalogs} == {
        "parquet": "parquet",
        "csv": "csv",
        "sql": "sql",
        "catalog": "unity-catalog",
    }
    assert all([column.name for table in c.tables for column in table.columns] == ["id"] for c in parsed.catalogs)
    assert len(calls) == 1 and calls[0][0] == endpoint + "/api/2.1/unity-catalog/tables"
    assert (calls[0][1]["headers"].get("Authorization") == "Bearer site-only-token") is managed
    assert "private-row" not in json.dumps(result) and "site-only-token" not in json.dumps(result)
