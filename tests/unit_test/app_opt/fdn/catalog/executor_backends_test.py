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

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.fdn.catalog.common.payload import (
    CATALOG_JOB_META,
    CATALOG_READ_KEY,
    ResultStatus,
    Selection,
    build_task_payload,
    parse_result,
)
from nvflare.app_common.fdn.catalog.site.executor import CatalogExecutor, MountResolver


@pytest.mark.parametrize("managed", [False, True], ids=["unity-catalog-oss", "databricks"])
def test_one_executor_reads_file_sql_and_unity_catalog_sources(tmp_path, monkeypatch, managed):
    """Real file/SQL metadata and a controlled REST boundary share the same task and publication policy."""
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
    monkeypatch.setenv("SITE_CATALOG_URL", endpoint + "/api/2.1/unity-catalog")
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
    args = {"url_env": "SITE_CATALOG_URL"}
    if managed:
        args["token_env"] = "SITE_CATALOG_TOKEN"
    config = {
        "format_version": 1,
        "connectors": {
            "catalog": {
                "class": "nvflare.app_opt.fdn.catalog.unity_catalog_connector.UnityCatalogConnector",
                "args": args,
            }
        },
        "studies": {
            "study1": {
                "csv": {"connector": "csv"},
                "sql": {"connector": "sql", "options": {"tables": ["patients"]}},
                "catalog": {"connector": "catalog", "options": {"catalog": "health", "schema": "ehr"}},
            }
        },
    }
    policy_root = tmp_path / "policy"
    (policy_root / "study1").mkdir(parents=True)
    (policy_root / "catalog_site.json").write_text(json.dumps(config))
    (policy_root / "study1" / "catalog_policy.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "datasets": {
                    name: {"share": True, "exclude_columns": {table: ["secret"]}}
                    for name, table in (
                        ("parquet", "parquet"),
                        ("csv", "csv"),
                        ("sql", "patients"),
                        ("catalog", "patients"),
                    )
                },
            }
        )
    )
    executor = CatalogExecutor()
    executor.policy_root = policy_root
    executor.resolver = MountResolver(str(root.parent))
    ctx = FLContext()
    ctx.set_prop(FLContextKey.JOB_META, {"study": "study1", CATALOG_JOB_META: True}, private=True, sticky=False)
    data = Shareable()
    data[CATALOG_READ_KEY] = build_task_payload(None, [])
    result = executor.execute("catalog.read", data, ctx, Signal())
    parsed = parse_result(result[CATALOG_READ_KEY], Selection(None))
    assert parsed.status == ResultStatus.OK and not parsed.unavailable and not parsed.withheld
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
