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
from types import SimpleNamespace

import pytest
import yaml

from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.fdn.catalog.common.limits import DEFAULT_LIMITS, ReadLimits
from nvflare.app_common.fdn.catalog.common.payload import (
    CATALOG_JOB_META,
    CATALOG_READ_KEY,
    TASK_NAME,
    ResultStatus,
    build_task_payload,
    parse_result,
)
from nvflare.app_common.fdn.catalog.site.executor import CatalogExecutor


@pytest.mark.parametrize("managed", [False, True], ids=["unity-catalog-oss", "databricks"])
def test_one_executor_reads_file_sql_and_unity_catalog_sources(tmp_path, monkeypatch, managed):
    """Real file/SQL metadata and a controlled REST boundary, configured in one datasets.yaml and read through the
    dataset configuration API, use the same task and source-owned metadata boundary."""
    import sqlite3

    arrow = pytest.importorskip("pyarrow")
    pytest.importorskip("datacontract")
    parquet = pytest.importorskip("pyarrow.parquet")
    from nvflare.app_common.fdn.catalog.site import executor
    from nvflare.app_opt.fdn.catalog.odcs_contract_builder import build_odcs_contract

    monkeypatch.setattr(executor, "run_catalog_discovery", build_odcs_contract)
    sdk = pytest.importorskip("databricks.sdk")
    from databricks.sdk.service.catalog import TableInfo
    from datacontract.data_contract import DataContract
    from datacontract.imports import unity_importer

    monkeypatch.setenv("DATACONTRACT_DATABRICKS_PROFILE", "other-study")
    formats = []
    upstream_import = DataContract.import_from_source

    def traced_import(**kwargs):
        formats.append(kwargs["format"])
        return upstream_import(**kwargs)

    monkeypatch.setattr(DataContract, "import_from_source", staticmethod(traced_import))

    root = tmp_path / "data" / "study1"
    root.mkdir(parents=True)
    parquet.write_table(arrow.table({"id": [1], "secret": ["private-row"]}), root / "parquet")
    (root / "csv").write_text("id,secret\n1,private-row\n")
    with sqlite3.connect(root / "sql") as connection:
        connection.execute("CREATE TABLE patients (id INTEGER PRIMARY KEY, secret TEXT)")
    endpoint = "https://workspace.cloud.databricks.com" if managed else "http://localhost:8080"
    monkeypatch.setenv("SITE_CATALOG_TOKEN", "site-only-token")
    calls = []

    class Tables:
        def list(self, catalog_name, schema_name):
            assert (catalog_name, schema_name) == ("health", "ehr")
            return [SimpleNamespace(name="patients")]

        def get(self, name):
            assert name == "health.ehr.patients"
            return TableInfo.from_dict(
                {
                    "name": "patients",
                    "catalog_name": "health",
                    "schema_name": "ehr",
                    "comment": "private-comment",
                    "storage_location": "s3://site-private-data",
                    "columns": [
                        {"name": "id", "type_text": "bigint", "nullable": False},
                        {"name": "secret", "type_text": "string", "nullable": True},
                    ],
                }
            )

    def workspace(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(tables=Tables())

    monkeypatch.setattr(sdk, "WorkspaceClient", workspace)
    monkeypatch.setattr(unity_importer, "WorkspaceClient", workspace)

    def dataset(type_, uri, **extra):
        return {
            "type": type_,
            "uri": uri,
            **extra,
        }

    catalog_entry = dataset(
        "unity-catalog",
        endpoint + "/api/2.1/unity-catalog",
        options={"catalog": "health", "schema": "ehr"},
    )
    if managed:
        catalog_entry["secrets"] = {"token": "SITE_CATALOG_TOKEN"}
    config = {
        "format_version": 1,
        "studies": {
            "study1": {
                "datasets": {
                    "parquet": dataset("parquet", (root / "parquet").as_uri()),
                    "csv": dataset("csv", (root / "csv").as_uri()),
                    "sql": dataset("sql", (root / "sql").as_uri(), options={"tables": ["patients"]}),
                    "catalog": catalog_entry,
                }
            }
        },
    }
    workspace = tmp_path / "workspace"
    (workspace / "local").mkdir(parents=True)
    (workspace / "local" / "datasets.yaml").write_text(yaml.safe_dump(config))  # the site file the job reads

    ctx = FLContext()
    engine = SimpleNamespace(get_workspace=lambda: SimpleNamespace(get_root_dir=lambda: str(workspace)))
    ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=False)
    ctx.set_prop(FLContextKey.JOB_META, {"study": "study1", CATALOG_JOB_META: True}, private=True, sticky=False)
    data = Shareable()
    data[CATALOG_READ_KEY] = build_task_payload(None)
    result = CatalogExecutor().execute(TASK_NAME, data, ctx, Signal())
    parsed = parse_result(result[CATALOG_READ_KEY], None, ReadLimits(**DEFAULT_LIMITS))
    assert parsed.status == ResultStatus.OK and not parsed.unavailable and not parsed.skipped
    assert {c["dataset"] for c in parsed.catalogs} == {"parquet", "csv", "sql", "catalog"}
    for entry in parsed.catalogs:
        assert [p["name"] for t in entry["contract"]["schema"] for p in t["properties"]] == ["id", "secret"]
        if entry["dataset"] == "csv":
            assert entry["contract"]["schema"][0]["properties"] == [{"name": "id"}, {"name": "secret"}]
    assert sorted(formats) == ["parquet", "sql", "unity"]
    assert len(calls) == 2  # discover names with the SDK, then run the upstream Unity importer
    assert all(call["host"] == endpoint for call in calls)
    assert all((call["token"] == "site-only-token") is managed for call in calls)
    for private in ("private-row", "private-comment", "site-private-data", "site-only-token"):
        assert private not in json.dumps(result)
