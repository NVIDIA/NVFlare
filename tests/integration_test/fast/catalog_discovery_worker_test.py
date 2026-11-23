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

"""Catalog discovery crosses a real process boundary and returns only the projected metadata."""

import pytest

from nvflare.apis.dataset_config_spec import DatasetConfig
from nvflare.app_common.fdn.catalog.common.limits import ReadLimits
from nvflare.app_opt.fdn.catalog._metadata import ReadBudget
from nvflare.app_opt.fdn.catalog.catalog_discovery_worker import run_catalog_discovery


def test_csv_worker_preserves_wide_headers_and_charges_each_batch(tmp_path):
    pytest.importorskip("datacontract.imports.csv_importer")
    path = tmp_path / "visits.csv"
    names = [f"visit_{i}" for i in range(129)]
    path.write_text(",".join(names) + "\n" + ",".join(["private"] * len(names)))
    dataset = DatasetConfig("study", "visits", "csv", path.as_uri(), str(path))
    budget = ReadBudget(ReadLimits(120, 4, 200))
    contract = run_catalog_discovery(dataset, None, budget, site="one")
    assert contract["schema"] == [{"name": "visits", "properties": [{"name": name} for name in names]}]
    assert budget.calls_left == 1


def test_sqlite_worker_discovers_current_schema_on_each_read(tmp_path):
    import sqlite3

    pytest.importorskip("datacontract")
    path = tmp_path / "registry.sqlite"
    dataset = DatasetConfig("study", "registry", "sql", path.as_uri(), str(path))
    for ddl, names in [
        ("CREATE TABLE visits (id INTEGER)", ["id"]),
        ("ALTER TABLE visits ADD COLUMN visit_date TEXT", ["id", "visit_date"]),
    ]:
        with sqlite3.connect(path) as db:
            db.execute(ddl)
        contract = run_catalog_discovery(dataset, None, ReadBudget(ReadLimits(60, 10, 100)), site="one")
        assert contract["name"] == "registry"
        (table,) = contract["schema"]
        assert table["name"] == "visits"
        assert [p["name"] for p in table["properties"]] == names


def test_executor_reads_configured_upstream_importer_from_site_yaml(tmp_path, monkeypatch):
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    from types import SimpleNamespace

    import yaml

    pytest.importorskip("datacontract.imports.odata_importer")
    from nvflare.apis.fl_constant import FLContextKey, ReservedKey
    from nvflare.apis.fl_context import FLContext
    from nvflare.apis.shareable import Shareable
    from nvflare.apis.signal import Signal
    from nvflare.app_common.fdn.catalog.common.payload import (
        CATALOG_JOB_META,
        CATALOG_READ_KEY,
        TASK_NAME,
        build_task_payload,
    )
    from nvflare.app_common.fdn.catalog.site.executor import CatalogExecutor

    requests = []

    class Service(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers.get("X-API-Key")))
            if self.path != "/service/$metadata":
                self.send_error(403)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                b"""<edmx:Edmx Version="4.0" xmlns:edmx="http://docs.oasis-open.org/odata/ns/edmx">
                <edmx:DataServices><Schema Namespace="Site" xmlns="http://docs.oasis-open.org/odata/ns/edm">
                <EntityType Name="Visit"><Key><PropertyRef Name="id"/></Key>
                <Property Name="id" Type="Edm.Int64" Nullable="false"/></EntityType>
                <EntityContainer Name="Catalog"><EntitySet Name="Visits" EntityType="Site.Visit"/></EntityContainer>
                </Schema></edmx:DataServices></edmx:Edmx>"""
            )

    monkeypatch.setenv("SITE_CATALOG_KEY", "test-site-key")
    with ThreadingHTTPServer(("127.0.0.1", 0), Service) as server:
        dataset = {
            "type": "odata",
            "uri": f"http://127.0.0.1:{server.server_port}/service/",
            "options": {"metadata_source": "uri", "odata_entity_set": ["Visits"], "config": {}},
            "secrets": {"DATACONTRACT_API_HEADER_X_API_KEY": "SITE_CATALOG_KEY"},
        }
        (tmp_path / "local").mkdir()
        (tmp_path / "local/datasets.yaml").write_text(
            yaml.safe_dump({"format_version": 1, "studies": {"study": {"datasets": {"visits": dataset}}}})
        )
        ctx = FLContext()
        engine = SimpleNamespace(get_workspace=lambda: SimpleNamespace(get_root_dir=lambda: str(tmp_path)))
        ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=False)
        ctx.set_prop(FLContextKey.JOB_META, {"study": "study", CATALOG_JOB_META: True}, private=True, sticky=False)
        task = Shareable()
        task[CATALOG_READ_KEY] = build_task_payload(None)
        listener = Thread(target=server.serve_forever)
        listener.start()
        try:
            result = CatalogExecutor().execute(TASK_NAME, task, ctx, Signal())[CATALOG_READ_KEY]
        finally:
            server.shutdown()
            listener.join()
    assert requests == [("/service/$metadata", "test-site-key")]
    assert not result["unavailable"] and not result["skipped"]
    (entry,) = result["catalogs"]
    assert entry["dataset"] == "visits"
    (table,) = entry["contract"]["schema"]
    assert [prop["name"] for prop in table["properties"]] == ["id"]
    assert "test-site-key" not in json.dumps(result)
