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

from nvflare.app_common.fdn.catalog.common.payload import CATALOG_JOB_META
from nvflare.recipe.catalog import CatalogRecipe


def export(tmp_path, **kw):
    CatalogRecipe(**kw).export(str(tmp_path))
    job = tmp_path / kw.get("name", "catalog_read")
    return (
        json.loads((job / "meta.json").read_text()),
        json.loads((job / "app" / "config" / "config_fed_server.json").read_text()),
        json.loads((job / "app" / "config" / "config_fed_client.json").read_text()),
    )


class TestCatalogRecipe:
    def test_defaults_read_every_site_and_every_shared_dataset(self, tmp_path):
        meta, server, client = export(tmp_path)
        assert meta["deploy_map"] == {"app": ["@ALL"]} and meta["min_clients"] == 1
        assert server["workflows"][0]["args"] == {}  # nothing to tell the controller: it decides from the sites
        assert client["executors"][0]["tasks"] == ["catalog.read"]
        assert server["workflows"][0]["path"] == "nvflare.app_common.fdn.catalog.server.controller.CatalogController"
        assert (
            client["executors"][0]["executor"]["path"] == "nvflare.app_common.fdn.catalog.site.executor.CatalogExecutor"
        )

    def test_the_job_carries_the_marker_the_server_looks_for(self, tmp_path):
        meta, _, _ = export(tmp_path)
        assert meta[CATALOG_JOB_META] is True

    def test_the_site_executor_is_never_configured_by_the_job(self, tmp_path):
        _, _, client = export(tmp_path, datasets={"site-a": ["omop"]}, task_timeout_s=30)
        assert client["executors"][0]["executor"]["args"] == {}

    def test_arguments_reach_the_controller(self, tmp_path):
        _, server, _ = export(tmp_path, datasets={"site-a": ["omop"]}, task_timeout_s=30)
        assert server["workflows"][0]["args"] == {"datasets": {"site-a": ["omop"]}, "task_timeout_s": 30}

    def test_named_sites_are_the_only_ones_deployed_to(self, tmp_path):
        CatalogRecipe(name="read_two", sites=["site-a", "site-b"]).export(str(tmp_path))
        meta = json.loads((tmp_path / "read_two" / "meta.json").read_text())
        assert meta["deploy_map"] == {"app_server": ["server"], "app_site-a": ["site-a"], "app_site-b": ["site-b"]}

    @pytest.mark.parametrize(
        "kw",
        [
            {"datasets": []},
            {"datasets": ["omop"]},
            {"sites": []},
            {"datasets": {"a": []}},
            {"task_timeout_s": 0},
            {"sites": ["a"], "datasets": {"b": ["x"]}},
        ],
    )
    def test_bad_arguments_are_refused_when_the_recipe_is_built(self, kw):
        with pytest.raises(ValueError):
            CatalogRecipe(**kw)


def test_reader_component_is_deployed_and_executor_references_it(tmp_path):
    from tests.unit_test.app_common.fdn.catalog.helpers import StubCatalogReader

    _, _, client = export(tmp_path, catalog_reader=StubCatalogReader())
    component = next(c for c in client["components"] if c["id"] == "catalog_reader")
    assert component["path"].endswith("StubCatalogReader")
    assert client["executors"][0]["executor"]["args"] == {"reader_id": "catalog_reader"}


def test_non_reader_is_rejected():
    with pytest.raises(TypeError, match="CatalogReader"):
        CatalogRecipe(catalog_reader=object())
