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
from unittest.mock import MagicMock

import pytest
import yaml

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.job_def import RunStatus
from nvflare.app_common.fdn.catalog.common.payload import (
    CATALOG_JOB_META,
    TASK_NAME,
    SiteEntry,
    SiteStatus,
    build_catalog_results,
)
from nvflare.recipe import catalog as catalog_module
from nvflare.recipe.catalog import DEFAULT_TASK_REQUEST_INTERVAL_S, CatalogReadError, CatalogRecipe, get_catalog
from nvflare.recipe.sim_env import SimEnv
from tests.unit_test.app_common.fdn.catalog.helpers import make_catalog


def export(tmp_path, **kw):
    CatalogRecipe(**kw).export(str(tmp_path))
    job = tmp_path / kw.get("name", "get_catalog")
    return (
        json.loads((job / "meta.json").read_text()),
        json.loads((job / "app" / "config" / "config_fed_server.json").read_text()),
        json.loads((job / "app" / "config" / "config_fed_client.json").read_text()),
    )


class TestCatalogRecipe:
    def test_defaults_read_every_site(self, tmp_path):
        meta, server, client = export(tmp_path)
        assert meta["name"] == "get_catalog" and meta["deploy_map"] == {"app": ["@ALL"]}
        assert meta[CATALOG_JOB_META] is True
        assert server["workflows"][0]["path"] == "nvflare.app_common.fdn.catalog.server.controller.CatalogController"
        assert server["workflows"][0]["args"] == {}
        assert client["executors"][0]["tasks"] == [TASK_NAME]
        assert (
            client["executors"][0]["executor"]["path"] == "nvflare.app_common.fdn.catalog.site.executor.CatalogExecutor"
        )
        assert client["executors"][0]["executor"]["args"] == {}

    def test_sites_notice_the_end_of_the_run_quickly(self, tmp_path):
        _, server, _ = export(tmp_path)
        assert server["server"] == {"task_request_interval": DEFAULT_TASK_REQUEST_INTERVAL_S}

    def test_the_signature_defaults_match_the_constant(self):
        import inspect

        for fn in (CatalogRecipe.__init__, get_catalog):
            default = inspect.signature(fn).parameters["task_request_interval_s"].default
            assert default == DEFAULT_TASK_REQUEST_INTERVAL_S

    def test_the_task_request_interval_is_configurable(self, tmp_path):
        _, server, _ = export(tmp_path, task_request_interval_s=1.5)
        assert server["server"] == {"task_request_interval": 1.5}

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
            {"task_request_interval_s": 0},
            {"task_request_interval_s": -1},
            {"task_request_interval_s": True},
            {"task_request_interval_s": "0.1"},
        ],
    )
    def test_bad_arguments_are_refused(self, kw):
        with pytest.raises(ValueError):
            CatalogRecipe(**kw)


class TestSimulation:
    def test_simulation_uses_the_simulation_executor(self, tmp_path):
        _, server, client = export(tmp_path, site_config_root=str(tmp_path))
        assert server["workflows"][0]["path"].endswith("server.controller.CatalogController")
        assert client["executors"][0]["executor"]["path"].endswith("simulation.SimulationCatalogExecutor")

    def test_site_config_root_only_with_simenv(self, tmp_path):
        from nvflare.recipe import ProdEnv, SimEnv

        with pytest.raises(ValueError, match="SimEnv only"):
            CatalogRecipe(site_config_root=str(tmp_path)).process_env(ProdEnv(startup_kit_location=str(tmp_path)))
        with pytest.raises(ValueError, match="needs site_config_root"):
            CatalogRecipe().process_env(SimEnv(clients=["site-1"]))

    def test_each_site_reads_its_own_configuration(self, tmp_path):
        from nvflare.app_common.fdn.catalog.simulation import SimulationCatalogExecutor

        for site, name in (("site-1", "a"), ("site-2", "b")):
            local = tmp_path / site / "local"
            local.mkdir(parents=True)
            doc = {
                "format_version": 1,
                "studies": {"default": {"datasets": {name: {"type": "csv", "uri": "file:///x"}}}},
            }
            (local / "datasets.yaml").write_text(yaml.safe_dump(doc))
        executor = SimulationCatalogExecutor(str(tmp_path))
        for site, name in (("site-1", "a"), ("site-2", "b")):
            ctx = SimpleNamespace(
                get_prop=lambda key, default=None: True if key == FLContextKey.SIMULATE_MODE else None,
                get_identity_name=lambda site=site: site,
            )
            assert executor.load_config(ctx).list_datasets() == [name]

    def test_simulation_executor_refuses_a_deployed_context(self, tmp_path):
        from nvflare.app_common.fdn.catalog.simulation import SimulationCatalogExecutor

        with pytest.raises(RuntimeError, match="only be used in SimEnv"):
            SimulationCatalogExecutor(str(tmp_path)).load_config(SimpleNamespace(get_prop=lambda *a: False))


class FakeRun:
    def __init__(self, result_dir, status=RunStatus.FINISHED_COMPLETED.value, exec_env=None):
        self.job_id = "job-1"
        self.result_dir, self.status, self.calls = result_dir, status, []
        self.exec_env = exec_env

    def get_result(self, timeout=0.0, clean_up=True):
        self.calls.append((timeout, clean_up))
        return self.result_dir

    def get_status(self):
        return self.status


def write_results(directory):
    directory.mkdir(parents=True, exist_ok=True)
    doc = build_catalog_results(
        "default", "job-1", "2026-11-09T00:00:00Z", {"site-1": SiteEntry(SiteStatus.OK, None, (make_catalog(),))}
    )
    (directory / "catalog_results.json").write_text(json.dumps(doc))
    return doc


class TestGetCatalogResults:
    def test_loads_and_saves_the_result(self, tmp_path):
        doc = write_results(tmp_path / "job-1" / "workspace" / "app_server")
        run = FakeRun(str(tmp_path / "job-1"))
        out = tmp_path / "saved.json"
        assert CatalogRecipe().get_catalog_results(run, output_path=str(out), timeout=5) == doc
        assert json.loads(out.read_text()) == doc
        assert run.calls == [(5, False)]

    def test_a_completed_read_keeps_only_the_result(self, tmp_path):
        job = tmp_path / "job-1"
        doc = write_results(job / "workspace" / "app_server")
        (job / "workspace" / "log.txt").write_text("log")
        (job / "meta.json").write_text("{}")
        CatalogRecipe().get_catalog_results(FakeRun(str(job)))
        assert [p.name for p in job.iterdir()] == ["catalog_results.json"]
        assert json.loads((job / "catalog_results.json").read_text()) == doc

    def test_a_simulation_keeps_its_live_workspace(self, tmp_path):
        job = tmp_path / "simulate_job"
        doc = write_results(job / "server" / "simulate_job" / "app_server")
        site_local = job / "site-1" / "local"
        site_local.mkdir(parents=True)
        (site_local / "datasets.yaml").write_text("format_version: 1\n")  # reused by later simulation runs
        (job / "site-1" / "startup").mkdir()
        out = tmp_path / "saved.json"

        results = CatalogRecipe().get_catalog_results(
            FakeRun(str(job), exec_env=MagicMock(spec=SimEnv)), output_path=str(out)
        )

        assert results == doc and json.loads(out.read_text()) == doc
        assert (site_local / "datasets.yaml").is_file() and (job / "site-1" / "startup").is_dir()

    def test_a_simulation_export_does_not_make_a_later_read_ambiguous(self, tmp_path):
        job = tmp_path / "simulate_job"
        doc = write_results(job / "server" / "simulate_job")  # the server's run folder in a SimEnv workspace
        run = FakeRun(str(job), exec_env=MagicMock(spec=SimEnv))
        recipe = CatalogRecipe()

        assert recipe.get_catalog_results(run, output_path=str(job / "catalog_results.json")) == doc
        assert recipe.get_catalog_results(run) == doc  # the exported copy is a second match

    def test_saving_to_the_kept_result_itself_does_not_raise(self, tmp_path):
        job = tmp_path / "job-1"
        doc = write_results(job / "workspace" / "app_server")

        results = CatalogRecipe().get_catalog_results(FakeRun(str(job)), output_path=str(job / "catalog_results.json"))

        assert results == doc and json.loads((job / "catalog_results.json").read_text()) == doc

    def test_a_failed_read_keeps_the_whole_workspace(self, tmp_path):
        job = tmp_path / "job-1"
        write_results(job / "workspace")
        (job / "workspace" / "log.txt").write_text("log")
        with pytest.raises(CatalogReadError):
            CatalogRecipe().get_catalog_results(FakeRun(str(job), RunStatus.FINISHED_ABORTED.value))
        assert (job / "workspace" / "log.txt").is_file()

    def test_a_job_that_did_not_complete_raises(self, tmp_path):
        write_results(tmp_path)
        with pytest.raises(CatalogReadError) as e:
            CatalogRecipe().get_catalog_results(FakeRun(str(tmp_path), RunStatus.FINISHED_ABORTED.value))
        assert e.value.job_id == "job-1" and e.value.status == RunStatus.FINISHED_ABORTED.value

    def test_a_missing_result_raises(self, tmp_path):
        with pytest.raises(CatalogReadError):
            CatalogRecipe().get_catalog_results(FakeRun(str(tmp_path)))
        with pytest.raises(CatalogReadError):
            CatalogRecipe().get_catalog_results(FakeRun(None))

    def test_an_unreadable_result_raises(self, tmp_path):
        (tmp_path / "catalog_results.json").write_text("{not json")
        with pytest.raises(CatalogReadError):
            CatalogRecipe().get_catalog_results(FakeRun(str(tmp_path)))


def test_the_requested_sites_reach_the_controller():
    recipe = CatalogRecipe(sites=["site-1", "site-2"])
    controller = recipe._job._deploy_map["server"].app_config.workflows[0].controller
    assert controller.sites == ["site-1", "site-2"]


def test_get_catalog_is_the_recipe_in_one_call(tmp_path, monkeypatch):
    seen = {}

    def execute(self, env):
        seen["recipe"], seen["env"], seen["run"] = self, env, FakeRun(str(tmp_path))
        return seen["run"]

    monkeypatch.setattr(CatalogRecipe, "execute", execute)
    doc = write_results(tmp_path)
    env = object()
    out = str(tmp_path / "o.json")
    assert (
        get_catalog(env, datasets={"site-1": ["omop"]}, timeout_s=30, output_path=out, task_request_interval_s=2) == doc
    )
    controller = seen["recipe"]._job._deploy_map["server"].app_config.workflows[0].controller
    assert seen["env"] is env and controller.datasets == {"site-1": ["omop"]} and controller.task_timeout_s == 30
    assert (tmp_path / "o.json").exists()
    assert seen["run"].calls == [(30 + catalog_module._WAIT_MARGIN_S, False)]
    seen["recipe"].export(str(tmp_path / "exported"))
    server = json.loads(
        (tmp_path / "exported" / "get_catalog" / "app" / "config" / "config_fed_server.json").read_text()
    )
    assert server["server"] == {"task_request_interval": 2}
