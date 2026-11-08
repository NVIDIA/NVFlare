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
from datetime import datetime, timezone
from pathlib import Path

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.job_def import RunStatus
from nvflare.app_common.fdn.catalog.common.payload import CATALOG_JOB_META, TASK_NAME
from nvflare.app_common.fdn.catalog.server.controller import CatalogController
from nvflare.app_common.fdn.catalog.site.executor import CatalogExecutor
from nvflare.app_common.fdn.catalog.site.reader import CatalogReader
from nvflare.job_config.api import FedJob
from nvflare.recipe.spec import Recipe


class CatalogRecipe(Recipe):
    """Reads the dataset catalogs of the sites in a Study into the server's federated data catalog.

    The Study is the one of the execution environment (for example `ProdEnv(study=...)`). Each site decides what it
    shares from its own policy and connector files. An optional reader supplies the metadata-reading logic. The server
    applies the result when the job completes; see it with `nvflare catalog show`.

    Args:
        name: the job name.
        sites: the sites to read. None reads every site of the Study.
        datasets: maps a site to the dataset names to read there; the site reads those it has and skips the others.
            A site not in the mapping reads everything its policy shares, and so does None. Sites have their own
            datasets, so the names differ by site.
        catalog_reader: optional CatalogReader component, instantiated at each site, like a FedStats generator.
            Site-owned policies and budgets still apply to every returned catalog.
        site_config_root: SimEnv-only parent of per-site directories containing local/fdn and runtime bindings.
        output_dir: SimEnv persistent catalog store and discovery.json output directory. SimEnv has no Study, so
            results are kept under the `default` Study. Runs that share this directory share one store; give
            separate runs separate directories.
        task_timeout_s: how long to wait for a site before recording it as not responding.

    Example:
        >>> recipe = CatalogRecipe()
        >>> recipe.execute(ProdEnv(startup_kit_location="/path/to/admin", study="oncology"))
    """

    def __init__(
        self,
        name: str = "catalog_read",
        sites: list[str] | None = None,
        datasets: dict[str, list[str]] | None = None,
        task_timeout_s: int = 300,
        catalog_reader: CatalogReader | None = None,
        site_config_root: str | None = None,
        output_dir: str = "/tmp/nvflare/catalog",
    ):
        if sites is not None and not sites:
            raise ValueError("sites must name at least one site; omit it to read every site of the Study")
        if datasets and sites and set(datasets) - set(sites):
            raise ValueError(f"datasets names sites that are not read: {sorted(set(datasets) - set(sites))}")
        if catalog_reader is not None and not isinstance(catalog_reader, CatalogReader):
            raise TypeError("catalog_reader must implement CatalogReader")
        self.site_config_root = str(Path(site_config_root).resolve()) if site_config_root is not None else None
        self.output_dir = str(Path(output_dir).resolve())
        self.catalog_results = None
        job = FedJob(name=name, min_clients=1, meta_props={CATALOG_JOB_META: True})
        if self.site_config_root is not None:
            from nvflare.app_common.fdn.catalog.simulation import SimulationCatalogController

            controller = SimulationCatalogController(self.output_dir, datasets, task_timeout_s)
        else:
            controller = CatalogController(datasets=datasets, task_timeout_s=task_timeout_s)
        job.to_server(controller)
        for site in sites if sites is not None else [None]:
            if catalog_reader is not None:
                if site is None:
                    job.to_clients(catalog_reader, id="catalog_reader")
                else:
                    job.to(catalog_reader, site, id="catalog_reader")
            reader_id = "catalog_reader" if catalog_reader is not None else None
            if self.site_config_root is not None:
                from nvflare.app_common.fdn.catalog.simulation import SimulationCatalogExecutor

                executor = SimulationCatalogExecutor(self.site_config_root, reader_id=reader_id)
            else:
                executor = CatalogExecutor(reader_id=reader_id)
            if site is None:
                job.to_clients(executor, tasks=[TASK_NAME])
            else:
                job.to(executor, site, tasks=[TASK_NAME])
        Recipe.__init__(self, job)

    def process_env(self, env):
        if self.site_config_root is not None:
            from nvflare.recipe.sim_env import SimEnv

            if not isinstance(env, SimEnv):
                raise ValueError("site_config_root is for SimEnv only; deployed sites own their configuration")
        return super().process_env(env)

    def run(self, env, server_exec_params=None, client_exec_params=None):
        self.catalog_results = None
        run = super().run(env, server_exec_params, client_exec_params)
        if self.site_config_root is not None:
            if run.get_status() != RunStatus.FINISHED_COMPLETED.value:
                raise RuntimeError(f"Catalog simulation failed: {run.get_status()}")
            self._collect_simulation(env)
        return run

    def _collect_simulation(self, env):
        from nvflare.app_common.fdn.catalog.common.payload import MANIFEST_FILE, Selection, parse_manifest
        from nvflare.app_common.fdn.catalog.common.schema import DEFAULT_LIMITS, ReadLimits
        from nvflare.app_common.fdn.catalog.server.reconcile import Reconciler
        from nvflare.app_common.fdn.catalog.server.search import SearchQuery, entry_to_dict
        from nvflare.app_common.fdn.catalog.server.store import CatalogStore

        manifests = list((Path(env.workspace_root) / self._job.name / "server").rglob(MANIFEST_FILE))
        if len(manifests) != 1:
            raise RuntimeError(f"Expected one simulation catalog manifest, found {len(manifests)}")
        entries = parse_manifest(json.loads(manifests[0].read_text())["manifest"])
        if not entries:
            raise RuntimeError("Catalog simulation returned no site results")
        store = CatalogStore(self.output_dir)
        reconciler = Reconciler(store, ReadLimits(**DEFAULT_LIMITS))
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for site, entry in entries.items():
            if entry.rc != ReturnCode.OK or entry.result is None or entry.result.get("status") != "OK":
                raise RuntimeError(f"Catalog simulation read failed for {site}")
            reconciler.apply("default", site, Selection(entry.datasets), entry.result, now)
        # Reopen persisted state rather than using only the in-memory search index.
        store = CatalogStore(self.output_dir)
        results, cursor = [], None
        while True:
            page = store.search("default", SearchQuery(page_size=100, cursor=cursor))
            results.extend(entry_to_dict(e) for e in page.entries)
            cursor = page.next_cursor
            if cursor is None:
                break
        self.catalog_results = {"entries": results, "total": len(results)}
        path = Path(self.output_dir) / "discovery.json"
        path.write_text(json.dumps(self.catalog_results, indent=2) + "\n")
