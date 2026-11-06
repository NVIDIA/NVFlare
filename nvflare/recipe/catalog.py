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

from nvflare.app_common.fdn.catalog.common.payload import CATALOG_JOB_META, TASK_NAME
from nvflare.app_common.fdn.catalog.server.controller import CatalogController
from nvflare.app_common.fdn.catalog.site.executor import CatalogExecutor
from nvflare.job_config.api import FedJob
from nvflare.recipe.spec import Recipe


class CatalogRecipe(Recipe):
    """Reads the dataset catalogs of the sites in a Study into the server's federated data catalog.

    The Study is the one of the execution environment (for example `ProdEnv(study=...)`). Each site decides what it
    shares from its own policy and connector files, so the recipe configures nothing about a site. The server
    applies the result when the job completes; see it with `nvflare catalog show`.

    Args:
        name: the job name.
        sites: the sites to read. None reads every site of the Study.
        datasets: maps a site to the dataset names to read there; the site reads those it has and skips the others.
            A site not in the mapping reads everything its policy shares, and so does None. Sites have their own
            datasets, so the names differ by site.
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
    ):
        if sites is not None and not sites:
            raise ValueError("sites must name at least one site; omit it to read every site of the Study")
        if datasets and sites and set(datasets) - set(sites):
            raise ValueError(f"datasets names sites that are not read: {sorted(set(datasets) - set(sites))}")
        job = FedJob(name=name, min_clients=1, meta_props={CATALOG_JOB_META: True})
        job.to_server(CatalogController(datasets=datasets, task_timeout_s=task_timeout_s))
        if sites is not None:
            for site in sites:
                job.to(CatalogExecutor(), site, tasks=[TASK_NAME])
        else:
            job.to_clients(CatalogExecutor(), tasks=[TASK_NAME])
        Recipe.__init__(self, job)
