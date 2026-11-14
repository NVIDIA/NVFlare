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
import os
import shutil
from pathlib import Path

from nvflare.apis.job_def import RunStatus
from nvflare.app_common.fdn.catalog.common.payload import (
    CATALOG_JOB_META,
    RESULTS_FILE,
    TASK_NAME,
    PayloadError,
    check_catalog_results,
)
from nvflare.app_common.fdn.catalog.server.controller import CatalogController
from nvflare.app_common.fdn.catalog.site.executor import CatalogExecutor
from nvflare.job_config.api import FedJob
from nvflare.recipe.spec import ExecEnv, Recipe

# Catalog tasks finish in well under a second; with the 2 s framework default each site would wait that long after
# returning its result before it notices that the run has ended.
DEFAULT_TASK_REQUEST_INTERVAL_S = 0.1
_WAIT_MARGIN_S = 60


class CatalogReadError(RuntimeError):
    """A catalog read job did not finish successfully, its wait expired, or its result is missing."""

    def __init__(self, message: str, job_id: str | None = None, status: str | None = None):
        super().__init__(message)
        self.job_id = job_id
        self.status = status


class CatalogRecipe(Recipe):
    """Reads the dataset catalogs of the sites in a Study.

    The Study is the one of the execution environment (for example ``ProdEnv(study=...)``). Each site reads the
    datasets its dataset configuration declares and shares what its catalog rules permit. The controller writes the
    result to ``catalog_results.json`` in the job workspace; ``get_catalog_results`` downloads and loads it.

    Args:
        name: the job name.
        sites: the sites to read. None reads every site of the Study.
        datasets: maps a site to the dataset names to read there; the site reads those it shares and skips the
            others. A site not in the mapping reads everything it shares, and so does None.
        task_timeout_s: how long to wait for a site before recording it as NO_RESULT.
        task_request_interval_s: how often, in seconds, sites ask the server for work during this job (the server's
            ``task_request_interval`` for this job only). The 0.1 s default lets sites notice the end of the run right
            after returning their results.
        site_config_root: SimEnv only. The parent folder of one folder per site, each holding that site's
            ``local/datasets.yaml``. A deployed site owns its configuration.

    Example:
        >>> recipe = CatalogRecipe(datasets={"site-1": ["omop"]})
        >>> run = recipe.execute(ProdEnv(startup_kit_location="/path/to/admin", study="oncology"))
        >>> results = recipe.get_catalog_results(run)
    """

    def __init__(
        self,
        name: str = "get_catalog",
        sites: list[str] | None = None,
        datasets: dict[str, list[str]] | None = None,
        task_timeout_s: int = 300,
        site_config_root: str | None = None,
        task_request_interval_s: float = 0.1,  # DEFAULT_TASK_REQUEST_INTERVAL_S; a literal so recipe metadata shows it
    ):
        if isinstance(task_request_interval_s, bool) or not isinstance(task_request_interval_s, (int, float)):
            raise ValueError("task_request_interval_s must be a positive number")
        if not task_request_interval_s > 0:
            raise ValueError("task_request_interval_s must be a positive number")
        if sites is not None and not sites:
            raise ValueError("sites must name at least one site; omit it to read every site of the Study")
        if datasets and sites and set(datasets) - set(sites):
            raise ValueError(f"datasets names sites that are not read: {sorted(set(datasets) - set(sites))}")
        self.task_timeout_s = task_timeout_s
        self.site_config_root = str(Path(site_config_root).resolve()) if site_config_root is not None else None
        job = FedJob(name=name, min_clients=1, meta_props={CATALOG_JOB_META: True})
        job.to_server(CatalogController(datasets=datasets, task_timeout_s=task_timeout_s))
        for site in sites if sites is not None else [None]:
            if self.site_config_root is not None:
                from nvflare.app_common.fdn.catalog.simulation import SimulationCatalogExecutor

                executor = SimulationCatalogExecutor(self.site_config_root)
            else:
                executor = CatalogExecutor()
            if site is None:
                job.to_clients(executor, tasks=[TASK_NAME])
            else:
                job.to(executor, site, tasks=[TASK_NAME])
        Recipe.__init__(self, job)
        self.add_server_config({"server": {"task_request_interval": task_request_interval_s}})

    def process_env(self, env):
        from nvflare.recipe.sim_env import SimEnv

        if self.site_config_root is not None and not isinstance(env, SimEnv):
            raise ValueError("site_config_root is for SimEnv only; deployed sites own their configuration")
        if self.site_config_root is None and isinstance(env, SimEnv):
            raise ValueError("SimEnv needs site_config_root: the folder with each simulated site's configuration")
        return super().process_env(env)

    def get_catalog_results(self, run, output_path: str | None = None, timeout: float = 0.0) -> dict:
        """Wait for the job, download its result and return the parsed catalog_results.json.

        When the read completes, a downloaded job workspace is reduced to its catalog_results.json; a failed
        job's workspace is kept whole for diagnosis. SimEnv returns its live simulation workspace, whose site
        configuration and inputs later runs reuse, so that workspace is left unchanged.

        Args:
            run: the Run returned by ``execute``.
            output_path: also copy catalog_results.json here.
            timeout: seconds to wait for the job after submission; 0 waits without limit.

        Raises:
            CatalogReadError: the job did not finish successfully, the wait expired, or the file is missing.
        """
        result_dir = run.get_result(timeout=timeout, clean_up=False)
        status = run.get_status()
        if status != RunStatus.FINISHED_COMPLETED.value:
            raise CatalogReadError(f"catalog read job {run.job_id} ended with status {status}", run.job_id, status)
        found = sorted(Path(result_dir).rglob(RESULTS_FILE)) if result_dir and os.path.isdir(result_dir) else []
        if len(found) != 1:
            raise CatalogReadError(f"catalog read job {run.job_id} has no single {RESULTS_FILE}", run.job_id, status)
        try:
            results = check_catalog_results(json.loads(found[0].read_text(encoding="utf-8")))
        except (OSError, ValueError, PayloadError) as e:
            raise CatalogReadError(f"catalog results are unreadable: {type(e).__name__}", run.job_id, status) from None
        from nvflare.recipe.sim_env import SimEnv

        if isinstance(getattr(run, "exec_env", None), SimEnv):
            kept = found[0]  # the live simulation workspace, not a downloaded copy: nothing to remove
        else:
            kept = _keep_only_the_result(Path(result_dir), found[0])
        if output_path is not None and not _same_file(kept, output_path):
            shutil.copyfile(kept, output_path)
        return results


def _same_file(path: Path, other: str) -> bool:
    try:
        return os.path.samefile(path, other)
    except OSError:
        return False  # e.g. other does not exist yet


def _keep_only_the_result(result_dir: Path, results_file: Path) -> Path:
    """Reduce a completed job's downloaded workspace to its catalog_results.json; return the kept file.

    The download holds the whole job workspace (configuration, logs, other artifacts). A completed read needs only
    the result; a failed one is left whole for diagnosis.
    """
    kept = result_dir / RESULTS_FILE
    if results_file != kept:
        os.replace(results_file, kept)
    for entry in result_dir.iterdir():
        if entry == kept:
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    return kept


def get_catalog(
    env: ExecEnv,
    *,
    sites: list[str] | None = None,
    datasets: dict[str, list[str]] | None = None,
    timeout_s: int = 300,
    output_path: str | None = None,
    site_config_root: str | None = None,
    task_request_interval_s: float = 0.1,  # DEFAULT_TASK_REQUEST_INTERVAL_S; a literal so recipe metadata shows it
) -> dict:
    """Read the catalog of the Study of ``env`` in one call and return the parsed catalog_results.json.

    Equivalent to building a ``CatalogRecipe`` with these arguments, ``recipe.execute(env)`` and
    ``recipe.get_catalog_results(run, output_path=...)``. Every call submits a new job that reads the sites again.

    Raises:
        CatalogReadError: the job did not finish successfully or the wait expired. Sites that fail do not raise;
            their status is in ``results["sites"]``.
    """
    recipe = CatalogRecipe(
        sites=sites,
        datasets=datasets,
        task_timeout_s=timeout_s,
        site_config_root=site_config_root,
        task_request_interval_s=task_request_interval_s,
    )
    run = recipe.execute(env)
    return recipe.get_catalog_results(run, output_path=output_path, timeout=timeout_s + _WAIT_MARGIN_S)
