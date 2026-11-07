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

"""NVFlare data-readiness job construction and direct entry point."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fedready.client import FedReadyTaskQueryExecutor
from fedready.data.extractor import ExtractionConfig
from fedready.data.parser import list_client_ids
from fedready.flare.channel import SIMULATION_SCHEMA_VERSION, TASK_QUERY_TASK_NAME
from fedready.server import FedReadyTaskQueryController
from fedready.utils.io import safe_path_slug
from fedready.utils.logging import timestamp_utc

from nvflare.job_config.api import FedJob
from nvflare.recipe import SimEnv
from nvflare.recipe.spec import Recipe

DEFAULT_WORKSPACE_ROOT = "workspace"


class FedReadyRecipe(Recipe):
    """Thin NVFlare Recipe wrapper for FedReady custom FedJob objects."""

    @property
    def job(self) -> FedJob:
        """Return the custom job wrapped by this research recipe."""

        return self._job


def export_fed_job_recipe(job: FedJob, job_root: str | Path) -> Path:
    """Export a custom FedJob through the NVFlare Recipe API."""

    root = Path(job_root)
    root.mkdir(parents=True, exist_ok=True)
    FedReadyRecipe(job).export(str(root))
    return root / job.name


def run_fed_job_recipe(
    job: FedJob,
    *,
    workspace_root: str | Path,
    clients: list[str],
    threads: int | None = None,
    log_config: str | None = "concise",
) -> dict[str, str]:
    """Run a custom FedJob through NVFlare Recipe + SimEnv semantics."""

    root = Path(workspace_root)
    root.mkdir(parents=True, exist_ok=True)
    env = SimEnv(
        clients=clients,
        num_threads=threads,
        log_config=log_config,
        workspace_root=str(root),
    )
    # Recipe.execute is the current public entry point for running a recipe in
    # an environment; it returns the Run used for status and result retrieval.
    run = FedReadyRecipe(job).execute(env)
    workspace = Path(run.get_result(clean_up=False) or (root / job.name))
    return {
        "schema_version": "fedready.nvflare_recipe_run.v1",
        "api": "nvflare.recipe",
        "environment": "SimEnv",
        "job_id": run.get_job_id(),
        "workspace_root": str(root),
        "workspace": str(workspace),
    }


def build_agent_task_query_job(
    *,
    site_meta_path: str | Path,
    task: str = "<TASK_DESCRIPTION>",
    project_root: str | Path | None = None,
    output_dir: str | Path = "runs",
    session_id: str | None = None,
    job_name: str = "fedready_task_query",
    min_count: int = 5,
    max_scan_files: int = 200_000,
    max_image_samples: int = 8,
    histogram_bins: int = 8,
    max_clients: int | None = None,
    total_rounds: int = 2,
    result_wait_timeout: int = 14_400,
    extraction_output_root: str = ExtractionConfig.output_root,
    extraction_output_name: str | None = None,
    extraction_max_samples: int | None = None,
    extraction_overwrite: bool = False,
    extraction_validation_fraction: float = ExtractionConfig.validation_fraction,
    agent_backend: str = "codex",
    agent_timeout_seconds: float = 3600.0,
    agent_poll_interval_seconds: float = 2.0,
    client_inquiry_prompt: str | None = None,
    profile_resume_run_dir: str | Path | None = None,
) -> tuple[FedJob, list[str]]:
    """Build an NVFlare Job API job for the FedReady task-query flow."""

    meta_path = Path(site_meta_path).resolve()
    root = Path(project_root).resolve() if project_root is not None else meta_path.parent.parent.resolve()
    run_output = Path(output_dir).resolve()
    client_ids = _server_visible_clients(str(meta_path), max_clients)

    job = FedJob(name=job_name, min_clients=len(client_ids), mandatory_clients=client_ids)
    controller_kwargs: dict[str, Any] = dict(
        site_meta_path=str(meta_path),
        task=task,
        project_root=str(root),
        output_dir=str(run_output),
        session_id=session_id,
        min_count=min_count,
        max_scan_files=max_scan_files,
        max_image_samples=max_image_samples,
        histogram_bins=histogram_bins,
        max_clients=max_clients,
        total_rounds=total_rounds,
        result_wait_timeout=result_wait_timeout,
        extraction_output_root=extraction_output_root,
        extraction_output_name=extraction_output_name,
        extraction_max_samples=extraction_max_samples,
        extraction_overwrite=extraction_overwrite,
        extraction_validation_fraction=extraction_validation_fraction,
        agent_backend=agent_backend,
        agent_timeout_seconds=agent_timeout_seconds,
        agent_poll_interval_seconds=agent_poll_interval_seconds,
        client_inquiry_prompt=client_inquiry_prompt,
        profile_resume_run_dir=(
            str(Path(profile_resume_run_dir).resolve()) if profile_resume_run_dir is not None else None
        ),
    )
    controller = FedReadyTaskQueryController(**controller_kwargs)
    executor = FedReadyTaskQueryExecutor(
        site_meta_path=str(meta_path),
        project_root=str(root),
        output_dir=str(run_output),
        min_count=min_count,
        max_scan_files=max_scan_files,
        max_image_samples=max_image_samples,
        histogram_bins=histogram_bins,
        total_rounds=total_rounds,
        extraction_output_root=extraction_output_root,
        extraction_output_name=extraction_output_name,
        extraction_max_samples=extraction_max_samples,
        extraction_overwrite=extraction_overwrite,
        extraction_validation_fraction=extraction_validation_fraction,
        agent_backend=agent_backend,
        agent_timeout_seconds=agent_timeout_seconds,
        agent_poll_interval_seconds=agent_poll_interval_seconds,
    )
    job.to_server(controller, id="fedready_task_query_controller")
    for client_id in client_ids:
        job.to(
            executor,
            client_id,
            id="fedready_task_query_executor",
            tasks=[TASK_QUERY_TASK_NAME],
        )
    _add_prompt_assets(job, client_ids=client_ids)
    return job, client_ids


def build_agent_task_query_recipe(**kwargs: Any) -> tuple[FedReadyRecipe, list[str]]:
    """Build an NVFlare Recipe wrapper for the FedReady task-query flow."""

    job, client_ids = build_agent_task_query_job(**kwargs)
    return FedReadyRecipe(job), client_ids


def _add_prompt_assets(job: FedJob, *, client_ids: list[str]) -> None:
    """Package non-Python prompt assets beside the collected prompt module."""

    prompt_dir = Path(__file__).resolve().parent / "prompts"
    for filename in ("client.json", "server.json"):
        source = str(prompt_dir / filename)
        job.add_file_to_server(source, dest_dir="fedready/prompts", app_folder_type="custom")
        for client_id in client_ids:
            job.add_file_to(
                source,
                client_id,
                dest_dir="fedready/prompts",
                app_folder_type="custom",
            )


def export_agent_task_query_job(
    *,
    job_root: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Export a FedReady NVFlare recipe job folder."""

    recipe, client_ids = build_agent_task_query_recipe(**kwargs)
    job_path = export_fed_job_recipe(recipe.job, job_root)
    return {
        "schema_version": SIMULATION_SCHEMA_VERSION,
        "job_name": recipe.job.name,
        "job_path": str(job_path),
        "client_ids": client_ids,
        "recipe_api": "nvflare.recipe",
    }


def run_agent_task_query_job(
    *,
    workspace: str | Path,
    threads: int | None = None,
    log_config: str | None = "concise",
    **kwargs: Any,
) -> dict[str, Any]:
    """Run a FedReady NVFlare Job API job in the local simulator."""

    run_kwargs = dict(kwargs)
    session = run_kwargs.get("session_id")
    if not session:
        session = _default_session_id(str(run_kwargs.get("task", "<TASK_DESCRIPTION>")))
        run_kwargs["session_id"] = session
    run_kwargs.setdefault("extraction_output_root", _run_scoped_extraction_output_root(str(session)))
    run_kwargs.setdefault("extraction_output_name", str(session))

    recipe, client_ids = build_agent_task_query_recipe(**run_kwargs)
    requested_workspace = Path(workspace)
    workspace_root = resolve_recipe_workspace_root(
        requested_workspace,
        session_id=str(session),
        job_name=recipe.job.name,
    )
    recipe_run = run_fed_job_recipe(
        recipe.job,
        workspace_root=workspace_root,
        clients=client_ids,
        threads=threads,
        log_config=log_config,
    )
    return {
        "schema_version": SIMULATION_SCHEMA_VERSION,
        "job_name": recipe.job.name,
        "session_id": session,
        "workspace_arg": str(requested_workspace),
        "workspace_root": recipe_run["workspace_root"],
        "workspace": recipe_run["workspace"],
        "recipe_run": recipe_run,
        "client_ids": client_ids,
        "run_dir": str(Path(run_kwargs.get("output_dir", "runs")).resolve() / str(session)),
    }


def resolve_recipe_workspace_root(workspace: str | Path, *, session_id: str, job_name: str) -> Path:
    """Resolve the workspace root passed to NVFlare ``SimEnv``.

    The latest recipe API writes simulator output under ``workspace_root / job.name``.
    When callers pass the default project-level ``workspace`` directory, keep runs
    grouped by FedReady session first, then let ``SimEnv`` add the job-name leaf.
    """

    workspace_path = Path(workspace)
    if workspace_path.name == DEFAULT_WORKSPACE_ROOT:
        return workspace_path / _safe_workspace_slug(session_id or job_name)
    return workspace_path


def resolve_experiment_workspace(workspace: str | Path, *, session_id: str, job_name: str) -> Path:
    """Resolve the concrete NVFlare recipe simulator workspace for one experiment."""

    return resolve_recipe_workspace_root(workspace, session_id=session_id, job_name=job_name) / job_name


def _default_session_id(task: str) -> str:
    timestamp = timestamp_utc().replace(":", "").replace(".", "_")
    return f"{_safe_workspace_slug(task)}_{timestamp}"


def _safe_workspace_slug(value: str) -> str:
    return safe_path_slug(value, fallback="fedready_experiment")


def _run_scoped_extraction_output_root(session_id: str) -> str:
    """Keep prepared outputs isolated so reruns never overwrite another run."""

    return str(Path("data") / "dataset_fl_runs" / _safe_workspace_slug(session_id))


def _server_visible_clients(site_meta_path: str, max_clients: int | None) -> list[str]:
    redacted = list_client_ids(site_meta_path)
    client_ids = [client["client_id"] for client in redacted["clients"]]
    if max_clients is not None:
        client_ids = client_ids[:max_clients]
    return client_ids


def main(argv: list[str] | None = None) -> int:
    """Preflight and run the data-readiness job with a minimal fixed interface."""

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) not in {2, 3}:
        print(
            "usage: python -m fedready.job_data SITE_META TASK [PROJECT_ROOT]",
            file=sys.stderr,
        )
        return 2
    site_meta, task = args[:2]
    project_root = Path(args[2] if len(args) == 3 else ".").resolve()
    session_id = _default_session_id(task)

    from fedready.agents import DEFAULT_LOCAL_VISION_MODEL
    from fedready.agents.bridge import LOCAL_VISION_API_BASE_URL
    from fedready.agents.preflight import run_live_runtime_preflight

    run_live_runtime_preflight(
        agent_backend="codex",
        task=task,
        project_root=project_root,
        output_dir=project_root / "runs",
        session_id=session_id,
        local_vlm_base_url=(
            os.environ.get("FEDREADY_VISION_AGENT_API_BASE_URL", LOCAL_VISION_API_BASE_URL).strip()
            or LOCAL_VISION_API_BASE_URL
        ),
        local_vlm_model=(
            os.environ.get("FEDREADY_VISION_AGENT_MODEL", DEFAULT_LOCAL_VISION_MODEL).strip()
            or DEFAULT_LOCAL_VISION_MODEL
        ),
        local_vlm_api_key_env=os.environ.get(
            "FEDREADY_VISION_AGENT_API_KEY_ENV",
            "FEDREADY_LOCAL_VISION_API_KEY",
        ).strip(),
    )
    result = run_agent_task_query_job(
        workspace=project_root / "workspace",
        site_meta_path=site_meta,
        task=task,
        project_root=project_root,
        output_dir=project_root / "runs",
        session_id=session_id,
        extraction_output_root=_run_scoped_extraction_output_root(session_id),
        extraction_output_name=session_id,
        agent_backend="codex",
    )
    print(
        json.dumps(
            {key: result[key] for key in ("session_id", "run_dir", "workspace", "client_ids")},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
