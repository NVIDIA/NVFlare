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


"""Compare exported recipes' fresh workers with assignments recorded in a job-mode run."""

import json
import shutil
from dataclasses import replace
from pathlib import Path

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.job_def import ALL_SITES
from nvflare.apis.signal import Signal
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.client.config import ExchangeFormat
from nvflare.client.decomposers import register_framework_decomposers
from nvflare.fuel.utils import fobs
from nvflare.job_config.task_execution import prepare_task_execution, task_constructor_args
from nvflare.private.fed.client.task_worker_executor import TaskWorkerExecutor
from tests.task_execution_test_utils import task_worker_workspace


class RecipeTaskRecorder(FLComponent):
    def __init__(self):
        super().__init__()
        self._index = 0

    def handle_event(self, event_type, fl_ctx):
        if event_type not in (EventType.BEFORE_TASK_EXECUTION, EventType.AFTER_TASK_EXECUTION):
            return
        run_dir = fl_ctx.get_workspace().get_run_dir(fl_ctx.get_job_id())
        folder = Path(run_dir) / "recipe_task_records"
        folder.mkdir(exist_ok=True)
        prefix = folder / f"{self._index:04d}"
        if event_type == EventType.BEFORE_TASK_EXECUTION:
            task_name = fl_ctx.get_prop(FLContextKey.TASK_NAME)
            fobs.dumpf((task_name, fl_ctx.get_prop(FLContextKey.TASK_DATA)), str(prefix.with_suffix(".input")))
        else:
            fobs.dumpf(fl_ctx.get_prop(FLContextKey.TASK_RESULT), str(prefix.with_suffix(".output")))
            self._index += 1


def replay_exported_recipe_tasks(recipe, baseline, root, compare, *, monkeypatch):
    export_dir = root / "export"
    recipe.export(str(export_dir))
    job = export_dir / recipe.name
    deploy_map = json.loads((job / "meta.json").read_text())["deploy_map"]
    for site in ("site-1", "site-2"):
        apps = [app for app, sites in deploy_map.items() if site in sites or ALL_SITES in sites]
        assert len(apps) == 1, deploy_map
        app = job / apps[0]
        config = json.loads((app / "config" / "config_fed_client.json").read_text())
        assert config["execution_lifetime"] == "task"
        [plan] = prepare_task_execution(config).executors
        args = task_constructor_args(plan.executor)
        files = sorted(p for p in baseline.rglob("recipe_task_records/*.input") if site in p.parts)
        assert files, (baseline, site)
        # Client API metadata and native checkpoints retain the original job scope.
        job_id = files[0].parent.parent.name
        with task_worker_workspace(root / site, monkeypatch, site=site, job_id=job_id) as (_, custom, ctx, policy):
            shutil.copytree(app / "custom", custom, dirs_exist_ok=True)
            register_framework_decomposers(
                args.get("params_exchange_format", ExchangeFormat.RAW),
                args.get("server_expected_format", ExchangeFormat.NUMPY),
            )
            policy = replace(
                policy,
                artifact_cleanup="retain",
                worker_timeout=90,
                environment_variables=("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"),
            )
            adapter = TaskWorkerExecutor(
                plan.executor, plan.components, policy, result_wait_timeout=plan.result_wait_timeout
            )
            try:
                adapter.handle_event(EventType.START_RUN, ctx)
                for index, path in enumerate(files):
                    name, data = fobs.loadf(str(path))
                    expected = fobs.loadf(str(path.with_suffix(".output")))
                    assert expected.get_return_code() == ReturnCode.OK
                    ctx.set_prop(FLContextKey.TASK_NAME, name, private=True, sticky=True)
                    ctx.set_prop(FLContextKey.TASK_ID, f"task-{index}", private=True, sticky=True)
                    ctx.set_prop(FLContextKey.TASK_ATTEMPT_ID, f"attempt-{index}", private=True, sticky=True)
                    result = adapter.execute(name, data, ctx, Signal())
                    assert result.get_return_code() == ReturnCode.OK, (site, name, index, result)
                    actual_model = FLModelUtils.from_shareable(result)
                    expected_model = FLModelUtils.from_shareable(expected)
                    assert actual_model.params_type == expected_model.params_type
                    compare(actual_model.params, expected_model.params)
                    compare(actual_model.metrics, expected_model.metrics)
            finally:
                adapter.handle_event(EventType.END_RUN, ctx)
                assert not adapter.supervisor.owns_execution, f"{site}: task worker is still active"
    return root
