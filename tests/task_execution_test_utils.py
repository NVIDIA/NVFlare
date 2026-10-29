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


"""Shared setup for policy unit tests and real process integration scenarios."""

import argparse
import json
import os
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.workspace import Workspace
from nvflare.app_common.app_constant import AppConstants
from nvflare.client.decomposers import register_framework_decomposers
from nvflare.job_config.task_execution import CLIENT_API_EXECUTOR_PATH, STATISTICS_EXECUTOR_PATH, prepare_task_execution
from nvflare.private.fed.client.client_json_config import ClientJsonConfigurator
from nvflare.private.fed.client.task_execution_policy import TaskExecutionPolicy
from nvflare.private.fed.client.task_worker_executor import TaskWorkerExecutor
from nvflare.private.fed.utils.fed_utils import nvflare_fobs_initialize


def _config(**args):
    return {
        "format_version": 3,
        "execution_lifetime": "task",
        "executors": [
            {
                "tasks": ["*"],
                "executor": {
                    "path": CLIENT_API_EXECUTOR_PATH,
                    "args": {"execution_mode": "in_process", "task_script_path": "train.py", **args},
                },
            }
        ],
    }


def _configure(tmp_path, config, authorizer=None):
    config_file = tmp_path / "config_fed_client.json"
    config_file.write_text(json.dumps(config))
    args = argparse.Namespace(
        sp_scheme="grpc",
        sp_target="localhost:8002",
        client_name="site-1",
        parent_url=None,
        job_id="job-1",
        workspace=str(tmp_path),
    )
    workspace = SimpleNamespace(
        get_app_custom_dir=lambda _: str(tmp_path / "custom"),
        get_app_config_dir=lambda _: str(tmp_path / "config"),
        get_resources_file_path=lambda: None,
        get_startup_kit_dir=lambda: str(tmp_path),
        get_site_config_dir=lambda: str(tmp_path),
    )
    configurator = ClientJsonConfigurator(workspace, str(config_file), args, str(tmp_path))
    if authorizer:
        configurator.set_component_build_authorizer(authorizer)
    configurator.configure()
    return configurator


@pytest.fixture
def native_task(tmp_path, monkeypatch, isolated_fobs):
    if not hasattr(os, "waitid"):
        pytest.skip("CPU Process task execution requires waitid/WNOWAIT")
    from nvflare.fuel.f3.cellnet.cell import Cell
    from nvflare.fuel.f3.cellnet.core_cell import CoreCell
    from nvflare.fuel.f3.stats_pool import StatsPoolManager

    nvflare_fobs_initialize()
    register_framework_decomposers("numpy", "numpy")
    root = tmp_path / "workspace"
    (root / "startup").mkdir(parents=True)
    (root / "local").mkdir()
    custom = root / "job-1" / "app_site-1" / "custom"
    custom.mkdir(parents=True)
    (custom.parent / "config").mkdir()
    (root / "job-1" / "meta.json").write_text(json.dumps({"byoc": True}))
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1]))
    monkeypatch.setenv("TASK_TEST_SECRET", "local-placeholder")
    workspace = Workspace(str(root), site_name="site-1")
    # Workers join the Client Job cell, so the executor needs a real one; it connects nowhere itself.
    # A unique name keeps it clear of cells other tests in this process may leave behind, and the
    # process-wide tables that stop() leaves alone are restored afterwards.
    original_cells, original_pools = dict(CoreCell.ALL_CELLS), dict(StatsPoolManager.pools)
    cell = Cell(f"site-1.job-{uuid.uuid4().hex}", "tcp://127.0.0.1:0", secure=False, credentials={})
    cell.start()
    try:
        context = FLContext()
        context.set_prop(
            ReservedKey.ENGINE,
            SimpleNamespace(get_workspace=lambda: workspace, get_cell=lambda: cell, fire_event=Mock()),
            private=True,
            sticky=True,
        )
        for key, value in (
            (ReservedKey.RUN_NUM, "job-1"),
            (ReservedKey.IDENTITY_NAME, "site-1"),
            (FLContextKey.WORKSPACE_OBJECT, workspace),
            (FLContextKey.TASK_ID, "task-1"),
            (FLContextKey.TASK_NAME, "train"),
            (FLContextKey.TASK_ATTEMPT_ID, "attempt-1"),
        ):
            context.set_prop(key, value, private=True, sticky=True)
        policy = TaskExecutionPolicy.from_config({"worker_timeout": 30, "environment_variables": ["TASK_TEST_SECRET"]})
        yield root, custom, context, policy
    finally:
        cell.stop()
        CoreCell.ALL_CELLS.clear(), CoreCell.ALL_CELLS.update(original_cells)
        StatsPoolManager.pools.clear(), StatsPoolManager.pools.update(original_pools)


SCRIPT = """
import os, sys, threading
import numpy as np
import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.apis.analytix import AnalyticsDataType
flare.init()
while flare.is_running():
    model = flare.receive()
    flare.log("accuracy", 0.75, AnalyticsDataType.SCALAR, global_step=model.current_round)
    # Larger than the download-reference threshold: must travel inline over the session.
    flare.log("image", np.zeros((1024, 1024), dtype=np.float32), AnalyticsDataType.IMAGE, global_step=0)
    result = FLModel(params={"w": model.params["w"] + 1},
        metrics={"pid": os.getpid(), "main_thread": threading.current_thread() is threading.main_thread(),
                 "argv": sys.argv[1:], "site": flare.get_site_name(), "job": flare.get_job_id(),
                 "metadata": model.meta},
        current_round=model.current_round)
    flare.send(result)
"""


def _adapter(custom, context, policy, script=SCRIPT, **args):
    (custom / "train.py").write_text(script)
    spec = _config(params_exchange_format="numpy", server_expected_format="numpy", **args)["executors"][0]["executor"]
    plan = prepare_task_execution({"execution_lifetime": "task", "executors": [{"executor": spec}]})
    adapter = TaskWorkerExecutor(spec, (), policy, result_wait_timeout=plan.executors[0].result_wait_timeout)
    adapter.handle_event(EventType.START_RUN, context)
    data = DXO(DataKind.WEIGHTS, {"w": np.array([1.0, 2.0])}).to_shareable()
    data.set_header(AppConstants.CURRENT_ROUND, 3)
    return adapter, data


def _statistics_config():
    config = _config()
    config["executors"][0]["executor"] = {"path": STATISTICS_EXECUTOR_PATH, "args": {"generator_id": "stats_generator"}}
    config["components"] = [
        {"id": "stats_generator", "path": "stats_probe.ProbeStatistics"},
        {
            "id": "min_count",
            "path": "nvflare.app_common.statistics.min_count_cleanser.MinCountCleanser",
            "args": {"min_count": 5},
        },
    ]
    config["task_result_filters"] = [
        {
            "tasks": ["fed_stats"],
            "filters": [
                {
                    "path": "nvflare.app_common.filters.statistics_privacy_filter.StatisticsPrivacyFilter",
                    "args": {"result_cleanser_ids": ["min_count"]},
                }
            ],
        }
    ]
    return config
