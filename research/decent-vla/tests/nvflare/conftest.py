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

"""Minimal fake `nvflare` namespace so CI tests the nvflare runtime without
nvflare. Only the symbols our codec/controller/client/job modules touch.
"""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

import pytest


class FakeParamsType:
    FULL = "FULL"
    DIFF = "DIFF"


class FakeFLModel:
    """Field-compatible stand-in for nvflare's FLModel (plain data object)."""

    def __init__(
        self,
        params_type=None,
        params=None,
        optimizer_params=None,
        metrics=None,
        start_round=0,
        current_round=None,
        total_rounds=None,
        meta=None,
    ):
        self.params_type = params_type
        self.params = params
        self.optimizer_params = optimizer_params
        self.metrics = metrics
        self.start_round = start_round
        self.current_round = current_round
        self.total_rounds = total_rounds
        self.meta = meta if meta is not None else {}


class FakeWorkspace:
    """The two nvflare Workspace lookups the controller makes."""

    def __init__(self, root):
        self.root = Path(root)

    def get_run_dir(self, job_id):
        return str(self.root / job_id)

    def get_app_config_dir(self, job_id):
        return str(self.root / job_id / "app_server" / "config")


class FakeEngine:
    def __init__(self, workspace):
        self._workspace = workspace

    def get_workspace(self):
        return self._workspace


class FakeFLContext:
    def get_job_id(self):
        return "simulate_job"


class FakeModelController:
    """Base-class double for nvflare's ModelController.

    Real broadcast_model ships FLModels over gRPC; tests attach a
    `loopback` callable that routes them to the client side in-process.
    `sent` records the target lists of every call (subset-send assertions),
    `min_responses` the quorum each call asked for.
    """

    def __init__(self, *args, persistor_id: str = "persistor", **kwargs):
        self.persistor_id = persistor_id
        self.init_kwargs = kwargs
        self.loopback = None
        self.sent: list[list[str]] = []
        self.min_responses: list = []
        self.fl_ctx = FakeFLContext()
        self.engine = FakeEngine(FakeWorkspace(Path(tempfile.gettempdir()) / "fake-nvflare-workspace"))

    def broadcast_model(self, data=None, task_name="train", targets=None, min_responses=None,
                        timeout=0, wait_time_after_min_received=0, blocking=True, callback=None):
        assert blocking, "the communicator waits for replies"
        self.sent.append(list(targets))
        self.min_responses.append(min_responses)
        assert self.loopback is not None, "test must attach a loopback router"
        return self.loopback(task_name, data, targets, timeout)


class FakeFlareClient(types.ModuleType):
    """Scripted `nvflare.client` module: tests queue FLModels in `incoming`;
    the loop runs until the queue drains. `sent` records flare.send calls."""

    def __init__(self):
        super().__init__("nvflare.client")
        self.incoming: list = []
        self.sent: list = []
        self.site_name = "c0"
        self.inited = False

    def init(self):
        self.inited = True

    def is_running(self):
        return bool(self.incoming)

    def receive(self):
        return self.incoming.pop(0)

    def send(self, model):
        # Like nvflare's flare.send(): the model's params are dropped once sent.
        self.sent.append(FakeFLModel(params=dict(model.params or {}), params_type=model.params_type,
                                     metrics=model.metrics, meta=model.meta))
        model.params = None

    def get_site_name(self):
        return self.site_name


class FakeScriptRunner:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeFedJob:
    """Records to_server/to_clients/simulator_run/export_job wiring."""

    instances: list = []

    def __init__(self, name="fed_job", min_clients=1, mandatory_clients=None, **kw):
        self.name = name
        self.min_clients = min_clients
        self.server_components: list = []
        self.client_components: list = []
        self.sim_calls: list[dict] = []
        self.exported: list[str] = []
        self.app_packages: list[str] = []
        FakeFedJob.instances.append(self)

    def set_app_packages(self, app_packages):
        self.app_packages = list(app_packages)

    def to_server(self, comp):
        self.server_components.append(comp)

    def to_clients(self, comp):
        self.client_components.append(comp)

    def simulator_run(self, workspace, n_clients=None, clients=None, threads=None,
                      gpu=None, log_config=None):
        self.sim_calls.append(
            {"workspace": workspace, "clients": clients, "threads": threads, "gpu": gpu}
        )

    def export_job(self, job_root):
        self.exported.append(job_root)


class _Enumish:
    """Stands in for nvflare's enums; identity of the value doesn't matter."""

    PYTORCH = "pytorch"
    NUMPY = "numpy"
    RAW = "raw"
    FULL = "FULL"
    DIFF = "DIFF"


@pytest.fixture(autouse=True)
def fake_nvflare(monkeypatch):
    root = types.ModuleType("nvflare")
    app_common = types.ModuleType("nvflare.app_common")
    abstract = types.ModuleType("nvflare.app_common.abstract")
    fl_model = types.ModuleType("nvflare.app_common.abstract.fl_model")
    fl_model.FLModel = FakeFLModel
    fl_model.ParamsType = FakeParamsType
    workflows = types.ModuleType("nvflare.app_common.workflows")
    model_controller = types.ModuleType("nvflare.app_common.workflows.model_controller")
    model_controller.ModelController = FakeModelController
    client = FakeFlareClient()
    client_config = types.ModuleType("nvflare.client.config")
    client_config.ExchangeFormat = _Enumish
    client_config.TransferType = _Enumish
    job_config = types.ModuleType("nvflare.job_config")
    job_api = types.ModuleType("nvflare.job_config.api")
    job_api.FedJob = FakeFedJob
    script_runner = types.ModuleType("nvflare.job_config.script_runner")
    script_runner.ScriptRunner = FakeScriptRunner
    script_runner.FrameworkType = _Enumish

    root.app_common, root.client, root.job_config = app_common, client, job_config
    app_common.abstract, app_common.workflows = abstract, workflows
    abstract.fl_model = fl_model
    workflows.model_controller = model_controller
    client.config = client_config
    job_config.api, job_config.script_runner = job_api, script_runner
    for name, mod in [
        ("nvflare", root),
        ("nvflare.app_common", app_common),
        ("nvflare.app_common.abstract", abstract),
        ("nvflare.app_common.abstract.fl_model", fl_model),
        ("nvflare.app_common.workflows", workflows),
        ("nvflare.app_common.workflows.model_controller", model_controller),
        ("nvflare.client", client),
        ("nvflare.client.config", client_config),
        ("nvflare.job_config", job_config),
        ("nvflare.job_config.api", job_api),
        ("nvflare.job_config.script_runner", script_runner),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    # force re-import of our modules against the fake
    for m in list(sys.modules):
        if m.startswith("decent_vla.runtime.nvflare"):
            monkeypatch.delitem(sys.modules, m, raising=False)
    FakeFedJob.instances = []
    yield root
