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

"""Smoke tests for the nvflare.collab public import surface and layering.

These pin the supported import paths so layout changes that break users
(or reintroduce the api -> runtime dependency cycle) fail fast.
"""

import subprocess
import sys


def test_top_level_exports():
    from nvflare.collab import (
        CollabClientAPI,
        CollabRecipe,
        InProcessEnv,
        InProcessRunner,
        MultiProcessEnv,
        collab,
        simple_logging,
    )

    for export in (
        collab,
        CollabClientAPI,
        CollabRecipe,
        InProcessEnv,
        MultiProcessEnv,
        InProcessRunner,
        simple_logging,
    ):
        assert export is not None


def test_api_surface():
    from nvflare.collab.api import (
        App,
        Backend,
        BackendType,
        CallFilter,
        CallOption,
        ClientApp,
        CollabWorkspace,
        Context,
        ContextKey,
        GroupCallContext,
        ModuleWrapper,
        PublishInterface,
        ResultFilter,
        ServerApp,
    )

    for export in (
        App,
        ClientApp,
        CollabWorkspace,
        ServerApp,
        Backend,
        BackendType,
        CallFilter,
        CallOption,
        Context,
        ContextKey,
        GroupCallContext,
        ModuleWrapper,
        PublishInterface,
        ResultFilter,
    ):
        assert export is not None


def test_runtime_surface():
    from nvflare.collab.runtime import FlareBackend, LocalBackend, SubprocessBackend
    from nvflare.collab.runtime.client_api import CollabClientAPI
    from nvflare.collab.runtime.lifecycle import run_server

    for export in (
        LocalBackend,
        SubprocessBackend,
        FlareBackend,
        CollabClientAPI,
        run_server,
    ):
        assert export is not None


def test_collab_public_surface_does_not_require_torch():
    """The collab public surface must work in a base installation without
    the PT extra; torch belongs only to app-level code built on top of it."""
    code = (
        "import sys\n"
        "class _BlockTorch:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'torch' or name.startswith('torch.'):\n"
        "            raise ImportError('torch is blocked for this test')\n"
        "sys.meta_path.insert(0, _BlockTorch())\n"
        "import nvflare.collab\n"
        "from nvflare.collab.api import CallFilter, Context, GroupCallContext, ResultFilter\n"
        "import nvflare.collab.runtime.flare.controller\n"
        "import nvflare.collab.runtime.flare.executor\n"
        "import nvflare.collab.runtime.local.app_runner\n"
        "import nvflare.collab.runtime.worker.worker\n"
        "from nvflare.collab.tracking import SummaryWriter\n"
        "from nvflare.collab import simple_logging\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_tracking_surface():
    from nvflare.collab.tracking import (
        AutoWriter,
        MLflowWriter,
        SummaryWriter,
        TensorBoardWriter,
        WandbWriter,
        get_auto_writer,
        mlflow,
        wandb,
    )

    assert mlflow.__name__ == "nvflare.collab.tracking.mlflow"
    assert wandb.__name__ == "nvflare.collab.tracking.wandb"
    for export in (SummaryWriter, TensorBoardWriter, MLflowWriter, WandbWriter, AutoWriter, get_auto_writer):
        assert export is not None


def test_api_layer_does_not_import_runtime():
    """The api package is the runtime-neutral contract layer; importing all of
    it (and the top-level facade export) must not pull in any runtime module."""
    code = (
        "import sys\n"
        "import nvflare.collab\n"
        "import nvflare.collab.api\n"
        "import nvflare.collab.api.app\n"
        "import nvflare.collab.api.backend\n"
        "import nvflare.collab.api.facade\n"
        "import nvflare.collab.api.filter\n"
        "import nvflare.collab.api.group\n"
        "import nvflare.collab.api.module_wrapper\n"
        "import nvflare.collab.api.proxy\n"
        "import nvflare.collab.api.proxy_list\n"
        "bad = sorted(m for m in sys.modules if m.startswith('nvflare.collab.runtime'))\n"
        "assert not bad, f'api layer imported runtime modules: {bad}'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
