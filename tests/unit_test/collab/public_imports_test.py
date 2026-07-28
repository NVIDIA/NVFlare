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
    from nvflare.collab import CollabCallError, CollabRecipe, collab, simple_logging

    for export in (
        collab,
        CollabCallError,
        CollabRecipe,
        simple_logging,
    ):
        assert export is not None


def test_api_surface():
    from nvflare.collab.api import (
        App,
        CallOption,
        ClientApp,
        CollabCallError,
        Context,
        ContextKey,
        GroupCallContext,
        ModuleWrapper,
        PublishInterface,
        ServerApp,
    )

    for export in (
        App,
        ClientApp,
        CollabCallError,
        ServerApp,
        CallOption,
        Context,
        ContextKey,
        GroupCallContext,
        ModuleWrapper,
        PublishInterface,
    ):
        assert export is not None


def test_runtime_surface():
    from nvflare.collab.runtime.controller import CollabController
    from nvflare.collab.runtime.executor import CollabExecutor
    from nvflare.collab.runtime.lifecycle import run_server

    for export in (
        CollabController,
        CollabExecutor,
        run_server,
    ):
        assert export is not None


def test_execution_details_are_not_public():
    import nvflare.collab as collab_package
    import nvflare.collab.api as collab_api
    import nvflare.collab.runtime as collab_runtime
    from nvflare.collab import CollabRecipe, collab

    for name in ("InProcessEnv", "MultiProcessEnv", "InProcessRunner"):
        assert not hasattr(collab_package, name)
    for name in ("Backend", "BackendType", "CallFilter", "ResultFilter"):
        assert not hasattr(collab_api, name)
    for name in ("LocalBackend", "FlareBackend", "SubprocessBackend"):
        assert not hasattr(collab_runtime, name)
    for name in (
        "call_filter",
        "in_call_filter",
        "out_call_filter",
        "result_filter",
        "in_result_filter",
        "out_result_filter",
        "filter_direction",
        "qual_func_name",
    ):
        assert not hasattr(collab, name)
    for name in (
        "add_server_outgoing_call_filters",
        "add_server_incoming_call_filters",
        "add_server_outgoing_result_filters",
        "add_server_incoming_result_filters",
        "add_client_outgoing_call_filters",
        "add_client_incoming_call_filters",
        "add_client_outgoing_result_filters",
        "add_client_incoming_result_filters",
    ):
        assert not hasattr(CollabRecipe, name)


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
        "from nvflare.collab.api import Context, GroupCallContext\n"
        "import nvflare.collab.runtime.controller\n"
        "import nvflare.collab.runtime.executor\n"
        "from nvflare.collab import simple_logging\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_api_layer_does_not_import_runtime():
    """The api package is the runtime-neutral contract layer; importing all of
    it (and the top-level facade export) must not pull in any runtime module."""
    code = (
        "import sys\n"
        "import nvflare.collab\n"
        "import nvflare.collab.api\n"
        "import nvflare.collab.api.app\n"
        "import nvflare.collab.api._invocation\n"
        "import nvflare.collab.api.facade\n"
        "import nvflare.collab.api.group\n"
        "import nvflare.collab.api.module_wrapper\n"
        "import nvflare.collab.api.proxy\n"
        "import nvflare.collab.api.proxy_list\n"
        "bad = sorted(m for m in sys.modules if m.startswith('nvflare.collab.runtime'))\n"
        "assert not bad, f'api layer imported runtime modules: {bad}'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
