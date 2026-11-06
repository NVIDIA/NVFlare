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

"""Local simulation adapters; production uses site workspaces and CatalogManager instead."""

from pathlib import Path

from nvflare.app_common.fdn.catalog.server.controller import CatalogController
from nvflare.app_common.fdn.catalog.server.store import CatalogStore
from nvflare.app_common.fdn.catalog.site.executor import CatalogExecutor


class SimulationCatalogExecutor(CatalogExecutor):
    def __init__(self, data_root: str):
        super().__init__()
        self.data_root = data_root

    def _workspace_root(self, fl_ctx):
        # SimEnv shares one host: resolve each site's external workspace by identity.
        return str(Path(self.data_root) / fl_ctx.get_identity_name())


class SimulationCatalogController(CatalogController):
    def __init__(self, store_root: str):
        super().__init__(task_timeout_s=60)
        self.store_root = store_root

    def _load_context(self, path, fl_ctx):
        # SimEnv has no parent CatalogManager. Recover previous names from the
        # demo store; job.py applies this local simulation's manifest afterward.
        self._previous_names = CatalogStore(self.store_root).indexed("default")
