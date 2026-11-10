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

"""SimEnv adapter: SimEnv runs every site on one host, so each site's dataset configuration is found under its own
folder instead of its workspace."""

from pathlib import Path

from nvflare.apis.dataset_config_spec import DATASET_CONFIG_FILE
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.job_def import get_job_meta_study
from nvflare.app_common.dataset_config.api import load_study_config
from nvflare.app_common.fdn.catalog.site.executor import CatalogExecutor


class SimulationCatalogExecutor(CatalogExecutor):
    """Reads one site's dataset configuration from ``<site_config_root>/<site>/local/datasets.yaml``."""

    def __init__(self, site_config_root: str):
        super().__init__()
        self.site_config_root = site_config_root

    def load_config(self, fl_ctx):
        if not fl_ctx.get_prop(FLContextKey.SIMULATE_MODE):
            raise RuntimeError("SimulationCatalogExecutor may only be used in SimEnv")
        path = Path(self.site_config_root) / fl_ctx.get_identity_name() / "local" / DATASET_CONFIG_FILE
        study = get_job_meta_study(fl_ctx.get_prop(FLContextKey.JOB_META))
        return load_study_config(str(path), study, missing_ok=True)
