# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import xgboost.federated as xgb_federated

from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.app_opt.xgboost.histogram_based_v2.runners.xgb_runner import AppRunner


class XGBServerRunner(AppRunner):
    def __init__(self):
        self._port = None
        self._world_size = None
        self._stopped = False

    def run(self, ctx: dict):
        self._port = ctx.get(Constant.RUNNER_CTX_PORT)
        self._world_size = ctx.get(Constant.RUNNER_CTX_WORLD_SIZE)

        xgb_federated.run_federated_server(
            n_workers=self._world_size,
            port=self._port,
        )
        self._stopped = True

    def stop(self):
        # no way to start currently
        pass

    def is_stopped(self) -> (bool, int):
        return self._stopped, 0
