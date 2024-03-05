# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Tuple

from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.app_opt.xgboost.histogram_based_v2.runner import XGBRunner
from nvflare.fuel.utils.import_utils import optional_import


class XGBServerRunner(XGBRunner):
    def __init__(self):
        self._stopped = True

    def run(self, ctx: dict):
        xgb_federated, flag = optional_import(module="xgboost.federated")
        if not flag:
            raise RuntimeError("Can't import xgboost.federated")

        _secure = ctx.get(Constant.RUNNER_CTX_SECURE, False)
        _port = ctx.get(Constant.RUNNER_CTX_PORT, None)
        _world_size = ctx.get(Constant.RUNNER_CTX_WORLD_SIZE, None)

        self._stopped = False
        if _secure:
            _server_key_path = ctx.get(Constant.RUNNER_CTX_SERVER_KEY_PATH, None)
            _server_cert_path = ctx.get(Constant.RUNNER_CTX_SERVER_CERT_PATH, None)
            _ca_cert_path = ctx.get(Constant.RUNNER_CTX_CA_CERT_PATH, None)
            if None in [_server_cert_path, _server_key_path, _ca_cert_path]:
                raise RuntimeError("Missing required certificates")
            xgb_federated.run_federated_server(
                port=_port,
                world_size=_world_size,
                server_key_path=_server_key_path,
                server_cert_path=_server_cert_path,
                client_cert_path=_ca_cert_path,
            )
        else:
            xgb_federated.run_federated_server(
                port=_port,
                world_size=_world_size,
            )
        self._stopped = True

    def stop(self):
        # currently no way to stop the runner
        pass

    def is_stopped(self) -> Tuple[bool, int]:
        return self._stopped, 0
