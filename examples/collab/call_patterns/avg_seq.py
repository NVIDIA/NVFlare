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

import os

from collab.call_patterns.np_utils import parse_array_def, save_np_model

from nvflare.collab import collab
from nvflare.fuel.utils.log_utils import get_obj_logger


class NPFedAvgSequential:

    def __init__(self, initial_model, num_rounds=10):
        self.name = "NPFedAvgSequential"
        self.num_rounds = num_rounds
        self.initial_model = initial_model
        self._initial_model = parse_array_def(initial_model)
        self.logger = get_obj_logger(self)
        self.client_weights = None

    @collab.init
    def init(self):
        self.logger.info("init NPFedAvgSequential")
        weight_config = collab.get_app_prop("client_weight_config", {})
        client_weights = {}
        total = 0
        for client in collab.clients:
            weight = weight_config.get(client.name, 100)
            client_weights[client.name] = weight
            total += weight

        for client in collab.clients:
            client_weights[client.name] = client_weights[client.name] / total

        self.client_weights = client_weights
        self.logger.info(f"client_weights: {client_weights}")

    @collab.main
    def execute(self):
        self.logger.info(f"[{collab.call_info}] Start training for {self.num_rounds} rounds")
        current_model = self._initial_model
        for current_round in range(self.num_rounds):
            current_model = self._do_one_round(current_round, current_model)

        file_name = os.path.join(collab.workspace.get_work_dir(), "model.npy")
        save_np_model(current_model, file_name)
        self.logger.info(f"FINAL RESULT: {current_model}")
        return current_model

    def _do_one_round(self, current_round, current_model):
        total = 0
        for client in collab.clients:
            result = client(expect_result=True, timeout=2.0, optional=True, secure=False).train(
                current_round, current_model
            )
            self.logger.info(
                f"[{collab.call_info}] round {current_round}: got result from client {client.name}: {result}"
            )
            total += result * self.client_weights[client.name]
        return total
