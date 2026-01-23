# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.collab import fox
from nvflare.collab.examples.np.mains.utils import parse_array_def, save_np_model
from nvflare.fuel.utils.log_utils import get_obj_logger


class NPFedAvgSequential:

    def __init__(self, initial_model, num_rounds=10):
        self.name = "NPFedAvgSequential"
        self.num_rounds = num_rounds
        self.initial_model = initial_model  # need to remember init for job API to work!
        self._initial_model = parse_array_def(initial_model)
        self.logger = get_obj_logger(self)
        self.client_weights = None

    @fox.init
    def init(self):
        self.logger.info("fox init NPFedAvgSequential")
        weight_config = fox.get_app_prop("client_weight_config", {})
        client_weights = {}
        total = 0
        for c in fox.clients:
            w = weight_config.get(c.name, 100)
            client_weights[c.name] = w
            total += w

        # normalize weights
        for c in fox.clients:
            client_weights[c.name] = client_weights[c.name] / total

        self.client_weights = client_weights
        self.logger.info("client_weights: {}".format(client_weights))

    @fox.main
    def execute(self):
        self.logger.info(f"[{fox.call_info}] Start training for {self.num_rounds} rounds")
        current_model = self._initial_model
        for i in range(self.num_rounds):
            current_model = self._do_one_round(i, current_model)

        # save model to work dir
        file_name = os.path.join(fox.workspace.get_work_dir(), "model.npy")
        save_np_model(current_model, file_name)
        self.logger.info(f"FINAL RESULT: {current_model}")
        return current_model

    def _do_one_round(self, r, current_model):
        total = 0
        for c in fox.clients:
            result = c(expect_result=True, timeout=2.0, optional=True, secure=False).train(r, current_model)
            self.logger.info(f"[{fox.call_info}] round {r}: got result from client {c.name}: {result}")
            total += result * self.client_weights[c.name]
        return total
