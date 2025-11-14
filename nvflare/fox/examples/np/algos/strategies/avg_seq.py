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

from nvflare.fox import fox
from nvflare.fox.api.constants import ContextKey
from nvflare.fox.api.ctx import Context
from nvflare.fox.api.strategy import Strategy
from nvflare.fox.examples.np.algos.utils import parse_array_def, save_np_model
from nvflare.fuel.utils.log_utils import get_obj_logger


class NPFedAvgSequential(Strategy):

    def __init__(self, initial_model, num_rounds=10):
        Strategy.__init__(self)
        self.name = "NPFedAvgSequential"
        self.num_rounds = num_rounds
        self.initial_model = initial_model  # need to remember init for job API to work!
        self._initial_model = parse_array_def(initial_model)
        self.logger = get_obj_logger(self)
        self.client_weights = None

    def fox_init(self, context: Context):
        weight_config = context.app.get_prop("client_weight_config", {})
        client_weights = {}
        total = 0
        for c in context.clients:
            w = weight_config.get(c.name, 100)
            client_weights[c.name] = w
            total += w

        # normalize weights
        for c in context.clients:
            client_weights[c.name] = client_weights[c.name] / total

        self.client_weights = client_weights
        self.logger.info("client_weights: {}".format(client_weights))

    def execute(self, context: Context):
        self.logger.info(f"[{context.header_str()}] Start training for {self.num_rounds} rounds")
        current_model = context.get_prop(ContextKey.INPUT, self._initial_model)
        for i in range(self.num_rounds):
            current_model = self._do_one_round(i, current_model)

        # save model to work dir
        file_name = os.path.join(context.workspace.get_work_dir(), "model.npy")
        save_np_model(current_model, file_name)
        return current_model

    def _do_one_round(self, r, current_model):
        total = 0
        for c in fox.clients:
            result = c(blocking=True, timeout=2.0, optional=True, secure=False).train(r, current_model)
            self.logger.info(f"[{fox.context.header_str()}] round {r}: got result from client {c.name}: {result}")
            total += result * self.client_weights[c.name]
        return total
