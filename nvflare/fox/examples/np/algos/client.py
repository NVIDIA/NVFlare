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
import random

from nvflare.fox import fox
from nvflare.fox.api.app import ClientApp
from nvflare.fox.api.ctx import Context
from nvflare.fox.api.group import all_children


class NPTrainer(ClientApp):

    def __init__(self, delta: float):
        ClientApp.__init__(self)
        self.delta = delta

    @fox.init
    def init_trainer(self):
        delta_config = fox.get_prop("client_delta", {})
        self.delta = delta_config.get(self.name, self.delta)
        self.logger.info(f"init_trainer: client {self.name}: delta={self.delta}")

    @fox.init
    def init_trainer2(self):
        self.logger.info(f"init_trainer2: client {self.name}: init again")

    @fox.collab
    def train(self, current_round, weights):
        if fox.is_aborted:
            self.logger.debug("training aborted")
            return 0
        self.logger.debug(f"[{fox.call_info}] EZ trained round {current_round}")

        # metric_receiver = self.server.get_target("metric_receiver")
        # if metric_receiver:
        #     self.server.accept_metric({"round": r, "y": 2})
        #     self.server.metric_receiver.accept_metric({"round": r, "y": 2})
        #
        self.server.fire_event("metrics", {"round": current_round, "y": 10}, _blocking=False)
        return weights + self.delta

    @fox.collab
    def evaluate(self, model):
        self.logger.debug(f"[{fox.call_info}] evaluate")
        return random.random()


class NPHierarchicalTrainer(ClientApp):

    def __init__(self, delta: float):
        ClientApp.__init__(self)
        self.delta = delta

    @fox.collab
    def train(self, current_round, weights, context: Context):
        if context.is_aborted():
            self.logger.debug("training aborted")
            return 0

        self.logger.debug(f"[{context.header_str()}] training round {current_round}")
        if context.app.has_children():
            total = 0
            results = all_children(context).train(current_round, weights)
            for n, v in results.items():
                total += v
            result = total / len(results)
            self.logger.debug(f"[{context.header_str()}]: aggr result from children of round {current_round}: {result}")
        else:
            result = self._local_train(current_round, weights, context)
            self.logger.debug(f"[{context.header_str()}]: local train result of round {current_round}: {result}")
        return result

    def _local_train(self, current_round, weights, context: Context):
        if context.is_aborted():
            self.logger.debug("training aborted")
            return 0
        self.logger.info(f"[{context.header_str()}] local trained round {current_round} {weights} {type(weights)}")
        return weights + self.delta

    @fox.collab
    def evaluate(self, model, context: Context):
        self.logger.debug(f"[{context.header_str()}] evaluate")
        return random.random()


class NPTrainerMaker(ClientApp):

    def __init__(self, delta):
        ClientApp.__init__(self)
        self.delta = delta

    def make_client_app(self, name: str) -> ClientApp:
        app = NPTrainer(self.delta)
        app.update_props(self.get_props())
        app.name = name
        return app
