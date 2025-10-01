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

from nvflare.fox.api.app import ClientApp, ClientAppFactory
from nvflare.fox.api.ctx import Context
from nvflare.fox.api.dec import collab


class NPTrainer(ClientApp):

    def __init__(self, delta: float):
        ClientApp.__init__(self)
        self.delta = delta

    @collab
    def train(self, current_round, weights, context: Context):
        if context.is_aborted():
            print("training aborted")
            return 0
        print(f"[{self.name}] called by {context.caller}: client {context.callee} trained round {current_round}")

        # metric_receiver = self.server.get_target("metric_receiver")
        # if metric_receiver:
        #     self.server.accept_metric({"round": r, "y": 2})
        #
        self.server.fire_event("metrics", {"round": current_round, "y": 10}, _blocking=False)
        return weights + self.delta

    @collab
    def evaluate(self, model, context: Context):
        print(f"[{self.name}] called by {context.caller}: client {context.callee} to evaluate")
        return random.random()


class TrainerFactory(ClientAppFactory):

    def __init__(self, delta):
        self.delta = delta

    def make_client_app(self, name: str) -> ClientApp:
        return NPTrainer(self.delta)
