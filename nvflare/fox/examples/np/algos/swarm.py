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
import threading
import traceback

from nvflare.fox.api.app import ClientApp
from nvflare.fox.api.ctx import Context
from nvflare.fox.api.dec import collab
from nvflare.fox.api.group import all_clients
from nvflare.fox.api.strategy import Strategy
from nvflare.fox.examples.np.algos.utils import parse_array_def
from nvflare.fuel.utils.log_utils import get_obj_logger


class NPSwarm(Strategy):

    def __init__(self, initial_model, num_rounds=10):
        self.num_rounds = num_rounds
        self.initial_model = initial_model
        self._initial_model = parse_array_def(initial_model)
        self.waiter = threading.Event()
        self.logger = get_obj_logger(self)

    def execute(self, context: Context):
        context.app.register_event_handler("all_done", self._all_done)

        # randomly pick a client to start
        start_client_idx = random.randint(0, len(context.clients) - 1)
        start_client = context.clients[start_client_idx]
        start_client.start(self.num_rounds, self._initial_model)
        while not context.is_aborted():
            if self.waiter.wait(timeout=0.5):
                break

    def _all_done(self, event_type: str, data, context: Context):
        self.logger.info(f"[{context.header_str()}]: received {event_type} from client: {context.caller}: {data}")
        self.all_done(data, context)

    @collab
    def all_done(self, reason: str, context: Context):
        self.logger.info(f"[{context.header_str()}]: all done from client: {context.caller}: {reason}")
        self.waiter.set()


class NPSwarmClient(ClientApp):

    def __init__(self, delta: float):
        super().__init__()
        self.delta = delta
        self.register_event_handler("final_model", self._accept_final_model)

    @collab
    def train(self, weights, current_round, context: Context):
        self.logger.info(f"[{context.header_str()}]: train asked by {context.caller}: {current_round=}")
        return weights + self.delta

    def sag(self, model, current_round, ctx: Context):
        results = all_clients(ctx, blocking=True).train(model, current_round)
        # results = all_other_clients(ctx, blocking=True).train(model, current_round)
        results = list(results.values())
        total = results[0]
        for i in range(1, len(results)):
            total += results[i]
        return total / len(results)

    @collab
    def swarm_learn(self, num_rounds, model, current_round, context: Context):
        self.logger.info(f"[{context.header_str()}]: swarm learn asked: {num_rounds=} {current_round=} {model=}")
        new_model = self.sag(model, current_round, context)

        self.logger.info(f"[{context.header_str()}]: trained model {new_model=}")
        if current_round == num_rounds - 1:
            # all done
            all_clients(context, blocking=False).fire_event("final_model", new_model)
            # self.server.fire_event("all_done", "OK", blocking=False)
            self.logger.info("notify server all done!")
            try:
                self.server.all_done("OK", _blocking=False)
            except:
                traceback.print_exc()
            self.logger.info("Swarm Training is DONE!")
            return

        # determine next client
        next_round = current_round + 1
        next_client_idx = random.randint(0, len(self.clients) - 1)
        self.logger.debug(f"chose aggr client for round {next_round}: {next_client_idx}")
        next_client = self.clients[next_client_idx]
        next_client.swarm_learn(num_rounds, new_model, next_round, _blocking=False)

    @collab
    def start(self, num_rounds, initial_model, context: Context):
        self.swarm_learn(num_rounds, initial_model, 0, context)

    def _accept_final_model(self, event_type: str, model, context: Context):
        # accept the final model
        # write model to disk
        self.logger.info(f"[{context.header_str()}]: received event '{event_type}' from {context.caller}: {model}")
