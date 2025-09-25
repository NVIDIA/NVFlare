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

from nvflare.focs.api.app import ClientApp, ServerApp
from nvflare.focs.api.ctx import Context
from nvflare.focs.api.group import all_clients
from nvflare.focs.api.strategy import Strategy
from nvflare.focs.examples.np.algos.utils import parse_array_def
from nvflare.focs.sim.runner import AppRunner


class NPSwarm(Strategy):

    def __init__(self, initial_model, num_rounds=10):
        self.num_rounds = num_rounds
        self.initial_model = parse_array_def(initial_model)
        self.waiter = threading.Event()

    def execute(self, context: Context):
        # randomly pick a client to start
        start_client_idx = random.randint(0, len(context.clients) - 1)
        start_client = context.clients[start_client_idx]
        start_client.start(self.num_rounds, self.initial_model)
        self.waiter.wait()

    def notify_done(self, context: Context):
        print(f"[{context.callee}]: received DONE from client: {context.caller}")
        self.waiter.set()


class NPSwarmClient(ClientApp):

    def __init__(self, delta: float):
        super().__init__()
        self.delta = delta
        self.register_event_handler("final_model", self._accept_final_model)

    def train(self, weights, current_round, context: Context):
        print(f"[{context.callee}]: train asked by {context.caller}: {current_round=}")
        return weights + self.delta

    def sag(self, model, current_round, ctx: Context):
        results = all_clients(ctx, blocking=True).train(model, current_round)
        results = list(results.values())
        total = results[0]
        for i in range(1, len(results)):
            total += results[i]
        return total / len(results)

    def swarm_learn(self, num_rounds, model, current_round, context: Context):
        print(f"[{context.callee}]: swarm learn asked by {context.caller}: {num_rounds=} {current_round=} {model=}")
        new_model = self.sag(model, current_round, context)

        print(f"[{context.callee}]: trained model {new_model=}")
        if current_round == num_rounds - 1:
            # all done
            all_clients(context, blocking=False).fire_event("final_model", new_model)
            self.server.notify_done()
            return

        # determine next client
        next_round = current_round + 1
        next_client_idx = random.randint(0, len(self.clients) - 1)
        next_client = self.clients[next_client_idx]
        next_client.swarm_learn(num_rounds, new_model, next_round, blocking=False)

    def start(self, num_rounds, initial_model, context: Context):
        self.swarm_learn(num_rounds, initial_model, 0, context)

    def _accept_final_model(self, event_type: str, model, context: Context):
        # accept the final model
        # write model to disk
        print(f"[{context.callee}]: received event '{event_type}' from {context.caller}: {model}")


def main():

    runner = AppRunner(
        server_app=ServerApp(strategy=NPSwarm(initial_model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], num_rounds=5)),
        client_app=NPSwarmClient(delta=1.0),
        num_clients=3,
    )

    runner.run()


if __name__ == "__main__":
    main()
