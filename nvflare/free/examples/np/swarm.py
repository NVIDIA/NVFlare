import random
import threading

import numpy as np

from nvflare.free.api.app import ServerApp, ClientApp
from nvflare.free.api.ctx import Context
from nvflare.free.api.runner import AppRunner


class NPSwarmServer(ServerApp):

    def __init__(self, num_rounds=10):
        ServerApp.__init__(self)
        self.num_rounds = num_rounds
        self.initial_model = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        self.waiter = threading.Event()

    def run(self, **kwargs):
        # randomly pick a client to start
        start_client_idx = random.randint(0, len(self.clients)-1)
        start_client = self.clients[start_client_idx]
        start_client.start(self.num_rounds, self.initial_model)
        self.waiter.wait()

    def notify_done(self, context: Context, **kwargs):
        print(f"[{context.callee}]: received DONE from client: {context.caller}")
        self.waiter.set()


class NPSwarmClient(ClientApp):

    def __init__(self, delta: float):
        super().__init__()
        self.delta = delta

    def train(self, weights):
        return weights + self.delta

    def sag(self, model):
        results = self.group(self.clients, blocking=True).train(model)
        results = list(results.values())
        total = results[0]
        for i in range(1, len(results)):
            total += results[i]
        return total / len(results)

    def swarm_learn(self, num_rounds, model, current_round, context: Context):
        print(f"[{context.callee}]: swarm learn asked by {context.caller}: {num_rounds=} {current_round=} {model=}")
        new_model = self.sag(model)

        print(f"[{context.callee}]: trained model {new_model=}")
        if current_round == num_rounds - 1:
            # all done
            self.group(self.clients).accept_final_model(new_model, blocking=False)
            self.server.notify_done()
            return

        # determine next client
        next_round = current_round+1
        next_client_idx = random.randint(0, len(self.clients) - 1)
        next_client = self.clients[next_client_idx]
        next_client.swarm_learn(num_rounds, new_model, next_round, blocking=False)

    def start(self, num_rounds, initial_model, context: Context):
        self.swarm_learn(num_rounds, initial_model, 0, context)

    def accept_final_model(self, model, context: Context):
        # accept the final model
        # write model to disk
        print(f"[{context.callee}]: received final model from {context.caller}: {model}")


def main():

    server_app = NPSwarmServer(num_rounds=5)

    runner = AppRunner(
        server_app=server_app,
        client_app=NPSwarmClient(delta=1.0),
        num_clients=3
    )

    runner.run()


if __name__ == "__main__":
    main()
