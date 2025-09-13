import random

import numpy as np

from nvflare.free.api.app import ServerApp, ClientApp
from nvflare.free.api.runner import AppRunner


class NPCyclic(ServerApp):

    def __init__(self, num_rounds=10):
        ServerApp.__init__(self)
        self.num_rounds = num_rounds
        self.initial_model = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

    def run(self):
        current_model = self.initial_model
        for _ in range(self.num_rounds):
            current_model = self._do_one_round(current_model)
        print(f"final result: {current_model}")

    def _do_one_round(self, current_model):
        random.shuffle(self.clients)
        for c in self.clients:
            current_model = c.train(current_model)
            print(f"result from {c.name}: {current_model}")
        return current_model


class NPTrainer(ClientApp):

    def __init__(self, delta: float):
        ClientApp.__init__(self)
        self.delta = delta

    def train(self, weights, **kwargs):
        return weights + self.delta

    def evaluate(self, model):
        pass


def main():

    runner = AppRunner(
        server_app=NPCyclic(num_rounds=2),
        client_app=NPTrainer(delta=1.0),
        num_clients=2
    )

    runner.run()


if __name__ == "__main__":
    main()
