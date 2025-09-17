import numpy as np
import random

from nvflare.free.api.app import ServerApp


class NPFedAvgSequential(ServerApp):

    def __init__(self, num_rounds=10):
        ServerApp.__init__(self)
        self.num_rounds = num_rounds
        self.initial_model = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

    def run(self, **kwargs):
        print(f"[{self.name}] Start training for {self.num_rounds} rounds")
        current_model = self.initial_model
        for i in range(self.num_rounds):
            current_model = self._do_one_round(i, current_model)

    def _do_one_round(self, r, current_model):
        total = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32)
        n = 0
        for c in self.clients:
            result = c.train(r, current_model)
            print(f"[{self.name}] round {r}: got result from client {c.name}: {result}")
            total += result
            n += 1
        return total / n


class NPFedAvgParallel(ServerApp):

    def __init__(self, num_rounds=10):
        ServerApp.__init__(self)
        self.num_rounds = num_rounds
        self.initial_model = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

    def run(self, **kwargs):
        print(f"[{self.name}] Start training for {self.num_rounds} rounds")
        current_model = self.initial_model
        for i in range(self.num_rounds):
            current_model = self._do_one_round(i, current_model)
            score = self._do_eval(current_model)
            print(f"[{self.name}]: eval score in round {i}: {score}")

    def _do_eval(self, model):
        results = self.group(self.clients).evaluate(model)
        total= 0.0
        for n, v in results.items():
            print(f"[{self.name}]: got eval result from client {n}: {v}")
            total += v
        return total / len(results)

    def _do_one_round(self, r, current_model):
        total = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32)
        results = self.group(self.clients).train(r, current_model)
        for n, v in results.items():
            print(f"[{self.name}] round {r}: got group result from client {n}: {v}")
            total += v
        return total / len(results)


class NPCyclic(ServerApp):

    def __init__(self, num_rounds=10):
        ServerApp.__init__(self)
        self.num_rounds = num_rounds
        self.initial_model = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

    def run(self):
        current_model = self.initial_model
        for current_round in range(self.num_rounds):
            current_model = self._do_one_round(current_round, current_model)
        print(f"final result: {current_model}")

    def _do_one_round(self, current_round, current_model):
        random.shuffle(self.clients)
        for c in self.clients:
            current_model = c.train(current_round, current_model)
            print(f"result from {c.name}: {current_model}")
        return current_model
