import numpy as np

from nvflare.free.api.app import ServerApp, ClientApp
from nvflare.free.api.runner import AppRunner
from nvflare.free.api.ctx import Context
from nvflare.apis.signal import Signal


class NPFedAvg(ServerApp):

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


class MetricReceiver:

    def accept_metric(self, metrics: dict, context: Context, **kwargs):
        print(f"[{context.callee}] received metric report from {context.caller}: {metrics}")


class NPTrainer(ClientApp):

    def __init__(self, delta: float):
        ClientApp.__init__(self)
        self.delta = delta

    def train(self, r, weights, abort_signal: Signal, context: Context, **kwargs):
        if abort_signal.triggered:
            print("training aborted")
            return 0
        print(f"[{self.name}] called by {context.caller}: client {context.callee} trained round {r}")

        self.server.metric_receiver.accept_metric({"round": r, "y": 2})
        return weights + self.delta


def main():

    server_app = NPFedAvg(num_rounds=2)
    server_app.add_target_object("metric_receiver", MetricReceiver())

    runner = AppRunner(
        server_app=server_app,
        client_app=NPTrainer(delta=1.0),
        num_clients=2
    )

    runner.run()


if __name__ == "__main__":
    main()
