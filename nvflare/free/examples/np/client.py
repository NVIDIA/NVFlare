import random

from nvflare.apis.signal import Signal
from nvflare.free.api.app import ClientApp
from nvflare.free.api.ctx import Context


class NPTrainer(ClientApp):

    def __init__(self, delta: float):
        ClientApp.__init__(self)
        self.delta = delta

    def train(self, r, weights, abort_signal: Signal, context: Context):
        if abort_signal.triggered:
            print("training aborted")
            return 0
        print(f"[{self.name}] called by {context.caller}: client {context.callee} trained round {r}")

        metric_receiver = self.server.get_target("metric_receiver")
        if metric_receiver:
            metric_receiver.accept_metric({"round": r, "y": 2})
        return weights + self.delta

    def evaluate(self, model, context: Context):
        print(f"[{self.name}] called by {context.caller}: client {context.callee} to evaluate")
        return random.random()
