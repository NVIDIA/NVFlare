from nvflare.free.examples.np.server import NPFedAvgSequential
from nvflare.free.examples.np.client import NPTrainer
from nvflare.free.api.runner import AppRunner
from nvflare.free.api.ctx import Context


class MetricReceiver:

    def accept_metric(self, metrics: dict, context: Context):
        print(f"[{context.callee}] received metric report from {context.caller}: {metrics}")


def main():

    server_app = NPFedAvgSequential(num_rounds=2)
    server_app.add_target_object("metric_receiver", MetricReceiver())

    runner = AppRunner(
        server_app=server_app,
        client_app=NPTrainer(delta=1.0),
        num_clients=2,
    )

    runner.run()


if __name__ == "__main__":
    main()
