from nvflare.free.examples.np.server import NPCyclic
from nvflare.free.examples.np.client import NPTrainer
from nvflare.free.api.runner import AppRunner


def main():

    runner = AppRunner(
        server_app=NPCyclic(num_rounds=2),
        client_app=NPTrainer(delta=1.0),
        num_clients=2
    )

    runner.run()


if __name__ == "__main__":
    main()
