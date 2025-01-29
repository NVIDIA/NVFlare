import pickle
import random
import matplotlib.pyplot as plt
from nvflare.job_config.api import FedJob

from nvdo.controllers import AlgorithmController
from nvdo.executors import ConsensusExecutor
from nvdo.types import Config
from nvdo.utils.config_generator import generate_random_network

class CustomConsensusExecutor(ConsensusExecutor):
    def __init__(self):
        super().__init__(initial_value=random.randint(0,10))


if __name__ == "__main__":
    # Create job
    job = FedJob(name="consensus")

    # generate random config
    num_clients = 6
    network, _ = generate_random_network(num_clients=num_clients)
    config = Config(network=network, extra={"iterations": 20})


    # send controller to server
    controller = AlgorithmController(config=config)
    job.to_server(controller)

    # Add clients
    for i in range(num_clients):
        executor = CustomConsensusExecutor()
        job.to(executor, f"site-{i + 1}")

    # run
    job.export_job("./tmp/job_configs")
    job.simulator_run("./tmp/runs/consensus")


    history = {
        f"site-{i + 1}": pickle.load(
            open(f"tmp/runs/consensus/site-{i + 1}/results.pkl", "rb")
        )
        for i in range(num_clients)
    }
    plt.figure()
    for i in range(num_clients):
        plt.plot(history[f"site-{i + 1}"], label=f"site-{i + 1}")
    plt.legend()
    plt.title("Evolution of local values")
    plt.show()
