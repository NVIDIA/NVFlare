import torch
from config import ITERATIONS, NUM_CLIENTS, STEPSIZE
from nvflare.job_config.api import FedJob
from utils import NeuralNetwork, get_dataloaders, plot_results

from nvdo.controllers import AlgorithmController
from nvdo.executors import DGDExecutor
from nvdo.types import Config
from nvdo.utils.config_generator import generate_random_network


class CustomDGDExecutor(DGDExecutor):
    def __init__(self, data_chunk: int | None = None):
        self._data_chunk = data_chunk
        train_dataloader, test_dataloader = get_dataloaders(data_chunk)
        super().__init__(
            model=NeuralNetwork(),
            loss=torch.nn.CrossEntropyLoss(),
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
        )


if __name__ == "__main__":
    # Create job
    job_name = "dgd"
    job = FedJob(name=job_name)

    # generate random config
    network, _ = generate_random_network(num_clients=NUM_CLIENTS)
    config = Config(
        network=network, extra={"iterations": ITERATIONS, "stepsize": STEPSIZE}
    )

    # send controller to server
    controller = AlgorithmController(config=config)
    job.to_server(controller)

    # Add clients
    for i in range(NUM_CLIENTS):
        executor = CustomDGDExecutor(data_chunk=i)
        job.to(executor, f"site-{i + 1}")

    # run
    job.export_job("./tmp/job_configs")
    job.simulator_run(f"./tmp/runs/{job_name}")

    # plot and save results
    plot_results(job_name, NUM_CLIENTS)
