import logging
import tarfile
from pathlib import Path

import torch
import nvflare.client as flare

from src.model import CIFAR10CNN
import train
from train import get_dataloaders, train_one_round, evaluate

logger = logging.getLogger(__name__)


def prepare_cifar10_data():
    data_dir = Path(train.__file__).resolve().parent / "data"
    archive = data_dir / "cifar-10-python.tar.gz"
    dataset_dir = data_dir / "cifar-10-batches-py"

    required_files = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
        "test_batch",
        "batches.meta",
    ]

    if all((dataset_dir / filename).is_file() for filename in required_files):
        logger.info("CIFAR-10 dataset already prepared")
        return

    if not archive.is_file():
        raise FileNotFoundError(
            f"CIFAR-10 archive not found: {archive}. "
            "Run prepare_data.sh first."
        )

    logger.info("Extracting CIFAR-10 dataset to %s", data_dir)

    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(data_dir)

    missing_files = [
        filename
        for filename in required_files
        if not (dataset_dir / filename).is_file()
    ]

    if missing_files:
        raise RuntimeError(
            f"CIFAR-10 extraction incomplete. "
            f"Missing files: {missing_files}"
        )

    logger.info("CIFAR-10 dataset prepared")


def load_model_parameters(model, params):
    state_dict = model.state_dict()

    for key in state_dict:
        if key not in params:
            raise KeyError(f"Missing model parameter: {key}")

        value = params[key]

        if not torch.is_tensor(value):
            value = torch.as_tensor(value)

        state_dict[key] = value.to(
            device=state_dict[key].device,
            dtype=state_dict[key].dtype,
        )

    model.load_state_dict(state_dict)


def get_model_parameters(model):
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    logger.info("Starting FedSCS NVFlare client")
    logger.info("Device: %s", device)

    flare.init()

    try:
        site_name = flare.get_site_name()
        logger.info("Site: %s", site_name)

        prepare_cifar10_data()

        train_loader, test_loader = get_dataloaders(
            site_name,
            batch_size=128,
        )

        while flare.is_running():
            input_model = flare.receive()

            if input_model is None:
                logger.info("No more model received. Exiting.")
                break

            logger.info(
                "Received global model for round %s",
                input_model.current_round,
            )

            model = CIFAR10CNN().to(device)

            if input_model.params is not None:
                load_model_parameters(
                    model,
                    input_model.params,
                )

            train_one_round(
                model=model,
                train_loader=train_loader,
                device=device,
                epochs=1,
                learning_rate=0.001,
            )

            accuracy = evaluate(
                model=model,
                test_loader=test_loader,
                device=device,
            )

            output_model = flare.FLModel(
                params=get_model_parameters(model),
                metrics={"accuracy": accuracy},
                start_round=input_model.start_round,
                current_round=input_model.current_round,
            )

            logger.info(
                "Sending local model for round %s",
                input_model.current_round,
            )

            flare.send(output_model)

    finally:
        flare.shutdown()

    logger.info("FedSCS client finished")


if __name__ == "__main__":
    main()
