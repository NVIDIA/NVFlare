import sys
from pathlib import Path
from typing import Any, Dict

from nvflare.app_opt.pt.recipes import FedAvgRecipe
from nvflare.client.config import ExchangeFormat
from nvflare.recipe import SimEnv

HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "src"
DATA_ARCHIVE = HERE / "data" / "cifar-10-python.tar.gz"


def create_recipe():
    source_dir = str(SOURCE_DIR)

    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

    from model import CIFAR10CNN
    from fedscs_aggregator import FedSCSAggregator
    from fedscs_controller import FedSCS

    class FedSCSRecipe(FedAvgRecipe):
        """FedSCS recipe using the custom FedSCS controller."""

        def _create_controller(
            self,
            persistor_id: str,
            model_params,
            model_aggregator,
        ):
            return FedSCS(
                num_clients=self.min_clients,
                num_rounds=self.num_rounds,
                persistor_id=persistor_id,
                model=model_params,
                save_filename=self.save_filename,
                aggregator=model_aggregator,
                stop_cond=self.stop_cond,
                patience=self.patience,
                task_name="train",
                exclude_vars=self.exclude_vars,
                aggregation_weights=self.aggregation_weights,
                memory_gc_rounds=self.server_memory_gc_rounds,
                enable_tensor_disk_offload=self.enable_tensor_disk_offload,
                **self._get_controller_kwargs(),
            )

    model = CIFAR10CNN()

    aggregator = FedSCSAggregator(
        eps=1e-12,
        max_update_norm=10.0,
    )

    recipe = FedSCSRecipe(
        name="fedscs_cifar10",
        model=model,
        min_clients=2,
        num_rounds=2,
        train_script=str(HERE / "client.py"),
        aggregator=aggregator,
        key_metric="accuracy",
        server_expected_format=ExchangeFormat.PYTORCH,
    )

    if not DATA_ARCHIVE.exists():
        raise FileNotFoundError(
            f"CIFAR-10 archive not found: {DATA_ARCHIVE}. "
            "Run prepare_data.sh first."
        )

    recipe._job.add_file_to_clients(
        str(DATA_ARCHIVE),
        dest_dir="data",
    )

    return recipe


def main():
    recipe = create_recipe()

    environment = SimEnv(
        num_clients=2,
        num_threads=2,
        workspace_root="/tmp/nvflare/fedscs",
    )

    run = recipe.execute(environment)

    print("Job status:", run.get_status())
    print("Results:", run.get_result(clean_up=False))


if __name__ == "__main__":
    main()
