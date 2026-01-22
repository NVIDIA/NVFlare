# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

from pydantic import BaseModel, conint

from nvflare.app_common.aggregators import CollectAndAssembleAggregator
from nvflare.app_opt.sklearn.joblib_model_param_persistor import JoblibModelParamPersistor
from nvflare.app_opt.sklearn.kmeans_assembler import KMeansAssembler
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.job_config.script_runner import FrameworkType
from nvflare.recipe.fedavg import FedAvgRecipe


# Internal — not part of the public API
class _KMeansValidator(BaseModel):
    # Allow custom types (e.g., Aggregator) in validation. Required by Pydantic v2.
    model_config = {"arbitrary_types_allowed": True}

    n_clusters: conint(gt=0)


class KMeansFedAvgRecipe(FedAvgRecipe):
    """A recipe for Federated K-Means Clustering with Scikit-learn.

    This recipe implements federated K-Means clustering using a mini-batch aggregation
    strategy. The aggregation follows the scheme defined in MiniBatchKMeans where each
    client's results are treated as a mini-batch for updating global centers.

    The recipe configures:
    - A federated job with initial n_clusters parameter
    - Scatter-and-gather controller for coordinating training rounds
    - Custom KMeansAssembler for mini-batch center aggregation
    - CollectAndAssembleAggregator for combining client updates
    - Script runners for client-side training execution

    Training Process:
    - Round 0: Each client generates initial centers using k-means++. The server
      collects all initial centers and performs one round of k-means to generate
      the initial global centers.
    - Subsequent rounds: Each client trains a local MiniBatchKMeans model starting
      from global centers. The server aggregates center and count information to
      update global centers using the mini-batch update rule.

    Args:
        name: Name of the federated learning job. Defaults to "kmeans_fedavg".
        min_clients: Minimum number of clients required to start a training round.
        num_rounds: Number of federated training rounds to execute. Defaults to 5.
        n_clusters: Number of clusters for K-Means. Defaults to 3.
        train_script: Path to the training script that will be executed on each client.
        train_args: Command line arguments to pass to the training script.
        launch_external_process: Whether to launch the script in external process. Defaults to False.
        command: If launch_external_process=True, command to run script (prepended to script).
            Defaults to "python3 -u".
        per_site_config: Per-site configuration for the federated learning job. Dictionary mapping
            site names to configuration dicts. If not provided, the same configuration will be used
            for all clients.

    Example:
        Basic usage with same config for all clients:

        ```python
        recipe = KMeansFedAvgRecipe(
            name="kmeans_iris",
            min_clients=3,
            num_rounds=5,
            n_clusters=3,
            train_script="src/kmeans_fl.py",
            train_args="--data_path /tmp/data/iris.csv",
        )

        from nvflare.recipe import SimEnv
        env = SimEnv(num_clients=3)
        run = recipe.execute(env)
        print("Result:", run.get_result())
        ```

        Per-site configuration:

        ```python
        from nvflare.app_opt.sklearn import KMeansFedAvgRecipe

        recipe = KMeansFedAvgRecipe(
            name="kmeans_iris",
            min_clients=3,
            num_rounds=5,
            n_clusters=3,
            train_script="src/kmeans_fl.py",
            per_site_config={
                "site-1": {"train_args": "--data_path /tmp/data/site1.csv --train_start 0 --train_end 50"},
                "site-2": {"train_args": "--data_path /tmp/data/site2.csv --train_start 50 --train_end 100"},
                "site-3": {"train_args": "--data_path /tmp/data/site3.csv --train_start 100 --train_end 150"},
            },
        )
        ```

    Note:
        This recipe uses a custom KMeansAssembler that implements the mini-batch
        K-Means aggregation logic. The assembler maintains historical center and
        count information across rounds for proper weighted averaging.
    """

    def __init__(
        self,
        *,
        name: str = "kmeans_fedavg",
        min_clients: int,
        num_rounds: int = 5,
        n_clusters: int = 3,
        train_script: str,
        train_args: str = "",
        launch_external_process: bool = False,
        command: str = "python3 -u",
        per_site_config: Optional[dict[str, dict]] = None,
    ):
        v = _KMeansValidator(n_clusters=n_clusters)
        self.n_clusters = v.n_clusters

        # Create KMeans-specific persistor
        persistor = JoblibModelParamPersistor(initial_params={"n_clusters": n_clusters})

        # K-Means uses custom assembler for mini-batch aggregation
        assembler = KMeansAssembler()
        assembler_id = "kmeans_assembler"
        aggregator = CollectAndAssembleAggregator(assembler_id=assembler_id)

        # Call the unified FedAvgRecipe with KMeans-specific settings
        super().__init__(
            name=name,
            min_clients=min_clients,
            num_rounds=num_rounds,
            train_script=train_script,
            train_args=train_args,
            aggregator=aggregator,
            launch_external_process=launch_external_process,
            command=command,
            framework=FrameworkType.RAW,
            server_expected_format=ExchangeFormat.RAW,
            params_transfer_type=TransferType.FULL,
            model_persistor=persistor,
            per_site_config=per_site_config,
        )
        self.job.to_server(assembler, id=assembler_id)
