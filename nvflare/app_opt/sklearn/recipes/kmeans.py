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

from pydantic import BaseModel

from nvflare import FedJob
from nvflare.app_common.aggregators import CollectAndAssembleAggregator
from nvflare.app_common.shareablegenerators import FullModelShareableGenerator
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.app_opt.sklearn.joblib_model_param_persistor import JoblibModelParamPersistor
from nvflare.app_opt.sklearn.kmeans_assembler import KMeansAssembler
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner
from nvflare.recipe.spec import Recipe


# Internal — not part of the public API
class _KMeansValidator(BaseModel):
    name: str
    min_clients: int
    num_rounds: int
    n_clusters: int
    train_script: str
    train_args: str
    launch_external_process: bool = False
    command: str = "python3 -u"


class KMeansFedAvgRecipe(Recipe):
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

    Example:
        ```python
        recipe = KMeansFedAvgRecipe(
            name="kmeans_iris",
            min_clients=3,
            num_rounds=5,
            n_clusters=3,
            train_script="src/kmeans_fl.py",
            train_args="--data_path /tmp/data/iris.csv --train_start 0 --train_end 50",
        )

        from nvflare.recipe import SimEnv
        env = SimEnv(num_clients=3)
        run = recipe.execute(env)
        print("Result:", run.get_result())
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
    ):
        # Validate inputs internally
        v = _KMeansValidator(
            name=name,
            min_clients=min_clients,
            num_rounds=num_rounds,
            n_clusters=n_clusters,
            train_script=train_script,
            train_args=train_args,
            launch_external_process=launch_external_process,
            command=command,
        )

        self.name = v.name
        self.min_clients = v.min_clients
        self.num_rounds = v.num_rounds
        self.n_clusters = v.n_clusters
        self.train_script = v.train_script
        self.train_args = v.train_args
        self.launch_external_process = v.launch_external_process
        self.command = v.command

        # Create FedJob
        job = FedJob(name=self.name, min_clients=self.min_clients)

        # Server components - K-Means specific
        persistor = JoblibModelParamPersistor(initial_params={"n_clusters": self.n_clusters})
        persistor_id = job.to_server(persistor, id="persistor")

        shareable_generator = FullModelShareableGenerator()
        shareable_generator_id = job.to_server(shareable_generator, id="shareable_generator")

        # K-Means uses custom assembler for mini-batch aggregation
        assembler = KMeansAssembler()
        assembler_id = job.to_server(assembler, id="kmeans_assembler")

        aggregator = CollectAndAssembleAggregator(assembler_id=assembler_id)
        aggregator_id = job.to_server(aggregator, id="aggregator")

        controller = ScatterAndGather(
            min_clients=self.min_clients,
            num_rounds=self.num_rounds,
            wait_time_after_min_received=0,
            aggregator_id=aggregator_id,
            persistor_id=persistor_id,
            shareable_generator_id=shareable_generator_id,
            train_task_name="train",
        )
        job.to_server(controller)

        # Client components
        executor = ScriptRunner(
            script=self.train_script,
            script_args=self.train_args,
            launch_external_process=self.launch_external_process,
            command=self.command,
            framework=FrameworkType.RAW,
            server_expected_format=ExchangeFormat.RAW,
            params_transfer_type=TransferType.FULL,
        )
        job.to_clients(executor)

        Recipe.__init__(self, job)


