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

from typing import Dict, Literal, Union

from pydantic import BaseModel

from nvflare import FedJob
from nvflare.app_common.aggregators import CollectAndAssembleAggregator
from nvflare.app_common.shareablegenerators import FullModelShareableGenerator
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.app_opt.sklearn.joblib_model_param_persistor import JoblibModelParamPersistor
from nvflare.app_opt.sklearn.svm_assembler import SVMAssembler
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner
from nvflare.recipe.spec import Recipe


# Internal — not part of the public API
class _SVMValidator(BaseModel):
    # Allow custom types (e.g., Aggregator) in validation. Required by Pydantic v2.
    model_config = {"arbitrary_types_allowed": True}

    name: str
    min_clients: int
    kernel: Literal["linear", "poly", "rbf", "sigmoid"]
    train_script: str
    train_args: Union[str, Dict[str, str]]
    backend: Literal["sklearn", "cuml"] = "sklearn"
    launch_external_process: bool = False
    command: str = "python3 -u"


class SVMFedAvgRecipe(Recipe):
    """A recipe for Federated SVM with Scikit-learn.

    This recipe implements federated SVM training using support vector aggregation.
    Unlike iterative algorithms, SVM training only requires one round:
    - Round 0: Each client trains a local SVM and sends their support vectors
    - Server aggregates all support vectors and trains a global SVM
    - Round 1: Clients validate using the global support vectors

    The recipe configures:
    - A federated job with kernel parameter
    - Scatter-and-gather controller (2 rounds)
    - Custom SVMAssembler for support vector aggregation
    - CollectAndAssembleAggregator for combining client updates
    - Script runners for client-side training execution

    Training Process:
    - Round 0 (Training): Each client trains a local SVM on their data and extracts
      support vectors. The server collects all support vectors, trains a global SVM,
      and extracts the global support vectors.
    - Round 1 (Validation): Each client validates using the global support vectors.

    Args:
        name: Name of the federated learning job. Defaults to "svm_fedavg".
        min_clients: Minimum number of clients required to start a training round.
        kernel: Kernel type for SVM. Options: 'linear', 'poly', 'rbf', 'sigmoid'.
            Defaults to 'rbf'.
        train_script: Path to the training script that will be executed on each client.
        train_args: Command line arguments to pass to the training script. Can be:
            - str: Same arguments for all clients (uses job.to_clients)
            - dict[str, str]: Per-client arguments mapping site names to args (uses job.to per site)
        backend: Backend library to use ('sklearn' or 'cuml'). Defaults to 'sklearn'.
        launch_external_process: Whether to launch the script in external process. Defaults to False.
        command: If launch_external_process=True, command to run script (prepended to script).
            Defaults to "python3 -u".

    Example:
        ```python
        recipe = SVMFedAvgRecipe(
            name="svm_cancer",
            min_clients=3,
            kernel="rbf",
            train_script="client.py",
            train_args="--data_path /tmp/data/cancer.csv --train_start 0 --train_end 100",
        )

        from nvflare.recipe import SimEnv
        env = SimEnv(num_clients=3)
        run = recipe.execute(env)
        print("Result:", run.get_result())
        ```

    Note:
        This recipe uses a custom SVMAssembler that implements support vector
        aggregation. The training only requires one round since SVM is not an
        iterative algorithm in the federated setting. A second round is included
        for validation purposes.
    """

    def __init__(
        self,
        *,
        name: str = "svm_fedavg",
        min_clients: int,
        kernel: str = "rbf",
        train_script: str,
        train_args: Union[str, Dict[str, str]] = "",
        backend: str = "sklearn",
        launch_external_process: bool = False,
        command: str = "python3 -u",
    ):
        # Validate inputs internally
        v = _SVMValidator(
            name=name,
            min_clients=min_clients,
            kernel=kernel,
            train_script=train_script,
            train_args=train_args,
            backend=backend,
            launch_external_process=launch_external_process,
            command=command,
        )

        self.name = v.name
        self.min_clients = v.min_clients
        self.kernel = v.kernel
        self.train_script = v.train_script
        self.train_args = v.train_args
        self.backend = v.backend
        self.launch_external_process = v.launch_external_process
        self.command = v.command

        # Create FedJob
        job = FedJob(name=self.name, min_clients=self.min_clients)

        # Server components - SVM specific
        persistor = JoblibModelParamPersistor(initial_params={"kernel": self.kernel})
        persistor_id = job.to_server(persistor, id="persistor")

        shareable_generator = FullModelShareableGenerator()
        shareable_generator_id = job.to_server(shareable_generator, id="shareable_generator")

        # SVM uses custom assembler for support vector aggregation
        assembler = SVMAssembler(kernel=self.kernel)
        assembler_id = job.to_server(assembler, id="svm_assembler")

        aggregator = CollectAndAssembleAggregator(assembler_id=assembler_id)
        aggregator_id = job.to_server(aggregator, id="aggregator")

        # SVM only needs 2 rounds: round 0 for training, round 1 for validation
        controller = ScatterAndGather(
            min_clients=self.min_clients,
            num_rounds=2,
            wait_time_after_min_received=0,
            aggregator_id=aggregator_id,
            persistor_id=persistor_id,
            shareable_generator_id=shareable_generator_id,
            train_task_name="train",
        )
        job.to_server(controller)

        # Client components
        if isinstance(self.train_args, dict):
            # Per-client configuration: add executor for each client with their specific args
            for site_name, site_args in self.train_args.items():
                executor = ScriptRunner(
                    script=self.train_script,
                    script_args=site_args,
                    launch_external_process=self.launch_external_process,
                    command=self.command,
                    framework=FrameworkType.RAW,
                    server_expected_format=ExchangeFormat.RAW,
                    params_transfer_type=TransferType.FULL,
                )
                job.to(executor, site_name)
        else:
            # Unified configuration: same args for all clients
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
