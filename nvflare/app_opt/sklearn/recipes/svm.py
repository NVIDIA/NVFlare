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

from typing import Literal, Optional

from pydantic import BaseModel

from nvflare.app_common.aggregators import CollectAndAssembleAggregator
from nvflare.app_opt.sklearn.joblib_model_param_persistor import JoblibModelParamPersistor
from nvflare.app_opt.sklearn.svm_assembler import SVMAssembler
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.job_config.script_runner import FrameworkType
from nvflare.recipe.fedavg import FedAvgRecipe


# Internal — not part of the public API
class _SVMValidator(BaseModel):
    # Allow custom types (e.g., Aggregator) in validation. Required by Pydantic v2.
    model_config = {"arbitrary_types_allowed": True}

    kernel: Literal["linear", "poly", "rbf", "sigmoid"]


class SVMFedAvgRecipe(FedAvgRecipe):
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
        recipe = SVMFedAvgRecipe(
            name="svm_cancer",
            min_clients=3,
            kernel="rbf",
            train_script="client.py",
            train_args="--data_path /tmp/data/cancer.csv",
        )

        from nvflare.recipe import SimEnv
        env = SimEnv(num_clients=3)
        run = recipe.execute(env)
        print("Result:", run.get_result())
        ```

        Per-site configuration:

        ```python
        from nvflare.app_opt.sklearn import SVMFedAvgRecipe

        recipe = SVMFedAvgRecipe(
            name="svm_cancer",
            min_clients=3,
            kernel="rbf",
            train_script="client.py",
            per_site_config={
                "site-1": {"train_args": "--data_path /tmp/data/site1.csv --train_start 0 --train_end 100"},
                "site-2": {"train_args": "--data_path /tmp/data/site2.csv --train_start 100 --train_end 200"},
                "site-3": {"train_args": "--data_path /tmp/data/site3.csv --train_start 200 --train_end 300"},
            },
        )
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
        kernel: Literal["linear", "poly", "rbf", "sigmoid"] = "rbf",
        train_script: str,
        train_args: str = "",
        launch_external_process: bool = False,
        command: str = "python3 -u",
        per_site_config: Optional[dict[str, dict]] = None,
    ):
        v = _SVMValidator(kernel=kernel)
        self.kernel = v.kernel

        # Create SVM-specific persistor
        persistor = JoblibModelParamPersistor(initial_params={"kernel": self.kernel})

        # Create SVM-specific aggregator with assembler
        assembler = SVMAssembler(kernel=self.kernel)
        assembler_id = "svm_assembler"
        aggregator = CollectAndAssembleAggregator(assembler_id=assembler_id)

        # Call the unified FedAvgRecipe with SVM-specific settings
        # Note: SVM only needs 2 rounds (round 0 for training, round 1 for validation)
        super().__init__(
            name=name,
            min_clients=min_clients,
            num_rounds=2,  # Fixed for SVM: training + validation
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
