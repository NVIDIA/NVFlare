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

from typing import List, Optional

from pydantic import BaseModel

from nvflare import FedJob
from nvflare.apis.dxo import DataKind
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.aggregators import InTimeAccumulateWeightedAggregator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.np.np_formatter import NPFormatter
from nvflare.app_common.np.np_model_locator import NPModelLocator
from nvflare.app_common.np.np_model_persistor import NPModelPersistor
from nvflare.app_common.np.np_trainer import NPTrainer
from nvflare.app_common.np.np_validator import NPValidator
from nvflare.app_common.shareablegenerators import FullModelShareableGenerator
from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner
from nvflare.recipe.spec import Recipe


# Internal — not part of the public API
class _FedAvgWithCSEValidator(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    name: str
    initial_model: Optional[list] = None
    min_clients: int
    num_rounds: int
    train_script: str
    train_args: str
    aggregator: Optional[Aggregator]
    aggregator_data_kind: Optional[DataKind]
    launch_external_process: bool = False
    command: str = "python3 -u"
    server_expected_format: ExchangeFormat = ExchangeFormat.NUMPY
    params_transfer_type: TransferType = TransferType.FULL
    cross_val_dir: str
    submit_model_timeout: int
    validation_timeout: int
    participating_clients: Optional[List[str]]
    client_model_dir: str
    client_model_name: str


class FedAvgWithCrossSiteEvalRecipe(Recipe):
    """A recipe for FedAvg training followed by Cross-Site Evaluation with NumPy in NVFlare.

    This recipe combines federated averaging (FedAvg) training with cross-site model evaluation
    in a single sequential workflow. After FedAvg training completes, the trained models are
    automatically evaluated across all client sites to create an all-to-all performance matrix.

    The recipe configures:
        - A federated job with optional initial model
        - Scatter-and-gather controller for FedAvg training
        - Weighted aggregator for combining client model updates
        - CrossSiteModelEval controller for evaluation (runs after training)
        - Model locator for finding trained models from persistor
        - Script runners for client-side training execution
        - Validators for client-side model evaluation
        - JSON generator for saving cross-validation results

    Args:
        name: Name of the federated learning job. Defaults to "fedavg_cse".
        initial_model: Initial model (as list or numpy array) to start federated training with.
            Lists are preferred for JSON serialization compatibility. If None,
            clients will start with their own local models. Defaults to None.
        min_clients: Minimum number of clients required to start training and evaluation.
        num_rounds: Number of federated training rounds to execute. Defaults to 2.
        train_script: Path to the training script that will be executed on each client.
        train_args: Command line arguments to pass to the training script. Defaults to "".
        aggregator: Aggregator for combining client updates. If None,
            uses InTimeAccumulateWeightedAggregator with aggregator_data_kind.
            Defaults to None.
        aggregator_data_kind: Data kind to use for the aggregator. Defaults to DataKind.WEIGHTS.
        launch_external_process: Whether to launch the script in external process. Defaults to False.
        command: If launch_external_process=True, command to run script (prepended to script).
            Defaults to "python3 -u".
        server_expected_format: What format to exchange the parameters between server and client.
            Defaults to ExchangeFormat.NUMPY.
        params_transfer_type: How to transfer the parameters. FULL means the whole model parameters
            are sent. DIFF means that only the difference is sent. Defaults to TransferType.FULL.
        cross_val_dir: Directory for cross-validation results. Defaults to "cross_site_val".
        submit_model_timeout: Timeout in seconds for submit_model task. Defaults to 600.
        validation_timeout: Timeout in seconds for validation task. Defaults to 6000.
        participating_clients: List of client names to participate. If None, all connected
            clients will participate. Defaults to None.
        client_model_dir: Directory where client models are saved. Defaults to "model".
        client_model_name: Name of the client model file. Defaults to "best_numpy.npy".

    Example:
        ```python
        recipe = FedAvgWithCrossSiteEvalRecipe(
            name="my_train_and_eval",
            min_clients=2,
            num_rounds=5,
            train_script="client.py",
            train_args="--epochs 1"
        )

        # Run in simulator
        from nvflare.recipe import SimEnv
        env = SimEnv(num_clients=2)
        run = recipe.execute(env)
        ```

    Note:
        This recipe runs two controllers sequentially:
        1. ScatterAndGather: Performs FedAvg training
        2. CrossSiteModelEval: Evaluates all models (server + clients) across all sites

        The trained models are automatically persisted by NPModelPersistor, and the
        CrossSiteModelEval controller uses NPModelLocator to find these models for evaluation.

        Results:
            - Training results: Available in the persistor's directory
            - Evaluation results: Saved as JSON in cross_val_dir, showing how each
              model performs on each client's dataset
    """

    def __init__(
        self,
        *,
        name: str = "fedavg_cse",
        initial_model: Optional[list] = None,
        min_clients: int,
        num_rounds: int = 2,
        train_script: str,
        train_args: str = "",
        aggregator: Optional[Aggregator] = None,
        aggregator_data_kind: Optional[DataKind] = DataKind.WEIGHTS,
        launch_external_process: bool = False,
        command: str = "python3 -u",
        server_expected_format: ExchangeFormat = ExchangeFormat.NUMPY,
        params_transfer_type: TransferType = TransferType.FULL,
        cross_val_dir: str = AppConstants.CROSS_VAL_DIR,
        submit_model_timeout: int = 600,
        validation_timeout: int = 6000,
        participating_clients: Optional[List[str]] = None,
        client_model_dir: str = "model",
        client_model_name: str = "best_numpy.npy",
    ):
        # Validate inputs internally
        v = _FedAvgWithCSEValidator(
            name=name,
            initial_model=initial_model,
            min_clients=min_clients,
            num_rounds=num_rounds,
            train_script=train_script,
            train_args=train_args,
            aggregator=aggregator,
            aggregator_data_kind=aggregator_data_kind,
            launch_external_process=launch_external_process,
            command=command,
            server_expected_format=server_expected_format,
            params_transfer_type=params_transfer_type,
            cross_val_dir=cross_val_dir,
            submit_model_timeout=submit_model_timeout,
            validation_timeout=validation_timeout,
            participating_clients=participating_clients,
            client_model_dir=client_model_dir,
            client_model_name=client_model_name,
        )

        self.name = v.name
        self.initial_model = v.initial_model
        self.min_clients = v.min_clients
        self.num_rounds = v.num_rounds
        self.train_script = v.train_script
        self.train_args = v.train_args
        self.aggregator = v.aggregator
        self.aggregator_data_kind = v.aggregator_data_kind
        self.launch_external_process = v.launch_external_process
        self.command = v.command
        self.server_expected_format: ExchangeFormat = v.server_expected_format
        self.params_transfer_type: TransferType = v.params_transfer_type
        self.cross_val_dir = v.cross_val_dir
        self.submit_model_timeout = v.submit_model_timeout
        self.validation_timeout = v.validation_timeout
        self.participating_clients = v.participating_clients
        self.client_model_dir = v.client_model_dir
        self.client_model_name = v.client_model_name

        # Create FedJob
        job = FedJob(name=self.name, min_clients=self.min_clients)

        # Server components for training
        if self.aggregator is None:
            self.aggregator = InTimeAccumulateWeightedAggregator(expected_data_kind=self.aggregator_data_kind)
        else:
            if not isinstance(self.aggregator, Aggregator):
                raise ValueError(f"Invalid aggregator type: {type(self.aggregator)}. Expected type: {Aggregator}")

        # Add persistor (required for both training and CSE)
        persistor_id = job.to_server(NPModelPersistor(initial_model=self.initial_model), id="persistor")

        # Add aggregator and shareable generator for training
        aggregator_id = job.to_server(self.aggregator, id="aggregator")
        shareable_generator = FullModelShareableGenerator()
        shareable_generator_id = job.to_server(shareable_generator, id="shareable_generator")

        # Workflow 1: Training (ScatterAndGather)
        controller = ScatterAndGather(
            min_clients=self.min_clients,
            num_rounds=self.num_rounds,
            wait_time_after_min_received=0,
            aggregator_id=aggregator_id,
            persistor_id=persistor_id,
            shareable_generator_id=shareable_generator_id,
            allow_empty_global_weights=self.initial_model is None,  # Allow empty weights if no initial model
        )
        job.to_server(controller)

        # Server components for cross-site evaluation
        # Model locator will find models from the persistor
        model_locator_id = job.to_server(NPModelLocator(), id="model_locator")
        formatter_id = job.to_server(NPFormatter(), id="formatter")
        job.to_server(ValidationJsonGenerator())

        # Workflow 2: Cross-Site Evaluation (runs after training)
        cse_controller = CrossSiteModelEval(
            model_locator_id=model_locator_id,
            formatter_id=formatter_id,
            cross_val_dir=self.cross_val_dir,
            submit_model_timeout=self.submit_model_timeout,
            validation_timeout=self.validation_timeout,
            participating_clients=self.participating_clients,
        )
        job.to_server(cse_controller)

        # Client components
        # Script runner for training
        executor = ScriptRunner(
            script=self.train_script,
            script_args=self.train_args,
            launch_external_process=self.launch_external_process,
            command=self.command,
            framework=FrameworkType.NUMPY,
            server_expected_format=self.server_expected_format,
            params_transfer_type=self.params_transfer_type,
        )
        job.to_clients(executor)

        # Trainer for submitting client models for CSE
        trainer = NPTrainer(
            train_task_name=AppConstants.TASK_TRAIN,
            submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL,
            model_name=self.client_model_name,
            model_dir=self.client_model_dir,
        )
        job.to_clients(trainer, tasks=[AppConstants.TASK_SUBMIT_MODEL])

        # Validator for evaluating models during CSE
        validator = NPValidator(
            validate_task_name=AppConstants.TASK_VALIDATION,
        )
        job.to_clients(validator, tasks=[AppConstants.TASK_VALIDATION])

        Recipe.__init__(self, job)
