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

from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, conint

from nvflare import FedJob
from nvflare.app_common.shareablegenerators import FullModelShareableGenerator
from nvflare.app_common.workflows.cyclic_ctl import CyclicController
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.fuel.utils.constants import FrameworkType
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.recipe.spec import Recipe


# Internal — not part of the public API
class _CyclicValidator(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    name: str
    model: Any
    initial_ckpt: Optional[str] = None
    num_rounds: int
    min_clients: conint(ge=2)
    train_script: str
    train_args: str
    launch_external_process: bool = False
    command: str = "python3 -u"
    server_expected_format: ExchangeFormat = ExchangeFormat.NUMPY
    params_transfer_type: TransferType = TransferType.FULL
    framework: FrameworkType = FrameworkType.NUMPY
    server_memory_gc_rounds: int = 1


class CyclicRecipe(Recipe):
    """Cyclic federated learning recipe for sequential model training across clients.

    This recipe implements a cyclic (sequential) federated learning approach where clients
    train one after another in a round-robin fashion, rather than training in parallel.
    Each client receives the model from the previous client, trains on their local data,
    and passes the updated model to the next client.

    The recipe uses the following key components:
    - CyclicController: Manages the sequential workflow and client coordination on the server
    - FullModelShareableGenerator: Handles serialization/deserialization of models for transfer
    - ScriptRunner: Executes client training scripts with specified parameters
    - FedJob: Orchestrates the overall federated learning job configuration

    Args:
        name: Name identifier for the federated learning job. Defaults to "cyclic".
        model: Starting model object to begin training. Can be:
            - Model instance (nn.Module, tf.keras.Model, np.ndarray, etc.)
            - Dict config: {"class_path": "module.ClassName", "args": {"param": value}}
            - None: no initial model
        initial_ckpt: Path to a pre-trained checkpoint file. Can be:
            - Relative path: file will be bundled into the job's custom/ directory.
            - Absolute path: treated as a server-side path, used as-is at runtime.
        num_rounds: Number of complete training rounds to execute. Defaults to 2.
        min_clients: Minimum number of clients required to participate. Must be >= 2.
        train_script: Path to the client training script to execute.
        train_args: Additional command-line arguments to pass to the training script.
        launch_external_process: Whether to run training in a separate process. Defaults to False.
        command: Shell command to execute the training script. Defaults to "python3 -u".
        framework: ML framework type for compatibility. Defaults to FrameworkType.NUMPY.
        server_expected_format: Data exchange format between server and clients.
            Defaults to ExchangeFormat.NUMPY.
        params_transfer_type: Method for transferring model parameters.
            Defaults to TransferType.FULL.
        server_memory_gc_rounds: Run memory cleanup (gc.collect + malloc_trim) every N rounds on server.
            Set to 0 to disable. Defaults to 1 (every round).

    Raises:
        ValidationError: If min_clients < 2 or other parameter validation fails.

    Example:
        >>> recipe = CyclicRecipe(
        ...     name="my_cyclic_job",
        ...     model=my_model,
        ...     num_rounds=5,
        ...     min_clients=3,
        ...     train_script="client_train.py",
        ...     train_args="--epochs 10 --lr 0.01"
        ... )
        >>> # The recipe can then be submitted to the federated learning system
    """

    def __init__(
        self,
        *,
        name: str = "cyclic",
        model: Union[Any, Dict[str, Any], None] = None,
        initial_ckpt: Optional[str] = None,
        num_rounds: int = 2,
        min_clients: int = 2,
        train_script: str,
        train_args: str = "",
        launch_external_process: bool = False,
        command: str = "python3 -u",
        framework: FrameworkType = FrameworkType.NUMPY,
        server_expected_format: ExchangeFormat = ExchangeFormat.NUMPY,
        params_transfer_type: TransferType = TransferType.FULL,
        server_memory_gc_rounds: int = 1,
    ):
        # Validate inputs internally
        v = _CyclicValidator(
            name=name,
            model=model,
            initial_ckpt=initial_ckpt,
            num_rounds=num_rounds,
            min_clients=min_clients,
            train_script=train_script,
            train_args=train_args,
            launch_external_process=launch_external_process,
            command=command,
            framework=framework,
            server_expected_format=server_expected_format,
            params_transfer_type=params_transfer_type,
            server_memory_gc_rounds=server_memory_gc_rounds,
        )

        self.name = v.name
        self.model = v.model
        self.initial_ckpt = v.initial_ckpt

        # Validate inputs using shared utilities
        from nvflare.recipe.utils import recipe_model_to_job_model, validate_ckpt

        validate_ckpt(self.initial_ckpt)
        if isinstance(self.model, dict):
            self.model = recipe_model_to_job_model(self.model)

        self.num_rounds = v.num_rounds
        self.train_script = v.train_script
        self.train_args = v.train_args
        self.launch_external_process = v.launch_external_process
        self.command = v.command
        self.framework = v.framework
        self.server_expected_format: ExchangeFormat = v.server_expected_format
        self.params_transfer_type: TransferType = v.params_transfer_type
        self.server_memory_gc_rounds = v.server_memory_gc_rounds

        # Validate that we have at least one model source
        if self.model is None and self.initial_ckpt is None:
            raise ValueError(
                "Must provide either model or initial_ckpt. " "Cannot create a job without a model source."
            )

        job = FedJob(name=name, min_clients=v.min_clients)

        # Setup model persistor first - subclasses override for framework-specific handling
        persistor_id = self._setup_model_and_persistor(job)

        # Use returned persistor_id or default to "persistor"
        if not persistor_id:
            persistor_id = "persistor"

        # Define the controller workflow and send to server
        controller = CyclicController(
            num_rounds=num_rounds,
            task_assignment_timeout=10,
            persistor_id=persistor_id,
            shareable_generator_id="shareable_generator",
            task_name="train",
            task_check_period=0.5,
            memory_gc_rounds=self.server_memory_gc_rounds,
        )
        job.to(controller, "server")

        shareable_generator = FullModelShareableGenerator()
        job.to_server(shareable_generator, id="shareable_generator")

        executor = ScriptRunner(
            script=self.train_script,
            script_args=self.train_args,
            launch_external_process=self.launch_external_process,
            framework=self.framework,
            server_expected_format=self.server_expected_format,
            params_transfer_type=self.params_transfer_type,
        )
        job.to_clients(executor)

        super().__init__(job)

    def _setup_model_and_persistor(self, job) -> str:
        """Setup framework-specific model components and persistor.

        Handles PTModel/TFModel wrappers passed by framework-specific subclasses.

        Returns:
            str: The persistor_id to be used by the controller.
        """
        if self.model is None:
            return ""

        # Check if model is a model wrapper (PTModel, TFModel)
        if hasattr(self.model, "add_to_fed_job"):
            # It's a model wrapper - use its add_to_fed_job method
            result = job.to_server(self.model, id="persistor")
            return result["persistor_id"]

        # Unknown model type
        raise TypeError(
            f"Unsupported model type: {type(self.model).__name__}. "
            f"Use a framework-specific recipe (PTCyclicRecipe, TFCyclicRecipe, etc.) "
            f"or wrap your model in PTModel/TFModel."
        )
