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

from typing import Any, Optional, Union

from pydantic import BaseModel

from nvflare.apis.dxo import DataKind
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.aggregators import InTimeAccumulateWeightedAggregator
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.app_opt.pt import PTFileModelPersistor
from nvflare.app_opt.pt.fedopt import PTFedOptModelShareableGenerator
from nvflare.app_opt.pt.file_model_locator import PTFileModelLocator
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.fuel.utils.constants import FrameworkType
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.recipe.spec import Recipe


# Internal — not part of the public API
class _FedOptValidator(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    name: str
    model: Any
    initial_ckpt: Optional[str] = None
    min_clients: int
    num_rounds: int
    train_script: str
    train_args: str
    aggregator: Optional[Aggregator]
    launch_external_process: bool = False
    command: str = "python3 -u"
    server_expected_format: ExchangeFormat = ExchangeFormat.NUMPY
    device: Optional[str] = None
    server_memory_gc_rounds: int = 1


class FedOptRecipe(Recipe):
    """A recipe for implementing Federated Optimization (FedOpt) in NVFlare.

    FedOpt is a federated learning algorithm that optimizes the global model using a server-side optimizer and learning rate scheduler.
    After each round, the global model is updated using the specified optimizer and learning rate scheduler.
    The algorithm is proposed in Reddi et al. "Adaptive Federated Optimization." arXiv preprint arXiv:2003.00295 (2020).

    Note: FedOpt is only implemented for params_transfer_type == TransferType.DIFF and DataKind.WEIGHT_DIFF in the aggregator.

    Args:
        name: Name of the federated learning job. Defaults to "fedopt".
        model: Initial model to start federated training with (REQUIRED). Can be:
            - nn.Module instance
            - Dict config: {"path": "module.ClassName", "args": {"param": value}}
            Note: FedOpt requires a model for the server-side optimizer to work.
        initial_ckpt: Absolute path to a pre-trained checkpoint file. The file may not
            exist locally as it could be on the server. Used to load initial weights.
            Note: PyTorch requires model when using initial_ckpt (for architecture).
        min_clients: Minimum number of clients required to start a training round.
        num_rounds: Number of federated training rounds to execute. Defaults to 2.
        train_script: Path to the training script that will be executed on each client.
        train_args: Command line arguments to pass to the training script.
        aggregator: Aggregator for combining client updates. If None,
            uses InTimeAccumulateWeightedAggregator with expected_data_kind=DataKind.WEIGHT_DIFF.
        launch_external_process (bool): Whether to launch the script in external process. Defaults to False.
        command (str): If launch_external_process=True, command to run script (prepended to script). Defaults to "python3".
        server_expected_format (str): What format to exchange the parameters between server and client.
        source_model (str): ID of the source model component. Defaults to "model".
        optimizer_args (dict): Configuration for server-side optimizer with keys:
            - path: Path to optimizer class (e.g., "torch.optim.SGD")
            - args: Dictionary of optimizer arguments (e.g., {"lr": 1.0, "momentum": 0.6})
            - config_type: Type of configuration, typically "dict"
        lr_scheduler_args (dict): Optional configuration for learning rate scheduler with keys:
            - path: Path to scheduler class (e.g., "torch.optim.lr_scheduler.CosineAnnealingLR")
            - args: Dictionary of scheduler arguments (e.g., {"T_max": 100, "eta_min": 0.9})
            - config_type: Type of configuration, typically "dict"
        device (str): Device to use for server-side optimization, e.g. "cpu" or "cuda:0".
            Defaults to None; will default to cuda if available and no device is specified.
        server_memory_gc_rounds: Run memory cleanup (gc.collect + malloc_trim) every N rounds on server.
            Set to 0 to disable. Defaults to 1 (every round).

    Example:
        ```python
        recipe = FedOptRecipe(
            name="my_fedopt_job",
            model=pretrained_model,
            min_clients=2,
            num_rounds=10,
            train_script="client.py",
            train_args="--epochs 5 --batch_size 32",
            device="cpu",
            source_model="model",
            optimizer_args={
                "path": "torch.optim.SGD",
                "args": {"lr": 1.0, "momentum": 0.6},
                "config_type": "dict"
            },
            lr_scheduler_args={
                "path": "torch.optim.lr_scheduler.CosineAnnealingLR",
                "args": {"T_max": "{num_rounds}", "eta_min": 0.9},
                "config_type": "dict"
            }
        )
        ```

    """

    def __init__(
        self,
        *,
        name: str = "fedopt",
        model: Union[Any, dict[str, Any], None] = None,
        initial_ckpt: Optional[str] = None,
        min_clients: int,
        num_rounds: int = 2,
        train_script: str,
        train_args: str = "",
        aggregator: Optional[Aggregator] = None,
        launch_external_process: bool = False,
        command: str = "python3 -u",
        server_expected_format: ExchangeFormat = ExchangeFormat.NUMPY,
        device: Optional[str] = None,
        source_model: str = "model",
        optimizer_args: Optional[dict] = None,
        lr_scheduler_args: Optional[dict] = None,
        server_memory_gc_rounds: int = 1,
    ):
        # Validate inputs internally
        v = _FedOptValidator(
            name=name,
            model=model,
            initial_ckpt=initial_ckpt,
            min_clients=min_clients,
            num_rounds=num_rounds,
            train_script=train_script,
            train_args=train_args,
            aggregator=aggregator,
            launch_external_process=launch_external_process,
            command=command,
            server_expected_format=server_expected_format,
            device=device,
            server_memory_gc_rounds=server_memory_gc_rounds,
        )

        self.name = v.name
        self.model = v.model
        self.initial_ckpt = v.initial_ckpt

        # Validate inputs using shared utilities
        from nvflare.recipe.utils import validate_dict_model_config, validate_initial_ckpt

        validate_initial_ckpt(self.initial_ckpt)
        validate_dict_model_config(self.model)

        self.min_clients = v.min_clients
        self.num_rounds = v.num_rounds
        self.train_script = v.train_script
        self.train_args = v.train_args
        self.aggregator = v.aggregator
        self.launch_external_process = v.launch_external_process
        self.command = v.command
        self.server_expected_format: ExchangeFormat = v.server_expected_format
        self.device = device
        self.source_model = source_model
        self.optimizer_args = optimizer_args
        self.lr_scheduler_args = lr_scheduler_args
        self.server_memory_gc_rounds = v.server_memory_gc_rounds

        # Replace {num_rounds} placeholder if present in lr_scheduler_args
        processed_lr_scheduler_args = None
        if self.lr_scheduler_args is not None:
            processed_lr_scheduler_args = self.lr_scheduler_args.copy()
            if "args" in processed_lr_scheduler_args:
                lr_args = processed_lr_scheduler_args["args"].copy()
                if "T_max" in lr_args and lr_args["T_max"] == "{num_rounds}":
                    lr_args["T_max"] = self.num_rounds
                processed_lr_scheduler_args["args"] = lr_args

        # Create BaseFedJob with initial model
        job = BaseFedJob(
            initial_model=None,
            name=self.name,
            min_clients=self.min_clients,
        )

        # FedOpt requires a model (either model or initial_ckpt must be provided)
        # The PTFedOptModelShareableGenerator needs source_model to exist
        if self.model is None:
            raise ValueError(
                "FedOpt requires model. Provide either:\n"
                "  - nn.Module instance\n"
                "  - Dict config: {'path': 'module.ClassName', 'args': {...}}\n"
                "Note: initial_ckpt alone is not sufficient for PyTorch (model architecture needed)."
            )

        # Handle dict config: instantiate model before registering as component
        # PTFileModelPersistor expects component ID to resolve to nn.Module, not dict
        model_to_register = self.model
        if isinstance(self.model, dict):
            from nvflare.fuel.utils.class_utils import instantiate_class

            class_path = self.model.get("path")
            class_args = self.model.get("args", {})
            try:
                model_to_register = instantiate_class(class_path, class_args)
            except Exception as e:
                raise RuntimeError(f"Failed to instantiate model from dict config: {e}")

        # Add initial model as a separate component
        job.to_server(model_to_register, id=self.source_model)

        # Add the persisted model to the job with checkpoint support
        from nvflare.recipe.utils import prepare_initial_ckpt

        ckpt_path = prepare_initial_ckpt(self.initial_ckpt, job)
        persistor = PTFileModelPersistor(
            model=self.source_model,
            source_ckpt_file_full_name=ckpt_path,
        )
        persistor_id = job.to_server(persistor, id="persistor")

        locator = PTFileModelLocator(pt_persistor_id=persistor_id)
        job.to_server(locator, id="locator")

        # Define the controller and send to server
        if self.aggregator is None:
            self.aggregator = InTimeAccumulateWeightedAggregator(
                expected_data_kind=DataKind.WEIGHT_DIFF
            )  # FedOpt only supports DataKind.WEIGHT_DIFF
        else:
            if not isinstance(self.aggregator, Aggregator):
                raise ValueError(f"Invalid aggregator type: {type(self.aggregator)}. Expected type: {Aggregator}")

        # Define the shareable generator with fedopt parameters
        shareable_generator = PTFedOptModelShareableGenerator(
            optimizer_args=self.optimizer_args,
            lr_scheduler_args=processed_lr_scheduler_args,
            source_model=self.source_model,
            device=self.device,
        )
        shareable_generator_id = job.to_server(shareable_generator, id="shareable_generator")
        aggregator_id = job.to_server(self.aggregator, id="aggregator")

        controller = ScatterAndGather(
            min_clients=self.min_clients,
            num_rounds=self.num_rounds,
            wait_time_after_min_received=0,
            aggregator_id=aggregator_id,
            persistor_id="persistor",
            shareable_generator_id=shareable_generator_id,
            memory_gc_rounds=self.server_memory_gc_rounds,
        )
        # Send the controller to the server
        job.to_server(controller)

        # Add clients
        executor = ScriptRunner(
            script=self.train_script,
            script_args=self.train_args,
            launch_external_process=self.launch_external_process,
            command=self.command,
            framework=FrameworkType.PYTORCH,
            server_expected_format=self.server_expected_format,
            params_transfer_type=TransferType.DIFF,
        )
        job.to_clients(executor)

        Recipe.__init__(self, job)
