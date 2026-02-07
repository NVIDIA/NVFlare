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

from typing import Any, List, Optional, Union

from pydantic import BaseModel

from nvflare.apis.dxo import DataKind
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.app_opt.he.intime_accumulate_model_aggregator import HEInTimeAccumulateWeightedAggregator
from nvflare.app_opt.he.model_decryptor import HEModelDecryptor
from nvflare.app_opt.he.model_encryptor import HEModelEncryptor
from nvflare.app_opt.he.model_serialize_filter import HEModelSerializeFilter
from nvflare.app_opt.he.model_shareable_generator import HEModelShareableGenerator
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.fuel.utils.constants import FrameworkType
from nvflare.job_config.defs import FilterType
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.recipe.spec import Recipe


# Internal — not part of the public API
class _FedAvgRecipeWithHEValidator(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    name: str
    model: Any
    initial_ckpt: Optional[str] = None
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
    encrypt_layers: Optional[Union[List[str], str]] = None
    server_memory_gc_rounds: int = 1


class FedAvgRecipeWithHE(Recipe):
    """A recipe for implementing Federated Averaging (FedAvg) with Homomorphic Encryption (HE) in NVFlare.

    FedAvg is a fundamental federated learning algorithm that aggregates model updates
    from multiple clients by computing a weighted average based on the amount of local
    training data. This recipe adds homomorphic encryption to preserve privacy during
    federated learning by allowing computations on encrypted data.

    The recipe configures:
    - A federated job with initial model (optional)
    - Scatter-and-gather controller for coordinating training rounds
    - HE-enabled weighted aggregator for combining encrypted client model updates
    - HE shareable generator for converting between Shareable and Learnable objects
    - HE model encryptor/decryptor filters on the client side
    - HE model serialization filter on the server side
    - Script runners for client-side training execution

    Args:
        name: Name of the federated learning job. Defaults to "fedavg".
        model: Initial model to start federated training with. Can be:
            - nn.Module instance
            - Dict config: {"class_path": "module.ClassName", "args": {"param": value}}
            - None: no initial model
        initial_ckpt: Path to a pre-trained checkpoint file. Can be:
            - Relative path: file will be bundled into the job's custom/ directory.
            - Absolute path: treated as a server-side path, used as-is at runtime.
            Note: PyTorch requires model when using initial_ckpt (for architecture).
        min_clients: Minimum number of clients required to start a training round.
        num_rounds: Number of federated training rounds to execute. Defaults to 2.
        train_script: Path to the training script that will be executed on each client.
        train_args: Command line arguments to pass to the training script.
        aggregator: Aggregator for combining client updates. If None,
            uses HEInTimeAccumulateWeightedAggregator with aggregator_data_kind.
        aggregator_data_kind: Data kind to use for the aggregator. Defaults to DataKind.WEIGHTS.
        launch_external_process (bool): Whether to launch the script in external process. Defaults to False.
        command (str): If launch_external_process=True, command to run script (prepended to script). Defaults to "python3".
        server_expected_format (str): What format to exchange the parameters between server and client.
        params_transfer_type (str): How to transfer the parameters. FULL means the whole model parameters are sent.
        DIFF means that only the difference is sent. Defaults to TransferType.FULL.
        encrypt_layers: if not specified (None), all layers are being encrypted;
                        if list of variable/layer names, only specified variables are encrypted;
                        if string containing regular expression (e.g. "conv"), only matched variables are
                        being encrypted.
        server_memory_gc_rounds: Run memory cleanup (gc.collect + malloc_trim) every N rounds on server.
            Set to 0 to disable. Defaults to 1 (every round).

    Example:
        ```python
        recipe = FedAvgRecipeWithHE(
            name="my_fedavg_he_job",
            model=pretrained_model,
            min_clients=2,
            num_rounds=10,
            train_script="client.py",
            train_args="--epochs 5 --batch_size 32"
        )
        ```

    Note:
        This recipe implements FedAvg with homomorphic encryption (HE) using TenSEAL library.
        HE allows computations to be performed on encrypted data, preserving client privacy.

        The following HE components are configured:
        - Server side: HEModelShareableGenerator, HEInTimeAccumulateWeightedAggregator, HEModelSerializeFilter
        - Client side: HEModelDecryptor (for incoming data), HEModelEncryptor (for outgoing results)

        Model updates are aggregated using weighted averaging based on the number of training
        samples provided by each client, with encryption/decryption handled transparently.

        If you want to use a custom aggregator, you can pass it in the aggregator parameter.
        The custom aggregator should support HE operations or be a subclass of HEInTimeAccumulateWeightedAggregator.
    """

    def __init__(
        self,
        *,
        name: str = "fedavg_he",
        model: Union[Any, dict[str, Any], None] = None,
        initial_ckpt: Optional[str] = None,
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
        encrypt_layers: Optional[Union[List[str], str]] = None,
        server_memory_gc_rounds: int = 1,
    ):
        # Validate inputs internally
        v = _FedAvgRecipeWithHEValidator(
            name=name,
            model=model,
            initial_ckpt=initial_ckpt,
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
            encrypt_layers=encrypt_layers,
            server_memory_gc_rounds=server_memory_gc_rounds,
        )

        self.name = v.name
        self.model = v.model
        self.initial_ckpt = v.initial_ckpt

        # Validate inputs using shared utilities
        from nvflare.recipe.utils import recipe_model_to_job_model, validate_ckpt, validate_dict_model_config

        validate_ckpt(self.initial_ckpt)
        validate_dict_model_config(self.model)
        if isinstance(self.model, dict):
            self.model = recipe_model_to_job_model(self.model)

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
        self.encrypt_layers: Optional[Union[List[str], str]] = v.encrypt_layers
        self.server_memory_gc_rounds = v.server_memory_gc_rounds

        # Create BaseFedJob without model first (model setup done manually below for HE)
        job = BaseFedJob(
            name=self.name,
            min_clients=self.min_clients,
        )

        # Create a persistor with HE serialization filter if initial model or checkpoint is provided
        if self.model is not None or self.initial_ckpt is not None:
            from nvflare.app_opt.pt.job_config.model import PTModel
            from nvflare.recipe.utils import prepare_initial_ckpt

            ckpt_path = prepare_initial_ckpt(self.initial_ckpt, job)
            model_persistor = PTFileModelPersistor(
                model=self.model,
                source_ckpt_file_full_name=ckpt_path,
                filter_id="model_serialize_filter",
            )
            pt_model = PTModel(model=self.model, persistor=model_persistor)
            job.comp_ids.update(job.to_server(pt_model))

        # Add HE model serialization filter (must be added before persistor uses it)
        if self.model is not None or self.initial_ckpt is not None:
            model_serialize_filter = HEModelSerializeFilter()
            job.to_server(model_serialize_filter, id="model_serialize_filter")

        # Define the HE-specific components for the server
        if self.aggregator is None:
            self.aggregator = HEInTimeAccumulateWeightedAggregator(
                expected_data_kind=self.aggregator_data_kind,
                weigh_by_local_iter=False,  # HE: weighting happens client-side in HEModelEncryptor (train task)
            )
        else:
            if not isinstance(self.aggregator, Aggregator):
                raise ValueError(f"Invalid aggregator type: {type(self.aggregator)}. Expected type: {Aggregator}")

        # Use HE-specific shareable generator
        shareable_generator = HEModelShareableGenerator()
        shareable_generator_id = job.to_server(shareable_generator, id="shareable_generator")
        aggregator_id = job.to_server(self.aggregator, id="aggregator")

        controller = ScatterAndGather(
            min_clients=self.min_clients,
            num_rounds=self.num_rounds,
            wait_time_after_min_received=0,
            aggregator_id=aggregator_id,
            persistor_id=(
                job.comp_ids.get("persistor_id", "")
                if (self.model is not None or self.initial_ckpt is not None)
                else ""
            ),
            shareable_generator_id=shareable_generator_id,
            memory_gc_rounds=self.server_memory_gc_rounds,
        )
        # Send the controller to the server
        job.to_server(controller)

        # Add clients with HE filters
        executor = ScriptRunner(
            script=self.train_script,
            script_args=self.train_args,
            launch_external_process=self.launch_external_process,
            command=self.command,
            framework=FrameworkType.PYTORCH,
            server_expected_format=self.server_expected_format,
            params_transfer_type=self.params_transfer_type,
        )
        job.to_clients(executor)

        # Add HE model decryptor as task data filter when training or validating (decrypt incoming data from server)
        job.to_clients(HEModelDecryptor(), tasks=["train", "validate"], filter_type=FilterType.TASK_DATA)

        # Add HE model encryptor as task result filter after training (encrypt outgoing results to server)
        job.to_clients(
            HEModelEncryptor(
                encrypt_layers=encrypt_layers,
                weigh_by_local_iter=True,  # Client-side weighting for HE (aggregator has weigh_by_local_iter=False)
            ),
            tasks=["train"],
            filter_type=FilterType.TASK_RESULT,
        )

        # Add HE model encryptor as task result filter when submitting model (encrypt outgoing results to server)
        job.to_clients(
            HEModelEncryptor(
                encrypt_layers=encrypt_layers, weigh_by_local_iter=False
            ),  # We don't need to weight by local iter when submitting model for evaluation
            tasks=["submit_model"],
            filter_type=FilterType.TASK_RESULT,
        )

        Recipe.__init__(self, job)
