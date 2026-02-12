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

from nvflare.app_opt.tf.job_config.model import TFModel
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.fuel.utils.constants import FrameworkType
from nvflare.recipe.cyclic import CyclicRecipe as BaseCyclicRecipe


class CyclicRecipe(BaseCyclicRecipe):
    """TensorFlow-specific Cyclic federated learning recipe.

    Args:
        name: Name identifier for the federated learning job. Defaults to "cyclic".
        model: Starting model object to begin training. Can be:
            - tf.keras.Model instance
            - Dict config: {"class_path": "module.ClassName", "args": {"param": value}}
            - TFModel instance (already wrapped)
            - None: no initial model
        initial_ckpt: Path to a pre-trained checkpoint file (.h5, .keras, or SavedModel dir). Can be:
            - Relative path: file will be bundled into the job's custom/ directory.
            - Absolute path: treated as a server-side path, used as-is at runtime.
            Note: TensorFlow can load full models from .h5/SavedModel without model.
        num_rounds: Number of complete training rounds to execute. Defaults to 2.
        min_clients: Minimum number of clients required to participate. Must be >= 2.
        train_script: Path to the client training script to execute.
        train_args: Additional command-line arguments to pass to the training script.
        launch_external_process: Whether to run training in a separate process. Defaults to False.
        command: Shell command to execute the training script. Defaults to "python3 -u".
        framework: ML framework type for compatibility. Defaults to FrameworkType.TENSORFLOW.
        server_expected_format: Data exchange format between server and clients.
        params_transfer_type: Method for transferring model parameters.
        server_memory_gc_rounds: Run memory cleanup every N rounds on server. Defaults to 1.
    """

    def __init__(
        self,
        *,
        name: str = "cyclic",
        model: Union[Any, dict[str, Any], None] = None,
        initial_ckpt: Optional[str] = None,
        num_rounds: int = 2,
        min_clients: int = 2,
        train_script: str,
        train_args: str = "",
        launch_external_process: bool = False,
        command: str = "python3 -u",
        framework: FrameworkType = FrameworkType.TENSORFLOW,
        server_expected_format: ExchangeFormat = ExchangeFormat.NUMPY,
        params_transfer_type: TransferType = TransferType.FULL,
        server_memory_gc_rounds: int = 1,
    ):
        # Validate initial_ckpt early (base class won't see it since we pass None)
        from nvflare.recipe.utils import validate_ckpt

        validate_ckpt(initial_ckpt)

        # Store initial_ckpt for _setup_model_and_persistor; wrap model with TFModel there
        self._tf_initial_ckpt = initial_ckpt
        if model is None and initial_ckpt is None:
            model_to_pass = None
        elif isinstance(model, TFModel):
            model_to_pass = model
            self._tf_initial_ckpt = None  # Already handled by TFModel wrapper
        else:
            # Don't wrap yet — prepare_initial_ckpt needs the job which doesn't exist yet
            model_to_pass = model

        super().__init__(
            name=name,
            model=model_to_pass,
            initial_ckpt=None,  # Handled in _setup_model_and_persistor
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

    def _setup_model_and_persistor(self, job) -> str:
        """Override to handle TensorFlow-specific model setup with relative ckpt support."""
        if self.model is None and self._tf_initial_ckpt is None:
            return ""

        # If model is already a TFModel wrapper (user passed TFModel directly), use as-is
        if hasattr(self.model, "add_to_fed_job"):
            result = job.to_server(self.model, id="persistor")
            return result["persistor_id"]

        from nvflare.recipe.utils import prepare_initial_ckpt

        ckpt_path = prepare_initial_ckpt(self._tf_initial_ckpt, job)
        tf_model = TFModel(model=self.model, initial_ckpt=ckpt_path)
        result = job.to_server(tf_model, id="persistor")
        return result["persistor_id"]
