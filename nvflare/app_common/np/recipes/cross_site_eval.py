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

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from nvflare import FedJob
from nvflare.app_common.app_constant import AppConstants, DefaultCheckpointFileName
from nvflare.app_common.np.np_formatter import NPFormatter
from nvflare.app_common.np.np_model_locator import NPModelLocator
from nvflare.app_common.np.np_model_persistor import NPModelPersistor
from nvflare.app_common.np.np_trainer import NPTrainer
from nvflare.app_common.np.np_validator import NPValidator
from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
from nvflare.app_common.workflows.cross_site_eval import CrossSiteEval
from nvflare.recipe.spec import Recipe


# Internal — not part of the public API
class _CrossSiteEvalValidator(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    name: str
    min_clients: int
    initial_model: Optional[Any]
    model_locator_config: Optional[Dict[str, Any]]
    server_models: List[str]
    cross_val_dir: str
    submit_model_timeout: int
    validation_timeout: int
    participating_clients: Optional[List[str]]
    client_model_dir: str
    client_model_name: str


class NumpyCrossSiteEvalRecipe(Recipe):
    """A recipe for implementing Cross-Site Evaluation with NumPy in NVFlare.

    Cross-site evaluation allows each client to evaluate models from other clients
    and the server, creating an all-to-all matrix of model performance. This enables
    comparison of how different models perform on different client datasets without
    sharing the data itself.

    The recipe configures:
    - A federated job with optional initial model or pre-trained models
    - CrossSiteEval controller for coordinating the evaluation workflow
    - Model locator for finding server models (optional)
    - Validators for client-side model evaluation
    - Trainers for submitting client models
    - JSON generator for saving cross-validation results

    Args:
        name: Name of the federated learning job. Defaults to "cross_site_eval".
        min_clients: Minimum number of clients required to start evaluation.
        initial_model: Initial model to evaluate. If None and model_locator_config is None,
            will only evaluate client models. Defaults to None.
        model_locator_config: Configuration for NPModelLocator to locate pre-trained models.
            Should be a dict with 'model_dir' and 'model_name' keys. If None,
            uses initial_model. Defaults to None.
        server_models: List of server model names to evaluate. Defaults to ["best_model"].
        cross_val_dir: Directory for cross-validation results. Defaults to "cross_site_val".
        submit_model_timeout: Timeout in seconds for submit_model task. Defaults to 600.
        validation_timeout: Timeout in seconds for validation task. Defaults to 6000.
        participating_clients: List of client names to participate. If None, all connected
            clients will participate. Defaults to None.
        client_model_dir: Directory where client models are saved. Defaults to "model".
        client_model_name: Name of the client model file. Defaults to "best_numpy.npy".

    Example:
        ```python
        # Standalone evaluation with pre-trained models
        recipe = NumpyCrossSiteEvalRecipe(
            name="numpy_cse",
            min_clients=2,
            model_locator_config={
                "model_dir": "/tmp/nvflare/server_pretrain_models",
                "model_name": {
                    "server_model_1": "server_1.npy",
                    "server_model_2": "server_2.npy"
                }
            }
        )

        # Evaluation with initial model
        recipe = NumpyCrossSiteEvalRecipe(
            name="numpy_cse",
            min_clients=2,
            initial_model=my_numpy_model,
        )

        # Evaluation of only client models (no server models)
        recipe = NumpyCrossSiteEvalRecipe(
            name="numpy_cse",
            min_clients=2,
            server_models=[],  # Empty list - no server models
        )
        ```

    Note:
        This recipe can be used standalone to evaluate pre-trained models, or
        combined with a training workflow (like FedAvg) to evaluate models
        after training completes.

        The results are saved as a JSON file in the cross_val_dir directory,
        creating an all-to-all matrix showing how each model performs on
        each client's dataset.
    """

    def __init__(
        self,
        *,
        name: str = "cross_site_eval",
        min_clients: int,
        initial_model: Optional[Any] = None,
        model_locator_config: Optional[Dict[str, Any]] = None,
        server_models: List[str] = None,
        cross_val_dir: str = AppConstants.CROSS_VAL_DIR,
        submit_model_timeout: int = 600,
        validation_timeout: int = 6000,
        participating_clients: Optional[List[str]] = None,
        client_model_dir: str = "model",
        client_model_name: str = "best_numpy.npy",
    ):
        # Set default for server_models
        if server_models is None:
            server_models = [DefaultCheckpointFileName.GLOBAL_MODEL]

        # Validate inputs internally
        v = _CrossSiteEvalValidator(
            name=name,
            min_clients=min_clients,
            initial_model=initial_model,
            model_locator_config=model_locator_config,
            server_models=server_models,
            cross_val_dir=cross_val_dir,
            submit_model_timeout=submit_model_timeout,
            validation_timeout=validation_timeout,
            participating_clients=participating_clients,
            client_model_dir=client_model_dir,
            client_model_name=client_model_name,
        )

        self.name = v.name
        self.min_clients = v.min_clients
        self.initial_model = v.initial_model
        self.model_locator_config = v.model_locator_config
        self.server_models = v.server_models
        self.cross_val_dir = v.cross_val_dir
        self.submit_model_timeout = v.submit_model_timeout
        self.validation_timeout = v.validation_timeout
        self.participating_clients = v.participating_clients
        self.client_model_dir = v.client_model_dir
        self.client_model_name = v.client_model_name

        # Create FedJob
        job = FedJob(name=self.name, min_clients=self.min_clients)

        # Decide on persistor and model locator
        persistor_id = ""
        model_locator_id = ""

        # If initial_model is provided, use persistor
        if self.initial_model is not None:
            persistor_id = job.to_server(NPModelPersistor(), id="persistor")
            job.to(self.initial_model, "server")
            # Use default model locator for the persistor's models
            model_locator = NPModelLocator()
            model_locator_id = job.to_server(model_locator, id="model_locator")

        # If model_locator_config is provided, create custom model locator
        elif self.model_locator_config is not None:
            model_dir = self.model_locator_config.get("model_dir", "models")
            model_name = self.model_locator_config.get("model_name", "server.npy")
            model_locator = NPModelLocator(model_dir=model_dir, model_name=model_name)
            model_locator_id = job.to_server(model_locator, id="model_locator")

        # Add formatter and validation JSON generator to server
        formatter_id = job.to_server(NPFormatter(), id="formatter")
        job.to_server(ValidationJsonGenerator())

        # Create CrossSiteEval controller
        controller = CrossSiteEval(
            persistor_id=persistor_id,
            cross_val_dir=self.cross_val_dir,
            submit_model_timeout=self.submit_model_timeout,
            validation_timeout=self.validation_timeout,
            server_models=self.server_models,
            participating_clients=self.participating_clients,
        )
        job.to_server(controller)

        # Add client components
        # Trainer for submitting client models
        trainer = NPTrainer(
            train_task_name=AppConstants.TASK_TRAIN,
            submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL,
            model_name=self.client_model_name,
            model_dir=self.client_model_dir,
        )
        job.to_clients(trainer, tasks=[AppConstants.TASK_SUBMIT_MODEL])

        # Validator for evaluating models
        validator = NPValidator(
            validate_task_name=AppConstants.TASK_VALIDATION,
        )
        job.to_clients(validator, tasks=[AppConstants.TASK_VALIDATION])

        Recipe.__init__(self, job)
