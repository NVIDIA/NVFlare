# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Dict, Optional

from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
from nvflare.job_config.api import FedJob
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner
from nvflare.recipe.spec import Recipe


class PyTorchCrossSiteEvalRecipe(Recipe):
    """Recipe for standalone cross-site evaluation with pre-trained PyTorch models.

    Creates a cross-site evaluation workflow that loads pre-trained models and evaluates
    them across all client sites without performing any training.

    Args:
        name: Name of the federated job. Defaults to "pytorch_cross_site_eval".
        min_clients: Minimum number of clients required to start the job. Defaults to 2.
        model: PyTorch model instance (required). Serves as architecture template for loading checkpoints.
        model_dir: Directory containing pre-trained models (server-side path, relative to
            workspace or absolute). Defaults to "/tmp/nvflare/server_pretrain_models".
        model_name: Dictionary mapping model identifiers to filenames, e.g.,
            {"model_1": "server_1.pt", "model_2": "server_2.pt"}.
            If None, defaults to {"server_model_1": "server_1.pt", "server_model_2": "server_2.pt"}.
        train_script: Path to the client training script that handles evaluation.
        train_args: Additional arguments to pass to the training script.
        submit_model_timeout: Timeout (seconds) for submitting models to clients. Defaults to 600.
        validation_timeout: Timeout (seconds) for validation tasks on clients. Defaults to 6000.

    Note:
        Unlike NumPy recipes, PyTorch CSE requires:
        - A model instance for the persistor to provide the architecture template
        - A train_script that implements the evaluation logic using Client API
        - The train_script must check `flare.is_evaluate()` and return metrics without training
    """

    def __init__(
        self,
        name: str = "pytorch_cross_site_eval",
        min_clients: int = 2,
        model=None,
        model_dir: str = "/tmp/nvflare/server_pretrain_models",
        model_name: Optional[Dict[str, str]] = None,
        train_script: str = "client.py",
        train_args: str = "",
        submit_model_timeout: int = 600,
        validation_timeout: int = 6000,
    ):
        if model is None:
            raise ValueError(
                "PyTorchCrossSiteEvalRecipe requires a model. "
                "Provide a PyTorch model instance as the architecture template."
            )

        if model_name is None:
            model_name = {"server_model_1": "server_1.pt", "server_model_2": "server_2.pt"}

        job = FedJob(name=name, min_clients=min_clients)

        # Add model locator that loads pre-trained PyTorch models directly from disk
        from nvflare.app_opt.pt.pt_model_locator import PTModelLocator

        model_locator_id = job.to_server(PTModelLocator(model_dir=model_dir, model_name=model_name))

        # Add cross-site evaluation controller
        job.to_server(
            CrossSiteModelEval(
                model_locator_id=model_locator_id,
                submit_model_timeout=submit_model_timeout,
                validation_timeout=validation_timeout,
            )
        )

        # Add validation JSON generator to save results
        from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator

        job.to_server(ValidationJsonGenerator())

        # Add client-side executor using ScriptRunner
        # PyTorch uses Client API pattern - the script handles evaluation via flare.is_evaluate()
        # Register for both submit_model and validation tasks
        executor = ScriptRunner(script=train_script, script_args=train_args)
        job.to_clients(executor, tasks=[AppConstants.TASK_SUBMIT_MODEL, AppConstants.TASK_VALIDATION])

        # Set framework for external API compatibility
        self.framework = FrameworkType.PYTORCH

        super().__init__(job)
