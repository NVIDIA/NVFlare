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

import os
from typing import Optional

from pydantic import BaseModel, field_validator

from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.np.np_model_locator import NPModelLocator
from nvflare.app_common.np.np_validator import NPValidator
from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
from nvflare.job_config.api import FedJob
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner
from nvflare.recipe.spec import Recipe
from nvflare.recipe.utils import validate_ckpt


# Internal validator
class _CrossSiteEvalValidator(BaseModel):
    name: str
    min_clients: int
    eval_script: Optional[str] = None
    eval_args: str = ""
    launch_external_process: bool = False
    command: str = "python3 -u"
    initial_ckpt: Optional[str] = None
    model_dir: Optional[str] = None
    model_name: Optional[dict] = None
    submit_model_timeout: int = 600
    validation_timeout: int = 6000
    client_memory_gc_rounds: int = 0
    cuda_empty_cache: bool = False

    @field_validator("initial_ckpt")
    @classmethod
    def validate_initial_ckpt(cls, v):
        if v is not None:
            if not os.path.isabs(v):
                raise ValueError(
                    f"initial_ckpt must be an absolute path for NumpyCrossSiteEvalRecipe, got: {v}. "
                    "Relative path support for this recipe is planned for a future release."
                )
            validate_ckpt(v)
        return v


class NumpyCrossSiteEvalRecipe(Recipe):
    """Recipe for standalone cross-site evaluation with pre-trained NumPy models.

    Creates a cross-site evaluation workflow that loads pre-trained models and evaluates
    them across all client sites without performing any training.

    Args:
        name: Name of the federated job. Defaults to "numpy_cross_site_eval".
        min_clients: Minimum number of clients required to start the job. Defaults to 2.
        eval_script: Path to the evaluation script that will be executed on each client.
            If not provided, uses a built-in dummy validator (for testing only).
        eval_args: Command line arguments to pass to the evaluation script. Defaults to "".
        launch_external_process: Whether to launch the script in external process. Defaults to False.
        command: If launch_external_process=True, command to run script (prepended to script).
            Defaults to "python3 -u".
        initial_ckpt: Absolute path to a pre-trained model file (.npy) on the server.
            If provided, this takes precedence over model_dir/model_name.
            The file may not exist locally (server-side path).
        model_dir: Directory containing pre-trained models (relative to run directory).
            Defaults to "models". Only used when initial_ckpt is not provided.
        model_name: Dictionary mapping model identifiers to filenames, e.g.,
            {"model_1": "model_1.npy", "model_2": "model_2.npy"}.
            If None, defaults to {"server": "server.npy"}.
            Only used when initial_ckpt is not provided.
        submit_model_timeout: Timeout (seconds) for submitting models to clients. Defaults to 600.
        validation_timeout: Timeout (seconds) for validation tasks on clients. Defaults to 6000.

    Example:
        Using eval_script with initial_ckpt:

        ```python
        recipe = NumpyCrossSiteEvalRecipe(
            eval_script="evaluate.py",
            eval_args="--data_root /path/to/data",
            initial_ckpt="/path/to/pretrained_model.npy",
            min_clients=2,
        )
        ```

        Using model_dir/model_name (models from training run):

        ```python
        recipe = NumpyCrossSiteEvalRecipe(
            eval_script="evaluate.py",
            model_dir="models",
            model_name={"server": "server.npy"},
            min_clients=2,
        )
        ```
    """

    def __init__(
        self,
        name: str = "numpy_cross_site_eval",
        min_clients: int = 2,
        eval_script: Optional[str] = None,
        eval_args: str = "",
        launch_external_process: bool = False,
        command: str = "python3 -u",
        initial_ckpt: Optional[str] = None,
        model_dir: Optional[str] = None,
        model_name: Optional[dict] = None,
        submit_model_timeout: int = 600,
        validation_timeout: int = 6000,
        client_memory_gc_rounds: int = 0,
        cuda_empty_cache: bool = False,
    ):
        # Validate all inputs
        _CrossSiteEvalValidator(
            name=name,
            min_clients=min_clients,
            eval_script=eval_script,
            eval_args=eval_args,
            launch_external_process=launch_external_process,
            command=command,
            initial_ckpt=initial_ckpt,
            model_dir=model_dir,
            model_name=model_name,
            submit_model_timeout=submit_model_timeout,
            validation_timeout=validation_timeout,
            client_memory_gc_rounds=client_memory_gc_rounds,
            cuda_empty_cache=cuda_empty_cache,
        )

        job = FedJob(name=name, min_clients=min_clients)

        # Determine model source
        if initial_ckpt is not None:
            # Use absolute path - pass directly to locator
            # Note: Relative path support deferred to future release (locator path resolution needed)
            locator_model_name = {NPModelLocator.SERVER_MODEL_NAME: initial_ckpt}
            locator_model_dir = model_dir if model_dir is not None else "models"
        else:
            # Use relative path pattern
            locator_model_name = model_name
            locator_model_dir = model_dir

        # Add model locator to locate pre-trained models
        model_locator_id = job.to_server(NPModelLocator(model_dir=locator_model_dir, model_name=locator_model_name))

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

        # Add validators to clients for validation tasks
        if eval_script is not None:
            # Use custom evaluation script via ScriptRunner
            executor = ScriptRunner(
                script=eval_script,
                script_args=eval_args,
                launch_external_process=launch_external_process,
                command=command,
                framework=FrameworkType.RAW,
                memory_gc_rounds=client_memory_gc_rounds,
                cuda_empty_cache=cuda_empty_cache,
            )
            job.to_clients(executor, tasks=[AppConstants.TASK_VALIDATION])
        else:
            # Use built-in dummy validator (for testing/demo only)
            job.to_clients(
                NPValidator(),
                tasks=[AppConstants.TASK_VALIDATION],
            )

        # Set framework for external API compatibility (e.g., add_cross_site_evaluation)
        self.framework = FrameworkType.RAW

        super().__init__(job)
