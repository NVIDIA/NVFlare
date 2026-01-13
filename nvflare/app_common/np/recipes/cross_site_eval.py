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

from typing import Optional

from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.np.np_model_locator import NPModelLocator
from nvflare.app_common.np.np_validator import NPValidator
from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
from nvflare.job_config.api import FedJob
from nvflare.job_config.script_runner import FrameworkType
from nvflare.recipe.spec import Recipe


class NumpyCrossSiteEvalRecipe(Recipe):
    """Recipe for standalone cross-site evaluation with pre-trained NumPy models.

    Creates a cross-site evaluation workflow that loads pre-trained models and evaluates
    them across all client sites without performing any training.

    Args:
        name: Name of the federated job. Defaults to "numpy_cross_site_eval".
        min_clients: Minimum number of clients required to start the job. Defaults to 2.
        model_dir: Directory containing pre-trained models (server-side path, relative to
            workspace or absolute). Defaults to "models".
        model_name: Dictionary mapping model identifiers to filenames, e.g.,
            {"model_1": "model_1.npy", "model_2": "model_2.npy"}.
            If None, defaults to {"server": "server.npy"}.
        submit_model_timeout: Timeout (seconds) for submitting models to clients. Defaults to 600.
        validation_timeout: Timeout (seconds) for validation tasks on clients. Defaults to 6000.
    """

    def __init__(
        self,
        name: str = "numpy_cross_site_eval",
        min_clients: int = 2,
        model_dir: Optional[str] = None,
        model_name: Optional[dict] = None,
        submit_model_timeout: int = 600,
        validation_timeout: int = 6000,
    ):
        job = FedJob(name=name, min_clients=min_clients)

        # Add model locator to locate pre-trained models
        model_locator_id = job.to_server(NPModelLocator(model_dir=model_dir, model_name=model_name))

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
        job.to_clients(
            NPValidator(),
            tasks=[AppConstants.TASK_VALIDATION],
        )

        # Set framework for external API compatibility (e.g., add_cross_site_evaluation)
        self.framework = FrameworkType.RAW

        super().__init__(job)
