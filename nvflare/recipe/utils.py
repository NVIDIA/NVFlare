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

import importlib

from nvflare.fuel.utils.import_utils import optional_import
from nvflare.recipe.spec import Recipe

TRACKING_REGISTRY = {
    "mlflow": {
        "package": "mlflow",
        "receiver_module": "nvflare.app_opt.tracking.mlflow.mlflow_receiver",
        "receiver_class": "MLflowReceiver",
    },
    "tensorboard": {
        "package": "tensorboard",
        "receiver_module": "nvflare.app_opt.tracking.tb.tb_receiver",
        "receiver_class": "TBAnalyticsReceiver",
    },
    "wandb": {
        "package": "wandb",
        "receiver_module": "nvflare.app_opt.tracking.wandb.wandb_receiver",
        "receiver_class": "WandBReceiver",
    },
}


def add_experiment_tracking(recipe: Recipe, tracking_type: str, tracking_config: dict = None):
    """Enable experiment tracking.

    Args:
        tracking_type: the type of tracking to enable
        tracking_config: the configuration for the tracking
    """
    tracking_config = tracking_config or {}
    if tracking_type not in TRACKING_REGISTRY:
        raise ValueError(f"Invalid tracking type: {tracking_type}")

    _, flag = optional_import(TRACKING_REGISTRY[tracking_type]["package"])
    if not flag:
        raise ValueError(
            f"{TRACKING_REGISTRY[tracking_type]['package']} is not installed. Please install it using `pip install {TRACKING_REGISTRY[tracking_type]['package']}`"
        )

    module = importlib.import_module(TRACKING_REGISTRY[tracking_type]["receiver_module"])
    receiver_class = getattr(module, TRACKING_REGISTRY[tracking_type]["receiver_class"])
    receiver = receiver_class(**tracking_config)
    recipe.job.to_server(receiver, "receiver")


def add_cross_site_evaluation(recipe: Recipe, persistor_id: str = None, submit_model_timeout: int = 600, validation_timeout: int = 6000):
    """Enable cross-site model evaluation.

    Args:
        recipe_or_job: Recipe object to add cross-site evaluation to
        persistor_id: The persistor ID to use for model location. If None, uses the default persistor_id from job.comp_ids
        submit_model_timeout: Timeout for model submission in seconds
        validation_timeout: Timeout for validation in seconds
    """
    from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
    from nvflare.app_opt.pt.file_model_locator import PTFileModelLocator
     
    # Use provided persistor_id or default from job.comp_ids
    if persistor_id is None:
        persistor_id = recipe.job.comp_ids["persistor_id"]
    
    # Create and add model locator
    model_locator_id = recipe.job.to_server(PTFileModelLocator(pt_persistor_id=persistor_id))
    
    # Create and add cross-site evaluation controller
    eval_controller = CrossSiteModelEval(
        model_locator_id=model_locator_id,
        submit_model_timeout=submit_model_timeout,
        validation_timeout=validation_timeout,
    )
    recipe.job.to_server(eval_controller)
    
