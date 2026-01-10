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
from typing import Optional

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

MODEL_LOCATOR_REGISTRY = {
    "pytorch": {
        "locator_module": "nvflare.app_opt.pt.file_model_locator",
        "locator_class": "PTFileModelLocator",
        "persistor_param": "pt_persistor_id",
    },
    "numpy": {
        "locator_module": "nvflare.app_common.np.np_model_locator",
        "locator_class": "NPModelLocator",
        "persistor_param": None,  # NPModelLocator doesn't use persistor_id
    },
}


def add_experiment_tracking(recipe: Recipe, tracking_type: str, tracking_config: Optional[dict] = None):
    """Add experiment tracking to a recipe.

    Adds a tracking receiver to the server to collect and log metrics from clients during training.

    Args:
        recipe: Recipe instance to augment with experiment tracking.
        tracking_type: Type of tracking to enable ("mlflow", "tensorboard", or "wandb").
        tracking_config: Optional configuration dict for the tracking receiver.
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


def add_cross_site_evaluation(
    recipe: Recipe,
    model_locator_type: str = "pytorch",
    model_locator_config: Optional[dict] = None,
    persistor_id: Optional[str] = None,
    submit_model_timeout: int = 600,
    validation_timeout: int = 6000,
):
    """Add cross-site evaluation to an existing recipe.

    This utility adds server-side CSE components (model locator, CrossSiteModelEval controller,
    and ValidationJsonGenerator) to a recipe that already includes training.

    **For standalone CSE without training**, use `NumpyCrossSiteEvalRecipe` instead of this utility.

    **IMPORTANT for NumPy recipes:** This function only adds server-side components. You MUST
    manually add NPValidator to clients BEFORE calling this function:

        ```python
        from nvflare.app_common.np.recipes import NumpyFedAvgRecipe
        from nvflare.app_common.np.np_validator import NPValidator
        from nvflare.app_common.app_constant import AppConstants
        from nvflare.recipe.utils import add_cross_site_evaluation

        recipe = NumpyFedAvgRecipe(
            name="my-job", min_clients=2, num_rounds=3, train_script="client.py"
        )

        # REQUIRED: Add validator to clients for NumPy
        validator = NPValidator(validate_task_name=AppConstants.TASK_VALIDATION)
        recipe.job.to_clients(validator, tasks=[AppConstants.TASK_VALIDATION])

        # Now add cross-site evaluation
        add_cross_site_evaluation(recipe, model_locator_type="numpy")
        ```

    For PyTorch recipes, client-side validators are typically included in the recipe already.

    Args:
        recipe: Recipe instance to augment with cross-site evaluation.
        model_locator_type: Type of model locator ("pytorch" or "numpy"). Defaults to "pytorch".
        model_locator_config: Optional configuration dict for the model locator
            (e.g., {"model_dir": "/path", "model_name": {...}}).
        persistor_id: Persistor ID for PyTorch model location. If None, auto-detected from
            recipe.job.comp_ids (PyTorch only; NumPy doesn't use persistor_id).
        submit_model_timeout: Timeout (seconds) for submitting models to clients. Defaults to 600.
        validation_timeout: Timeout (seconds) for validation tasks on clients. Defaults to 6000.
    """
    from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
    from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval

    if model_locator_type not in MODEL_LOCATOR_REGISTRY:
        raise ValueError(
            f"Invalid model locator type: {model_locator_type}. Available types: {list(MODEL_LOCATOR_REGISTRY.keys())}"
        )

    # Get model locator configuration from registry
    locator_config = MODEL_LOCATOR_REGISTRY[model_locator_type]

    # Import and create model locator
    module = importlib.import_module(locator_config["locator_module"])
    locator_class = getattr(module, locator_config["locator_class"])

    # Create model locator with appropriate parameters
    locator_kwargs = {}
    if locator_config["persistor_param"] is not None:
        # For PyTorch locator, use persistor_id
        # If not provided, try to get it from comp_ids (only available in BaseFedJob)
        if persistor_id is None:
            if hasattr(recipe.job, "comp_ids"):
                persistor_id = recipe.job.comp_ids.get("persistor_id", "")
            else:
                persistor_id = ""
        locator_kwargs[locator_config["persistor_param"]] = persistor_id

    # Merge in custom config if provided
    if model_locator_config:
        locator_kwargs.update(model_locator_config)

    model_locator = locator_class(**locator_kwargs)
    model_locator_id = recipe.job.to_server(model_locator)

    # Add validation JSON generator
    recipe.job.to_server(ValidationJsonGenerator())

    # Create and add cross-site evaluation controller
    eval_controller = CrossSiteModelEval(
        model_locator_id=model_locator_id,
        submit_model_timeout=submit_model_timeout,
        validation_timeout=validation_timeout,
    )
    recipe.job.to_server(eval_controller)
