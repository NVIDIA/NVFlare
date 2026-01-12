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
    submit_model_timeout: int = 600,
    validation_timeout: int = 6000,
):
    """Add cross-site evaluation to an existing recipe.

    This utility automatically configures cross-site evaluation by:
    - Auto-detecting the framework from the recipe
    - Adding the appropriate model locator
    - Adding the CrossSiteModelEval controller
    - Adding ValidationJsonGenerator for results
    - Auto-adding the appropriate validator to clients (for NumPy recipes)

    **For standalone CSE without training**, use `NumpyCrossSiteEvalRecipe` instead.

    **Note**: This utility is designed for adding CSE to training recipes. If you call it on
    a CSE-only recipe (e.g., `NumpyCrossSiteEvalRecipe`), it will detect this and skip
    adding duplicate validators automatically.

    Example:
        ```python
        from nvflare.app_common.np.recipes import NumpyFedAvgRecipe
        from nvflare.recipe.utils import add_cross_site_evaluation

        recipe = NumpyFedAvgRecipe(
            name="my-job", min_clients=2, num_rounds=3, train_script="client.py"
        )

        # That's it! Framework auto-detected, validator auto-added
        add_cross_site_evaluation(recipe)
        ```

    Args:
        recipe: Recipe instance to augment with cross-site evaluation.
        submit_model_timeout: Timeout (seconds) for submitting models to clients. Defaults to 600.
        validation_timeout: Timeout (seconds) for validation tasks on clients. Defaults to 6000.

    Raises:
        ValueError: If the recipe doesn't have a framework attribute or uses an unsupported framework.

    Note:
        - Currently supports PyTorch and NumPy frameworks. TensorFlow support may be added in the future.
        - For NumPy recipes, validators are automatically added to clients. This is skipped for
          CSE-only recipes (like `NumpyCrossSiteEvalRecipe`) which already have validators configured.
        - For PyTorch recipes, client-side validators are typically already configured in the recipe.
    """
    from nvflare.app_common.app_constant import AppConstants
    from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
    from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
    from nvflare.job_config.script_runner import FrameworkType

    # Auto-detect framework from recipe
    if not hasattr(recipe, "framework"):
        raise ValueError(
            f"Recipe {type(recipe).__name__} does not have a 'framework' attribute. "
            "Ensure you're using a Recipe class that declares its framework (e.g., NumpyFedAvgRecipe, FedAvgRecipe)."
        )

    framework = recipe.framework

    # Map framework to model locator type
    framework_to_locator = {
        FrameworkType.PYTORCH: "pytorch",
        FrameworkType.RAW: "numpy",  # NumPy uses RAW framework type
    }

    if framework not in framework_to_locator:
        # Build user-friendly error message with supported frameworks
        supported_list = []
        for fw_type in framework_to_locator.keys():
            # Format: "pytorch (FrameworkType.PYTORCH)" and "numpy (FrameworkType.RAW)"
            supported_list.append(f'"{fw_type.value}" (FrameworkType.{fw_type.name})')
        supported_str = ", ".join(supported_list)

        raise ValueError(
            f"Unsupported framework for cross-site evaluation: {framework}. "
            f"Currently supported: {supported_str}. "
            f"TensorFlow support may be added in the future."
        )

    model_locator_type = framework_to_locator[framework]

    # Get model locator configuration from registry
    locator_config = MODEL_LOCATOR_REGISTRY[model_locator_type]

    # Import and create model locator
    module = importlib.import_module(locator_config["locator_module"])
    locator_class = getattr(module, locator_config["locator_class"])

    # Create model locator with appropriate parameters
    locator_kwargs = {}
    if locator_config["persistor_param"] is not None:
        # For PyTorch locator, get persistor_id from comp_ids
        if hasattr(recipe.job, "comp_ids"):
            persistor_id = recipe.job.comp_ids.get("persistor_id", "")
            locator_kwargs[locator_config["persistor_param"]] = persistor_id

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

    # Auto-add validators for NumPy recipes (if not already a CSE-only recipe)
    if framework == FrameworkType.RAW:
        # Check if this is already a standalone CSE recipe (which already has validators)
        # NumpyCrossSiteEvalRecipe is CSE-only and already configures validators
        from nvflare.app_common.np.np_validator import NPValidator

        # Check if this is a CSE-only recipe by checking the recipe class name
        # CSE-only recipes already have validators configured, so we skip adding them
        recipe_class_name = type(recipe).__name__
        is_cse_only_recipe = "CrossSiteEval" in recipe_class_name or "CSE" in recipe_class_name

        if not is_cse_only_recipe:
            # For training recipes (e.g., NumpyFedAvgRecipe), add validator for CSE
            validator = NPValidator()
            recipe.job.to_clients(validator, tasks=[AppConstants.TASK_VALIDATION])
