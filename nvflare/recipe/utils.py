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

import copy
import importlib
import os
from typing import Any, List, Optional

from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE
from nvflare.fuel.utils.import_utils import optional_import
from nvflare.job_config.api import FedJob
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
    "tensorflow": {
        "locator_module": "nvflare.app_opt.tf.file_model_locator",
        "locator_class": "TFFileModelLocator",
        "persistor_param": "tf_persistor_id",
    },
}


def add_experiment_tracking(
    recipe: Recipe,
    tracking_type: str,
    tracking_config: Optional[dict] = None,
    client_side: bool = False,
    server_side: bool = True,
):
    """Add experiment tracking to a recipe.

    Adds tracking receivers to the server and/or clients to collect and log metrics during training.

    Args:
        recipe: Recipe instance to augment with experiment tracking.
        tracking_type: Type of tracking to enable ("mlflow", "tensorboard", or "wandb").
        tracking_config: Optional configuration dict for the tracking receiver.
        client_side: If True, add tracking to all clients (each client tracks locally).
        server_side: If True, add tracking to server (aggregates metrics from all clients). Default: True.

    Examples:
        # Server-side tracking (default - federated metrics)
        add_experiment_tracking(recipe, "mlflow", {"tracking_uri": "..."})

        # Client-side tracking only (each client tracks independently)
        add_experiment_tracking(recipe, "mlflow", {...}, client_side=True, server_side=False)

        # Both server and client tracking
        add_experiment_tracking(recipe, "mlflow", {...}, client_side=True, server_side=True)
    """
    tracking_config = tracking_config or {}
    if tracking_type not in TRACKING_REGISTRY:
        raise ValueError(f"Invalid tracking type: {tracking_type}")

    if not server_side and not client_side:
        raise ValueError("At least one of server_side or client_side must be True")

    _, flag = optional_import(TRACKING_REGISTRY[tracking_type]["package"])
    if not flag:
        raise ValueError(
            f"{TRACKING_REGISTRY[tracking_type]['package']} is not installed. Please install it using `pip install {TRACKING_REGISTRY[tracking_type]['package']}`"
        )

    module = importlib.import_module(TRACKING_REGISTRY[tracking_type]["receiver_module"])
    receiver_class = getattr(module, TRACKING_REGISTRY[tracking_type]["receiver_class"])

    # Add server-side tracking
    if server_side:
        receiver = receiver_class(**tracking_config)
        recipe.job.to_server(receiver, "receiver")

    # Add client-side tracking
    if client_side:
        # For client-side tracking, need to configure local events
        # Deep copy to avoid shared mutable state (tracking_config may contain nested dicts)
        client_config = copy.deepcopy(tracking_config)
        # Override events to track local analytics (not federated)
        if "events" not in client_config:
            client_config["events"] = [ANALYTIC_EVENT_TYPE]

        client_receiver = receiver_class(**client_config)
        recipe.job.to_clients(client_receiver, id="client_receiver")


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

    **WARNING**: Do not call this function multiple times on the same recipe instance.
    This function is idempotent and will raise a RuntimeError if called more than once
    on the same recipe to prevent duplicate component registration.

    **IMPORTANT for PyTorch**: Your client training script must handle validation tasks by
    checking `flare.is_evaluate()` and returning metrics without training. Example pattern:

        ```python
        # In your client script:
        while flare.is_running():
            input_model = flare.receive()
            model.load_state_dict(input_model.params)

            # Evaluate model (always required)
            metrics = evaluate(model, test_loader)

            # Handle CSE validation task
            if flare.is_evaluate():
                output_model = flare.FLModel(metrics=metrics)
                flare.send(output_model)
                continue  # Skip training for validation-only tasks

            # Normal training code here...
        ```

    Example (NumPy - fully automatic):
        ```python
        from nvflare.app_common.np.recipes import NumpyFedAvgRecipe
        from nvflare.recipe.utils import add_cross_site_evaluation

        recipe = NumpyFedAvgRecipe(
            name="my-job", model=[1.0, 2.0, 3.0], min_clients=2, num_rounds=3, train_script="client.py"
        )

        # That's it! Framework auto-detected, validator auto-added
        add_cross_site_evaluation(recipe)
        ```

    Example (PyTorch - requires client script support):
        ```python
        from nvflare.app_opt.pt.recipes import FedAvgRecipe
        from nvflare.recipe.utils import add_cross_site_evaluation

        recipe = FedAvgRecipe(
            name="my-job", min_clients=2, num_rounds=3,
            model=MyModel(), train_script="client.py"
        )

        # Note: client.py must handle flare.is_evaluate() for validation
        add_cross_site_evaluation(recipe)
        ```

    Example (TensorFlow - Client API pattern, recommended):
        ```python
        from nvflare.app_opt.tf.recipes import FedAvgRecipe
        from nvflare.recipe.utils import add_cross_site_evaluation

        recipe = FedAvgRecipe(
            name="my-job", min_clients=2, num_rounds=3,
            model=MyTFModel(), train_script="client.py"
        )

        # Note: client.py must handle flare.is_evaluate() for validation
        add_cross_site_evaluation(recipe)
        ```

    Example (TensorFlow - Component-based alternative):
        ```python
        from nvflare.app_opt.tf.recipes import FedAvgRecipe
        from nvflare.app_opt.tf.tf_validator import TFValidator
        from nvflare.recipe.utils import add_cross_site_evaluation

        recipe = FedAvgRecipe(
            name="my-job", min_clients=2, num_rounds=3,
            model=MyTFModel(), train_script="client.py"
        )

        add_cross_site_evaluation(recipe)

        # Optional: manually add TFValidator for component-based validation
        validator = TFValidator(model=my_model, data_loader=test_loader)
        recipe.job.to_clients(validator, tasks=["validate"])
        ```

    Args:
        recipe: Recipe instance to augment with cross-site evaluation.
        submit_model_timeout: Timeout (seconds) for submitting models to clients. Defaults to 600.
        validation_timeout: Timeout (seconds) for validation tasks on clients. Defaults to 6000.

    Raises:
        ValueError: If the recipe doesn't have a framework attribute or uses an unsupported framework.
        RuntimeError: If cross-site evaluation has already been added to this recipe.

    Note:
        - Currently supports PyTorch, NumPy, and TensorFlow frameworks.
        - **NumPy recipes**: Validators (NPValidator) are automatically added to clients to handle
          validation tasks. The function intelligently detects if validators are already configured
          by checking for executors handling TASK_VALIDATION, avoiding duplicates for CSE-only recipes
          (like `NumpyCrossSiteEvalRecipe`).
        - **PyTorch recipes**: No separate validator component is needed. The client training script
          handles validation tasks through the Client API's `flare.is_evaluate()` check. See the
          hello-pt example for implementation pattern.
        - **TensorFlow recipes**: Similar to PyTorch, uses the Client API pattern. The client script
          should handle validation tasks via `flare.is_evaluate()` check.
    """
    from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
    from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
    from nvflare.job_config.script_runner import FrameworkType

    # Idempotency check: prevent multiple calls on the same recipe
    if hasattr(recipe, "_cse_added") and recipe._cse_added:
        name = recipe.name if hasattr(recipe, "name") else "cross-site-evaluation job"
        raise RuntimeError(
            f"Cross-site evaluation has already been added to recipe '{name}'. "
            "Calling add_cross_site_evaluation() multiple times would create duplicate "
            "model locators, validators, and controllers, which can cause unexpected behavior. "
            "Please call this function only once per recipe instance."
        )

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
        FrameworkType.TENSORFLOW: "tensorflow",
    }

    if framework not in framework_to_locator:
        # Build user-friendly error message with supported frameworks
        supported_list = []
        for fw_type in framework_to_locator.keys():
            # Format: "pytorch (FrameworkType.PYTORCH)" and "numpy (FrameworkType.RAW)"
            supported_list.append(f'"{fw_type.value}" (FrameworkType.{fw_type.name})')
        supported_str = ", ".join(supported_list)

        raise ValueError(
            f"Unsupported framework for cross-site evaluation: {framework}. " f"Currently supported: {supported_str}."
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
        # For frameworks requiring persistor_id (PyTorch, TensorFlow), get it from comp_ids
        if hasattr(recipe.job, "comp_ids"):
            persistor_id = recipe.job.comp_ids.get("persistor_id", "")
            if not persistor_id:
                raise ValueError(
                    f"Cross-site evaluation requires a persistor for {framework_to_locator[framework]} recipes, "
                    f"but no persistor_id found in recipe.job.comp_ids. "
                    f"Ensure your recipe includes an model to create a persistor."
                )
            locator_kwargs[locator_config["persistor_param"]] = persistor_id
        else:
            raise ValueError(
                f"Recipe {type(recipe).__name__} does not have comp_ids. "
                f"Cross-site evaluation requires recipes that track component IDs."
            )

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

    # Let recipe handle framework-specific validator setup if needed
    # NumPy recipes implement add_cse_validator_if_needed() to add NPValidator automatically
    # PyTorch/TensorFlow recipes use Client API pattern (flare.is_evaluate()) and handle
    # validation in the training script itself, so no validator component is needed
    if hasattr(recipe, "add_cse_validator_if_needed"):
        recipe.add_cse_validator_if_needed()

    # Mark that CSE has been added to prevent duplicate calls
    recipe._cse_added = True


def _has_task_executor(job, task_name: str) -> bool:
    """Check if any executor is already configured for the specified task.

    This function inspects the job's internal structure to determine if a validator
    or executor is already handling the specified task. It uses defensive programming
    to handle potential variations in the internal API structure.

    IMPORTANT: This function accesses the private attribute job._deploy_map because:
    1. No public API exists in FedJob to query configured executors
    2. This check is necessary to avoid adding duplicate validators for CSE
    3. Without this, we'd rely on fragile string matching on recipe class names

    The implementation uses defensive programming (hasattr checks, try-except) to
    minimize fragility. If FedJob's internal structure changes, this function will
    gracefully return False rather than crashing.

    Future improvement: FedJob could provide a public method like get_executors(target)
    to make this check safer and more maintainable.

    Args:
        job: FedJob instance to check
        task_name: Task name to check for (e.g., AppConstants.TASK_VALIDATION)

    Returns:
        True if an executor is already configured for this task, False otherwise
    """
    # Access _deploy_map (private attribute) - see docstring for justification
    # Defensive check: ensure _deploy_map exists before accessing
    if not hasattr(job, "_deploy_map"):
        return False

    for target, app in job._deploy_map.items():
        # Skip server apps, only check client apps
        if target == "server":
            continue

        # Get the client app configuration
        if hasattr(app, "app_config"):
            app_config = app.app_config
            # Check if it's a ClientAppConfig with executors
            if hasattr(app_config, "executors"):
                for executor_def in app_config.executors:
                    # Defensive check: ensure executor_def has tasks attribute
                    if not hasattr(executor_def, "tasks"):
                        continue

                    try:
                        # Check if this executor handles the task
                        # Wildcard executors (["*"]) can handle any task
                        if "*" in executor_def.tasks or task_name in executor_def.tasks:
                            return True
                    except (TypeError, AttributeError):
                        # Handle case where tasks is not iterable or comparable
                        # This could happen if tasks has an unexpected type
                        continue
    return False


def _collect_non_local_scripts(job: FedJob) -> List[str]:
    """Collect scripts that don't exist locally.

    This utility function is used by ExecEnv subclasses to validate script resources
    before deployment. Scripts are considered "non-local" if they are absolute paths
    that don't exist on the local machine.

    Args:
        job: The FedJob to check for non-local scripts.

    Returns:
        List of absolute script paths that don't exist on the local machine.
    """
    non_local_scripts = []
    for app in job._deploy_map.values():
        for script in app.app_config.ext_scripts:
            if os.path.isabs(script) and not os.path.exists(script):
                non_local_scripts.append(script)
    return non_local_scripts


def validate_initial_ckpt(initial_ckpt: Optional[str]) -> None:
    """Validate that initial_ckpt is an absolute path if provided.

    Args:
        initial_ckpt: Checkpoint file path to validate.

    Raises:
        ValueError: If initial_ckpt is not an absolute path.
    """
    if initial_ckpt is not None:
        if not os.path.isabs(initial_ckpt):
            raise ValueError(
                f"initial_ckpt must be an absolute path, got: {initial_ckpt}. "
                "Use absolute paths like '/workspace/model.pt' for server-side checkpoints."
            )


def validate_dict_model_config(model: Any) -> None:
    """Validate dict model config structure.

    Args:
        model: Model input to validate.

    Raises:
        ValueError: If dict config is missing 'path' key or 'path' is not a string.
    """
    if isinstance(model, dict):
        if "path" not in model:
            raise ValueError("Dict model config must have 'path' key with fully qualified class path. " f"Got: {model}")
        if not isinstance(model["path"], str):
            raise ValueError(f"Dict model config 'path' must be a string, got: {type(model['path'])}")
