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
import json
import os
import warnings
from typing import Any, Dict, List, Optional

from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE
from nvflare.apis.job_def import USER_SETTABLE_JOB_META_KEYS, JobMetaKey
from nvflare.fuel.utils.import_utils import optional_import
from nvflare.job_config.api import FedJob
from nvflare.job_config.fed_job_config import FedJobConfig
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


# User-settable keys whose values are dicts keyed by site name with dict values.
_SITE_KEYED_META_KEYS = frozenset({JobMetaKey.RESOURCE_SPEC, JobMetaKey.JOB_LAUNCHER_SPEC})


def merge_config_overrides(defaults: Dict[str, Any], overrides: Optional[Dict[str, Any]], name: str) -> Dict[str, Any]:
    """Return a shallow merge of recipe defaults and user overrides."""
    if overrides is None:
        return dict(defaults)
    if not isinstance(overrides, dict):
        raise TypeError(f"{name} must be a dict, but got {type(overrides).__name__}")
    for key in overrides:
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings, but got {type(key).__name__}")

    result = dict(defaults)
    result.update(overrides)
    return result


def _normalize_recipe_meta_key(key: Any) -> str:
    if not isinstance(key, JobMetaKey):
        raise TypeError(f"recipe meta key must be a JobMetaKey, got {type(key).__name__}")
    if key not in USER_SETTABLE_JOB_META_KEYS:
        raise ValueError(f"recipe meta key {key.value!r} cannot be set through set_recipe_meta")
    return key.value


def _normalize_recipe_meta_value(key: JobMetaKey, key_str: str, value: Any) -> Any:
    """Validate ``value`` against the key's shape contract and return a
    JSON-normalized, caller-independent copy.

    Per-key shapes: SCOPE is a plain string; RESOURCE_SPEC and JOB_LAUNCHER_SPEC
    are dicts keyed by site name with dict values; the remaining user-settable
    keys (CUSTOM_PROPS) are dicts. Catching shape errors here gives an immediate,
    contextual error instead of a late failure at server-side submission
    (JobMetaValidator) or job launch (e.g. PrivacyService scope lookup).

    For dict values, the round-trip through JSON validates nested
    serializability, rejects non-finite floats, and produces a value with
    exactly the semantics the generated ``meta.json`` will have (e.g. dict keys
    coerced to strings) with no aliasing to the caller's object -- so no
    separate ``deepcopy`` is needed.
    """
    if key is JobMetaKey.SCOPE:
        if not isinstance(value, str):
            raise TypeError(f"recipe meta value for key {key_str!r} must be a str, got {type(value).__name__}")
        return value
    # All other user-settable keys are dict-shaped.
    if key in _SITE_KEYED_META_KEYS:
        try:
            _validate_per_site_config_shape(value)
        except TypeError as e:
            raise TypeError(f"recipe meta value for key {key_str!r}: {e}") from None
    elif not isinstance(value, dict):
        raise TypeError(f"recipe meta value for key {key_str!r} must be a dict, got {type(value).__name__}")
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except TypeError as e:
        raise TypeError(f"recipe meta value for key {key_str!r} must be JSON-serializable: {e}") from e
    except ValueError as e:
        raise ValueError(f"recipe meta value for key {key_str!r} must be JSON-serializable: {e}") from e


def _get_recipe_job_config(recipe: Recipe) -> FedJobConfig:
    job = getattr(recipe, "job", None)
    job_config = getattr(job, "job", None)
    if not isinstance(job_config, FedJobConfig):
        raise TypeError("recipe must provide a FedJob through recipe.job")
    return job_config


def set_recipe_meta(recipe: Recipe, key: JobMetaKey, value: Any) -> None:
    """Set one generated job metadata value through ``meta_props``.

    The key must be one of :data:`nvflare.apis.job_def.USER_SETTABLE_JOB_META_KEYS`.
    Keys with dedicated ``FedJob`` constructor fields, such as ``MIN_CLIENTS`` and
    ``MANDATORY_CLIENTS``, are not accepted here -- set those through the
    recipe/``FedJob`` constructor so the controller, scheduler, and metadata stay
    in sync. ``STUDY`` is not accepted either: the server assigns the study from
    the admin session's active study at job submission, so a recipe-set value
    would be silently overwritten.

    The value shape depends on the key: ``SCOPE`` takes a string;
    ``RESOURCE_SPEC`` and ``JOB_LAUNCHER_SPEC`` take a dict keyed by site name
    with dict values; ``CUSTOM_PROPS`` takes a dict. Dict values must be
    completely JSON-serializable, cannot contain non-finite floats, and have
    their keys coerced to strings as they will appear in ``meta.json``. The
    value is stored in ``meta_props`` and replaces any existing ``meta_props``
    value for that key.
    """
    key_str = _normalize_recipe_meta_key(key)
    normalized_value = _normalize_recipe_meta_value(key, key_str, value)
    job_config = _get_recipe_job_config(recipe)

    # RESOURCE_SPEC also has a dedicated FedJobConfig field (populated via
    # add_resource_spec). meta.json is generated by merging meta_props last, so a
    # meta_props value silently replaces any per-site specs registered there. Warn
    # rather than merge, since the two shapes are not guaranteed to be compatible.
    # Note this check is point-in-time: specs registered after this call are still
    # overridden at export, without a warning.
    if key is JobMetaKey.RESOURCE_SPEC and job_config.resource_specs:
        warnings.warn(
            "set_recipe_meta(RESOURCE_SPEC, ...) overrides the per-site resource specs registered "
            "on the FedJob (via add_resource_spec); those specs will not appear in the generated meta.json.",
            stacklevel=2,
        )

    if job_config.meta_props is None:
        job_config.meta_props = {}
    job_config.meta_props[key_str] = normalized_value


def _validate_per_site_config_shape(config: Any) -> Dict[str, Dict]:
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dict, got {type(config).__name__}")

    for site_name, site_config in config.items():
        if not isinstance(site_name, str):
            raise TypeError(f"per-site config key must be a str, got {type(site_name).__name__}")
        if not isinstance(site_config, dict):
            raise TypeError(f"per-site config for site {site_name!r} must be a dict, got {type(site_config).__name__}")

    return config


def set_per_site_config(recipe: Recipe, config: Dict[str, Dict]) -> None:
    """Set site-keyed configuration on a recipe.

    The helper only validates the generic shape:
    - top-level keys are site names
    - values are recipe-specific dictionaries

    Each recipe is responsible for validating and interpreting the fields inside
    each site's dictionary. The execution environment still controls which
    clients are present for a run.
    """
    recipe.set_per_site_config(_validate_per_site_config_shape(config))


def _has_cross_site_eval_workflow(job: FedJob) -> bool:
    """Check if CrossSiteModelEval workflow is already configured on server."""
    from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval

    deploy_map = getattr(job, "_deploy_map", {})
    server_app = deploy_map.get("server")
    if not server_app or not hasattr(server_app, "app_config"):
        return False

    workflows = getattr(server_app.app_config, "workflows", [])
    for w in workflows:
        # Server stores workflow definitions as wrapper objects (e.g. WorkFlow)
        # with the actual controller on `controller`.
        if isinstance(w, CrossSiteModelEval):
            return True
        controller = getattr(w, "controller", None)
        if controller is not None and isinstance(controller, CrossSiteModelEval):
            return True
    return False


def add_experiment_tracking(
    recipe: Recipe,
    tracking_type: str,
    tracking_config: Optional[dict] = None,
    client_side: bool = False,
    server_side: bool = True,
    clients: Optional[List[str]] = None,
):
    """Add experiment tracking to a recipe.

    Adds tracking receivers to the server and/or clients to collect and log metrics during training.

    Args:
        recipe: Recipe instance to augment with experiment tracking.
        tracking_type: Type of tracking to enable ("mlflow", "tensorboard", or "wandb").
        tracking_config: Optional configuration dict for the tracking receiver. For MLflow,
            omitting this uses a local file store and derives ``experiment_name`` and
            ``run_name`` from the recipe name.
        client_side: If True, add tracking to clients (each client tracks locally).
        server_side: If True, add tracking to server (aggregates metrics from all clients). Default: True.
        clients: Optional list of client names for client-side tracking. If None, the
            client-side receiver is added to all clients. Only valid with client_side=True.
            To give sites different receiver configs (e.g. per-site tracking_uri), call this
            function once per site with that site's tracking_config and clients=[site].
            Targeting specific clients requires the recipe's client apps to be per-site
            (e.g. recipes constructed with the per_site_config constructor argument), and
            each name must match an existing per-site client app; with the default
            all-clients topology or unknown site names, targeted placement raises ValueError.

    Examples:
        # Server-side MLflow tracking with local storage and recipe-derived names
        add_experiment_tracking(recipe, "mlflow")

        # Client-side tracking only (each client tracks independently)
        add_experiment_tracking(recipe, "mlflow", client_side=True, server_side=False)

        # Both server and client tracking
        add_experiment_tracking(recipe, "mlflow", {...}, client_side=True, server_side=True)

        # Per-site client tracking configs (one call per site)
        add_experiment_tracking(
            recipe, "mlflow", {"tracking_uri": "file:///tmp/site-1/mlruns"},
            client_side=True, server_side=False, clients=["site-1"],
        )
        add_experiment_tracking(
            recipe, "mlflow", {"tracking_uri": "file:///tmp/site-2/mlruns"},
            client_side=True, server_side=False, clients=["site-2"],
        )
    """
    if tracking_type not in TRACKING_REGISTRY:
        raise ValueError(f"Invalid tracking type: {tracking_type}")

    tracking_config = copy.deepcopy(tracking_config) if tracking_config else {}
    if tracking_type == "mlflow":
        kw_args = tracking_config.get("kw_args")
        if kw_args is None:
            kw_args = {}
            tracking_config["kw_args"] = kw_args
        elif not isinstance(kw_args, dict):
            raise TypeError(f"MLflow kw_args must be a dict, got {type(kw_args).__name__}")
        recipe_name = getattr(recipe, "name", None) or getattr(recipe.job, "name", None) or "nvflare"
        kw_args.setdefault("experiment_name", f"{recipe_name}-experiment")
        kw_args.setdefault("run_name", f"{recipe_name}-Client")

    if not server_side and not client_side:
        raise ValueError("At least one of server_side or client_side must be True")

    if clients is not None:
        if not client_side:
            raise ValueError("clients is only used for client-side tracking; set client_side=True")
        if not isinstance(clients, list) or not all(isinstance(c, str) for c in clients):
            raise TypeError(f"clients must be a list of str, got {clients!r}")
        if not clients:
            raise ValueError("clients must not be empty; omit it to add tracking to all clients")

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
        # Route through the recipe placement layer so existing per-site client apps
        # are preserved (to_clients would target ALL_SITES even when per-site apps exist).
        recipe._add_to_client_apps(client_receiver, clients=clients, id="client_receiver")


def add_final_global_evaluation(
    recipe: Recipe,
    participating_clients: Optional[List[str]] = None,
    validation_timeout: int = 6000,
) -> None:
    """Evaluate a PyTorch recipe's final global model on selected clients.

    Unlike full cross-site evaluation, this helper does not ask clients to
    submit their local models. It locates the recipe's persisted global model
    and sends only that model for validation after training.

    Args:
        recipe: PyTorch recipe to augment with final global model evaluation.
        participating_clients: Optional client names to run validation. If not
            provided, all clients connected when the controller starts are used.
        validation_timeout: Timeout in seconds for validation tasks.

    Raises:
        TypeError: If ``participating_clients`` is not a list of strings.
        ValueError: If ``participating_clients`` is empty, the recipe is not
            PyTorch, or the recipe has no model persistor.
        RuntimeError: If a cross-site evaluation workflow is already configured.
    """
    from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
    from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
    from nvflare.app_opt.pt.file_model_locator import PTFileModelLocator
    from nvflare.job_config.script_runner import FrameworkType

    if getattr(recipe, "_cse_added", False) or _has_cross_site_eval_workflow(recipe.job):
        raise RuntimeError("a cross-site evaluation workflow is already configured for this recipe")

    if getattr(recipe, "framework", None) != FrameworkType.PYTORCH:
        raise ValueError("final global evaluation currently supports PyTorch recipes only")

    if participating_clients is not None:
        if not isinstance(participating_clients, list) or not all(
            isinstance(client, str) for client in participating_clients
        ):
            raise TypeError(f"participating_clients must be a list of str, got {participating_clients!r}")
        if not participating_clients:
            raise ValueError("participating_clients must not be empty; use None to evaluate on all clients")

    comp_ids = getattr(recipe.job, "comp_ids", None)
    if not isinstance(comp_ids, dict):
        raise ValueError("final global evaluation requires a recipe that tracks component IDs")

    model_locator_id = comp_ids.get("locator_id", "")
    if not model_locator_id:
        persistor_id = comp_ids.get("persistor_id", "")
        if not persistor_id:
            raise ValueError("final global evaluation requires a PyTorch model persistor")
        model_locator_id = recipe.job.to_server(
            PTFileModelLocator(pt_persistor_id=persistor_id), id="final_model_locator"
        )
        if not isinstance(model_locator_id, str) or not model_locator_id:
            raise RuntimeError("failed to register the final global model locator")
        comp_ids["locator_id"] = model_locator_id

    recipe.job.to_server(ValidationJsonGenerator())
    recipe.job.to_server(
        CrossSiteModelEval(
            model_locator_id=model_locator_id,
            submit_model_task_name="",
            validation_timeout=validation_timeout,
            participating_clients=participating_clients,
        )
    )
    recipe._cse_added = True


def add_cross_site_evaluation(
    recipe: Recipe,
    submit_model_timeout: int = 600,
    validation_timeout: int = 6000,
    participating_clients: Optional[List[str]] = None,
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

    TensorFlow component-based validators are executors, not plain components.
    Use the lower-level Job API when explicit ``TFValidator`` placement is required;
    Recipe-based jobs should use the Client API pattern above.

    Args:
        recipe: Recipe instance to augment with cross-site evaluation.
        submit_model_timeout: Timeout (seconds) for submitting models to clients. Defaults to 600.
        validation_timeout: Timeout (seconds) for validation tasks on clients. Defaults to 6000.
        participating_clients: Optional list of client names to include in cross-site evaluation. If not provided,
            all clients connected at controller start are used.

    Raises:
        ValueError: If the recipe doesn't have a framework attribute or uses an unsupported framework.
        RuntimeError: If cross-site evaluation has already been added to this recipe.

    Note:
        - Currently supports PyTorch, NumPy, and TensorFlow frameworks.
        - **NumPy recipes using `NumpyFedAvgRecipe`**: Validators (NPValidator) are automatically
          added to clients to handle validation tasks. The function intelligently detects if validators
          are already configured by checking for executors handling TASK_VALIDATION, avoiding duplicates
          for CSE-only recipes (like `NumpyCrossSiteEvalRecipe`).
        - **Unified `FedAvgRecipe` with `framework=FrameworkType.NUMPY`**: Uses the same Client API
          validation pattern as PyTorch and TensorFlow. Your client script should handle
          `flare.is_evaluate()` and return metrics for validation tasks.
        - **PyTorch recipes**: No separate validator component is needed. The client training script
          handles validation tasks through the Client API's `flare.is_evaluate()` check. See the
          hello-pt example for implementation pattern.
        - **TensorFlow recipes**: Similar to PyTorch, uses the Client API pattern. The client script
          should handle validation tasks via `flare.is_evaluate()` check.
    """
    from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
    from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
    from nvflare.job_config.script_runner import FrameworkType

    # Idempotency check: prevent multiple calls on the same recipe.
    # Keep the explicit flag fast-path, but also verify server workflow state so
    # protection remains effective even if dynamic attributes are lost.
    if getattr(recipe, "_cse_added", False) or _has_cross_site_eval_workflow(recipe.job):
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
        FrameworkType.NUMPY: "numpy",
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
                    f"Ensure your recipe includes a model to create a persistor."
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
        participating_clients=participating_clients,
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


def collect_non_local_scripts(job: FedJob) -> List[str]:
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


def ensure_config_type_dict(config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Ensure a component config dict has config_type 'dict' and is normalized for the config layer.

    Used by FedOpt-style recipes for optimizer_args and lr_scheduler_args: those dicts have 'path' or
    'class_path' plus 'args', and would otherwise be treated as component configs and instantiated
    during config scan (e.g. torch.optim.SGD without params). This function:
    - Accepts either 'path' or 'class_path' (for consistency with recipe model_config); if only
      'class_path' is set, copies it to 'path' so the component builder and runtime code work unchanged.
    - Sets config_type to 'dict' when missing so the component builder does not instantiate at load time;
      the optimizer/scheduler is instantiated at runtime when params/optimizer are available.

    Args:
        config: A component-style config dict (e.g. {'class_path': 'torch.optim.SGD', 'args': {'lr': 1.0}}
                or {'path': '...', 'args': {...}}) or None.

    Returns:
        A copy of config with config_type 'dict' if missing and path set from class_path if needed; None if config is None.
    """
    if config is None:
        return None
    out = copy.copy(config)
    if out.get("path") is None and out.get("class_path") is not None:
        out["path"] = out["class_path"]
    if out.get("config_type") is None:
        out["config_type"] = "dict"
    return out


def validate_ckpt(ckpt: Optional[str]) -> None:
    """Validate a checkpoint path if provided.

    For absolute paths: no local existence check (file may be a server-side path).
    For relative paths: verifies the file exists locally (it will be bundled into the job).

    Args:
        ckpt: Checkpoint file path to validate (e.g. initial_ckpt or eval_ckpt).

    Raises:
        ValueError: If relative path does not exist locally.
    """
    if ckpt is not None:
        if not os.path.isabs(ckpt):
            if not os.path.isfile(ckpt):
                raise ValueError(
                    f"Checkpoint relative path does not exist locally: {ckpt}. "
                    "Relative paths are treated as local files that will be bundled into the job. "
                    "Use an absolute path for server-side checkpoints."
                )


def prepare_initial_ckpt(initial_ckpt: Optional[str], job) -> Optional[str]:
    """Prepare initial_ckpt for job deployment.

    - Relative path: treated as a local file. The file is bundled into the server
      app's custom directory and the basename is returned for runtime resolution.
    - Absolute path: treated as a server-side (remote) path and returned as-is.
      The file is expected to exist on the server at runtime.

    Args:
        initial_ckpt: Checkpoint file path (absolute or relative).
        job: BaseFedJob instance to add the file to.

    Returns:
        The checkpoint path to pass to the persistor:
        - None if initial_ckpt is None
        - Basename for relative paths (file is bundled into app/custom/)
        - Absolute path as-is for server-side checkpoints
    """
    if initial_ckpt is None:
        return None
    if os.path.isabs(initial_ckpt):
        # Absolute path: server-side checkpoint, use as-is
        return initial_ckpt
    # Relative path: bundle local file into server app's custom/ directory
    job.add_file_to_server(initial_ckpt)
    return os.path.basename(initial_ckpt)


def extract_persistor_id(result: Any) -> str:
    if isinstance(result, dict):
        persistor_id = result.get("persistor_id", "")
        return persistor_id if isinstance(persistor_id, str) else ""
    if isinstance(result, str):
        return result
    return ""


def resolve_initial_ckpt(initial_ckpt: Optional[str], prepared_initial_ckpt: Optional[str], job) -> Optional[str]:
    if prepared_initial_ckpt is not None:
        return prepared_initial_ckpt
    return prepare_initial_ckpt(initial_ckpt, job)


def setup_custom_persistor(*, job, model_persistor=None) -> str:
    if model_persistor is None:
        return ""
    return extract_persistor_id(job.to_server(model_persistor, id="persistor"))


def _resolve_recipe_model_class_path(recipe_model: Dict[str, Any]) -> str:
    if "class_path" in recipe_model:
        key = "class_path"
    elif "path" in recipe_model:
        key = "path"
    else:
        raise ValueError(
            "Dict model config must have 'class_path' or 'path' key with fully qualified class path. "
            f"Got: {recipe_model}"
        )

    class_path = recipe_model[key]
    if not isinstance(class_path, str):
        raise ValueError(f"Dict model config '{key}' must be a string, got: {type(class_path)}")
    return class_path


def validate_dict_model_config(model: Any) -> None:
    """Validate recipe dict model config structure.

    Recipes accept model config with ``class_path`` or the ``path`` alias.
    The job/config layer uses ``path``.

    Args:
        model: Model input to validate.

    Raises:
        ValueError: If dict config is missing 'class_path'/'path' or value is not a string.
    """
    if isinstance(model, dict):
        _resolve_recipe_model_class_path(model)


def recipe_model_to_job_model(recipe_model: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and convert recipe model dict to job/config format (path).

    Calls :func:`validate_dict_model_config` internally so callers do not need to
    validate separately. Recipes accept {"class_path": "module.Class", "args": {...}}
    or {"path": "module.Class", "args": {...}}.
    The Job API and config parsing expect {"path": "module.Class", "args": {...}}.

    Args:
        recipe_model: Dict with 'class_path' or 'path' and optional 'args'.

    Returns:
        Dict with 'path' and 'args' for use by PTModel, persistors, etc.
    """
    return {"path": _resolve_recipe_model_class_path(recipe_model), "args": recipe_model.get("args", {})}
