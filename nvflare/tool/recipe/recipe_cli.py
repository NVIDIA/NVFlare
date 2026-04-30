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

import ast
import importlib
import inspect
import pkgutil
import sys
from enum import Enum
from pathlib import Path

from nvflare.tool.cli_output import output_usage_error

_RECIPE_PACKAGE_ROOTS = [
    {"package": "nvflare.recipe", "framework": "core"},
    {"package": "nvflare.app_opt.pt.recipes", "framework": "pytorch"},
    {"package": "nvflare.app_opt.tf.recipes", "framework": "tensorflow"},
    {"package": "nvflare.app_opt.sklearn.recipes", "framework": "sklearn"},
    {"package": "nvflare.app_opt.xgboost.recipes", "framework": "xgboost"},
]
_FILTER_KEYS = {"framework", "privacy", "algorithm", "aggregation", "state_exchange"}
_JSON_OUTPUT_MODES = ["json"]
_NO_RETRY_TOKEN_SCHEMA = {"supported": False}
_LIST_METADATA_KEYS = {"privacy"}
_CATALOG_RECIPE_CLASS_KEY = "_recipe_cls"
_CORE_FRAMEWORK_SUPPORT = {
    "cyclic": ["pytorch", "tensorflow", "numpy", "raw"],
    "fedavg": ["pytorch", "tensorflow", "sklearn", "numpy", "raw"],
    "fedstats": ["framework_agnostic"],
}
_NVFLARE_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_DOCUMENTED_RECIPE_SPECS = {
    "fedavg-pt": {
        "module": "nvflare.app_opt.pt.recipes.fedavg",
        "class": "FedAvgRecipe",
        "description": "A recipe for implementing Federated Averaging (FedAvg) for PyTorch.",
        "framework": "pytorch",
        "algorithm": "fedavg",
        "aggregation": "weighted_average",
        "state_exchange": "full_model",
    },
    "fedavg-tf": {
        "module": "nvflare.app_opt.tf.recipes.fedavg",
        "class": "FedAvgRecipe",
        "description": "A recipe for implementing Federated Averaging (FedAvg) for TensorFlow.",
        "framework": "tensorflow",
        "algorithm": "fedavg",
        "aggregation": "weighted_average",
        "state_exchange": "full_model",
    },
    "fedavg-numpy": {
        "module": "nvflare.app_common.np.recipes.fedavg",
        "class": "NumpyFedAvgRecipe",
        "description": "A recipe for implementing Federated Averaging (FedAvg) for NumPy-based models.",
        "framework": "numpy",
        "algorithm": "fedavg",
        "aggregation": "weighted_average",
        "state_exchange": "full_model",
        "framework_support": ["numpy", "raw"],
    },
    "fedavg-sklearn": {
        "module": "nvflare.app_opt.sklearn.recipes.fedavg",
        "class": "SklearnFedAvgRecipe",
        "description": "A recipe for implementing Federated Averaging (FedAvg) with Scikit-learn.",
        "framework": "sklearn",
        "algorithm": "fedavg",
        "aggregation": "weighted_average",
        "state_exchange": "full_model",
    },
    "fedavg-he-pt": {
        "module": "nvflare.app_opt.pt.recipes.fedavg_he",
        "class": "FedAvgRecipeWithHE",
        "description": "A recipe for implementing Federated Averaging (FedAvg) with Homomorphic Encryption.",
        "framework": "pytorch",
        "algorithm": "fedavg",
        "aggregation": "weighted_average",
        "state_exchange": "full_model",
        "privacy": ["homomorphic_encryption"],
        "privacy_compatible": ["homomorphic_encryption"],
        "optional_dependencies": ["pip install nvflare[PT]", "pip install torch", "pip install tenseal"],
    },
    "fedprox-pt": {
        "module": "nvflare.app_opt.pt.recipes.fedavg",
        "class": "FedAvgRecipe",
        "description": "FedProx pattern using the PyTorch FedAvg recipe with a FedProx client loss.",
        "framework": "pytorch",
        "algorithm": "fedprox",
        "aggregation": "weighted_average",
        "state_exchange": "full_model",
        "heterogeneity_support": ["non_iid"],
    },
    "fedprox-tf": {
        "module": "nvflare.app_opt.tf.recipes.fedavg",
        "class": "FedAvgRecipe",
        "description": "FedProx pattern using the TensorFlow FedAvg recipe with a FedProx client loss.",
        "framework": "tensorflow",
        "algorithm": "fedprox",
        "aggregation": "weighted_average",
        "state_exchange": "full_model",
        "heterogeneity_support": ["non_iid"],
    },
    "fedopt-pt": {
        "module": "nvflare.app_opt.pt.recipes.fedopt",
        "class": "FedOptRecipe",
        "description": "A recipe for implementing Federated Optimization (FedOpt) in PyTorch.",
        "framework": "pytorch",
        "algorithm": "fedopt",
        "aggregation": "server_optimizer",
        "state_exchange": "weight_diff",
    },
    "fedopt-tf": {
        "module": "nvflare.app_opt.tf.recipes.fedopt",
        "class": "FedOptRecipe",
        "description": "A recipe for implementing Federated Optimization (FedOpt) in TensorFlow.",
        "framework": "tensorflow",
        "algorithm": "fedopt",
        "aggregation": "server_optimizer",
        "state_exchange": "weight_diff",
    },
    "scaffold-pt": {
        "module": "nvflare.app_opt.pt.recipes.scaffold",
        "class": "ScaffoldRecipe",
        "description": "A recipe for implementing SCAFFOLD in PyTorch.",
        "framework": "pytorch",
        "algorithm": "scaffold",
        "aggregation": "weighted_average",
        "state_exchange": "full_model",
        "heterogeneity_support": ["non_iid"],
    },
    "scaffold-tf": {
        "module": "nvflare.app_opt.tf.recipes.scaffold",
        "class": "ScaffoldRecipe",
        "description": "A recipe for implementing SCAFFOLD in TensorFlow.",
        "framework": "tensorflow",
        "algorithm": "scaffold",
        "aggregation": "weighted_average",
        "state_exchange": "full_model",
        "heterogeneity_support": ["non_iid"],
    },
    "cyclic-pt": {
        "module": "nvflare.app_opt.pt.recipes.cyclic",
        "class": "CyclicRecipe",
        "description": "PyTorch-specific cyclic federated learning recipe.",
        "framework": "pytorch",
        "algorithm": "cyclic",
        "aggregation": None,
        "state_exchange": "full_model",
    },
    "cyclic-tf": {
        "module": "nvflare.app_opt.tf.recipes.cyclic",
        "class": "CyclicRecipe",
        "description": "TensorFlow-specific cyclic federated learning recipe.",
        "framework": "tensorflow",
        "algorithm": "cyclic",
        "aggregation": None,
        "state_exchange": "full_model",
    },
    "xgb-horizontal": {
        "module": "nvflare.app_opt.xgboost.recipes.histogram",
        "class": "XGBHorizontalRecipe",
        "description": "Histogram-based federated XGBoost for horizontal data partitioning.",
        "framework": "xgboost",
        "algorithm": "xgboost_horizontal",
        "aggregation": "tree_ensemble",
        "state_exchange": "trees",
        "heterogeneity_support": ["horizontal"],
        "privacy_compatible": ["homomorphic_encryption"],
    },
    "xgb-bagging": {
        "module": "nvflare.app_opt.xgboost.recipes.bagging",
        "class": "XGBBaggingRecipe",
        "description": "Tree-based federated XGBoost using bagging.",
        "framework": "xgboost",
        "algorithm": "xgboost_bagging",
        "aggregation": "tree_ensemble",
        "state_exchange": "trees",
        "heterogeneity_support": ["horizontal"],
    },
    "xgb-vertical": {
        "module": "nvflare.app_opt.xgboost.recipes.vertical",
        "class": "XGBVerticalRecipe",
        "description": "Federated XGBoost for vertical data partitioning.",
        "framework": "xgboost",
        "algorithm": "xgboost_vertical",
        "aggregation": "tree_ensemble",
        "state_exchange": "trees",
        "heterogeneity_support": ["vertical"],
        "privacy_compatible": ["homomorphic_encryption", "private_set_intersection"],
    },
    "kmeans-sklearn": {
        "module": "nvflare.app_opt.sklearn.recipes.kmeans",
        "class": "KMeansFedAvgRecipe",
        "description": "A recipe for Federated K-Means Clustering with Scikit-learn.",
        "framework": "sklearn",
        "algorithm": "kmeans",
        "aggregation": "cluster_centers",
        "state_exchange": "cluster_centers",
    },
    "svm-sklearn": {
        "module": "nvflare.app_opt.sklearn.recipes.svm",
        "class": "SVMFedAvgRecipe",
        "description": "A recipe for Federated SVM with Scikit-learn.",
        "framework": "sklearn",
        "algorithm": "svm",
        "aggregation": "support_vectors",
        "state_exchange": "support_vectors",
    },
    "lr": {
        "module": "nvflare.app_common.np.recipes.lr.fedavg",
        "class": "FedAvgLrRecipe",
        "description": "A recipe for federated logistic regression.",
        "framework": "numpy",
        "algorithm": "fedavg_logistic_regression",
        "aggregation": "weighted_average",
        "state_exchange": "model_weights",
        "framework_support": ["numpy", "sklearn"],
    },
    "fedstats": {
        "module": "nvflare.recipe.fedstats",
        "class": "FedStatsRecipe",
        "description": "A recipe for federated statistics computation.",
        "framework": "core",
        "algorithm": "fedstats",
        "aggregation": None,
        "state_exchange": None,
        "framework_support": ["framework_agnostic"],
        "heterogeneity_support": ["cross_site_statistics"],
    },
    "fedeval-pt": {
        "module": "nvflare.app_opt.pt.recipes.fedeval",
        "class": "FedEvalRecipe",
        "description": "A recipe for federated evaluation of a PyTorch model across multiple sites.",
        "framework": "pytorch",
        "algorithm": "fedeval",
        "aggregation": None,
        "state_exchange": "full_model",
    },
}


def _normalize_filter_value(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def _normalize_recipe_name(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


def _recipe_attr(recipe_cls, name: str, default=None):
    value = getattr(recipe_cls, f"recipe_{name}", None)
    if value is None:
        value = getattr(recipe_cls, name, None)
    return default if value is None else value


def _as_string_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [_normalize_filter_value(value)] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [_normalize_filter_value(v) for v in value if str(v).strip()]
    return [_normalize_filter_value(value)]


def _as_preserved_string_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)]


def _module_source_path(module_name: str):
    if not module_name:
        return None
    parts = module_name.split(".")
    if not parts or parts[0] != "nvflare":
        return None
    return _NVFLARE_PACKAGE_ROOT.joinpath(*parts[1:]).with_suffix(".py")


def _ast_default_value(node):
    if node is None:
        return None
    try:
        return _json_safe_value(ast.literal_eval(node))
    except (ValueError, TypeError):
        try:
            return ast.unparse(node)
        except Exception:
            return None


def _ast_annotation_to_string(node):
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _ast_parameter(name: str, annotation, default_node, kind: str, required: bool) -> dict:
    return {
        "name": name,
        "type": _ast_annotation_to_string(annotation),
        "required": required,
        "default": _ast_default_value(default_node),
        "kind": kind,
    }


def _static_recipe_parameters(module_name: str, class_name: str) -> list:
    path = _module_source_path(module_name)
    if not path or not path.is_file():
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []

    class_node = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name), None)
    if class_node is None:
        return []
    init_node = next(
        (
            node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__"
        ),
        None,
    )
    if init_node is None:
        return []

    params = []
    args = init_node.args
    positional = list(args.posonlyargs) + list(args.args)
    positional_defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for index, (arg, default_node) in enumerate(zip(positional, positional_defaults)):
        if arg.arg == "self":
            continue
        kind = "positional_only" if index < len(args.posonlyargs) else "positional_or_keyword"
        params.append(_ast_parameter(arg.arg, arg.annotation, default_node, kind, default_node is None))

    if args.vararg is not None:
        params.append(_ast_parameter(args.vararg.arg, args.vararg.annotation, None, "var_positional", False))

    for arg, default_node in zip(args.kwonlyargs, args.kw_defaults):
        params.append(_ast_parameter(arg.arg, arg.annotation, default_node, "keyword_only", default_node is None))

    if args.kwarg is not None:
        params.append(_ast_parameter(args.kwarg.arg, args.kwarg.annotation, None, "var_keyword", False))

    return params


def _try_import_recipe_class(module_name: str, class_name: str):
    try:
        module = importlib.import_module(module_name)
    except (ImportError, SyntaxError):
        return None
    return getattr(module, class_name, None)


def _infer_algorithm(cli_name: str, recipe_cls) -> str:
    text = _normalize_filter_value(f"{cli_name} {recipe_cls.__name__} {recipe_cls.__module__}")
    algorithm_markers = [
        ("kmeans", "kmeans"),
        ("svm", "svm"),
        ("fedavg", "fedavg"),
        ("fedopt", "fedopt"),
        ("scaffold", "scaffold"),
        ("cyclic", "cyclic"),
        ("swarm", "swarm"),
        ("fedstats", "fedstats"),
        ("fedeval", "fedeval"),
        ("cross_site_eval", "cross_site_eval"),
        ("cross_site", "cross_site_eval"),
        ("bagging", "xgboost_bagging"),
        ("vertical", "xgboost_vertical"),
        ("histogram", "xgboost_horizontal"),
        ("xgb", "xgboost"),
        ("psi", "psi"),
    ]
    for marker, algorithm in algorithm_markers:
        if marker in text:
            return algorithm
    return None


def _infer_aggregation(algorithm: str) -> str:
    if algorithm in {"fedavg", "scaffold"}:
        return "weighted_average"
    if algorithm == "fedopt":
        return "server_optimizer"
    if algorithm == "kmeans":
        return "cluster_centers"
    if algorithm == "svm":
        return "support_vectors"
    if algorithm and algorithm.startswith("xgboost"):
        return "tree_ensemble"
    return None


def _infer_state_exchange(algorithm: str) -> str:
    if algorithm == "fedopt":
        return "weight_diff"
    if algorithm in {"fedavg", "scaffold", "cyclic", "swarm", "fedeval"}:
        return "full_model"
    if algorithm == "kmeans":
        return "cluster_centers"
    if algorithm == "svm":
        return "support_vectors"
    if algorithm and algorithm.startswith("xgboost"):
        return "trees"
    return None


def _infer_privacy(cli_name: str, recipe_cls) -> list:
    text = _normalize_filter_value(f"{cli_name} {recipe_cls.__name__} {recipe_cls.__module__}")
    privacy = []
    if "_he" in text or "he_" in text or "withhe" in text or "homomorphic" in text:
        privacy.append("homomorphic_encryption")
    if "differential_privacy" in text or "_dp" in text or " dp" in text:
        privacy.append("differential_privacy")
    return privacy


def _recipe_metadata(cli_name: str, recipe_cls) -> dict:
    algorithm = _recipe_attr(recipe_cls, "algorithm") or _infer_algorithm(cli_name, recipe_cls)
    aggregation = _recipe_attr(recipe_cls, "aggregation") or _infer_aggregation(algorithm)
    state_exchange = _recipe_attr(recipe_cls, "state_exchange") or _infer_state_exchange(algorithm)
    privacy = _as_string_list(_recipe_attr(recipe_cls, "privacy")) or _infer_privacy(cli_name, recipe_cls)
    return {
        "algorithm": _normalize_filter_value(algorithm) if algorithm else None,
        "aggregation": _normalize_filter_value(aggregation) if aggregation else None,
        "state_exchange": _normalize_filter_value(state_exchange) if state_exchange else None,
        "privacy": privacy,
    }


def _json_safe_value(value):
    if value is inspect.Signature.empty or value is inspect.Parameter.empty:
        return None
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    return repr(value)


def _annotation_to_string(annotation) -> str:
    if annotation is inspect.Signature.empty or annotation is inspect.Parameter.empty:
        return None
    return inspect.formatannotation(annotation).replace("typing.", "")


def _recipe_parameters(recipe_cls) -> list:
    try:
        signature = inspect.signature(recipe_cls.__init__)
    except (TypeError, ValueError):
        return []

    parameters = []
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        required = param.default is inspect.Parameter.empty and param.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
        parameters.append(
            {
                "name": name,
                "type": _annotation_to_string(param.annotation),
                "required": required,
                "default": _json_safe_value(param.default),
                "kind": param.kind.name.lower(),
            }
        )
    return parameters


def _entry_parameters(entry: dict, recipe_cls) -> list:
    if recipe_cls:
        params = _recipe_parameters(recipe_cls)
        if params:
            return params
    return _static_recipe_parameters(entry.get("module"), entry.get("class"))


def _framework_support(entry: dict, recipe_cls) -> list:
    explicit = _as_string_list(
        entry.get("framework_support")
        or _recipe_attr(recipe_cls, "framework_support")
        or _recipe_attr(recipe_cls, "frameworks")
        or _recipe_attr(recipe_cls, "supported_frameworks")
    )
    if explicit:
        return explicit

    framework = entry.get("framework")
    if framework == "core":
        return _CORE_FRAMEWORK_SUPPORT.get(entry.get("algorithm"), ["framework_agnostic"])
    return [framework] if framework else []


def _optional_dependencies(entry: dict, recipe_cls) -> list:
    explicit = entry.get("optional_dependencies") or _recipe_attr(recipe_cls, "optional_dependencies")
    if explicit is not None:
        return _as_preserved_string_list(explicit)

    framework = entry.get("framework")
    if framework and framework != "core":
        return _framework_install_hint(framework)
    return []


def _heterogeneity_support(entry: dict, recipe_cls) -> list:
    explicit = _as_string_list(
        _recipe_attr(recipe_cls, "heterogeneity_support")
        or _recipe_attr(recipe_cls, "supported_heterogeneity")
        or entry.get("heterogeneity_support")
    )
    if explicit:
        return explicit

    algorithm = entry.get("algorithm")
    if algorithm == "xgboost_vertical":
        return ["vertical"]
    if algorithm in {"xgboost_bagging", "xgboost_horizontal"}:
        return ["horizontal"]
    if algorithm == "fedstats":
        return ["cross_site_statistics"]
    if algorithm == "psi":
        return ["sample_intersection"]
    return ["horizontal"]


def _privacy_compatible(entry: dict, parameters: list, recipe_cls) -> list:
    privacy = set(_as_string_list(entry.get("privacy")))
    privacy.update(_as_string_list(entry.get("privacy_compatible")))
    privacy.update(_as_string_list(_recipe_attr(recipe_cls, "privacy_compatible")))
    parameter_names = {p["name"] for p in parameters}
    if "secure" in parameter_names:
        privacy.add("homomorphic_encryption")
    if entry.get("algorithm") == "xgboost_vertical":
        privacy.add("private_set_intersection")
    return sorted(privacy)


def _client_requirements(entry: dict, parameters: list) -> dict:
    by_name = {p["name"]: p for p in parameters}
    requirements = {
        "state_exchange": entry.get("state_exchange"),
        "requires_training_script": "train_script" in by_name,
        "requires_per_site_config": "per_site_config" in by_name,
        "requires_site_list": "sites" in by_name,
    }
    for name in ("min_clients", "sites", "label_owner", "client_ranks"):
        parameter = by_name.get(name)
        if parameter:
            requirements[name] = {
                "required": parameter["required"],
                "default": parameter["default"],
            }
    return requirements


def _recipe_detail(entry: dict) -> dict:
    recipe_cls = entry.get(_CATALOG_RECIPE_CLASS_KEY)
    parameters = _entry_parameters(entry, recipe_cls)
    detail = {
        "name": entry.get("name"),
        "description": entry.get("description"),
        "framework": entry.get("framework"),
        "module": entry.get("module"),
        "class": entry.get("class"),
        "algorithm": entry.get("algorithm"),
        "aggregation": entry.get("aggregation"),
        "state_exchange": entry.get("state_exchange"),
        "privacy": entry.get("privacy"),
        "client_requirements": _client_requirements(entry, parameters),
        "framework_support": _framework_support(entry, recipe_cls),
        "heterogeneity_support": _heterogeneity_support(entry, recipe_cls),
        "privacy_compatible": _privacy_compatible(entry, parameters, recipe_cls),
        "parameters": parameters,
        "optional_dependencies": _optional_dependencies(entry, recipe_cls),
        "template_references": _as_preserved_string_list(
            entry.get("template_references") or _recipe_attr(recipe_cls, "template_references")
        ),
    }
    return detail


def _parse_recipe_filters(raw_filters: list) -> dict:
    parsed = {}
    for raw_filter in raw_filters or []:
        if "=" not in raw_filter:
            raise ValueError(f"invalid filter '{raw_filter}'; expected key=value")
        key, value = raw_filter.split("=", 1)
        key = _normalize_filter_value(key)
        value = _normalize_filter_value(value)
        if key not in _FILTER_KEYS:
            raise ValueError(f"unsupported filter key '{key}'")
        if not value:
            raise ValueError(f"filter '{key}' requires a non-empty value")
        parsed.setdefault(key, set()).add(value)
    return parsed


def _entry_matches_filters(entry: dict, filters: dict) -> bool:
    for key, expected_values in filters.items():
        actual_value = entry.get(key)
        if key in _LIST_METADATA_KEYS:
            actual_values = set(_as_string_list(actual_value))
            if not actual_values.intersection(expected_values):
                return False
        else:
            if _normalize_filter_value(actual_value) not in expected_values:
                return False
    return True


def _filter_catalog(catalog: list, filters: dict) -> list:
    if not filters:
        return catalog
    return [entry for entry in catalog if _entry_matches_filters(entry, filters)]


def _documented_recipe_entry(name: str, spec: dict, include_recipe_class: bool = False) -> dict:
    entry = {
        "name": name,
        "description": spec.get("description"),
        "framework": spec.get("framework"),
        "module": spec.get("module"),
        "class": spec.get("class"),
        "algorithm": spec.get("algorithm"),
        "aggregation": spec.get("aggregation"),
        "state_exchange": spec.get("state_exchange"),
        "privacy": _as_string_list(spec.get("privacy")),
    }
    for key in (
        "framework_support",
        "heterogeneity_support",
        "privacy_compatible",
        "optional_dependencies",
        "template_references",
    ):
        if key in spec:
            entry[key] = spec[key]

    if include_recipe_class:
        recipe_cls = _try_import_recipe_class(spec.get("module"), spec.get("class"))
        if recipe_cls is not None:
            entry[_CATALOG_RECIPE_CLASS_KEY] = recipe_cls
    return entry


def _apply_documented_recipe_specs(catalog: list, include_recipe_class: bool = False) -> list:
    by_name = {_normalize_recipe_name(entry["name"]): entry for entry in catalog}
    for name, spec in _DOCUMENTED_RECIPE_SPECS.items():
        normalized_name = _normalize_recipe_name(name)
        spec_entry = _documented_recipe_entry(name, spec, include_recipe_class=include_recipe_class)
        if normalized_name in by_name:
            existing = by_name[normalized_name]
            recipe_cls = existing.get(_CATALOG_RECIPE_CLASS_KEY)
            existing.update({k: v for k, v in spec_entry.items() if v is not None})
            if recipe_cls is not None:
                existing[_CATALOG_RECIPE_CLASS_KEY] = recipe_cls
            elif include_recipe_class and spec_entry.get(_CATALOG_RECIPE_CLASS_KEY) is not None:
                existing[_CATALOG_RECIPE_CLASS_KEY] = spec_entry[_CATALOG_RECIPE_CLASS_KEY]
        else:
            catalog.append(spec_entry)
            by_name[normalized_name] = spec_entry
    catalog.sort(key=lambda entry: entry["name"])
    return catalog


def _framework_install_hint(framework: str = None) -> list[str]:
    if framework in {"core", "numpy"}:
        return []
    if framework == "pytorch":
        return ["pip install nvflare[PT]", "pip install torch"]
    if framework == "sklearn":
        return ["pip install nvflare[SKLEARN]", "pip install scikit-learn"]
    if framework == "tensorflow":
        return ["pip install tensorflow"]
    if framework == "xgboost":
        return ["pip install xgboost"]
    return [
        "pip install nvflare[PT,SKLEARN]",
        "pip install tensorflow xgboost",
    ]


def _framework_install_hint_text(framework: str = None) -> str:
    return "Try: " + " ; ".join(_framework_install_hint(framework))


def _recipe_cli_name(module_name: str, framework: str) -> str:
    stem = module_name.rsplit(".", 1)[-1].replace("_", "-")
    if framework == "core":
        return stem
    if framework == "pytorch":
        return f"{stem}-pt"
    if framework == "tensorflow":
        return f"{stem}-tf"
    if framework == "sklearn":
        return f"{stem}-sklearn"
    if framework == "xgboost":
        if stem == "histogram":
            return "xgb-horizontal"
        return f"xgb-{stem}"
    return stem


def _recipe_description(recipe_cls) -> str:
    doc = inspect.getdoc(recipe_cls) or ""
    if not doc:
        return f"{recipe_cls.__name__} recipe"
    return next((line.strip() for line in doc.splitlines() if line.strip()), f"{recipe_cls.__name__} recipe")


def _iter_recipe_classes(module):
    from nvflare.recipe.spec import Recipe

    for _name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        if obj is Recipe:
            continue
        if issubclass(obj, Recipe):
            yield obj


def _select_recipe_class(module):
    """Select the single CLI-exposed recipe class for a module.

    Recipe discovery is module-oriented: the CLI name comes from the module name, so a
    module contributes at most one catalog entry. If a module defines both a reusable base
    recipe and a concrete subclass, prefer the leaf subclass.
    """

    classes = list(_iter_recipe_classes(module))
    if not classes:
        return None
    if len(classes) == 1:
        return classes[0]

    leaf_classes = [cls for cls in classes if not any(cls is not other and issubclass(other, cls) for other in classes)]
    return leaf_classes[0] if leaf_classes else classes[0]


def _load_catalog(framework: str = None, include_recipe_class: bool = False) -> list:
    """Return available recipes, filtered by framework if given.

    Dynamically discovered recipes are supplemented with documented recipe
    metadata so recipes remain queryable before optional dependencies are installed.
    """
    results = []
    seen = set()
    for root in _RECIPE_PACKAGE_ROOTS:
        if framework and root["framework"] != framework:
            continue
        try:
            package = importlib.import_module(root["package"])
        except (ImportError, SyntaxError):
            pass

        else:
            for _finder, module_name, _is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
                if module_name.endswith(".__init__"):
                    continue
                try:
                    mod = importlib.import_module(module_name)
                except (ImportError, SyntaxError):
                    continue

                recipe_cls = _select_recipe_class(mod)
                if recipe_cls is None:
                    continue

                cli_name = _recipe_cli_name(module_name, root["framework"])
                if cli_name in seen:
                    continue
                seen.add(cli_name)
                entry = {
                    "name": cli_name,
                    "description": _recipe_description(recipe_cls),
                    "framework": root["framework"],
                    "module": module_name,
                    "class": recipe_cls.__name__,
                }
                entry.update(_recipe_metadata(cli_name, recipe_cls))
                if include_recipe_class:
                    entry[_CATALOG_RECIPE_CLASS_KEY] = recipe_cls
                results.append(entry)

    results.sort(key=lambda entry: entry["name"])
    results = _apply_documented_recipe_specs(results, include_recipe_class=include_recipe_class)
    if framework:
        results = [entry for entry in results if entry.get("framework") == framework]
    return results


def cmd_recipe_list(cmd_args):
    from nvflare.tool.cli_output import is_json_mode, output_error_message, output_ok, print_human
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        _recipe_list_parser,
        "nvflare recipe list",
        [
            "nvflare recipe list",
            "nvflare recipe list --framework pytorch",
            "nvflare recipe list --filter framework=pytorch --filter algorithm=fedavg",
        ],
        sys.argv[1:],
        output_modes=_JSON_OUTPUT_MODES,
        streaming=False,
        mutating=False,
        idempotent=True,
        retry_token=_NO_RETRY_TOKEN_SCHEMA,
    )

    framework = getattr(cmd_args, "framework", None)
    try:
        filters = _parse_recipe_filters(getattr(cmd_args, "filters", None))
    except ValueError as e:
        output_usage_error(
            _recipe_list_parser,
            str(e),
            exit_code=4,
            hint=f"Use --filter key=value with keys: {', '.join(sorted(_FILTER_KEYS))}.",
        )
        raise SystemExit(4)

    if framework:
        normalized_framework = _normalize_filter_value(framework)
        filter_frameworks = filters.get("framework")
        if filter_frameworks and normalized_framework not in filter_frameworks:
            output_usage_error(
                _recipe_list_parser,
                f"--framework {framework} conflicts with --filter framework={','.join(sorted(filter_frameworks))}",
                exit_code=4,
                hint="Use either --framework or matching framework filters.",
            )
            raise SystemExit(4)
        filters.setdefault("framework", set()).add(normalized_framework)

    catalog = _load_catalog(framework=framework)

    if framework and not catalog:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _framework_install_hint_text(framework),
            None,
            exit_code=4,
            detail=f"no installed recipes found for framework '{framework}'",
        )
        raise SystemExit(4)

    catalog = _filter_catalog(catalog, filters)

    if is_json_mode():
        output_ok(catalog)
        return

    if filters and not catalog:
        filter_desc = ", ".join(f"{key}={','.join(sorted(values))}" for key, values in sorted(filters.items()))
        print_human(f"No recipes matched filters: {filter_desc}")
        print_human()
        return

    if not catalog:
        print_human("No recipes are currently available.")
        install_hints = _framework_install_hint(framework)
        if framework:
            print_human(f"Install the optional dependencies for '{framework}' recipes, then try again.")
        else:
            print_human("Install optional framework dependencies to make recipe entries available.")
        for hint in install_hints:
            print_human(f"  e.g. {hint}")
        print_human()
        return

    # Human-readable table to human stream (stdout by default; stderr in agent mode)
    name_w = max(len(e["name"]) for e in catalog) + 2 if catalog else 20
    fw_w = max(len(e["framework"]) for e in catalog) + 2 if catalog else 12
    print_human(f"\n  {'RECIPE':<{name_w}} {'FRAMEWORK':<{fw_w}} DESCRIPTION")
    print_human(f"  {'-' * (name_w + fw_w + 40)}")
    for entry in catalog:
        print_human(f"  {entry['name']:<{name_w}} {entry['framework']:<{fw_w}} {entry['description']}")
    print_human()


def cmd_recipe_show(cmd_args):
    from nvflare.tool.cli_output import is_json_mode, output_error_message, output_ok, print_human
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        _recipe_show_parser,
        "nvflare recipe show",
        [
            "nvflare recipe show fedavg-pt",
            "nvflare recipe show fedavg-pt --format json",
        ],
        sys.argv[1:],
        output_modes=_JSON_OUTPUT_MODES,
        streaming=False,
        mutating=False,
        idempotent=True,
        retry_token=_NO_RETRY_TOKEN_SCHEMA,
    )

    requested_name = _normalize_recipe_name(getattr(cmd_args, "name", ""))
    catalog = _load_catalog(include_recipe_class=True)
    entry = next((e for e in catalog if _normalize_recipe_name(e["name"]) == requested_name), None)
    if entry is None:
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            "Run 'nvflare recipe list --format json' to see available recipe names.",
            None,
            exit_code=4,
            detail=f"unknown recipe '{getattr(cmd_args, 'name', '')}'",
        )
        raise SystemExit(4)

    detail = _recipe_detail(entry)
    if is_json_mode():
        output_ok(detail)
        return

    print_human(f"\n  recipe: {detail['name']}")
    print_human(f"  description: {detail['description']}")
    print_human(f"  algorithm: {detail['algorithm']}")
    print_human(f"  aggregation: {detail['aggregation']}")
    print_human(f"  state_exchange: {detail['state_exchange']}")
    print_human(f"  framework_support: {', '.join(detail['framework_support'])}")
    print_human(f"  privacy_compatible: {', '.join(detail['privacy_compatible']) or 'none'}")
    print_human(f"  parameters: {len(detail['parameters'])}")
    print_human()


_recipe_list_parser = None
_recipe_show_parser = None
_recipe_root_parser = None


def def_recipe_parser(sub_cmd):
    global _recipe_list_parser, _recipe_root_parser, _recipe_show_parser
    cmd = "recipe"
    parser = sub_cmd.add_parser(cmd, help="list available FL job recipes")
    _recipe_root_parser = parser
    recipe_subparser = parser.add_subparsers(title="recipe subcommands", metavar="", dest="recipe_sub_cmd")

    list_parser = recipe_subparser.add_parser("list", help="list available recipes (default)")
    list_parser.add_argument(
        "--framework",
        type=str,
        default=None,
        choices=["core", "numpy", "pytorch", "tensorflow", "sklearn", "xgboost"],
        help="filter by framework",
    )
    list_parser.add_argument(
        "--filter",
        dest="filters",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="filter by metadata; repeatable keys: framework, privacy, algorithm, aggregation, state_exchange",
    )
    list_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    list_parser.set_defaults(recipe_sub_cmd="list")
    _recipe_list_parser = list_parser

    show_parser = recipe_subparser.add_parser("show", help="show structured metadata for a recipe")
    show_parser.add_argument("name", type=str, help="recipe name returned by 'nvflare recipe list'")
    show_parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")
    show_parser.set_defaults(recipe_sub_cmd="show")
    _recipe_show_parser = show_parser

    parser.set_defaults(recipe_sub_cmd="list")

    return {cmd: parser}


def handle_recipe_cmd(args):
    sub_cmd = getattr(args, "recipe_sub_cmd", None)
    if sub_cmd == "list":
        cmd_recipe_list(args)
    elif sub_cmd == "show":
        cmd_recipe_show(args)
    else:
        output_usage_error(_recipe_root_parser, "recipe subcommand required", exit_code=4)
        raise SystemExit(4)
