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
import json
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
_CATALOG_RECIPE_ATTRS_KEY = "_recipe_attrs"
_RECIPE_CATALOG_PATH = Path(__file__).with_name("recipe_catalog.json")
_RECIPE_CATALOG_SCHEMA_VERSION = 1
_RECIPE_DETAIL_ATTR_ALIASES = {
    "framework_support": (
        "recipe_framework_support",
        "framework_support",
        "recipe_frameworks",
        "frameworks",
        "recipe_supported_frameworks",
        "supported_frameworks",
    ),
    "optional_dependencies": ("recipe_optional_dependencies", "optional_dependencies"),
    "heterogeneity_support": (
        "recipe_heterogeneity_support",
        "heterogeneity_support",
        "recipe_supported_heterogeneity",
        "supported_heterogeneity",
    ),
    "privacy_compatible": ("recipe_privacy_compatible", "privacy_compatible"),
    "notes": ("recipe_notes", "notes"),
    "template_references": ("recipe_template_references", "template_references"),
}
_CATALOG_SUMMARY_REQUIRED_FIELDS = {
    "name": str,
    "description": str,
    "framework": str,
    "module": str,
    "class": str,
    "algorithm": (str, type(None)),
    "aggregation": (str, type(None)),
    "state_exchange": (str, type(None)),
    "privacy": list,
}
_CATALOG_DETAIL_REQUIRED_FIELDS = {
    **_CATALOG_SUMMARY_REQUIRED_FIELDS,
    "client_requirements": dict,
    "framework_support": list,
    "heterogeneity_support": list,
    "privacy_compatible": list,
    "notes": list,
    "parameters": list,
    "optional_dependencies": list,
    "template_references": list,
}
_CATALOG_DETAIL_STRING_LIST_FIELDS = (
    "framework_support",
    "heterogeneity_support",
    "privacy_compatible",
    "notes",
    "optional_dependencies",
    "template_references",
)
_CATALOG_PARAMETER_REQUIRED_FIELDS = {
    "name": str,
    "type": (str, type(None)),
    "required": bool,
    "kind": str,
}
_CORE_FRAMEWORK_SUPPORT = {
    "cyclic": ["pytorch", "tensorflow", "numpy", "raw"],
    "fedavg": ["pytorch", "tensorflow", "sklearn", "numpy", "raw"],
    "fedstats": ["framework_agnostic"],
}
_NVFLARE_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_RECIPE_BASE_CLASS = ("nvflare.recipe.spec", "Recipe")
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
    "fedce-pt": {
        "module": "nvflare.app_opt.pt.recipes.fedce",
        "class": "FedCERecipe",
        "description": "PyTorch federated training via client contribution estimation (FedCE).",
        "framework": "pytorch",
        "algorithm": "fedce",
        "aggregation": "contribution_weighted_average",
        "state_exchange": "weight_diff",
        "heterogeneity_support": ["non_iid", "contribution_fairness"],
        "notes": ["Client scripts must return fedce_minus_val metadata; FedCE is not a passive FedAvg flag."],
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
        "module": "nvflare.app_opt.pt.recipes.fedprox",
        "class": "FedProxRecipe",
        "description": "A recipe for implementing FedProx with PyTorch.",
        "framework": "pytorch",
        "algorithm": "fedprox",
        "aggregation": "weighted_average",
        "state_exchange": "full_model",
        "heterogeneity_support": ["non_iid"],
        "notes": [
            "Patched PyTorch Lightning clients apply FedProx automatically; raw PyTorch clients must consume "
            "FEDPROX_MU metadata and integrate PTFedProxLoss."
        ],
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


def _recipe_metadata_attr(metadata: dict, name: str, default=None):
    for alias in _RECIPE_DETAIL_ATTR_ALIASES[name]:
        value = (metadata or {}).get(alias)
        if value is not None:
            return value
    return default


def _as_string_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [_normalize_filter_value(value)] if value.strip() else []
    if isinstance(value, set):
        return sorted(_normalize_filter_value(v) for v in value if str(v).strip())
    if isinstance(value, (list, tuple)):
        return [_normalize_filter_value(v) for v in value if str(v).strip()]
    return [_normalize_filter_value(value)]


def _as_preserved_string_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, set):
        return sorted(str(v) for v in value if str(v).strip())
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)]


def _module_source_path(module_name: str):
    if not module_name:
        return None
    parts = module_name.split(".")
    if not parts or parts[0] != "nvflare":
        return None
    path = _NVFLARE_PACKAGE_ROOT.joinpath(*parts[1:])
    module_path = path.with_suffix(".py")
    if module_path.is_file():
        return module_path
    package_path = path / "__init__.py"
    return package_path if package_path.is_file() else module_path


def _package_source_path(package_name: str):
    if not package_name:
        return None
    parts = package_name.split(".")
    if not parts or parts[0] != "nvflare":
        return None
    return _NVFLARE_PACKAGE_ROOT.joinpath(*parts[1:])


def _static_module_tree(module_name: str):
    path = _module_source_path(module_name)
    if not path or not path.is_file():
        return None
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None


def _resolve_import_from_module(module_name: str, node: ast.ImportFrom):
    if not node.level:
        return node.module
    path = _module_source_path(module_name)
    package_parts = module_name.split(".")
    if not path or path.name != "__init__.py":
        package_parts = package_parts[:-1]
    parent_levels = node.level - 1
    if parent_levels > len(package_parts):
        return None
    base_parts = package_parts[: len(package_parts) - parent_levels]
    if node.module:
        base_parts.extend(node.module.split("."))
    return ".".join(base_parts)


def _static_imports(tree: ast.Module, module_name: str = None) -> dict:
    imports = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            imported_module = _resolve_import_from_module(module_name, node) if module_name else node.module
            if not imported_module:
                continue
            for imported in node.names:
                imports[imported.asname or imported.name] = (imported_module, imported.name)
        elif isinstance(node, ast.Import):
            for imported in node.names:
                local_name = imported.asname or imported.name.split(".", 1)[0]
                imports[local_name] = (imported.name, None)
    return imports


def _static_base_references(module_name: str, tree: ast.Module, class_node: ast.ClassDef) -> list:
    imports = _static_imports(tree, module_name)
    local_classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    references = []
    for base in class_node.bases:
        if isinstance(base, ast.Name):
            if base.id in local_classes:
                references.append((module_name, base.id))
            elif base.id in imports and imports[base.id][1]:
                references.append(imports[base.id])
        elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
            imported = imports.get(base.value.id)
            if imported and imported[1] is None:
                references.append((imported[0], base.attr))
    return references


def _static_class_node(module_name: str, class_name: str, visiting=None):
    reference = (module_name, class_name)
    visiting = set(visiting or ())
    if reference in visiting:
        return module_name, None, None
    visiting.add(reference)

    tree = _static_module_tree(module_name)
    if tree is None:
        return module_name, None, None
    class_node = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name), None)
    if class_node is not None:
        return module_name, tree, class_node
    imported = _static_imports(tree, module_name).get(class_name)
    if imported and imported[1]:
        return _static_class_node(imported[0], imported[1], visiting)
    return module_name, tree, None


def _static_class_is_recipe(module_name: str, class_name: str, visiting=None) -> bool:
    reference = (module_name, class_name)
    if reference == _RECIPE_BASE_CLASS:
        return True
    visiting = set(visiting or ())
    if reference in visiting:
        return False
    visiting.add(reference)

    resolved_module, tree, class_node = _static_class_node(module_name, class_name)
    if tree is None or class_node is None:
        return False
    return any(
        _static_class_is_recipe(*base, visiting) for base in _static_base_references(resolved_module, tree, class_node)
    )


def _static_class_doc(module_name: str, class_name: str, visiting=None) -> str:
    reference = (module_name, class_name)
    visiting = set(visiting or ())
    if reference in visiting:
        return ""
    visiting.add(reference)

    resolved_module, tree, class_node = _static_class_node(module_name, class_name)
    if tree is None or class_node is None:
        return ""
    doc = ast.get_docstring(class_node)
    if doc:
        return doc
    for base in _static_base_references(resolved_module, tree, class_node):
        doc = _static_class_doc(*base, visiting)
        if doc:
            return doc
    return ""


def _static_recipe_class(module_name: str):
    tree = _static_module_tree(module_name)
    if tree is None:
        return None

    class_nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name != "Recipe"]
    candidate_names = {node.name for node in class_nodes if _static_class_is_recipe(module_name, node.name)}
    candidates = [node for node in class_nodes if node.name in candidate_names]
    if not candidates:
        return None

    inherited_names = {
        base_name
        for candidate in candidates
        for base_module, base_name in _static_base_references(module_name, tree, candidate)
        if base_module == module_name and base_name in candidate_names
    }
    leaf_candidates = [candidate for candidate in candidates if candidate.name not in inherited_names]
    recipe_class = leaf_candidates[0] if leaf_candidates else candidates[0]
    doc = _static_class_doc(module_name, recipe_class.name)
    description = next((line.strip() for line in doc.splitlines() if line.strip()), f"{recipe_class.name} recipe")
    return recipe_class, description


def _static_class_attr(class_node: ast.ClassDef, *names):
    assignments = {}
    for node in class_node.body:
        value_node = None
        target_names = []
        if isinstance(node, ast.Assign):
            value_node = node.value
            target_names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value_node = node.value
            target_names = [node.target.id]
        if value_node is None:
            continue
        for target_name in target_names:
            if target_name in names:
                assignments[target_name] = value_node

    for name in names:
        value_node = assignments.get(name)
        if value_node is None:
            continue
        try:
            return ast.literal_eval(value_node)
        except (ValueError, TypeError):
            continue
    return None


def _static_recipe_attrs(module_name: str, class_name: str, visiting=None) -> dict:
    reference = (module_name, class_name)
    visiting = set(visiting or ())
    if reference in visiting:
        return {}
    visiting.add(reference)

    resolved_module, tree, class_node = _static_class_node(module_name, class_name)
    if tree is None or class_node is None:
        return {}
    attrs = {}
    for base in _static_base_references(resolved_module, tree, class_node):
        attrs.update(_static_recipe_attrs(*base, visiting))
    for aliases in _RECIPE_DETAIL_ATTR_ALIASES.values():
        for name in aliases:
            value = _static_class_attr(class_node, name)
            if value is not None:
                attrs[name] = _json_safe_value(value)
    return attrs


def _infer_algorithm(cli_name: str, class_name: str, module_name: str) -> str:
    text = _normalize_filter_value(f"{cli_name} {class_name} {module_name}")
    algorithm_markers = [
        ("kmeans", "kmeans"),
        ("svm", "svm"),
        ("fedprox", "fedprox"),
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
    if algorithm in {"fedavg", "fedprox", "scaffold"}:
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
    if algorithm in {"fedavg", "fedprox", "scaffold", "cyclic", "swarm", "fedeval"}:
        return "full_model"
    if algorithm == "kmeans":
        return "cluster_centers"
    if algorithm == "svm":
        return "support_vectors"
    if algorithm and algorithm.startswith("xgboost"):
        return "trees"
    return None


def _infer_privacy(cli_name: str, class_name: str, module_name: str) -> list:
    text = _normalize_filter_value(f"{cli_name} {class_name} {module_name}")
    privacy = []
    if "_he" in text or "he_" in text or "withhe" in text or "homomorphic" in text:
        privacy.append("homomorphic_encryption")
    if "differential_privacy" in text or "_dp" in text or " dp" in text:
        privacy.append("differential_privacy")
    return privacy


def _static_recipe_metadata(cli_name: str, module_name: str, class_node: ast.ClassDef) -> dict:
    algorithm = _static_class_attr(class_node, "recipe_algorithm", "algorithm") or _infer_algorithm(
        cli_name, class_node.name, module_name
    )
    aggregation = _static_class_attr(class_node, "recipe_aggregation", "aggregation") or _infer_aggregation(algorithm)
    state_exchange = _static_class_attr(class_node, "recipe_state_exchange", "state_exchange") or _infer_state_exchange(
        algorithm
    )
    privacy = _as_string_list(_static_class_attr(class_node, "recipe_privacy", "privacy")) or _infer_privacy(
        cli_name, class_node.name, module_name
    )
    return {
        "algorithm": _normalize_filter_value(algorithm) if algorithm else None,
        "aggregation": _normalize_filter_value(aggregation) if aggregation else None,
        "state_exchange": _normalize_filter_value(state_exchange) if state_exchange else None,
        "privacy": privacy,
    }


def _static_member_value(module_name: str, class_name: str, member_name: str):
    _, _, class_node = _static_class_node(module_name, class_name)
    if class_node is None:
        return None
    for node in class_node.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == member_name for target in node.targets
        ):
            try:
                return ast.literal_eval(node.value)
            except (ValueError, TypeError):
                return None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == member_name:
            try:
                return ast.literal_eval(node.value)
            except (ValueError, TypeError):
                return None
    return None


def _ast_default_value(node, imports: dict = None):
    if node is None:
        return None
    try:
        return _json_safe_value(ast.literal_eval(node))
    except (ValueError, TypeError):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            imported = (imports or {}).get(node.value.id)
            if imported:
                value = _static_member_value(imported[0], imported[1], node.attr)
                if value is not None:
                    return _json_safe_value(value)
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


def _ast_parameter(name: str, annotation, default_node, kind: str, required: bool, imports: dict) -> dict:
    return {
        "name": name,
        "type": _ast_annotation_to_string(annotation),
        "required": required,
        "default": _ast_default_value(default_node, imports),
        "kind": kind,
    }


def _static_class_function(module_name: str, class_name: str, function_name: str, visiting=None):
    reference = (module_name, class_name)
    visiting = set(visiting or ())
    if reference in visiting:
        return None, None
    visiting.add(reference)

    resolved_module, tree, class_node = _static_class_node(module_name, class_name)
    if tree is None or class_node is None:
        return None, None
    function_node = next(
        (
            node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
        ),
        None,
    )
    if function_node is not None:
        return resolved_module, function_node
    for base in _static_base_references(resolved_module, tree, class_node):
        function_module, function_node = _static_class_function(*base, function_name, visiting)
        if function_node is not None:
            return function_module, function_node
    return None, None


def _static_recipe_parameters(module_name: str, class_name: str) -> list:
    init_module, init_node = _static_class_function(module_name, class_name, "__init__")
    if init_node is None:
        return []

    tree = _static_module_tree(init_module)
    imports = _static_imports(tree, init_module)
    params = []
    args = init_node.args
    positional = list(args.posonlyargs) + list(args.args)
    positional_defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for index, (arg, default_node) in enumerate(zip(positional, positional_defaults)):
        if arg.arg == "self":
            continue
        kind = "positional_only" if index < len(args.posonlyargs) else "positional_or_keyword"
        params.append(_ast_parameter(arg.arg, arg.annotation, default_node, kind, default_node is None, imports))

    if args.vararg is not None:
        params.append(_ast_parameter(args.vararg.arg, args.vararg.annotation, None, "var_positional", False, imports))

    for arg, default_node in zip(args.kwonlyargs, args.kw_defaults):
        params.append(
            _ast_parameter(arg.arg, arg.annotation, default_node, "keyword_only", default_node is None, imports)
        )

    if args.kwarg is not None:
        params.append(_ast_parameter(args.kwarg.arg, args.kwarg.annotation, None, "var_keyword", False, imports))

    return params


def _json_safe_value(value):
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, set):
        values = [_json_safe_value(v) for v in value]
        return sorted(values, key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    return repr(value)


def _entry_parameters(entry: dict) -> list:
    return _static_recipe_parameters(entry.get("module"), entry.get("class"))


def _framework_support(entry: dict, metadata: dict) -> list:
    explicit = _as_string_list(entry.get("framework_support") or _recipe_metadata_attr(metadata, "framework_support"))
    if explicit:
        return explicit

    framework = entry.get("framework")
    if framework == "core":
        return _CORE_FRAMEWORK_SUPPORT.get(entry.get("algorithm"), ["framework_agnostic"])
    return [framework] if framework else []


def _optional_dependencies(entry: dict, metadata: dict) -> list:
    explicit = entry.get("optional_dependencies") or _recipe_metadata_attr(metadata, "optional_dependencies")
    if explicit is not None:
        return _as_preserved_string_list(explicit)

    framework = entry.get("framework")
    if framework and framework != "core":
        return _framework_install_hint(framework)
    return []


def _heterogeneity_support(entry: dict, metadata: dict) -> list:
    explicit = _as_string_list(
        entry.get("heterogeneity_support") or _recipe_metadata_attr(metadata, "heterogeneity_support")
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


def _privacy_compatible(entry: dict, parameters: list, metadata: dict) -> list:
    privacy = set(_as_string_list(entry.get("privacy")))
    privacy.update(_as_string_list(entry.get("privacy_compatible")))
    privacy.update(_as_string_list(_recipe_metadata_attr(metadata, "privacy_compatible")))
    parameter_names = {p["name"] for p in parameters}
    if "secure" in parameter_names:
        privacy.add("homomorphic_encryption")
    if entry.get("algorithm") == "xgboost_vertical":
        privacy.add("private_set_intersection")
    return sorted(privacy)


def _client_requirements(entry: dict, parameters: list) -> dict:
    by_name = {p["name"]: p for p in parameters}
    per_site_config = by_name.get("per_site_config")
    requirements = {
        "state_exchange": entry.get("state_exchange"),
        "requires_training_script": "train_script" in by_name,
        "requires_per_site_config": bool(per_site_config and per_site_config["required"]),
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


def _generate_recipe_detail(entry: dict) -> dict:
    recipe_metadata = entry.get(_CATALOG_RECIPE_ATTRS_KEY)
    parameters = _entry_parameters(entry)
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
        "framework_support": _framework_support(entry, recipe_metadata),
        "heterogeneity_support": _heterogeneity_support(entry, recipe_metadata),
        "privacy_compatible": _privacy_compatible(entry, parameters, recipe_metadata),
        "notes": _as_preserved_string_list(entry.get("notes") or _recipe_metadata_attr(recipe_metadata, "notes")),
        "parameters": parameters,
        "optional_dependencies": _optional_dependencies(entry, recipe_metadata),
        "template_references": _as_preserved_string_list(
            entry.get("template_references") or _recipe_metadata_attr(recipe_metadata, "template_references")
        ),
    }
    return detail


def _recipe_parameter(detail: dict, name: str) -> dict:
    return next((p for p in detail.get("parameters", []) if p.get("name") == name), {})


def _state_exchange_text(detail: dict) -> str:
    state_exchange = detail.get("state_exchange") or "none"
    transfer_param = _recipe_parameter(detail, "params_transfer_type")
    if transfer_param:
        default = transfer_param.get("default")
        if default:
            return f"{state_exchange} (default; params_transfer_type={default}, supports FULL or DIFF)"
        return f"{state_exchange} (configurable with params_transfer_type)"
    return state_exchange


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


def _documented_recipe_entry(name: str, spec: dict) -> dict:
    entry = {"name": name}
    for key in (
        "description",
        "framework",
        "module",
        "class",
        "algorithm",
        "aggregation",
        "state_exchange",
        "framework_support",
        "heterogeneity_support",
        "privacy_compatible",
        "optional_dependencies",
        "template_references",
        "notes",
    ):
        if key in spec:
            entry[key] = spec[key]
    if "privacy" in spec:
        entry["privacy"] = _as_string_list(spec["privacy"])

    return entry


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


def _apply_documented_recipe_specs(catalog: list) -> list:
    by_name = {_normalize_recipe_name(entry["name"]): entry for entry in catalog}
    for name, spec in _DOCUMENTED_RECIPE_SPECS.items():
        normalized_name = _normalize_recipe_name(name)
        spec_entry = _documented_recipe_entry(name, spec)
        if normalized_name in by_name:
            by_name[normalized_name].update(spec_entry)
        else:
            core_keys = (
                "name",
                "description",
                "framework",
                "module",
                "class",
                "algorithm",
                "aggregation",
                "state_exchange",
            )
            completed_entry = {key: spec_entry.get(key) for key in core_keys}
            completed_entry["privacy"] = spec_entry.get("privacy", [])
            completed_entry.update({key: value for key, value in spec_entry.items() if key not in completed_entry})
            spec_entry = completed_entry
            catalog.append(spec_entry)
            by_name[normalized_name] = spec_entry
    return sorted(catalog, key=lambda entry: entry["name"])


def _discover_recipe_catalog() -> list:
    """Discover recipe metadata from source without importing recipe modules."""
    results = []
    seen = set()
    for root in _RECIPE_PACKAGE_ROOTS:
        package_path = _package_source_path(root["package"])
        if not package_path or not package_path.is_dir():
            continue
        for module_path in sorted(package_path.glob("*.py")):
            if module_path.name == "__init__.py":
                continue
            module_name = f"{root['package']}.{module_path.stem}"
            recipe_info = _static_recipe_class(module_name)
            if recipe_info is None:
                continue
            recipe_class, description = recipe_info
            cli_name = _recipe_cli_name(module_name, root["framework"])
            if cli_name in seen:
                continue
            seen.add(cli_name)
            entry = {
                "name": cli_name,
                "description": description,
                "framework": root["framework"],
                "module": module_name,
                "class": recipe_class.name,
            }
            entry.update(_static_recipe_metadata(cli_name, module_name, recipe_class))
            recipe_attrs = _static_recipe_attrs(module_name, recipe_class.name)
            if recipe_attrs:
                entry[_CATALOG_RECIPE_ATTRS_KEY] = recipe_attrs
            results.append(entry)
    return _apply_documented_recipe_specs(results)


def _generate_recipe_catalog() -> dict:
    discovered = _discover_recipe_catalog()
    return {
        "schema_version": _RECIPE_CATALOG_SCHEMA_VERSION,
        "recipes": [
            {
                "summary": {key: value for key, value in entry.items() if not key.startswith("_")},
                "detail": _generate_recipe_detail(entry),
            }
            for entry in discovered
        ],
    }


def _has_required_field_types(value: dict, required_fields: dict) -> bool:
    return isinstance(value, dict) and all(
        name in value and isinstance(value[name], expected_type) for name, expected_type in required_fields.items()
    )


def _is_string_list(value) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _is_valid_catalog_parameter(parameter) -> bool:
    return _has_required_field_types(parameter, _CATALOG_PARAMETER_REQUIRED_FIELDS) and "default" in parameter


def _is_valid_catalog_entry(entry) -> bool:
    if not isinstance(entry, dict):
        return False
    summary = entry.get("summary")
    detail = entry.get("detail")
    if not _has_required_field_types(summary, _CATALOG_SUMMARY_REQUIRED_FIELDS) or not _has_required_field_types(
        detail, _CATALOG_DETAIL_REQUIRED_FIELDS
    ):
        return False
    if not _is_string_list(summary["privacy"]) or not _is_string_list(detail["privacy"]):
        return False
    if any(not _is_string_list(detail[name]) for name in _CATALOG_DETAIL_STRING_LIST_FIELDS):
        return False
    if any(not _is_valid_catalog_parameter(parameter) for parameter in detail["parameters"]):
        return False
    return summary["name"] == detail["name"] and summary["framework"] == detail["framework"]


def _read_recipe_catalog() -> list:
    try:
        payload = json.loads(_RECIPE_CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise RuntimeError(f"unable to read generated recipe catalog at {_RECIPE_CATALOG_PATH}: {e}") from e
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid generated recipe catalog at {_RECIPE_CATALOG_PATH}")
    recipes = payload.get("recipes")
    if (
        payload.get("schema_version") != _RECIPE_CATALOG_SCHEMA_VERSION
        or not isinstance(recipes, list)
        or any(not _is_valid_catalog_entry(entry) for entry in recipes)
    ):
        raise RuntimeError(f"invalid generated recipe catalog at {_RECIPE_CATALOG_PATH}")
    return recipes


def _load_catalog(framework: str = None, include_recipe_detail: bool = False) -> list:
    """Return generated recipe metadata, filtered by framework if given.

    Runtime catalog loading only reads JSON. Source discovery and AST inspection
    are reserved for catalog generation and freshness tests.
    """
    entry_type = "detail" if include_recipe_detail else "summary"
    recipes = [entry[entry_type] for entry in _read_recipe_catalog()]
    return [entry for entry in recipes if not framework or entry.get("framework") == framework]


def _load_catalog_for_cli(framework: str = None, include_recipe_detail: bool = False) -> list:
    from nvflare.tool.cli_output import output_error_message

    try:
        kwargs = {}
        if framework is not None:
            kwargs["framework"] = framework
        if include_recipe_detail:
            kwargs["include_recipe_detail"] = True
        return _load_catalog(**kwargs)
    except RuntimeError as e:
        output_error_message(
            "INTERNAL_ERROR",
            "Unable to load recipe metadata.",
            "Reinstall NVFLARE. For a source checkout, regenerate the catalog with "
            "'python -m nvflare.tool.recipe.generate_recipe_catalog'.",
            None,
            exit_code=5,
            detail=str(e),
        )
        raise SystemExit(5)


def cmd_recipe_list(cmd_args):
    from nvflare.tool.cli_output import is_json_mode, is_jsonl_mode, output_error_message, output_ok, print_human
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

    if not is_json_mode() and not is_jsonl_mode():
        print_human("Loading installed recipe catalog...", flush=True)

    catalog = _load_catalog_for_cli(framework=framework)

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
    from nvflare.tool.cli_output import is_json_mode, is_jsonl_mode, output_error_message, output_ok, print_human
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
    if not is_json_mode() and not is_jsonl_mode():
        print_human(f"Loading installed recipe metadata for '{getattr(cmd_args, 'name', '')}'...", flush=True)

    catalog = _load_catalog_for_cli(include_recipe_detail=True)
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

    detail = entry
    if is_json_mode():
        output_ok(detail)
        return

    print_human(f"\n  recipe: {detail['name']}")
    print_human(f"  description: {detail['description']}")
    print_human(f"  algorithm: {detail['algorithm']}")
    print_human(f"  aggregation: {detail['aggregation']}")
    print_human(f"  state_exchange: {_state_exchange_text(detail)}")
    print_human(f"  framework_support: {', '.join(detail['framework_support'])}")
    print_human(f"  privacy: {', '.join(detail['privacy']) or 'none enabled by default'}")
    print_human(
        f"  privacy compatibility: {', '.join(detail['privacy_compatible']) or 'not declared in recipe metadata'}"
    )
    print_human(
        f"  parameters: {len(detail['parameters'])} available; run 'nvflare recipe show {detail['name']} --format json'"
    )
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
