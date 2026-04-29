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

import importlib
import inspect
import pkgutil
import sys

from nvflare.tool.cli_output import output_usage_error

_RECIPE_PACKAGE_ROOTS = [
    {"package": "nvflare.recipe", "framework": "core"},
    {"package": "nvflare.app_opt.pt.recipes", "framework": "pytorch"},
    {"package": "nvflare.app_opt.tf.recipes", "framework": "tensorflow"},
    {"package": "nvflare.app_opt.sklearn.recipes", "framework": "sklearn"},
    {"package": "nvflare.app_opt.xgboost.recipes", "framework": "xgboost"},
]
_FILTER_KEYS = {"framework", "privacy", "algorithm", "aggregation", "state_exchange"}
_LIST_METADATA_KEYS = {"privacy"}


def _normalize_filter_value(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


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


def _framework_install_hint(framework: str = None) -> list[str]:
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


def _load_catalog(framework: str = None) -> list:
    """Return available recipes, filtered by framework if given.

    Recipes whose optional dependencies are not installed are silently skipped.
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
                results.append(entry)

    results.sort(key=lambda entry: entry["name"])
    return results


def cmd_recipe_list(cmd_args):
    from nvflare.tool.cli_output import is_json_mode, output_error_message, output_ok, print_human
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        _recipe_parser,
        "nvflare recipe list",
        [
            "nvflare recipe list",
            "nvflare recipe list --framework pytorch",
            "nvflare recipe list --filter framework=pytorch --filter algorithm=fedavg",
        ],
        sys.argv[1:],
    )

    framework = getattr(cmd_args, "framework", None)
    try:
        filters = _parse_recipe_filters(getattr(cmd_args, "filters", None))
    except ValueError as e:
        output_usage_error(
            _recipe_parser,
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
                _recipe_parser,
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


_recipe_parser = None
_recipe_root_parser = None


def def_recipe_parser(sub_cmd):
    global _recipe_parser, _recipe_root_parser
    cmd = "recipe"
    parser = sub_cmd.add_parser(cmd, help="list available FL job recipes")
    _recipe_root_parser = parser
    recipe_subparser = parser.add_subparsers(title="recipe subcommands", metavar="", dest="recipe_sub_cmd")

    list_parser = recipe_subparser.add_parser("list", help="list available recipes (default)")
    list_parser.add_argument(
        "--framework",
        type=str,
        default=None,
        choices=["pytorch", "tensorflow", "sklearn", "xgboost"],
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
    _recipe_parser = list_parser
    parser.set_defaults(recipe_sub_cmd="list")

    return {cmd: parser}


def handle_recipe_cmd(args):
    sub_cmd = getattr(args, "recipe_sub_cmd", None)
    if sub_cmd == "list":
        cmd_recipe_list(args)
    else:
        output_usage_error(_recipe_root_parser, "recipe subcommand required", exit_code=4)
        raise SystemExit(4)
