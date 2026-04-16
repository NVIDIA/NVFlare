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
import sys

from nvflare.tool.cli_output import output_usage_error

# Static catalog of all known recipes.
# Each entry: name (CLI id), description, framework tag, module path, class name.
# _load_catalog() filters this to entries whose dependencies are actually installed.
_KNOWN_RECIPES = [
    # ── Unified (framework-agnostic) ─────────────────────────────────────────
    {
        "name": "fedavg",
        "description": "Federated Averaging — parallel aggregation across clients (any framework)",
        "framework": "any",
        "module": "nvflare.recipe.fedavg",
        "class": "FedAvgRecipe",
    },
    {
        "name": "cyclic",
        "description": "Cyclic federated learning — sequential round-robin training across clients",
        "framework": "any",
        "module": "nvflare.recipe.cyclic",
        "class": "CyclicRecipe",
    },
    {
        "name": "fedstats",
        "description": "Federated statistics — compute dataset statistics across sites without sharing data",
        "framework": "any",
        "module": "nvflare.recipe.fedstats",
        "class": "FedStatsRecipe",
    },
    # ── PyTorch ──────────────────────────────────────────────────────────────
    {
        "name": "fedavg-pt",
        "description": "FedAvg for PyTorch nn.Module models",
        "framework": "pytorch",
        "module": "nvflare.app_opt.pt.recipes.fedavg",
        "class": "FedAvgRecipe",
    },
    {
        "name": "fedopt-pt",
        "description": "FedOpt for PyTorch — server-side optimizer (FedAdam, FedYogi, FedAdagrad)",
        "framework": "pytorch",
        "module": "nvflare.app_opt.pt.recipes.fedopt",
        "class": "FedOptRecipe",
    },
    {
        "name": "scaffold-pt",
        "description": "SCAFFOLD for PyTorch — reduces client drift with control variates",
        "framework": "pytorch",
        "module": "nvflare.app_opt.pt.recipes.scaffold",
        "class": "ScaffoldRecipe",
    },
    {
        "name": "cyclic-pt",
        "description": "Cyclic federated learning for PyTorch nn.Module models",
        "framework": "pytorch",
        "module": "nvflare.app_opt.pt.recipes.cyclic",
        "class": "CyclicRecipe",
    },
    {
        "name": "swarm-pt",
        "description": "Swarm learning for PyTorch — peer-to-peer federated learning without central server",
        "framework": "pytorch",
        "module": "nvflare.app_opt.pt.recipes.swarm",
        "class": "SwarmLearningRecipe",
    },
    {
        "name": "fedeval-pt",
        "description": "Federated evaluation for PyTorch — evaluate a pre-trained model across sites",
        "framework": "pytorch",
        "module": "nvflare.app_opt.pt.recipes.fedeval",
        "class": "FedEvalRecipe",
    },
    # ── TensorFlow ───────────────────────────────────────────────────────────
    {
        "name": "fedavg-tf",
        "description": "FedAvg for TensorFlow / Keras models",
        "framework": "tensorflow",
        "module": "nvflare.app_opt.tf.recipes.fedavg",
        "class": "FedAvgRecipe",
    },
    {
        "name": "fedopt-tf",
        "description": "FedOpt for TensorFlow — server-side optimizer",
        "framework": "tensorflow",
        "module": "nvflare.app_opt.tf.recipes.fedopt",
        "class": "FedOptRecipe",
    },
    {
        "name": "scaffold-tf",
        "description": "SCAFFOLD for TensorFlow — reduces client drift with control variates",
        "framework": "tensorflow",
        "module": "nvflare.app_opt.tf.recipes.scaffold",
        "class": "ScaffoldRecipe",
    },
    {
        "name": "cyclic-tf",
        "description": "Cyclic federated learning for TensorFlow / Keras models",
        "framework": "tensorflow",
        "module": "nvflare.app_opt.tf.recipes.cyclic",
        "class": "CyclicRecipe",
    },
    # ── Scikit-learn ─────────────────────────────────────────────────────────
    {
        "name": "fedavg-sklearn",
        "description": "FedAvg for scikit-learn models (linear, SVM, etc.)",
        "framework": "sklearn",
        "module": "nvflare.app_opt.sklearn.recipes.fedavg",
        "class": "SklearnFedAvgRecipe",
    },
    {
        "name": "kmeans-sklearn",
        "description": "Federated K-Means clustering with scikit-learn",
        "framework": "sklearn",
        "module": "nvflare.app_opt.sklearn.recipes.kmeans",
        "class": "KMeansFedAvgRecipe",
    },
    {
        "name": "svm-sklearn",
        "description": "Federated SVM with support vector aggregation",
        "framework": "sklearn",
        "module": "nvflare.app_opt.sklearn.recipes.svm",
        "class": "SVMFedAvgRecipe",
    },
    # ── XGBoost ──────────────────────────────────────────────────────────────
    {
        "name": "xgb-bagging",
        "description": "XGBoost with bagging — tree-based horizontal federated learning",
        "framework": "xgboost",
        "module": "nvflare.app_opt.xgboost.recipes.bagging",
        "class": "XGBBaggingRecipe",
    },
    {
        "name": "xgb-histogram",
        "description": "XGBoost histogram-based horizontal federated learning",
        "framework": "xgboost",
        "module": "nvflare.app_opt.xgboost.recipes.histogram",
        "class": "XGBHorizontalRecipe",
    },
    {
        "name": "xgb-vertical",
        "description": "XGBoost vertical federated learning (label on one site)",
        "framework": "xgboost",
        "module": "nvflare.app_opt.xgboost.recipes.vertical",
        "class": "XGBVerticalRecipe",
    },
]


def _load_catalog(framework: str = None) -> list:
    """Return available recipes, filtered by framework if given.

    Recipes whose optional dependencies are not installed are silently skipped.
    """
    results = []
    for entry in _KNOWN_RECIPES:
        if framework and entry["framework"] not in (framework, "any"):
            continue
        try:
            mod = importlib.import_module(entry["module"])
            getattr(mod, entry["class"])  # confirm class exists
            results.append(entry)
        except (ImportError, AttributeError):
            pass
    return results


def cmd_recipe_list(cmd_args):
    from nvflare.tool.cli_output import is_json_mode, output_error, output_ok, print_human
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        _recipe_parser,
        "nvflare recipe list",
        ["nvflare recipe list", "nvflare recipe list --framework pytorch"],
        sys.argv[1:],
    )

    framework = getattr(cmd_args, "framework", None)
    catalog = _load_catalog(framework=framework)

    if framework and not catalog:
        output_error("INVALID_ARGS", exit_code=4, detail=f"no installed recipes found for framework '{framework}'")

    if is_json_mode():
        output_ok(catalog)
        return

    if not catalog:
        print_human("No recipes are currently available.")
        print_human("Install optional framework dependencies to make recipe entries available.")
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
        choices=["any", "pytorch", "tensorflow", "sklearn", "xgboost"],
        help="filter by framework",
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
