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
    {"package": "nvflare.recipe", "framework": "any"},
    {"package": "nvflare.app_opt.pt.recipes", "framework": "pytorch"},
    {"package": "nvflare.app_opt.tf.recipes", "framework": "tensorflow"},
    {"package": "nvflare.app_opt.sklearn.recipes", "framework": "sklearn"},
    {"package": "nvflare.app_opt.xgboost.recipes", "framework": "xgboost"},
]


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
    if framework == "any":
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


def _load_catalog(framework: str = None) -> list:
    """Return available recipes, filtered by framework if given.

    Recipes whose optional dependencies are not installed are silently skipped.
    """
    results = []
    seen = set()
    for root in _RECIPE_PACKAGE_ROOTS:
        if framework and root["framework"] not in (framework, "any"):
            continue
        try:
            package = importlib.import_module(root["package"])
        except ModuleNotFoundError:
            pass

        else:
            for _finder, module_name, _is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
                if module_name.endswith(".__init__"):
                    continue
                try:
                    mod = importlib.import_module(module_name)
                except ModuleNotFoundError:
                    continue

                for recipe_cls in _iter_recipe_classes(mod):
                    cli_name = _recipe_cli_name(module_name, root["framework"])
                    if cli_name in seen:
                        continue
                    seen.add(cli_name)
                    results.append(
                        {
                            "name": cli_name,
                            "description": _recipe_description(recipe_cls),
                            "framework": root["framework"],
                            "module": module_name,
                            "class": recipe_cls.__name__,
                        }
                    )

    results.sort(key=lambda entry: entry["name"])
    return results


def cmd_recipe_list(cmd_args):
    from nvflare.tool.cli_output import is_json_mode, output_error_message, output_ok, print_human
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
        output_error_message(
            "INVALID_ARGS",
            "Invalid arguments.",
            _framework_install_hint_text(framework),
            None,
            exit_code=4,
            detail=f"no installed recipes found for framework '{framework}'",
        )

    if is_json_mode():
        output_ok(catalog)
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
