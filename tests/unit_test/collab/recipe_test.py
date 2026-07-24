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
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from nvflare.collab import CollabRecipe, collab
from nvflare.collab.api import ClientApp, ModuleWrapper, ServerApp
from nvflare.collab.api.decorators import supports_context


@collab.main
def module_run():
    return None


@collab.publish
def module_train(value):
    return value


def _make_module(name):
    module = ModuleType(name)
    module.value = None

    @collab.init
    def initialize_value():
        module.value = "initialized"

    @collab.final
    def clear_value():
        module.value = None

    @collab.main
    def run():
        return None

    @collab.publish
    def train(value):
        return value

    module.initialize_value = initialize_value
    module.clear_value = clear_value
    module.run = run
    module.train = train
    return module


def test_recipe_uses_caller_module_for_missing_server_and_client():
    explicit_module = _make_module("explicit_recipe_module")

    recipe = CollabRecipe(job_name="default_recipe_apps_test")
    assert recipe.server_app.obj.module_name == __name__
    assert recipe.client_app.obj.module_name == __name__

    recipe = CollabRecipe(job_name="default_recipe_server_test", client=explicit_module)
    assert recipe.server_app.obj.module_name == __name__
    assert recipe.client_app.obj.module_name == "explicit_recipe_module"

    recipe = CollabRecipe(job_name="default_recipe_client_test", server=explicit_module)
    assert recipe.server_app.obj.module_name == "explicit_recipe_module"
    assert recipe.client_app.obj.module_name == __name__


def test_apps_automatically_wrap_primary_and_named_modules():
    main_module = _make_module("main_app_module")
    extra_module = _make_module("extra_app_module")

    server_app = ServerApp(main_module)
    client_app = ClientApp(main_module)
    server_app.add_collab_object("extra", extra_module)
    client_app.add_collab_object("extra", extra_module)

    assert isinstance(server_app.obj, ModuleWrapper)
    assert isinstance(client_app.obj, ModuleWrapper)
    assert isinstance(server_app.get_collab_objects()["extra"], ModuleWrapper)
    assert isinstance(client_app.get_collab_objects()["extra"], ModuleWrapper)
    assert [name for name, _ in server_app.mains] == ["run"]
    assert client_app.get_collab_interface()["client"] == {
        "train": [{"name": "value", "kind": "POSITIONAL_OR_KEYWORD", "required": True}]
    }

    context = server_app.new_context("server", "server")
    server_app.initialize(context)
    assert main_module.value == "initialized"
    server_app.finalize(context)
    assert main_module.value is None


def test_server_app_requires_exactly_one_main_function():
    no_main_module = ModuleType("no_main_module")
    with pytest.raises(ValueError, match=r"exactly one @collab\.main function but got 0"):
        ServerApp(no_main_module)

    multiple_main_module = _make_module("multiple_main_module")

    @collab.main
    def run_again():
        return None

    multiple_main_module.run_again = run_again
    with pytest.raises(ValueError, match=r"exactly one @collab\.main function but got 2"):
        ServerApp(multiple_main_module)


def test_module_main_wrapper_forwards_context():
    module = ModuleType("context_main_module")
    received = []

    @collab.main
    def run(context):
        received.append(context)

    module.run = run
    app = ServerApp(module)
    _, main_func = app.mains[0]
    context = object()

    assert supports_context(main_func)
    main_func(context=context)
    assert received == [context]


def test_app_runs_multiple_init_functions_in_name_order():
    calls = []
    module = ModuleType("multiple_init_module")

    @collab.init
    def init_second():
        calls.append("second")

    @collab.init
    def init_first():
        calls.append("first")

    module.init_second = init_second
    module.init_first = init_first

    app = ClientApp(module)
    app.initialize(app.new_context("site-1", "site-1"))

    assert calls == ["first", "second"]


def test_recipe_accepts_modules_for_all_collab_objects():
    main_module = _make_module("main_recipe_module")
    extra_module = _make_module("extra_recipe_module")

    recipe = CollabRecipe(
        job_name="module_recipe_test",
        server=main_module,
        client=main_module,
        server_objects={"extra": extra_module},
        client_objects={"extra": extra_module},
    )

    assert isinstance(recipe.server, ModuleWrapper)
    assert isinstance(recipe.client, ModuleWrapper)
    assert isinstance(recipe.server_objects["extra"], ModuleWrapper)
    assert isinstance(recipe.client_objects["extra"], ModuleWrapper)


def test_recipe_materializes_only_each_sites_config_into_targeted_executor_props():
    module = _make_module("per_site_config_module")
    recipe = CollabRecipe(job_name="per_site_config_test", server=module, client=module)
    config = {
        "site-1": {"learning_rate": 0.01},
        "site-2": {"learning_rate": 0.02},
    }

    recipe.set_per_site_config(config)
    recipe.set_client_prop("shared", "value")
    config["site-1"]["learning_rate"] = 1.0

    job = recipe.finalize()

    assert "@ALL" not in job._deploy_map
    site_1_executor = job._deploy_map["site-1"].app_config.executors[0].executor
    site_2_executor = job._deploy_map["site-2"].app_config.executors[0].executor
    assert site_1_executor.props == {"shared": "value", "learning_rate": 0.01}
    assert site_2_executor.props == {"shared": "value", "learning_rate": 0.02}
    assert "__per_site_config__" not in site_1_executor.props
    assert "__per_site_config__" not in site_2_executor.props


def test_exported_module_package_imports_in_clean_process(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    package = source_root / "collab_export_pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "helper.py").write_text("VALUE = 42\n", encoding="utf-8")
    (package / "app.py").write_text(
        "from nvflare.collab import collab\n"
        "from .helper import VALUE\n"
        "@collab.main\n"
        "def run():\n"
        "    return VALUE\n"
        "@collab.publish\n"
        "def train():\n"
        "    return VALUE\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(source_root))
    module = importlib.import_module("collab_export_pkg.app")
    export_root = tmp_path / "export"

    CollabRecipe("package_export_test", server=module, client=module).export(str(export_root))

    custom_dir = export_root / "package_export_test" / "app" / "custom"
    assert (custom_dir / "collab_export_pkg" / "__init__.py").is_file()
    assert (custom_dir / "collab_export_pkg" / "helper.py").is_file()
    assert (custom_dir / "collab_export_pkg" / "app.py").is_file()
    repo_root = Path(__file__).resolve().parents[3]
    code = (
        "import pathlib, sys; "
        "sys.path.insert(0, sys.argv[1]); "
        "sys.path.insert(0, sys.argv[2]); "
        "import collab_export_pkg.app as app; "
        "assert app.VALUE == 42; "
        "assert pathlib.Path(app.__file__).is_relative_to(pathlib.Path(sys.argv[1]))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(custom_dir), str(repo_root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
