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

from types import ModuleType

import pytest

from nvflare.collab import CollabRecipe, collab
from nvflare.collab.api import ClientApp, ModuleWrapper, ServerApp
from nvflare.collab.api.constants import PER_SITE_CONFIG_PROP


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
    assert client_app.get_collab_interface()["client"] == {"train": ["value"]}

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


def test_recipe_public_per_site_config_reaches_executor_props():
    module = _make_module("per_site_config_module")
    recipe = CollabRecipe(job_name="per_site_config_test", server=module, client=module)
    config = {
        "site-1": {"learning_rate": 0.01},
        "site-2": {"learning_rate": 0.02},
    }

    recipe.set_per_site_config(config)
    config["site-1"]["learning_rate"] = 1.0

    props = recipe._client_props_with_per_site_config()
    assert props[PER_SITE_CONFIG_PROP] == {
        "site-1": {"learning_rate": 0.01},
        "site-2": {"learning_rate": 0.02},
    }
