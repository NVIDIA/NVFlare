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

"""Migration contracts for Edge recipe names and unchanged runtime settings."""

import json
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from nvflare.edge.tools.edge_fed_buff_recipe import DeviceManagerConfig, EdgeFedBuffRecipe, ModelManagerConfig


@pytest.fixture(params=["edge", "et"])
def recipe_factory(request):
    if request.param == "edge":
        yield lambda *args, **kwargs: EdgeFedBuffRecipe(*args, model=torch.nn.Linear(2, 1), **kwargs)
    else:
        from nvflare.edge.models.model import DeviceModel
        from nvflare.edge.tools.et_fed_buff_recipe import ETFedBuffRecipe

        # Exercise the real ET job construction/forwarding without running ExecuTorch export.
        with patch("nvflare.edge.tools.et_fed_buff_recipe.importlib.util.find_spec", return_value=object()):
            yield lambda *args, **kwargs: ETFedBuffRecipe(
                *args,
                device_model=DeviceModel(torch.nn.Linear(2, 1)),
                input_shape=(1, 2),
                output_shape=(1,),
                **kwargs,
            )


def _export_configs(recipe, root):
    recipe.export(str(root))
    job = root / recipe.name
    return {str(path.relative_to(job)): json.loads(path.read_text()) for path in job.glob("*/config/*.json")}


def test_new_names_preserve_legacy_export(recipe_factory, tmp_path):
    """The public spelling must not change the generated server or hierarchical client configs."""
    legacy = recipe_factory(
        job_name="migration",
        model_manager_config=ModelManagerConfig(
            max_model_version=101,
            num_updates_for_model=20,
            global_lr=0.1,
            max_model_history=None,
            max_num_active_model_versions=7,
            update_timeout=500,
            staleness_weight=True,
        ),
        device_manager_config=DeviceManagerConfig(
            device_selection_size=200,
            min_hole_to_fill=10,
            device_reuse=False,
        ),
        device_wait_timeout=300,
    )
    modern = recipe_factory(
        name="migration",
        num_rounds=100,
        buffer_size=20,
        concurrency=200,
        min_open_slots=10,
        server_lr=0.1,
        model_manager_config=ModelManagerConfig(
            max_model_history=None,
            max_num_active_model_versions=7,
            update_timeout=500,
            staleness_weight=True,
        ),
        device_manager_config=DeviceManagerConfig(device_reuse=False),
        device_wait_timeout=300,
    )
    assert _export_configs(modern, tmp_path / "new") == _export_configs(legacy, tmp_path / "old")


def test_defaults_preserve_legacy_export(recipe_factory, tmp_path):
    legacy = recipe_factory(
        job_name="defaults",
        model_manager_config=ModelManagerConfig(),
        device_manager_config=DeviceManagerConfig(),
    )
    modern = recipe_factory(name="defaults")
    assert modern.model_manager_config.max_model_version == 20
    assert modern.model_manager_config.global_lr == 0.01
    assert _export_configs(modern, tmp_path / "new") == _export_configs(legacy, tmp_path / "old")


def test_explicit_keywords_override_copies(recipe_factory):
    model_config = ModelManagerConfig(max_model_version=101, global_lr=0.5, update_timeout=500)
    device_config = DeviceManagerConfig(device_selection_size=200, min_hole_to_fill=10, device_reuse=False)
    recipe = recipe_factory(
        name="override",
        num_rounds=1,
        buffer_size=2,
        concurrency=3,
        min_open_slots=1,
        server_lr=0.1,
        model_manager_config=model_config,
        device_manager_config=device_config,
    )
    assert model_config.max_model_version == 101
    assert model_config.num_updates_for_model == 100
    assert model_config.global_lr == 0.5
    assert device_config.device_selection_size == 200
    assert device_config.min_hole_to_fill == 10
    assert recipe.model_manager_config.max_model_version == 2
    assert recipe.model_manager_config.update_timeout == 500
    assert not recipe.device_manager_config.device_reuse


def test_legacy_positional_order():
    recipe = EdgeFedBuffRecipe("positional", torch.nn.Linear(2, 1), ModelManagerConfig(), DeviceManagerConfig())
    assert recipe.name == recipe.job_name == "positional"


def test_et_legacy_positional_order():
    from nvflare.edge.models.model import DeviceModel
    from nvflare.edge.tools.et_fed_buff_recipe import ETFedBuffRecipe

    with patch("nvflare.edge.tools.et_fed_buff_recipe.importlib.util.find_spec", return_value=object()):
        recipe = ETFedBuffRecipe(
            "positional",
            DeviceModel(torch.nn.Linear(2, 1)),
            (1, 2),
            (1,),
            ModelManagerConfig(),
            DeviceManagerConfig(),
        )
    assert recipe.name == recipe.job_name == "positional"


def test_name_conflicts_and_matching_alias(recipe_factory):
    with pytest.raises(ValueError, match="name and job_name must agree"):
        recipe_factory(name="new", job_name="old")
    assert recipe_factory(name="same", job_name="same").name == "same"
    with pytest.raises(ValueError, match="required"):
        recipe_factory()


@pytest.mark.parametrize("key", ["num_rounds", "buffer_size", "concurrency", "min_open_slots"])
@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_positive_integer_keywords(recipe_factory, key, bad):
    with pytest.raises(ValueError, match=f"{key} must be a positive integer"):
        recipe_factory(name="invalid", **{key: bad})


@pytest.mark.parametrize("bad", [0, -1, True, float("inf"), float("nan"), "0.1"])
def test_invalid_server_lr(recipe_factory, bad):
    with pytest.raises(ValueError, match="server_lr must be a finite positive number"):
        recipe_factory(name="invalid", server_lr=bad)


@pytest.mark.parametrize(
    "options",
    [
        {"concurrency": 2, "min_open_slots": 3},
        {"concurrency": 2, "device_manager_config": DeviceManagerConfig(min_hole_to_fill=3)},
        {"buffer_size": 101},
    ],
)
def test_effective_cross_field_constraints(recipe_factory, options):
    with pytest.raises(ValueError, match="min_open_slots|num_updates_for_model"):
        recipe_factory(name="invalid", **options)
