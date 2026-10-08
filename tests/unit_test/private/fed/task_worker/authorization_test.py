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

import json
from types import SimpleNamespace

import pytest

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.fuel.common.excepts import ComponentNotAuthorized
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.private.fed.utils.worker_component_builder import WorkerComponentBuilder


@pytest.fixture
def builder(tmp_path):
    ConfigService.reset()
    resources = tmp_path / "resources.json"
    resources.write_text(json.dumps({"class_allow_list": ["allowed."]}))
    workspace = SimpleNamespace(get_resources_file_path=lambda: str(resources))
    ctx = FLContext()
    ctx.set_prop(FLContextKey.JOB_META, {"byoc": False})
    yield WorkerComponentBuilder(fl_ctx=ctx, workspace=workspace)
    ConfigService.reset()


@pytest.mark.parametrize(
    "nested",
    [
        {"helper": {"path": "forbidden.Helper"}},
        {"helpers": [{"class_path": "forbidden.Helper"}]},
        {"path": "ordinary-argument", "helper": {"path": "forbidden.Helper"}},
        {
            "holder": {
                "config_type": "dict",
                "components": [{"id": "nested", "config_type": "dict", "path": "forbidden.Helper"}],
            }
        },
    ],
)
def test_worker_authorization_traverses_original_nested_graph(builder, nested, monkeypatch):
    imported = []
    monkeypatch.setattr(
        "nvflare.fuel.utils.component_builder.instantiate_class",
        lambda *args: imported.append(args),
    )
    with pytest.raises(ComponentNotAuthorized, match="forbidden.Helper.*allow_list"):
        builder.build_component({"path": "allowed.Executor", "args": nested})
    assert imported == []


def test_worker_preserves_plain_data_and_component_metadata_traversal(builder):
    builder.authorize_tree(
        {
            "path": "allowed.Executor",
            "metadata": {"path": "not.a.Component"},
            "args": {"options": {"config_type": "dict", "path": "ordinary.data", "numbers": [1, 2]}},
        }
    )


def test_explicit_component_list_entry_cannot_bypass_policy_with_config_type(builder):
    config = {"id": "helper", "config_type": "dict", "path": "forbidden.Helper"}
    with pytest.raises(ComponentNotAuthorized, match="components.#2"):
        builder.authorize_tree(config, builder.make_component_node(config, 2))


def test_byoc_uses_authoritative_context_metadata(builder):
    builder.fl_ctx.set_prop(FLContextKey.JOB_META, {"byoc": True})
    builder.authorize_tree({"path": "custom.Executor", "args": {"helper": {"path": "custom.Helper"}}})


def test_missing_site_policy_uses_current_default_allowlist(tmp_path):
    resources = tmp_path / "resources.json"
    resources.write_text("{}")
    builder = WorkerComponentBuilder(workspace=SimpleNamespace(get_resources_file_path=lambda: str(resources)))
    builder.authorize_tree({"path": "nvflare.app_common.np.np_trainer.NPTrainer"})
    with pytest.raises(ComponentNotAuthorized, match="allow_list"):
        builder.authorize_tree({"path": "subprocess.Popen"})


def test_warn_policy_is_preserved(builder):
    path = builder.workspace.get_resources_file_path()
    with open(path, "w") as stream:
        json.dump({"class_allow_list": [], "class_list_enforcement_mode": "warn"}, stream)
    builder.authorize_tree({"path": "unlisted.Executor"})
