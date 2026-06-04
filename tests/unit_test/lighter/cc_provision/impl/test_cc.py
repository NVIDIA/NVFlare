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
import os

import pytest

from nvflare.lighter.cc_provision.cc_constants import CCConfigKey, CCConfigValue
from nvflare.lighter.cc_provision.impl.cc import CCBuilder
from nvflare.lighter.constants import PropKey, ProvFileName
from nvflare.lighter.ctx import ProvisionContext
from nvflare.lighter.entity import Project


def _project_with_server(props=None):
    project = Project("test_project", "A testing project")
    project.set_server("server", "org", props or {})
    return project


def _write_resources(ctx, participant, resources):
    local_dir = ctx.get_local_dir(participant)
    os.makedirs(local_dir, exist_ok=True)
    resources_file = os.path.join(local_dir, ProvFileName.RESOURCES_JSON_DEFAULT)
    with open(resources_file, "w") as f:
        json.dump(resources, f)
    return resources_file


def test_cc_builder_loads_class_allow_list_from_cc_config_file(tmp_path):
    cc_config_file = tmp_path / "cc_server.yml"
    cc_config_file.write_text(
        "\n".join(
            [
                f"{CCConfigKey.COMPUTE_ENV}: {CCConfigValue.MOCK}",
                f"{CCConfigKey.CLASS_ALLOW_LIST}:",
                "  - hello_cyclic.",
                "",
            ]
        )
    )
    project = _project_with_server({PropKey.CC_CONFIG: str(cc_config_file)})
    ctx = ProvisionContext(str(tmp_path), project)
    builder = CCBuilder()

    builder.initialize(project, ctx)

    server = project.get_server()
    assert server.get_prop(PropKey.CC_ENABLED) is True
    assert server.get_prop(PropKey.CC_CONFIG_DICT)[CCConfigKey.CLASS_ALLOW_LIST] == ["hello_cyclic."]


def test_cc_builder_extends_generated_class_allow_list(tmp_path):
    project = _project_with_server(
        {
            PropKey.CC_ENABLED: True,
            PropKey.CC_CONFIG_DICT: {
                CCConfigKey.COMPUTE_ENV: CCConfigValue.MOCK,
                CCConfigKey.CLASS_ALLOW_LIST: ["hello_cyclic.", "nvflare."],
            },
        }
    )
    ctx = ProvisionContext(str(tmp_path), project)
    server = project.get_server()
    resources_file = _write_resources(
        ctx,
        server,
        {
            "format_version": 2,
            "class_allow_list": ["nvflare."],
            "components": [],
        },
    )
    builder = CCBuilder()
    builder._cc_enabled_sites = [server]

    builder.build(project, ctx)

    with open(resources_file, "r") as f:
        resources = json.load(f)
    assert resources["class_allow_list"] == ["nvflare.", "hello_cyclic."]
    with open(resources_file, "rb") as f:
        assert f.read().endswith(b"\n")
    assert not os.path.exists(f"{resources_file}.{os.getpid()}.tmp")


def test_cc_builder_rejects_invalid_class_allow_list(tmp_path):
    project = _project_with_server(
        {
            PropKey.CC_ENABLED: True,
            PropKey.CC_CONFIG_DICT: {
                CCConfigKey.COMPUTE_ENV: CCConfigValue.MOCK,
                CCConfigKey.CLASS_ALLOW_LIST: "hello_cyclic.",
            },
        }
    )
    ctx = ProvisionContext(str(tmp_path), project)
    server = project.get_server()
    _write_resources(ctx, server, {"format_version": 2, "class_allow_list": ["nvflare."]})
    builder = CCBuilder()
    builder._cc_enabled_sites = [server]

    with pytest.raises(ValueError, match=CCConfigKey.CLASS_ALLOW_LIST):
        builder.build(project, ctx)
