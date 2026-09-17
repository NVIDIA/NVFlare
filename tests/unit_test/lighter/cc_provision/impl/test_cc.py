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
from pathlib import Path
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.confidential_computing.cc_authorizer import CCAuthorizer
from nvflare.app_opt.confidential_computing.cc_manager import CC_NAMESPACE, CC_TOKEN, CCManager
from nvflare.lighter.cc_provision.cc_constants import CC_AUTHORIZERS_KEY, CCConfigKey, CCConfigValue
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


def test_legacy_heterogeneous_issuers_generate_per_site_requirements(tmp_path):
    project = Project("heterogeneous", "CPU server and CPU plus GPU client")
    server = project.set_server("server.example.com", "org", {})
    client = project.add_client("client1", "org", {})
    ctx = ProvisionContext(str(tmp_path), project)
    builder = CCBuilder()
    builder._cc_enabled_sites = [server, client]
    for participant, ids in ((server, ["snp"]), (client, ["snp", "gpu"])):
        participant.set_prop(PropKey.CC_ENABLED, True)
        participant.set_prop(PropKey.CC_CONFIG_DICT, {CCConfigKey.COMPUTE_ENV: CCConfigValue.ONPREM_CVM})
        participant.set_prop(PropKey.CC_ISSUERS, [{"id": v, "token_expiration": 300} for v in ids])
        _write_resources(ctx, participant, {"components": []})
    ctx[CC_AUTHORIZERS_KEY] = [{"id": "snp"}, {"id": "gpu"}]
    for participant in (server, client):
        builder._build_cc_manager_component(participant, ctx)
        args = json.loads((Path(ctx.get_local_dir(participant)) / "cc_manager__p_resources.json").read_text())[
            "components"
        ][0]["args"]
        assert args["required_site_verifier_ids"] == {"server": ["snp"], "client1": ["snp", "gpu"]}
        assert set(args["cc_verifier_ids"]) == {"snp", "gpu"}
        manager = CCManager(**args)
        verifiers = {}
        for name in ("snp", "gpu"):
            verifier = Mock(spec=CCAuthorizer)
            verifier.get_namespace.return_value = name
            verifier.verify_for_site.return_value = True
            verifiers[name] = verifier
        context = Mock(spec=FLContext)
        context.get_engine.return_value.get_component.side_effect = verifiers.get
        manager._setup_cc_authorizers(context)
        tokens = {
            "server": [{CC_NAMESPACE: "snp", CC_TOKEN: "proof"}],
            "client1": [{CC_NAMESPACE: "snp", CC_TOKEN: "proof"}, {CC_NAMESPACE: "gpu", CC_TOKEN: "proof"}],
        }
        assert manager._verify_participants_tokens(tokens).error == ""
        tokens["client1"].pop()
        assert manager._verify_participants_tokens(tokens).error
