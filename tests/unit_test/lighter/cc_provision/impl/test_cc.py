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
from unittest.mock import Mock, patch

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.confidential_computing.cc_authorizer import CCAuthorizer
from nvflare.app_opt.confidential_computing.cc_manager import CC_INFO, CC_NAMESPACE, CC_TOKEN, CCManager
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
                f"{CCConfigKey.COMPUTE_ENV}: {CCConfigValue.AZURE_CVM}",
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


def test_cc_builder_rejects_removed_onprem_compute_environment(tmp_path):
    cc_config_file = tmp_path / "cc_server.yml"
    cc_config_file.write_text(f"{CCConfigKey.COMPUTE_ENV}: onprem_cvm\n")
    project = _project_with_server({PropKey.CC_CONFIG: str(cc_config_file)})
    ctx = ProvisionContext(str(tmp_path), project)

    with pytest.raises(ValueError, match="Invalid compute environment: onprem_cvm"):
        CCBuilder().initialize(project, ctx)


def test_cc_builder_builds_azure_authorizer_and_manager_resources(tmp_path):
    cc_config_file = tmp_path / "cc_server.yml"
    cc_config_file.write_text(
        "\n".join(
            [
                f"{CCConfigKey.COMPUTE_ENV}: {CCConfigValue.AZURE_CVM}",
                f"{CCConfigKey.CC_ISSUERS}:",
                "  - id: az_cvm_authorizer",
                "    path: nvflare.app_opt.confidential_computing.az_cvm_authorizer.AZCVMAuthorizer",
                "    token_expiration: 100",
                f"{CCConfigKey.CC_ATTESTATION_CONFIG}:",
                "  check_frequency: 120",
                "",
            ]
        )
    )
    project = _project_with_server({PropKey.CC_CONFIG: str(cc_config_file)})
    ctx = ProvisionContext(str(tmp_path), project)
    server = project.get_server()
    _write_resources(ctx, server, {"format_version": 2, "components": []})
    builder = CCBuilder()

    builder.initialize(project, ctx)
    builder.build(project, ctx)

    local_dir = ctx.get_local_dir(server)
    with open(os.path.join(local_dir, "az_cvm_authorizer__p_resources.json"), "r") as f:
        authorizer = json.load(f)["components"][0]
    assert authorizer["id"] == "az_cvm_authorizer"

    with open(os.path.join(local_dir, "cc_manager__p_resources.json"), "r") as f:
        manager_args = json.load(f)["components"][0]["args"]
    assert manager_args["cc_verifier_ids"] == ["az_cvm_authorizer"]
    assert manager_args["verify_frequency"] == 120
    assert server.get_prop(PropKey.AUTHZ_SECTION_KEY) == "cc_authz"


def test_cc_builder_extends_generated_class_allow_list(tmp_path):
    project = _project_with_server(
        {
            PropKey.CC_ENABLED: True,
            PropKey.CC_CONFIG_DICT: {
                CCConfigKey.COMPUTE_ENV: CCConfigValue.AZURE_CVM,
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
                CCConfigKey.COMPUTE_ENV: CCConfigValue.AZURE_CVM,
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


def test_heterogeneous_issuers_generate_per_site_requirements(tmp_path):
    project = Project("heterogeneous", "CPU server and CPU plus GPU client")
    server = project.set_server("server.example.com", "org", {})
    client = project.add_client("client1", "org", {})
    ctx = ProvisionContext(str(tmp_path), project)
    builder = CCBuilder()
    builder._cc_enabled_sites = [server, client]
    for participant, ids in ((server, ["snp"]), (client, ["snp", "gpu"])):
        participant.set_prop(PropKey.CC_ENABLED, True)
        participant.set_prop(PropKey.CC_CONFIG_DICT, {CCConfigKey.COMPUTE_ENV: CCConfigValue.AZURE_CVM})
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


@pytest.mark.parametrize("server_name", ["server.example.com", "server1"])
@pytest.mark.parametrize("case", ["missing", "wrong_identity", "invalid_proof", "valid_proof"])
def test_provisioned_client_validates_server_registration(tmp_path, server_name, case):
    cc_config_file = tmp_path / "cc.yml"
    cc_config_file.write_text(
        json.dumps(
            {
                "compute_env": CCConfigValue.AZURE_CVM,
                "cc_issuers": [
                    {
                        "id": "mock_authorizer",
                        "path": "nvflare.app_opt.confidential_computing.mock_authorizer.MockAuthorizer",
                        "token_expiration": 300,
                    }
                ],
            }
        )
    )
    project = Project("registration", "Protected server registration")
    project.set_server(server_name, "org", {PropKey.CC_CONFIG: str(cc_config_file)})
    client = project.add_client("client1", "org", {PropKey.CC_CONFIG: str(cc_config_file)})
    ctx = ProvisionContext(str(tmp_path), project)
    for participant in project.get_all_participants():
        _write_resources(ctx, participant, {"components": []})
        (Path(ctx.get_local_dir(participant)) / ProvFileName.LOG_CONFIG_DEFAULT).write_text("{}")
    builder = CCBuilder()
    builder.initialize(project, ctx)
    builder.build(project, ctx)
    args = json.loads((Path(ctx.get_local_dir(client)) / "cc_manager__p_resources.json").read_text())["components"][0][
        "args"
    ]
    assert set(args["cc_enabled_sites"]) == {"server", "client1"}
    assert args["required_site_verifier_ids"] == {"server": ["mock_authorizer"], "client1": ["mock_authorizer"]}

    manager = CCManager(**args)
    verifier = Mock(spec=CCAuthorizer)
    verifier.get_namespace.return_value = "test-cc"
    verifier.verify_for_site.side_effect = lambda token, site: token == "valid-proof"
    engine = Mock()
    engine.get_component.return_value = verifier
    context = FLContext()
    context.set_prop(ReservedKey.ENGINE, engine)
    context.set_prop(ReservedKey.IDENTITY_NAME, client.name)
    manager.handle_event(EventType.SYSTEM_BOOTSTRAP, context)

    if case != "missing":
        identity = server_name if case == "wrong_identity" else "server"
        token = "invalid-proof" if case == "invalid_proof" else "valid-proof"
        context.set_prop(CC_INFO, {identity: [{CC_NAMESPACE: "test-cc", CC_TOKEN: token}]})
    with patch.object(manager, "_shutdown_system") as shutdown:
        manager.handle_event(EventType.AFTER_CLIENT_REGISTER, context)
    if case == "valid_proof":
        shutdown.assert_not_called()
    else:
        shutdown.assert_called_once()
        assert "CC info validation failed" in shutdown.call_args.args[0]
    if case in ("invalid_proof", "valid_proof"):
        verifier.verify_for_site.assert_called_once_with(token, "server")
    else:
        verifier.verify_for_site.assert_not_called()
