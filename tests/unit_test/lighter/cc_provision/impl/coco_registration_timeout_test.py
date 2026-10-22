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
from unittest.mock import Mock, patch

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.apis.workspace import Workspace
from nvflare.app_opt.confidential_computing.cc_timeouts import resolve_token_timeouts
from nvflare.fuel.data_event.data_bus import DataBus
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.message import Message
from nvflare.lighter.cc_provision.impl.coco import CoCoBuilder
from nvflare.lighter.constants import ProvFileName
from nvflare.lighter.provision import prepare_project
from nvflare.private.defs import CellMessageHeaderKeys
from nvflare.private.fed.app.fl_conf import FLClientStarterConfiger


@pytest.fixture(autouse=True)
def preserve_data_bus():
    prior_data = dict(DataBus().data_store)
    yield
    DataBus().data_store.clear()
    DataBus().data_store.update(prior_data)


def build_resources(tmp_path, protected_server=True, registration_timeout=300.0, current_timeout=None):
    project = prepare_project(
        {
            "api_version": 3,
            "name": "coco-registration",
            "participants": [
                {"type": "server", "name": "server.example.com", "org": "example"},
                {"type": "client", "name": "site-1", "org": "example"},
                {"type": "client", "name": "plain-client", "org": "example"},
            ],
        }
    )
    kits, originals = {}, {}
    for participant in project.get_all_participants():
        kit = kits[participant.name] = tmp_path / participant.name
        local = kit / "local"
        startup = kit / "startup"
        local.mkdir(parents=True)
        startup.mkdir()
        (startup / "sub_start.sh").write_text("#!/usr/bin/env bash\ntrue\n")
        if participant.type == "client":
            client = {"retry_timeout": 30, "compression": "Gzip"}
            if current_timeout is not None:
                client["maint_msg_timeout"] = current_timeout
            resources = {"client": client, "components": [], "custom_resource": "preserved"}
            (startup / "fed_client.json").write_text(
                json.dumps(
                    {
                        "client": {},
                        "servers": [{"name": project.name, "service": {"target": "server.example.com:8002"}}],
                    }
                )
            )
        else:
            resources = {"components": [], "custom_resource": "preserved"}
        target = local / ProvFileName.RESOURCES_JSON_DEFAULT
        target.write_text(json.dumps(resources))
        originals[participant.name] = target.read_bytes()

    timeouts = resolve_token_timeouts(registration_token_timeout=registration_timeout)
    setting = ({"max_token_age_seconds": 300}, "http://127.0.0.1:8006/aa/token", 120, timeouts)
    builder = CoCoBuilder()
    builder.settings = {"site-1": setting}
    if protected_server:
        builder.settings["server.example.com"] = setting
    context = Mock()
    context.get_local_dir.side_effect = lambda participant: kits[participant.name] / "local"
    context.get_ws_dir.side_effect = lambda participant: kits[participant.name]
    builder.build(project, context)
    return kits, originals, timeouts


@pytest.mark.parametrize("registration_timeout", [300.0, 180.5])
@pytest.mark.parametrize("name", ["site-1", "plain-client"])
def test_protected_server_budget_reaches_registration_request(tmp_path, registration_timeout, name):
    kits, originals, timeouts = build_resources(tmp_path, registration_timeout=registration_timeout)
    kit = kits[name]
    expected = registration_timeout + 30.0
    resources = json.loads((kit / "local" / ProvFileName.RESOURCES_JSON_DEFAULT).read_text())
    assert resources["client"] == {"retry_timeout": 30, "compression": "Gzip", "maint_msg_timeout": expected}
    assert resources["custom_resource"] == "preserved"
    assert (kits["server.example.com"] / "local" / ProvFileName.RESOURCES_JSON_DEFAULT).read_bytes() == originals[
        "server.example.com"
    ]
    for participant_kit in kits.values():
        manager = json.loads((participant_kit / "local/cc_manager__p_resources.json").read_text())["components"][0][
            "args"
        ]
        assert {key: manager[key] for key in timeouts} == timeouts

    # Load the emitted resource through the real startup configurator and
    # deployer, then capture the actual Authenticator RPC timeout.
    args = SimpleNamespace(workspace=str(kit), job_id=None, set=[])
    configer = FLClientStarterConfiger(Workspace(str(kit), name), args, kv_list=[f"uid={name}"])
    with patch("nvflare.private.fed.app.fl_conf.ConfigService.initialize"):
        configer.finalize_config(Mock())
    client = configer.base_deployer.create_fed_client(args)
    cell = Mock(spec=Cell, send_request=Mock(), register_request_cb=Mock(), core_cell=Mock())
    cell.send_request.return_value = Message(
        payload={CellMessageHeaderKeys.TOKEN: "fixture-token", CellMessageHeaderKeys.SSID: "fixture-session"}
    )
    client.communicator.set_cell(cell)
    with patch("nvflare.private.fed.authenticator._get_client_ip", return_value="127.0.0.1"):
        client.communicator.client_registration(name, "coco-registration", FLContext())
    assert cell.send_request.call_args.kwargs["timeout"] == expected
    assert cell.send_request.call_count == 1


@pytest.mark.parametrize("current_timeout,expected", [(10, 330.0), (600.0, 600.0), ("600", 600.0), ("12.5", 330.0)])
def test_protected_server_preserves_larger_explicit_maintenance_timeout(tmp_path, current_timeout, expected):
    kits, _, _ = build_resources(tmp_path, current_timeout=current_timeout)
    for name in ("site-1", "plain-client"):
        config = json.loads((kits[name] / "local" / ProvFileName.RESOURCES_JSON_DEFAULT).read_text())
        assert config["client"]["maint_msg_timeout"] == expected


@pytest.mark.parametrize("current_timeout", [None, 12.0, 600.0, "600"])
def test_ordinary_server_leaves_all_client_resource_files_unchanged(tmp_path, current_timeout):
    kits, originals, _ = build_resources(tmp_path, protected_server=False, current_timeout=current_timeout)
    for name, kit in kits.items():
        assert (kit / "local" / ProvFileName.RESOURCES_JSON_DEFAULT).read_bytes() == originals[name]


@pytest.mark.parametrize("current_timeout", [0, -1, True, "invalid", float("nan"), float("inf")])
def test_protected_server_rejects_invalid_explicit_maintenance_timeout(tmp_path, current_timeout):
    with pytest.raises(ValueError, match="maint_msg_timeout must be finite and positive"):
        build_resources(tmp_path, current_timeout=current_timeout)
