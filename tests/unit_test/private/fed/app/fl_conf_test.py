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
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from nvflare.apis.fl_constant import ConnectionSecurity, ConnPropKey
from nvflare.fuel.data_event.data_bus import DataBus
from nvflare.fuel.data_event.utils import get_scope_property
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.lighter.constants import CtxKey
from nvflare.lighter.impl.static_file import StaticFileBuilder
from nvflare.lighter.impl.workspace import WorkspaceBuilder
from nvflare.lighter.provision import prepare_project
from nvflare.lighter.provisioner import Provisioner
from nvflare.private.fed.app.fl_conf import FLClientStarterConfiger


@pytest.fixture(autouse=True)
def clear_data_bus():
    DataBus().data_store.clear()
    yield
    DataBus().data_store.clear()


def test_cp_conn_props_include_root_auth_identity():
    configer = FLClientStarterConfiger.__new__(FLClientStarterConfiger)
    configer.args = SimpleNamespace()
    configer.logger = logging.getLogger(__name__)
    configer.config_data = {
        "servers": [
            {
                "service": {
                    "scheme": "grpc",
                    "target": "server.example.com:8002",
                },
                ConnPropKey.IDENTITY: FQCN.ROOT_SERVER,
                ConnPropKey.AUTH_IDENTITY: "custom-server-cn",
            }
        ],
        "client": {
            ConnPropKey.IDENTITY: "site-1",
            ConnPropKey.AUTH_IDENTITY: "custom-site-cn",
            ConnPropKey.CONNECTION_SECURITY: ConnectionSecurity.MTLS,
        },
    }

    configer._determine_conn_props("site-1", configer.config_data)

    root_conn_props = get_scope_property("site-1", ConnPropKey.ROOT_CONN_PROPS)
    assert root_conn_props == {
        ConnPropKey.FQCN: FQCN.ROOT_SERVER,
        ConnPropKey.IDENTITY: "custom-server-cn",
        ConnPropKey.AUTH_IDENTITY: "custom-server-cn",
        ConnPropKey.CONNECTION_SECURITY: ConnectionSecurity.MTLS,
        ConnPropKey.URL: "grpcs://server.example.com:8002",
    }

    cp_conn_props = get_scope_property("site-1", ConnPropKey.CP_CONN_PROPS)
    assert cp_conn_props[ConnPropKey.AUTH_IDENTITY] == "custom-site-cn"


@pytest.mark.parametrize("security", [ConnectionSecurity.CLEAR, ConnectionSecurity.TLS, ConnectionSecurity.MTLS])
@pytest.mark.parametrize("routed", [False, True])
def test_provisioned_async_grpc_client_conn_props(tmp_path, security, routed):
    client = {"name": "site-1", "type": "client", "org": "org", "use_aio": True, "connection_security": security}
    participants = [
        {"name": "server.example.com", "type": "server", "org": "org", "connection_security": security},
        client,
    ]
    if routed:
        client["connect_to"] = {"name": "relay-1"}
        participants.append(
            {
                "name": "relay-1",
                "type": "relay",
                "org": "org",
                "listening_host": {
                    "default_host": "relay-1",
                    "port": 8004,
                    "scheme": "grpc",
                    "connection_security": security,
                },
            }
        )

    project = prepare_project({"api_version": 3, "name": "async-grpc", "participants": participants})
    ctx = Provisioner(str(tmp_path), [WorkspaceBuilder(), StaticFileBuilder(scheme="grpc")]).provision(project)
    assert not ctx.get(CtxKey.BUILD_ERROR)

    kit = Path(ctx[CtxKey.CURRENT_PROD_DIR]) / "site-1"
    config = json.loads((kit / "startup/fed_client.json").read_text())
    assert config["servers"][0]["service"]["scheme"] == "agrpc"
    if routed:
        config.update(json.loads((kit / "local/relay__resources.json").read_text()))

    configer = FLClientStarterConfiger.__new__(FLClientStarterConfiger)
    configer.args = SimpleNamespace()
    configer.logger = logging.getLogger(__name__)
    configer.config_data = config
    configer._determine_conn_props("site-1", config)

    scheme = "agrpc" if security == ConnectionSecurity.CLEAR else "agrpcs"
    root = get_scope_property("site-1", ConnPropKey.ROOT_CONN_PROPS)
    assert root[ConnPropKey.URL] == f"{scheme}://server.example.com:8002"
    assert root[ConnPropKey.CONNECTION_SECURITY] == security
    assert root[ConnPropKey.AUTH_IDENTITY] == "server.example.com"
