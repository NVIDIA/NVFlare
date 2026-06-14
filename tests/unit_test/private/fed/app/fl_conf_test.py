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

import logging
from types import SimpleNamespace

import pytest

from nvflare.apis.fl_constant import ConnectionSecurity, ConnPropKey
from nvflare.fuel.data_event.data_bus import DataBus
from nvflare.fuel.data_event.utils import get_scope_property
from nvflare.fuel.f3.cellnet.fqcn import FQCN
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
