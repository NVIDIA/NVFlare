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
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.fl_constant import ConnectionSecurity, ConnPropKey, SecureTrainConst, SiteType
from nvflare.apis.workspace import Workspace
from nvflare.fuel.data_event.data_bus import DataBus
from nvflare.fuel.data_event.utils import get_scope_property
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.private.fed.app.fl_conf import FLClientStarterConfiger, FLServerStarterConfiger
from nvflare.private.fed.utils.job_cert_utils import job_cert_paths, write_job_cert


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


def _workspace_with_kit(tmp_path, site_name):
    (tmp_path / "startup").mkdir()
    (tmp_path / "local").mkdir()
    return Workspace(str(tmp_path), site_name=site_name)


@pytest.mark.parametrize(
    ("job_id", "with_job_cert", "expect_job_cert"),
    [("job-1", True, True), ("job-1", False, False), (None, True, False)],
)
def test_server_configer_points_job_process_at_job_credential(tmp_path, job_id, with_job_cert, expect_job_cert):
    workspace = _workspace_with_kit(tmp_path, SiteType.SERVER)
    run_dir = workspace.get_run_dir("job-1")
    if with_job_cert:
        write_job_cert(run_dir, b"cert", b"key")
    configer = FLServerStarterConfiger.__new__(FLServerStarterConfiger)
    configer.args = SimpleNamespace(job_id=job_id)
    configer.workspace = workspace
    configer.server_config_file_names = ["fed_server.json"]
    server = {
        SecureTrainConst.SSL_ROOT_CERT: "rootCA.pem",
        SecureTrainConst.SSL_CERT: "server.crt",
        SecureTrainConst.PRIVATE_KEY: "server.key",
    }
    configer.config_data = {"servers": [server]}

    with patch("nvflare.private.fed.app.fl_conf.JsonConfigurator.start_config"):
        configer.start_config(MagicMock())

    startup = workspace.get_startup_kit_dir()
    assert server[SecureTrainConst.SSL_ROOT_CERT] == os.path.join(startup, "rootCA.pem")
    if expect_job_cert:
        assert (server[SecureTrainConst.SSL_CERT], server[SecureTrainConst.PRIVATE_KEY]) == job_cert_paths(run_dir)
    else:
        assert server[SecureTrainConst.SSL_CERT] == os.path.join(startup, "server.crt")
        assert server[SecureTrainConst.PRIVATE_KEY] == os.path.join(startup, "server.key")


@pytest.mark.parametrize(
    ("job_id", "with_job_cert", "expect_job_cert"),
    [("job-1", True, True), ("job-1", False, False), (None, True, False)],
)
def test_client_configer_points_job_process_at_job_credential(tmp_path, job_id, with_job_cert, expect_job_cert):
    workspace = _workspace_with_kit(tmp_path, "site-1")
    run_dir = workspace.get_run_dir("job-1")
    if with_job_cert:
        write_job_cert(run_dir, b"cert", b"key")
    configer = FLClientStarterConfiger.__new__(FLClientStarterConfiger)
    configer.args = SimpleNamespace(
        job_id=job_id,
        sp_scheme="grpc",
        sp_target="server.example.com:8002",
        parent_url="stcp://localhost:8102",
        parent_conn_sec="mtls",
    )
    configer.workspace = workspace
    configer.logger = logging.getLogger(__name__)
    configer.cmd_vars = {"uid": "site-1"}
    configer.client_config_file_names = ["fed_client.json"]
    client = {
        ConnPropKey.IDENTITY: "site-1",
        SecureTrainConst.SSL_ROOT_CERT: "rootCA.pem",
        SecureTrainConst.SSL_CERT: "client.crt",
        SecureTrainConst.PRIVATE_KEY: "client.key",
    }
    configer.config_data = {
        "servers": [{"service": {"scheme": "grpc", "target": "server.example.com:8002"}}],
        "client": client,
    }

    with patch("nvflare.private.fed.app.fl_conf.JsonConfigurator.start_config"):
        configer.start_config(MagicMock())

    startup = workspace.get_startup_kit_dir()
    assert client[SecureTrainConst.SSL_ROOT_CERT] == os.path.join(startup, "rootCA.pem")
    if expect_job_cert:
        assert (client[SecureTrainConst.SSL_CERT], client[SecureTrainConst.PRIVATE_KEY]) == job_cert_paths(run_dir)
    else:
        assert client[SecureTrainConst.SSL_CERT] == os.path.join(startup, "client.crt")
        assert client[SecureTrainConst.PRIVATE_KEY] == os.path.join(startup, "client.key")
