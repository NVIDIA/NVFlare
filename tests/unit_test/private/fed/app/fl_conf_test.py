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

import argparse
import builtins
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.fl_constant import ConnectionSecurity, ConnPropKey, SecureTrainConst, SiteType, SystemConfigs
from nvflare.apis.workspace import Workspace
from nvflare.fuel.data_event.data_bus import DataBus
from nvflare.fuel.data_event.utils import get_scope_property
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.lighter.constants import CtxKey
from nvflare.lighter.impl.static_file import StaticFileBuilder
from nvflare.lighter.impl.workspace import WorkspaceBuilder
from nvflare.lighter.provision import prepare_project
from nvflare.lighter.provisioner import Provisioner
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


@pytest.mark.parametrize("is_server", [True, False])
def test_starter_restart_never_searches_persisted_jobs_for_site_config(tmp_path, monkeypatch, is_server):
    site_name = SiteType.SERVER if is_server else "site-1"
    workspace = _workspace_with_kit(tmp_path, site_name)
    startup_name = "fed_server.json" if is_server else "fed_client.json"
    startup_config = {"servers": []} if is_server else {"client": {}, "servers": []}
    (tmp_path / "startup" / startup_name).write_text(json.dumps(startup_config))
    (tmp_path / "local" / "resources.json.default").write_text('{"site_owned": true}')
    job_app = Path(workspace.get_app_dir("old-job"))
    (job_app / "config").mkdir(parents=True)
    secret = tmp_path / "synthetic-secret.conf"
    secret.write_text('probe = "SYNTHETIC_FILE_SECRET"')
    monkeypatch.setenv("NVFLARE_RESTART_TEST_SECRET", "SYNTHETIC_ENV_SECRET")
    payload = f'include file("{secret}")\nprobe_env = ${{NVFLARE_RESTART_TEST_SECRET}}'
    (job_app / "resources.conf").write_text(payload)
    (job_app / "config" / "comm_config.conf").write_text(payload)

    configer_type = FLServerStarterConfiger if is_server else FLClientStarterConfiger
    configer = configer_type.__new__(configer_type)
    configer.workspace = workspace
    configer.args = argparse.Namespace(workspace=str(tmp_path), job_id=None)
    configer.cmd_vars = {}
    configer.app_validator = None
    configer.components = {}
    configer.handlers = []
    configer.config_data = startup_config
    if is_server:
        configer.server_config_file_names = workspace.get_config_files_for_startup(True, False)
        configer.snapshot_persistor = None
    else:
        configer.client_config_file_names = workspace.get_config_files_for_startup(False, False)

    real_open = builtins.open

    def site_only_open(path, *args, **kwargs):
        resolved = os.path.realpath(path)
        assert not resolved.startswith(str(job_app)), "startup read persisted job configuration"
        assert resolved != str(secret), "startup evaluated the persisted job include"
        return real_open(path, *args, **kwargs)

    ConfigService.reset()
    try:
        with patch("builtins.open", side_effect=site_only_open):
            configer.finalize_config(MagicMock())
            assert ConfigService.get_section(SystemConfigs.RESOURCES_CONF) == {"site_owned": True}
            assert ConfigService.load_configuration("comm_config") is None
            (tmp_path / "local" / "comm_config.conf").write_text('{"trusted": "site"}')
            assert ConfigService.load_configuration("comm_config").to_dict() == {"trusted": "site"}
    finally:
        ConfigService.reset()


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


@pytest.mark.parametrize("security", [ConnectionSecurity.CLEAR, ConnectionSecurity.TLS, ConnectionSecurity.MTLS])
@pytest.mark.parametrize("routed", [False, True])
def test_provisioned_async_tcp_client_conn_props(tmp_path, security, routed):
    client = {"name": "site-1", "type": "client", "org": "org", "use_aio": True, "connection_security": security}
    if routed:
        client["connect_to"] = {"name": "relay-1"}
    project = prepare_project(
        {
            "api_version": 3,
            "name": "async-tcp",
            "participants": [
                {"name": "server.example.com", "type": "server", "org": "org", "connection_security": security},
                client,
                {
                    "name": "relay-1",
                    "type": "relay",
                    "org": "org",
                    "use_aio": True,
                    "listening_host": {
                        "default_host": "relay-1",
                        "port": 8004,
                        "scheme": "atcp",
                        "connection_security": security,
                    },
                },
            ],
        }
    )
    ctx = Provisioner(str(tmp_path), [WorkspaceBuilder(), StaticFileBuilder(scheme="tcp")]).provision(project)
    assert not ctx.get(CtxKey.BUILD_ERROR)
    kit = Path(ctx[CtxKey.CURRENT_PROD_DIR]) / "site-1"
    config = json.loads((kit / "startup/fed_client.json").read_text())
    assert config["servers"][0]["service"]["scheme"] == "atcp"
    if routed:
        config.update(json.loads((kit / "local/relay__resources.json").read_text()))

    configer = FLClientStarterConfiger.__new__(FLClientStarterConfiger)
    configer.args = SimpleNamespace()
    configer.logger = logging.getLogger(__name__)
    configer.config_data = config
    configer._determine_conn_props("site-1", config)

    scheme = "atcp" if security == ConnectionSecurity.CLEAR else "satcp"
    root = get_scope_property("site-1", ConnPropKey.ROOT_CONN_PROPS)
    assert root[ConnPropKey.URL] == f"{scheme}://server.example.com:8002"
    assert root[ConnPropKey.CONNECTION_SECURITY] == security
    assert root[ConnPropKey.AUTH_IDENTITY] == "server.example.com"
    if routed:
        relay = get_scope_property("site-1", ConnPropKey.RELAY_CONN_PROPS)
        assert relay[ConnPropKey.URL] == f"{scheme}://relay-1:8004"
        assert relay[ConnPropKey.CONNECTION_SECURITY] == security
        assert relay[ConnPropKey.AUTH_IDENTITY] == "relay-1"
