# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import collections
import copy
import json
import os

import pytest
import yaml

from nvflare.cli_exception import CLIException
from nvflare.lighter.utils import update_project_server_name_config
from nvflare.tool.poc import poc_commands
from nvflare.tool.poc.poc_commands import (
    add_poc_docker_runtime,
    client_gpu_assignments,
    get_gpu_ids,
    get_service_command,
    get_service_config,
    local_provision,
    prepare_builders,
    prepare_env,
    prepare_poc_provision,
    update_clients,
)
from nvflare.tool.poc.service_constants import FlareServiceConstants as SC


class TestPOCCommands:
    def test_client_gpu_assignments(self):
        clients = [f"site-{i}" for i in range(0, 12)]
        gpu_ids = [0, 1, 2, 0, 3]
        assignments = client_gpu_assignments(clients, gpu_ids)
        assert assignments == {
            "site-0": [0],
            "site-1": [1],
            "site-2": [2],
            "site-3": [0],
            "site-4": [3],
            "site-5": [0],
            "site-6": [1],
            "site-7": [2],
            "site-8": [0],
            "site-9": [3],
            "site-10": [0],
            "site-11": [1],
        }

    clients = [f"site-{i}" for i in range(0, 4)]
    gpu_ids = []
    assignments = client_gpu_assignments(clients, gpu_ids)
    assert assignments == {"site-0": [], "site-1": [], "site-2": [], "site-3": []}
    clients = [f"site-{i}" for i in range(0, 4)]
    gpu_ids = [0, 1, 2, 0, 3]
    assignments = client_gpu_assignments(clients, gpu_ids)
    assert assignments == {"site-0": [0, 3], "site-1": [1], "site-2": [2], "site-3": [0]}

    def test_get_gpu_ids(self):
        host_gpu_ids = [0]
        gpu_ids = get_gpu_ids(-1, host_gpu_ids)
        assert gpu_ids == [0]
        gpu_ids = get_gpu_ids([0], host_gpu_ids)
        assert gpu_ids == [0]
        with pytest.raises(CLIException) as e:
            # gpu id =1 is not valid GPU ID as the host only has 1 gpu where id = 0
            gpu_ids = get_gpu_ids([0, 1], host_gpu_ids)

    def test_prepare_env_docker_single_gpu(self):
        my_env = prepare_env("site-1", [0], {SC.IS_DOCKER_RUN: True})
        assert my_env["CUDA_VISIBLE_DEVICES"] == "0"
        assert my_env["GPU2USE"] == '--gpus="device=0"'
        assert my_env["SVR_NAME"] == "site-1"
        assert "MY_DATA_DIR" in my_env

    def test_prepare_env_docker_multi_gpu(self):
        my_env = prepare_env("site-1", [0, 1], {SC.IS_DOCKER_RUN: True})
        assert my_env["CUDA_VISIBLE_DEVICES"] == "0,1"
        assert my_env["GPU2USE"] == '--gpus="device=0,1"'

    def test_prepare_env_docker_no_gpu(self, monkeypatch):
        monkeypatch.delenv("GPU2USE", raising=False)
        my_env = prepare_env("site-1", [], {SC.IS_DOCKER_RUN: True})
        assert "GPU2USE" not in my_env
        assert my_env["SVR_NAME"] == "site-1"
        assert "MY_DATA_DIR" in my_env

    def test_prepare_env_non_docker_with_gpu(self, monkeypatch):
        monkeypatch.delenv("GPU2USE", raising=False)
        monkeypatch.delenv("SVR_NAME", raising=False)
        my_env = prepare_env("site-1", [0], {})
        assert my_env["CUDA_VISIBLE_DEVICES"] == "0"
        assert "GPU2USE" not in my_env
        assert "SVR_NAME" not in my_env

    def test_get_package_command(self):
        cmd = get_service_command(SC.CMD_START, "/tmp/nvflare/poc", SC.FLARE_SERVER, {})
        assert "/tmp/nvflare/poc/server/startup/start.sh" == cmd

        cmd = get_service_command(SC.CMD_START, "/tmp/nvflare/poc", SC.FLARE_PROJ_ADMIN, {})
        assert "/tmp/nvflare/poc/admin@nvidia.com/startup/fl_admin.sh" == cmd

        cmd = get_service_command(SC.CMD_START, "/tmp/nvflare/poc", "site-2000", {})
        assert "/tmp/nvflare/poc/site-2000/startup/start.sh" == cmd

    def test_static_file_docker_image_does_not_enable_poc_docker_mode(self):
        project_config = {
            "api_version": 3,
            "name": "example_project",
            "description": "NVIDIA FLARE sample project yaml file",
            "participants": [
                {"name": "server", "type": "server", "org": "nvidia", "fed_learn_port": 8002, "admin_port": 8003},
                {"name": "admin@nvidia.com", "type": "admin", "org": "nvidia", "role": "project_admin"},
                {"name": "lead@nvidia.com", "type": "admin", "org": "nvidia", "role": "lead"},
                {"name": "site-1", "type": "client", "org": "nvidia"},
                {"name": "site-2000", "type": "client", "org": "nvidia"},
            ],
            "builders": [
                {
                    "path": "nvflare.lighter.impl.static_file.StaticFileBuilder",
                    "args": {"config_folder": "config", "docker_image": "nvflare/nvflare"},
                },
            ],
        }

        project_config = collections.OrderedDict(project_config)
        global_packages = get_service_config(project_config)
        assert global_packages[SC.IS_DOCKER_RUN] is False
        assert global_packages[SC.DOCKER_RUN_MODE] == ""

        cmd = get_service_command(SC.CMD_START, "/tmp/nvflare/poc", SC.FLARE_SERVER, global_packages)
        assert "/tmp/nvflare/poc/server/startup/start.sh" == cmd

        cmd = get_service_command(SC.CMD_START, "/tmp/nvflare/poc", SC.FLARE_PROJ_ADMIN, global_packages)
        assert "/tmp/nvflare/poc/admin@nvidia.com/startup/fl_admin.sh" == cmd

        cmd = get_service_command(SC.CMD_START, "/tmp/nvflare/poc", "site-2000", global_packages)
        assert "/tmp/nvflare/poc/site-2000/startup/start.sh" == cmd

        cmd = get_service_command(SC.CMD_STOP, "/tmp/nvflare/poc", SC.FLARE_SERVER, global_packages)
        assert "touch /tmp/nvflare/poc/server/shutdown.fl" == cmd

        cmd = get_service_command(SC.CMD_STOP, "/tmp/nvflare/poc", SC.FLARE_PROJ_ADMIN, global_packages)
        assert "touch /tmp/nvflare/poc/admin@nvidia.com/shutdown.fl" == cmd

        cmd = get_service_command(SC.CMD_STOP, "/tmp/nvflare/poc", "site-2000", global_packages)
        assert "touch /tmp/nvflare/poc/site-2000/shutdown.fl" == cmd

    def test_get_package_command_docker_deploy_mode(self):
        project_config = {
            "api_version": 3,
            "name": "example_project",
            "description": "NVIDIA FLARE sample project yaml file",
            "participants": [
                {"name": "server", "type": "server", "org": "nvidia", "fed_learn_port": 8002, "admin_port": 8002},
                {"name": "admin@nvidia.com", "type": "admin", "org": "nvidia", "role": "project_admin"},
                {"name": "site-1", "type": "client", "org": "nvidia"},
            ],
            "poc_runtime": {"runtime": "docker", "docker_image": "nvflare/nvflare"},
        }

        project_config = collections.OrderedDict(project_config)
        global_packages = get_service_config(project_config)
        assert global_packages[SC.IS_DOCKER_RUN] is True
        assert global_packages[SC.DOCKER_RUN_MODE] == SC.DOCKER_RUN_MODE_DEPLOY

        cmd = get_service_command(SC.CMD_START, "/tmp/nvflare/poc", SC.FLARE_SERVER, global_packages)
        assert "/tmp/nvflare/poc/server/startup/start_docker.sh" == cmd

        cmd = get_service_command(SC.CMD_START, "/tmp/nvflare/poc", "site-1", global_packages)
        assert "/tmp/nvflare/poc/site-1/startup/start_docker.sh" == cmd

        cmd = get_service_command(SC.CMD_STOP, "/tmp/nvflare/poc", "site-1", global_packages)
        assert "docker stop site-1" == cmd

    def test_add_poc_docker_runtime_preserves_static_builder(self):
        project_config = collections.OrderedDict(
            {
                "participants": [
                    {"name": "server", "type": "server", "org": "nvidia", "fed_learn_port": 8002, "admin_port": 8003},
                    {"name": "admin@nvidia.com", "type": "admin", "org": "nvidia", "role": "project_admin"},
                    {"name": "site-1", "type": "client", "org": "nvidia"},
                ],
                "builders": [
                    {"path": "nvflare.lighter.impl.workspace.WorkspaceBuilder", "args": {}},
                    {"path": "nvflare.lighter.impl.static_file.StaticFileBuilder", "args": {"config_folder": "config"}},
                    {"path": "nvflare.lighter.impl.cert.CertBuilder", "args": {}},
                    {"path": "nvflare.lighter.impl.signature.SignatureBuilder", "args": {}},
                ],
            }
        )

        result = add_poc_docker_runtime("nvflare/site:latest", project_config)

        paths = [builder["path"] for builder in result["builders"]]
        assert paths == [
            "nvflare.lighter.impl.workspace.WorkspaceBuilder",
            "nvflare.lighter.impl.static_file.StaticFileBuilder",
            "nvflare.lighter.impl.cert.CertBuilder",
            "nvflare.lighter.impl.signature.SignatureBuilder",
        ]
        assert result["builders"][1]["args"] == {"config_folder": "config"}
        assert result["poc_runtime"] == {
            "runtime": "docker",
            "docker_image": "nvflare/site:latest",
            "network": "nvflare-network",
        }

    def test_get_docker_run_mode_rejects_invalid_poc_runtime(self):
        with pytest.raises(CLIException, match="poc_runtime must be a mapping"):
            poc_commands.get_docker_run_mode({"poc_runtime": "docker"})

        with pytest.raises(CLIException, match="poc_runtime.runtime must be a string"):
            poc_commands.get_docker_run_mode({"poc_runtime": {"runtime": 1}})

    def test_get_poc_docker_runtime_config_rejects_invalid_nested_mappings(self):
        with pytest.raises(CLIException, match="poc_runtime.parent must be a mapping"):
            poc_commands.get_poc_docker_runtime_config({"poc_runtime": {"runtime": "docker", "parent": "server"}})

        with pytest.raises(CLIException, match="poc_runtime.job_launcher must be a mapping"):
            poc_commands.get_poc_docker_runtime_config({"poc_runtime": {"runtime": "docker", "job_launcher": "bad"}})

    def test_prepare_poc_provision_prepares_docker_before_storage_update(self, monkeypatch, tmp_path):
        project_config = collections.OrderedDict(
            {
                "name": "example_project",
                "participants": [
                    {"name": "server", "type": "server", "org": "nvidia"},
                    {"name": "admin@nvidia.com", "type": "admin", "org": "nvidia", "role": "project_admin"},
                    {"name": "site-1", "type": "client", "org": "nvidia"},
                ],
                "poc_runtime": {"runtime": "docker", "docker_image": "nvflare/site:latest"},
            }
        )
        service_config = {
            SC.FLARE_SERVER: "server",
            SC.FLARE_PROJ_ADMIN: "admin@nvidia.com",
            SC.FLARE_OTHER_ADMINS: [],
            SC.FLARE_CLIENTS: ["site-1"],
            SC.IS_DOCKER_RUN: True,
            SC.DOCKER_RUN_MODE: SC.DOCKER_RUN_MODE_DEPLOY,
        }
        calls = []

        monkeypatch.setattr(poc_commands, "local_provision", lambda *_args, **_kwargs: (project_config, service_config))
        monkeypatch.setattr(poc_commands, "get_prod_dir", lambda *_args, **_kwargs: str(tmp_path / "prod_00"))
        monkeypatch.setattr(poc_commands, "_prepare_poc_docker_deployments", lambda *_args: calls.append("docker"))
        monkeypatch.setattr(poc_commands, "update_storage_locations", lambda **_kwargs: calls.append("storage"))
        monkeypatch.setattr(poc_commands, "get_examples_dir", lambda _examples_dir: None)

        prepare_poc_provision([], 1, str(tmp_path), "nvflare/site:latest")

        assert calls == ["docker", "storage"]

    def test_prepare_poc_docker_deployments_passes_workspace(self, monkeypatch, tmp_path):
        project_config = collections.OrderedDict(
            {
                "name": "example_project",
                "participants": [
                    {"name": "server", "type": "server", "org": "nvidia"},
                    {"name": "site-1", "type": "client", "org": "nvidia"},
                ],
                "poc_runtime": {"runtime": "docker", "docker_image": "nvflare/site:latest"},
            }
        )
        calls = []

        monkeypatch.setattr(poc_commands, "get_prod_dir", lambda *_args: str(tmp_path / "prod_00"))
        monkeypatch.setattr(
            poc_commands,
            "_prepare_poc_docker_kit",
            lambda kit, config, workspace: calls.append((kit, config, workspace)),
        )

        assert poc_commands._prepare_poc_docker_deployments(str(tmp_path), project_config) is True
        assert len(calls) == 2
        assert {workspace for _kit, _config, workspace in calls} == {str(tmp_path)}

    def test_write_poc_docker_study_data_maps_workspace_data(self, tmp_path):
        kit_dir = tmp_path / "prod_00" / "site-1"
        local_dir = kit_dir / "local"
        local_dir.mkdir(parents=True)
        study_data_file = local_dir / "study_data.yaml"
        study_data_file.write_text("{}\n", encoding="utf-8")

        poc_workspace = tmp_path / "poc"
        poc_commands._write_poc_docker_study_data(kit_dir, str(poc_workspace))

        assert yaml.safe_load(study_data_file.read_text(encoding="utf-8")) == {
            "default": {"poc": {"source": os.path.realpath(poc_workspace / "data"), "mode": "rw"}}
        }

    def test_local_provision_keeps_project_config_after_prepare_project_mutation(self, monkeypatch, tmp_path):
        project_config = collections.OrderedDict(
            {
                "name": "example_project",
                "participants": [
                    {"name": "server", "type": "server", "org": "nvidia", "fed_learn_port": 8002, "admin_port": 8003},
                    {"name": "admin@nvidia.com", "type": "admin", "org": "nvidia", "role": "project_admin"},
                ],
                "builders": [],
            }
        )

        monkeypatch.setattr(poc_commands, "gen_project_config_file", lambda _workspace: str(tmp_path / "project.yml"))
        monkeypatch.setattr(poc_commands, "load_yaml", lambda _path: copy.deepcopy(project_config))
        monkeypatch.setattr(poc_commands, "save_project_config", lambda *_args: None)
        monkeypatch.setattr(poc_commands, "update_server_name", lambda config: config)
        monkeypatch.setattr(poc_commands, "update_clients", lambda _clients, _n_clients, config: config)
        monkeypatch.setattr(poc_commands, "add_he_builder", lambda _use_he, config: config)
        monkeypatch.setattr(poc_commands, "update_server_default_host", lambda config, _host: config)
        monkeypatch.setattr(poc_commands, "prepare_builders", lambda _config: [])
        monkeypatch.setattr(poc_commands, "prepare_packager", lambda _config: object())

        def fake_prepare_project(config):
            config["participants"] = [{"__comm_config_args__": {}}]
            return object()

        class FakeProvisioner:
            def __init__(self, *_args, **_kwargs):
                pass

            def provision(self, *_args, **_kwargs):
                pass

        monkeypatch.setattr(poc_commands, "prepare_project", fake_prepare_project)
        monkeypatch.setattr(poc_commands, "Provisioner", FakeProvisioner)

        result, _service_config = local_provision([], 1, str(tmp_path), "nvflare/site:latest")

        assert result["participants"][0]["name"] == "server"
        assert result["participants"][0]["type"] == "server"
        assert result["poc_runtime"]["runtime"] == "docker"

    def test_patch_poc_docker_client_target_uses_server_alias(self, tmp_path):
        startup_dir = tmp_path / "startup"
        startup_dir.mkdir()
        fed_client_file = startup_dir / "fed_client.json"
        fed_client_file.write_text(
            """
{
  "servers": [
    {"service": {"target": "localhost:8002"}},
    {"service": {"target": "external.example.com:8002"}}
  ]
}
""",
            encoding="utf-8",
        )

        poc_commands._patch_poc_docker_client_target(tmp_path)

        patched = json.loads(fed_client_file.read_text(encoding="utf-8"))
        assert patched["servers"][0]["service"]["target"] == "server:8002"
        assert patched["servers"][1]["service"]["target"] == "external.example.com:8002"

    def test_update_server_name(self):

        project_config = {
            "participants": [
                {"name": "server1", "org": "nvidia", "type": "server"},
                {"name": "admin@nvidia.com", "org": "nvidia", "role": "project_admin", "type": "admin"},
                {"name": "client-1", "org": "nvidia", "type": "client"},
            ]
        }

        project_config = collections.OrderedDict(project_config)

        old_server_name = "server1"
        server_name = "server"
        update_project_server_name_config(project_config, old_server_name, server_name)
        servers = [p for p in project_config["participants"] if p["type"] == "server"]
        assert len(servers) == 1
        assert servers[0]["name"] == server_name

    def test_update_clients(self):
        project_config = {
            "participants": [
                {"name": "server1", "org": "nvidia", "type": "server"},
                {"name": "admin@nvidia.com", "org": "nvidia", "role": "project_admin", "type": "admin"},
                {"name": "client-1", "org": "nvidia", "type": "client"},
            ]
        }

        project_config = collections.OrderedDict(project_config)
        clients = []
        n_clients = 3
        project_config = update_clients(clients, n_clients, project_config)
        result_clients = [p["name"] for p in project_config["participants"] if p["type"] == "client"]
        assert len(result_clients) == 3
        assert result_clients == ["site-1", "site-2", "site-3"]

        clients = ["client-1", "client-2", "client-3", "client-4"]
        n_clients = 3
        project_config = update_clients(clients, n_clients, project_config)
        result_clients = [p["name"] for p in project_config["participants"] if p["type"] == "client"]
        assert len(result_clients) == len(clients)
        assert result_clients == clients

    def test_prepare_builders(self):
        project_config = {
            "participants": [
                {"name": "server1", "org": "nvidia", "type": "server"},
                {"name": "admin@nvidia.com", "org": "nvidia", "role": "project_admin", "type": "admin"},
                {"name": "client-1", "org": "nvidia", "type": "client"},
            ],
            "builders": [
                {
                    "path": "nvflare.lighter.impl.static_file.StaticFileBuilder",
                    "args": {"overseer_agent": {"args": {"sp_end_point": "server1: 8002: 8003"}}},
                },
                {"path": "nvflare.lighter.impl.cert.CertBuilder", "args": {}},
            ],
        }

        project_config = collections.OrderedDict(project_config)
        builders = prepare_builders(project_config)
        assert len(builders) == 2

    def test_get_packages_config(self):
        project_config = {
            "participants": [
                {"name": "server1", "org": "nvidia", "type": "server"},
                {"name": "admin@nvidia.com", "org": "nvidia", "role": "project_admin", "type": "admin"},
                {"name": "client-1", "org": "nvidia", "type": "client"},
                {"name": "client-2", "org": "nvidia", "type": "client"},
            ],
            "builders": [
                {
                    "path": "nvflare.lighter.impl.static_file.StaticFileBuilder",
                    "args": {"overseer_agent": {"args": {"sp_end_point": "server1: 8002: 8003"}}},
                },
                {"path": "nvflare.lighter.impl.cert.CertBuilder", "args": {}},
            ],
        }

        project_config = collections.OrderedDict(project_config)

        global_config = get_service_config(project_config)
        assert "server1" == global_config[SC.FLARE_SERVER]
        assert "admin@nvidia.com" == global_config[SC.FLARE_PROJ_ADMIN]
        assert ["client-1", "client-2"] == global_config[SC.FLARE_CLIENTS]
