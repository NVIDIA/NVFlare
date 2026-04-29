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

from nvflare.lighter.constants import PropKey
from nvflare.lighter.ctx import ProvisionContext
from nvflare.lighter.entity import Project
from nvflare.lighter.impl.docker_launcher import DockerLauncherBuilder


def _make_project():
    project = Project(name="test_project", description="test")
    project.set_server("server-1", "nvidia", {"fed_learn_port": 8002, "admin_port": 8002})
    project.add_client("site-1", "hospital", {})
    return project


def _seed_kit_dirs(ctx, project):
    resources = {
        "components": [
            {
                "id": "resource_manager",
                "path": "nvflare.app_common.resource_managers.gpu_resource_manager.GPUResourceManager",
                "args": {"num_of_gpus": 2},
            },
            {
                "id": "resource_consumer",
                "path": "nvflare.app_common.resource_consumers.gpu_resource_consumer.GPUResourceConsumer",
                "args": {},
            },
            {
                "id": "process_launcher",
                "path": "nvflare.app_common.job_launcher.client_process_launcher.ClientProcessJobLauncher",
                "args": {},
            },
            {
                "id": "keep_me",
                "path": "example.Component",
                "args": {"x": 1},
            },
        ]
    }

    for participant in project.get_all_participants():
        participant.set_prop(PropKey.COMM_CONFIG_ARGS, {})
        os.makedirs(ctx.get_local_dir(participant), exist_ok=True)
        os.makedirs(ctx.get_kit_dir(participant), exist_ok=True)
        with open(os.path.join(ctx.get_local_dir(participant), "resources.json.default"), "wt") as f:
            json.dump(resources, f)


def _load_resources(ctx, participant):
    with open(os.path.join(ctx.get_local_dir(participant), "resources.json.default"), "rt") as f:
        return json.load(f)


def test_docker_launcher_builder_uses_custom_args_and_cleans_process_mode_components(tmp_path):
    project = _make_project()
    ctx = ProvisionContext(str(tmp_path), project)
    _seed_kit_dirs(ctx, project)

    builder = DockerLauncherBuilder(
        docker_image="repo/nvflare:1.2.3",
        network="my-overlay",
        python_path="/opt/bin/python3",
        default_job_container_kwargs={"shm_size": "8g", "ipc_mode": "host"},
        default_job_env={"NCCL_P2P_DISABLE": "1"},
    )
    builder.initialize(project, ctx)
    builder.build(project, ctx)

    server = project.get_server()
    client = project.get_clients()[0]

    for participant in (server, client):
        resources = _load_resources(ctx, participant)
        components = resources["components"]
        component_ids = {c["id"] for c in components}

        assert "process_launcher" not in component_ids
        assert "resource_consumer" not in component_ids
        assert "keep_me" in component_ids

        resource_manager = next(c for c in components if c["id"] == "resource_manager")
        assert (
            resource_manager["path"]
            == "nvflare.app_common.resource_managers.passthrough_resource_manager.PassthroughResourceManager"
        )

        docker_launcher = next(c for c in components if c["id"] == "docker_launcher")
        assert docker_launcher["args"]["network"] == "my-overlay"
        assert docker_launcher["args"]["python_path"] == "/opt/bin/python3"
        assert docker_launcher["args"]["default_job_env"] == {"NCCL_P2P_DISABLE": "1"}
        assert docker_launcher["args"]["default_job_container_kwargs"] == {
            "shm_size": "8g",
            "ipc_mode": "host",
        }

        start_docker = os.path.join(ctx.get_kit_dir(participant), "start_docker.sh")
        with open(start_docker, "rt") as f:
            script = f.read()
        assert 'NETWORK_NAME="my-overlay"' in script
        assert "repo/nvflare:1.2.3" in script

    assert server.get_prop(PropKey.COMM_CONFIG_ARGS)["host"] == "0.0.0.0"
    assert client.get_prop(PropKey.COMM_CONFIG_ARGS)["host"] == "0.0.0.0"


def test_docker_launcher_builder_can_preserve_existing_resource_manager_when_requested(tmp_path):
    project = _make_project()
    ctx = ProvisionContext(str(tmp_path), project)
    _seed_kit_dirs(ctx, project)

    builder = DockerLauncherBuilder(docker_image="repo/nvflare:1.2.3", set_passthrough_resource_manager=False)
    builder.initialize(project, ctx)
    builder.build(project, ctx)

    client = project.get_clients()[0]
    resources = _load_resources(ctx, client)
    components = resources["components"]
    component_ids = {c["id"] for c in components}

    assert "process_launcher" not in component_ids
    assert "resource_consumer" in component_ids
    resource_manager = next(c for c in components if c["id"] == "resource_manager")
    assert resource_manager["path"] == "nvflare.app_common.resource_managers.gpu_resource_manager.GPUResourceManager"
