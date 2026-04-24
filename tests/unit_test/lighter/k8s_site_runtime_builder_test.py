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

import yaml

from nvflare.lighter.constants import PropKey
from nvflare.lighter.ctx import ProvisionContext
from nvflare.lighter.entity import Project
from nvflare.lighter.impl.k8s_runtime import K8sRuntimeBuilder


def _make_project():
    project = Project(name="test_project", description="test")
    project.set_server("server-1", "nvidia", {"fed_learn_port": 8002, "admin_port": 8002})
    project.add_client("site-1", "hospital", {})
    return project


def _seed_local_resources(ctx, project):
    common_components = [
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
    server_resources = {
        "snapshot_persistor": {
            "args": {
                "storage": {
                    "args": {
                        "root_dir": "/tmp/nvflare/snapshot-storage",
                    }
                }
            }
        },
        "components": [
            {
                "id": "job_manager",
                "path": "nvflare.apis.impl.job_def_manager.SimpleJobDefManager",
                "args": {"uri_root": "/tmp/nvflare/jobs-storage", "job_store_id": "job_store"},
            },
            *common_components,
        ],
    }

    for participant in project.get_all_participants():
        participant.set_prop(PropKey.COMM_CONFIG_ARGS, {})
        os.makedirs(ctx.get_local_dir(participant), exist_ok=True)
        os.makedirs(ctx.get_kit_dir(participant), exist_ok=True)
        with open(os.path.join(ctx.get_local_dir(participant), "resources.json.default"), "wt") as f:
            json.dump(server_resources if participant == project.get_server() else {"components": common_components}, f)


def _load_resources(ctx, participant):
    with open(os.path.join(ctx.get_local_dir(participant), "resources.json.default"), "rt") as f:
        return json.load(f)


def test_k8s_site_runtime_builder_generates_charts_and_patches_runtime(tmp_path):
    project = _make_project()
    ctx = ProvisionContext(str(tmp_path), project)
    _seed_local_resources(ctx, project)

    builder = K8sRuntimeBuilder(
        docker_image="repo/nvflare:2.0.0",
        namespace="nvflare",
        parent_port=19002,
        workspace_pvc="shared-ws",
        workspace_mount_path="/workspace",
        study_data_pvc={"default": "nvfldata", "study-a": "study-a-pvc"},
    )
    builder.build(project, ctx)

    server = project.get_server()
    client = project.get_clients()[0]

    for participant, expected_launcher in [
        (server, "nvflare.app_opt.job_launcher.k8s_launcher.ServerK8sJobLauncher"),
        (client, "nvflare.app_opt.job_launcher.k8s_launcher.ClientK8sJobLauncher"),
    ]:
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

        k8s_launcher = next(c for c in components if c["id"] == "k8s_launcher")
        assert k8s_launcher["path"] == expected_launcher
        assert k8s_launcher["args"]["namespace"] == "nvflare"
        assert (
            k8s_launcher["args"]["study_data_pvc_file_path"] == "/var/tmp/nvflare/workspace/local/study_data_pvc.yaml"
        )

        with open(os.path.join(ctx.get_local_dir(participant), "study_data_pvc.yaml"), "rt") as f:
            pvc_config = yaml.safe_load(f)
        assert pvc_config == {"default": "nvfldata", "study-a": "study-a-pvc"}

        chart_dir = os.path.join(ctx.get_ws_dir(participant), "helm_chart")
        assert os.path.isdir(chart_dir)
        assert os.path.isfile(os.path.join(chart_dir, "Chart.yaml"))
        assert os.path.isfile(os.path.join(chart_dir, "values.yaml"))


def test_k8s_site_runtime_builder_can_preserve_existing_resource_manager_when_requested(tmp_path):
    project = _make_project()
    ctx = ProvisionContext(str(tmp_path), project)
    _seed_local_resources(ctx, project)

    builder = K8sRuntimeBuilder(docker_image="repo/nvflare:2.0.0", set_passthrough_resource_manager=False)
    builder.build(project, ctx)

    client = project.get_clients()[0]
    resources = _load_resources(ctx, client)
    components = resources["components"]
    component_ids = {c["id"] for c in components}

    assert "process_launcher" not in component_ids
    assert "resource_consumer" in component_ids
    resource_manager = next(c for c in components if c["id"] == "resource_manager")
    assert resource_manager["path"] == "nvflare.app_common.resource_managers.gpu_resource_manager.GPUResourceManager"
