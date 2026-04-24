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

from nvflare.app_opt.job_launcher.k8s_launcher import ClientK8sJobLauncher, ServerK8sJobLauncher
from nvflare.lighter import utils
from nvflare.lighter.constants import ProvFileName
from nvflare.lighter.impl.helm_chart import HelmChartBuilder
from nvflare.lighter.spec import Builder, Project, ProvisionContext

_LAUNCHER_IDS = {"process_launcher", "docker_launcher", "k8s_launcher"}
_PASSTHROUGH_RESOURCE_MANAGER_PATH = (
    "nvflare.app_common.resource_managers.passthrough_resource_manager.PassthroughResourceManager"
)
_DEFAULT_STUDY_DATA_PVC = {"default": "nvfldata"}
_STUDY_DATA_PVC_FILE_PATH = "/var/tmp/nvflare/workspace/local/study_data_pvc.yaml"


class K8sRuntimeBuilder(Builder):
    """Apply K8s runtime packaging to a site by reusing HelmChartBuilder and patching runtime config."""

    def __init__(
        self,
        docker_image: str,
        namespace: str = "default",
        parent_port: int = 8102,
        workspace_pvc: str = "nvflws",
        workspace_mount_path: str = "/var/tmp/nvflare/workspace",
        study_data_pvc: dict = None,
        config_file_path: str = None,
        pending_timeout: int = None,
        python_path: str = "/usr/local/bin/python",
        security_context: dict = None,
        set_passthrough_resource_manager: bool = True,
    ):
        self.namespace = namespace
        self.study_data_pvc = study_data_pvc or dict(_DEFAULT_STUDY_DATA_PVC)
        self.config_file_path = config_file_path
        self.pending_timeout = pending_timeout
        self.python_path = python_path
        self.security_context = security_context
        self.set_passthrough_resource_manager = set_passthrough_resource_manager
        self.helm_builder = HelmChartBuilder(
            docker_image=docker_image,
            parent_port=parent_port,
            workspace_pvc=workspace_pvc,
            workspace_mount_path=workspace_mount_path,
        )

    def _inject_launcher(self, dest_dir: str, path: str):
        resources_file = os.path.join(dest_dir, ProvFileName.RESOURCES_JSON_DEFAULT)
        with open(resources_file, "rt") as f:
            resources = json.load(f)

        components = resources.get("components", [])
        resources["components"] = [c for c in components if c.get("id") not in _LAUNCHER_IDS]
        args = {
            "config_file_path": self.config_file_path,
            "study_data_pvc_file_path": _STUDY_DATA_PVC_FILE_PATH,
            "namespace": self.namespace,
            "python_path": self.python_path,
        }
        if self.pending_timeout is not None:
            args["pending_timeout"] = self.pending_timeout
        if self.security_context:
            args["security_context"] = self.security_context
        resources["components"].append({"id": "k8s_launcher", "path": path, "args": args})
        utils.write(resources_file, json.dumps(resources, indent=4), "t")

    def _set_resource_manager(self, dest_dir: str):
        resources_file = os.path.join(dest_dir, ProvFileName.RESOURCES_JSON_DEFAULT)
        with open(resources_file, "rt") as f:
            resources = json.load(f)

        components = resources.get("components", [])
        new_components = []
        found_resource_manager = False
        for component in components:
            component_id = component.get("id")
            if component_id == "resource_consumer":
                continue
            if component_id == "resource_manager":
                component = {
                    "id": "resource_manager",
                    "path": _PASSTHROUGH_RESOURCE_MANAGER_PATH,
                    "args": {},
                }
                found_resource_manager = True
            new_components.append(component)

        if not found_resource_manager:
            new_components.insert(
                0,
                {
                    "id": "resource_manager",
                    "path": _PASSTHROUGH_RESOURCE_MANAGER_PATH,
                    "args": {},
                },
            )

        resources["components"] = new_components
        utils.write(resources_file, json.dumps(resources, indent=4), "t")

    def _write_study_data_pvc_file(self, dest_dir: str):
        utils.write(
            os.path.join(dest_dir, "study_data_pvc.yaml"), yaml.safe_dump(self.study_data_pvc, sort_keys=False), "t"
        )

    def _patch_server(self, ctx: ProvisionContext, server):
        dest_dir = ctx.get_local_dir(server)
        if self.set_passthrough_resource_manager:
            self._set_resource_manager(dest_dir)
        self._inject_launcher(dest_dir, ServerK8sJobLauncher.__module__ + ".ServerK8sJobLauncher")
        self._write_study_data_pvc_file(dest_dir)

    def _patch_client(self, ctx: ProvisionContext, client):
        dest_dir = ctx.get_local_dir(client)
        if self.set_passthrough_resource_manager:
            self._set_resource_manager(dest_dir)
        self._inject_launcher(dest_dir, ClientK8sJobLauncher.__module__ + ".ClientK8sJobLauncher")
        self._write_study_data_pvc_file(dest_dir)

    def build(self, project: Project, ctx: ProvisionContext):
        self.helm_builder.build(project, ctx)

        server = project.get_server()
        if server:
            self._patch_server(ctx, server)

        for client in project.get_clients():
            self._patch_client(ctx, client)
