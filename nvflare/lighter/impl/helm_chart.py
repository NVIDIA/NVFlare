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

import os

import yaml

from nvflare.lighter.constants import CtxKey, PropKey, ProvFileName, TemplateSectionKey
from nvflare.lighter.entity import Participant
from nvflare.lighter.spec import Builder, Project, ProvisionContext


class HelmChartBuilder(Builder):
    def __init__(self, docker_image):
        """Build Helm Chart."""
        self.docker_image = docker_image
        self.helm_chart_directory = None
        self.service_overseer = None
        self.service_server = None
        self.deployment_server = None
        self.deployment_overseer = None
        self.helm_chart_templates_directory = None

    def initialize(self, project: Project, ctx: ProvisionContext):
        ctx.load_templates("master_template.yml")
        self.helm_chart_directory = os.path.join(ctx.get_wip_dir(), ProvFileName.HELM_CHART_DIR)
        os.mkdir(self.helm_chart_directory)

    def _build_overseer(self, overseer: Participant):
        protocol = overseer.get_prop(PropKey.PROTOCOL, "http")
        default_port = "443" if protocol == "https" else "80"
        port = overseer.get_prop(PropKey.PORT, default_port)
        self.deployment_overseer["spec"]["template"]["spec"]["volumes"][0]["hostPath"][
            "path"
        ] = "{{ .Values.workspace }}"
        self.deployment_overseer["spec"]["template"]["spec"]["containers"][0]["ports"][0]["containerPort"] = port
        self.deployment_overseer["spec"]["template"]["spec"]["containers"][0]["image"] = self.docker_image
        self.deployment_overseer["spec"]["template"]["spec"]["containers"][0]["command"][
            0
        ] = f"/workspace/{overseer.name}/startup/start.sh"
        with open(os.path.join(self.helm_chart_templates_directory, ProvFileName.DEPLOYMENT_OVERSEER_YAML), "wt") as f:
            yaml.dump(self.deployment_overseer, f)

        self.service_overseer["spec"]["ports"][0]["port"] = port
        self.service_overseer["spec"]["ports"][0]["targetPort"] = port
        with open(os.path.join(self.helm_chart_templates_directory, ProvFileName.SERVICE_OVERSEER_YAML), "wt") as f:
            yaml.dump(self.service_overseer, f)

    def _build_server(self, server: Participant, ctx: ProvisionContext, idx: int):
        fed_learn_port = ctx.get(CtxKey.FED_LEARN_PORT, 30002)
        admin_port = ctx.get(CtxKey.ADMIN_PORT, 30003)

        self.deployment_server["metadata"]["name"] = f"{server.name}"
        self.deployment_server["metadata"]["labels"]["system"] = f"{server.name}"

        self.deployment_server["spec"]["selector"]["matchLabels"]["system"] = f"{server.name}"

        self.deployment_server["spec"]["template"]["metadata"]["labels"]["system"] = f"{server.name}"
        self.deployment_server["spec"]["template"]["spec"]["volumes"][0]["hostPath"]["path"] = "{{ .Values.workspace }}"
        self.deployment_server["spec"]["template"]["spec"]["volumes"][1]["hostPath"]["path"] = "{{ .Values.persist }}"
        self.deployment_server["spec"]["template"]["spec"]["containers"][0]["name"] = f"{server.name}"
        self.deployment_server["spec"]["template"]["spec"]["containers"][0]["image"] = self.docker_image
        self.deployment_server["spec"]["template"]["spec"]["containers"][0]["ports"][0][
            "containerPort"
        ] = fed_learn_port
        self.deployment_server["spec"]["template"]["spec"]["containers"][0]["ports"][1]["containerPort"] = admin_port
        cmd_args = self.deployment_server["spec"]["template"]["spec"]["containers"][0]["args"]
        for i, item in enumerate(cmd_args):
            if "/workspace/server" in item:
                cmd_args[i] = f"/workspace/{server.name}"
            if "__org_name__" in item:
                cmd_args[i] = f"org={server.org}"
        self.deployment_server["spec"]["template"]["spec"]["containers"][0]["args"] = cmd_args
        with open(os.path.join(self.helm_chart_templates_directory, f"deployment_server{idx}.yaml"), "wt") as f:
            yaml.dump(self.deployment_server, f)

        self.service_server["metadata"]["name"] = f"{server.name}"
        self.service_server["metadata"]["labels"]["system"] = f"{server.name}"

        self.service_server["spec"]["selector"]["system"] = f"{server.name}"
        self.service_server["spec"]["ports"][0]["name"] = "fl-port"
        self.service_server["spec"]["ports"][0]["port"] = fed_learn_port
        self.service_server["spec"]["ports"][0]["targetPort"] = fed_learn_port
        self.service_server["spec"]["ports"][1]["name"] = "admin-port"
        self.service_server["spec"]["ports"][1]["port"] = admin_port
        self.service_server["spec"]["ports"][1]["targetPort"] = admin_port

        with open(os.path.join(self.helm_chart_templates_directory, f"service_server{idx}.yaml"), "wt") as f:
            yaml.dump(self.service_server, f)

    def build(self, project: Project, ctx: ProvisionContext):
        with open(os.path.join(self.helm_chart_directory, ProvFileName.CHART_YAML), "wt") as f:
            yaml.dump(ctx.yaml_load_template_section(TemplateSectionKey.HELM_CHART_CHART), f)

        with open(os.path.join(self.helm_chart_directory, ProvFileName.VALUES_YAML), "wt") as f:
            yaml.dump(ctx.yaml_load_template_section(TemplateSectionKey.HELM_CHART_VALUES), f)

        self.service_overseer = ctx.yaml_load_template_section(TemplateSectionKey.HELM_CHART_SERVICE_OVERSEER)
        self.service_server = ctx.yaml_load_template_section(TemplateSectionKey.HELM_CHART_SERVICE_SERVER)

        self.deployment_overseer = ctx.yaml_load_template_section(TemplateSectionKey.HELM_CHART_DEPLOYMENT_OVERSEER)
        self.deployment_server = ctx.yaml_load_template_section(TemplateSectionKey.HELM_CHART_DEPLOYMENT_SERVER)
        self.helm_chart_templates_directory = os.path.join(
            self.helm_chart_directory, ProvFileName.HELM_CHART_TEMPLATES_DIR
        )
        os.mkdir(self.helm_chart_templates_directory)
        overseer = project.get_overseer()
        if overseer:
            self._build_overseer(overseer)

        server = project.get_server()
        if server:
            self._build_server(server, ctx, 0)
