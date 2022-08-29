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

from nvflare.lighter.spec import Builder


class HelmChartBuilder(Builder):
    def __init__(self, docker_image):
        """Build Helm Chart."""
        self.docker_image = docker_image

    def initialize(self, ctx):
        self.helm_chart_directory = os.path.join(self.get_wip_dir(ctx), "nvflare_hc")
        os.mkdir(self.helm_chart_directory)

    def _build_overseer(self, overseer, ctx):
        protocol = overseer.props.get("protocol", "http")
        default_port = "443" if protocol == "https" else "80"
        port = overseer.props.get("port", default_port)
        self.deployment_overseer["spec"]["template"]["spec"]["volumes"][0]["hostPath"][
            "path"
        ] = "{{ .Values.workspace }}"
        self.deployment_overseer["spec"]["template"]["spec"]["containers"][0]["ports"][0]["containerPort"] = port
        self.deployment_overseer["spec"]["template"]["spec"]["containers"][0]["image"] = self.docker_image
        self.deployment_overseer["spec"]["template"]["spec"]["containers"][0]["command"][
            0
        ] = f"/workspace/{overseer.name}/startup/start.sh"
        with open(os.path.join(self.helm_chart_templates_directory, "deployment_overseer.yaml"), "wt") as f:
            yaml.dump(self.deployment_overseer, f)

        self.service_overseer["spec"]["ports"][0]["port"] = port
        self.service_overseer["spec"]["ports"][0]["targetPort"] = port
        with open(os.path.join(self.helm_chart_templates_directory, "service_overseer.yaml"), "wt") as f:
            yaml.dump(self.service_overseer, f)

    def _build_server(self, server, ctx):
        fed_learn_port = server.props.get("fed_learn_port", 30002)
        admin_port = server.props.get("admin_port", 30003)
        idx = ctx["index"]

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

    def build(self, project, ctx):
        self.template = ctx.get("template")
        with open(os.path.join(self.helm_chart_directory, "Chart.yaml"), "wt") as f:
            yaml.dump(yaml.safe_load(self.template.get("helm_chart_chart")), f)

        with open(os.path.join(self.helm_chart_directory, "values.yaml"), "wt") as f:
            yaml.dump(yaml.safe_load(self.template.get("helm_chart_values")), f)

        self.service_overseer = yaml.safe_load(self.template.get("helm_chart_service_overseer"))
        self.service_server = yaml.safe_load(self.template.get("helm_chart_service_server"))

        self.deployment_overseer = yaml.safe_load(self.template.get("helm_chart_deployment_overseer"))
        self.deployment_server = yaml.safe_load(self.template.get("helm_chart_deployment_server"))

        self.helm_chart_templates_directory = os.path.join(self.helm_chart_directory, "templates")
        os.mkdir(self.helm_chart_templates_directory)
        overseer = project.get_participants_by_type("overseer")
        self._build_overseer(overseer, ctx)
        servers = project.get_participants_by_type("server", first_only=False)
        for index, server in enumerate(servers):
            ctx["index"] = index
            self._build_server(server, ctx)
