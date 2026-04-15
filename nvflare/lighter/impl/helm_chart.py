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

import os
import shutil

import yaml

import nvflare.lighter as prov
from nvflare.lighter.constants import CommConfigArg, ConnSecurity, CtxKey, PropKey, ProvFileName
from nvflare.lighter.entity import Participant
from nvflare.lighter.spec import Builder, Project, ProvisionContext

_HELM_TEMPLATES_DIR = os.path.join(os.path.dirname(prov.__file__), "templates", "helm")


def _split_image(docker_image: str):
    """Split ``'repo:tag'`` into ``(repo, tag)``.  Returns ``('repo', '')`` when no tag."""
    if ":" in docker_image:
        repo, tag = docker_image.rsplit(":", 1)
        return repo, tag
    return docker_image, ""


def _helm_src(role: str, filename: str) -> str:
    """Return the absolute path of a Helm template file shipped with this package."""
    return os.path.join(_HELM_TEMPLATES_DIR, role, filename)


class HelmChartBuilder(Builder):
    def __init__(
        self,
        docker_image: str,
        parent_port: int = 8102,
        workspace_pvc: str = "nvflws",
        etc_pvc: str = "nvfletc",
        workspace_mount_path: str = "/var/tmp/nvflare/workspace",
        etc_mount_path: str = "/var/tmp/nvflare/etc",
    ):
        """Build Helm charts for the FL server and all FL clients.

        Both the server chart and client charts follow the same construction
        pattern: ``Chart.yaml`` and ``values.yaml`` are built from Python dicts,
        and Kubernetes manifests are copied from package template files.

        **Server chart** — written to ``<wip>/<server-name>/helm_chart/``.  Uses a
        Kubernetes Deployment backed by PersistentVolumeClaims.  A tcp-services
        ConfigMap instructs the nginx ingress controller to open ``fedLearnPort``
        for raw TCP passthrough to the ``nvflare-server`` Service on the same port.

        **Client charts** — one chart per client, written to
        ``<wip>/<client-name>/helm_chart/``.  Uses a Kubernetes Pod
        backed by PersistentVolumeClaims.  The ``uid=`` argument that identifies
        the client to the FL server is rendered via ``{{ .Values.name }}`` in
        the pod template so a single ``--set name=<site>`` override is
        sufficient to re-target the chart.

        ``COMM_CONFIG_ARGS`` on each client participant is updated with
        ``host=client.name`` and ``port=parent_port`` (when pre-seeded by
        ``StaticFileBuilder``) so that ``comm_config.json`` uses the Kubernetes
        Service DNS name and the matching port.

        Args:
            docker_image: container image used for all participants, e.g.
                ``myregistry/nvflare:2.7.0``.
            parent_port: port job pods use to talk back to the client process
                (default 8102).  Exposed as ``containerPort`` in the Pod and
                as ``port``/``targetPort`` in the client Service.
            workspace_pvc: PVC claim name for the runtime workspace volume.
            etc_pvc: PVC claim name for the startup-kit/etc volume.
            workspace_mount_path: in-container mount path for the workspace PVC.
            etc_mount_path: in-container mount path for the etc PVC.
        """
        self.docker_image = docker_image
        self.parent_port = parent_port
        self.workspace_pvc = workspace_pvc
        self.etc_pvc = etc_pvc
        self.workspace_mount_path = workspace_mount_path
        self.etc_mount_path = etc_mount_path

    # ------------------------------------------------------------------
    # Builder lifecycle
    # ------------------------------------------------------------------

    def build(self, project: Project, ctx: ProvisionContext):
        """Generate the server Helm chart and one chart per client."""
        self._build_server_chart(project, ctx)
        self._build_client_charts(project, ctx)

    # ------------------------------------------------------------------
    # Server chart
    # ------------------------------------------------------------------

    def _build_server_chart(self, project: Project, ctx: ProvisionContext):
        server = project.get_server()
        if not server:
            return

        chart_dir = os.path.join(ctx.get_ws_dir(server), ProvFileName.HELM_CHART)
        templates_dir = os.path.join(chart_dir, "templates")
        os.makedirs(templates_dir, exist_ok=True)

        fed_learn_port = ctx.get(CtxKey.FED_LEARN_PORT, 8002)
        admin_port = ctx.get(CtxKey.ADMIN_PORT, 8003)

        # Align comm_config.json with the chart so that internal jobs can reach the
        # parent at <nvflare-server>:<parent_port> within the cluster namespace.
        comm_config_args = server.get_prop(PropKey.COMM_CONFIG_ARGS)
        if comm_config_args is not None:
            comm_config_args.update(
                {
                    CommConfigArg.HOST: "nvflare-server",
                    CommConfigArg.PORT: self.parent_port,
                    CommConfigArg.SCHEME: "tcp",
                    CommConfigArg.CONN_SEC: ConnSecurity.CLEAR,
                }
            )

        self._write_server_chart_yaml(chart_dir, server)
        self._write_server_values_yaml(chart_dir, server, fed_learn_port, admin_port)
        self._write_server_template_files(templates_dir)

    def _write_server_chart_yaml(self, chart_dir: str, server: Participant):
        _, tag = _split_image(self.docker_image)
        chart = {
            "apiVersion": "v2",
            "name": "nvflare-server",
            "description": f"NVFlare federated learning server for {server.name}",
            "type": "application",
            "version": "0.1.0",
            "appVersion": tag or "latest",
            "keywords": ["nvflare", "federated-learning"],
            "maintainers": [],
        }
        with open(os.path.join(chart_dir, ProvFileName.CHART_YAML), "wt") as f:
            yaml.dump(chart, f, default_flow_style=False)

    def _write_server_values_yaml(self, chart_dir: str, server: Participant, fed_learn_port: int, admin_port: int):
        repo, tag = _split_image(self.docker_image)
        args = [
            "-u",
            "-m",
            "nvflare.private.fed.app.server.server_train",
            "-m",
            self.workspace_mount_path,
            "-s",
            "fed_server.json",
            "--set",
            "secure_train=true",
            "config_folder=config",
            f"org={server.org}",
        ]
        values = {
            "name": server.name,
            "image": {
                "repository": repo,
                "tag": tag,
                "pullPolicy": "IfNotPresent",
            },
            "serviceAccount": {
                "create": True,
                "annotations": {},
                "automountServiceAccountToken": True,
            },
            "rbac": {
                "create": True,
            },
            "persistence": {
                "etc": {
                    "claimName": self.etc_pvc,
                    "friendlyName": self.etc_pvc,
                    "mountPath": self.etc_mount_path,
                },
                "workspace": {
                    "claimName": self.workspace_pvc,
                    "friendlyName": self.workspace_pvc,
                    "mountPath": self.workspace_mount_path,
                },
            },
            "fedLearnPort": fed_learn_port,
            "adminPort": admin_port if admin_port != fed_learn_port else None,
            "parentPort": self.parent_port,
            "resources": {
                "requests": {
                    "cpu": "2",
                    "memory": "8Gi",
                },
            },
            "securityContext": {},
            "hostPortEnabled": True,
            "tcpConfigMapEnabled": True,
            "service": {
                "type": "ClusterIP",
                "loadBalancerIP": None,
            },
            "command": ["/usr/local/bin/python3"],
            "args": args,
        }
        with open(os.path.join(chart_dir, ProvFileName.VALUES_YAML), "wt") as f:
            yaml.dump(values, f, default_flow_style=False)

    def _write_server_template_files(self, templates_dir: str):
        for src, dst in [
            (_helm_src("server", "_helpers.tpl"), "_helpers.tpl"),
            (_helm_src("server", "deployment.yaml"), "server-deployment.yaml"),
            (_helm_src("server", "service.yaml"), "server-service.yaml"),
            (_helm_src("server", "tcp-services.yaml"), "server-tcp-services.yaml"),
            (_helm_src("server", "serviceaccount.yaml"), "serviceaccount.yaml"),
            (_helm_src("server", "role.yaml"), "role.yaml"),
        ]:
            shutil.copy(src, os.path.join(templates_dir, dst))

    # ------------------------------------------------------------------
    # Client charts
    # ------------------------------------------------------------------

    def _build_client_charts(self, project: Project, ctx: ProvisionContext):
        server = project.get_server()
        if not server:
            raise ValueError("project has no server; cannot build client Helm charts")

        for client in project.get_clients():
            self._build_one_client_chart(client, ctx)

    def _build_one_client_chart(self, client: Participant, ctx: ProvisionContext):
        chart_dir = os.path.join(ctx.get_ws_dir(client), ProvFileName.HELM_CHART)
        templates_dir = os.path.join(chart_dir, "templates")
        os.makedirs(templates_dir, exist_ok=True)

        # Align comm_config.json with the chart so that job pods can reach
        # the client at <client.name>:<parent_port> within the cluster namespace.
        comm_config_args = client.get_prop(PropKey.COMM_CONFIG_ARGS)
        if comm_config_args is not None:
            comm_config_args.update(
                {
                    CommConfigArg.HOST: client.name,
                    CommConfigArg.PORT: self.parent_port,
                    CommConfigArg.SCHEME: "tcp",
                    CommConfigArg.CONN_SEC: ConnSecurity.CLEAR,
                }
            )

        self._write_client_chart_yaml(chart_dir, client)
        self._write_client_values_yaml(chart_dir, client)
        self._write_client_template_files(templates_dir)

    def _write_client_chart_yaml(self, chart_dir: str, client: Participant):
        _, tag = _split_image(self.docker_image)
        chart = {
            "apiVersion": "v2",
            "name": "nvflare-client",
            "description": f"NVFlare federated learning client pod and service for {client.name}",
            "type": "application",
            "version": "0.1.0",
            "appVersion": tag or "latest",
            "keywords": ["nvflare", "federated-learning"],
            "maintainers": [],
        }
        with open(os.path.join(chart_dir, ProvFileName.CHART_YAML), "wt") as f:
            yaml.dump(chart, f, default_flow_style=False)

    def _write_client_values_yaml(self, chart_dir: str, client: Participant):
        repo, tag = _split_image(self.docker_image)
        args = [
            "-u",
            "-m",
            "nvflare.private.fed.app.client.client_train",
            "-m",
            self.workspace_mount_path,
            "-s",
            "fed_client.json",
            "--set",
            "secure_train=true",
            "config_folder=config",
            f"org={client.org}",
        ]
        values = {
            "name": client.name,
            "image": {
                "repository": repo,
                "tag": tag,
                "pullPolicy": "Always",
            },
            "serviceAccount": {
                "create": True,
                "annotations": {},
                "automountServiceAccountToken": True,
            },
            "rbac": {
                "create": True,
            },
            "persistence": {
                "etc": {
                    "claimName": self.etc_pvc,
                    "friendlyName": self.etc_pvc,
                    "mountPath": self.etc_mount_path,
                },
                "workspace": {
                    "claimName": self.workspace_pvc,
                    "friendlyName": self.workspace_pvc,
                    "mountPath": self.workspace_mount_path,
                },
            },
            "port": self.parent_port,
            "securityContext": {},
            "resources": {
                "requests": {
                    "cpu": "2",
                    "memory": "8Gi",
                },
            },
            "command": ["/usr/local/bin/python3"],
            "args": args,
            "restartPolicy": "Never",
        }
        with open(os.path.join(chart_dir, ProvFileName.VALUES_YAML), "wt") as f:
            yaml.dump(values, f, default_flow_style=False)

    def _write_client_template_files(self, templates_dir: str):
        for src, dst in [
            (_helm_src("client", "_helpers.tpl"), "_helpers.tpl"),
            (_helm_src("client", "pod.yaml"), "client-pod.yaml"),
            (_helm_src("client", "service.yaml"), "service.yaml"),
            (_helm_src("client", "serviceaccount.yaml"), "serviceaccount.yaml"),
            (_helm_src("client", "role.yaml"), "role.yaml"),
        ]:
            shutil.copy(src, os.path.join(templates_dir, dst))
