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

import yaml

from nvflare.lighter.constants import CommConfigArg, ConnSecurity, CtxKey, PropKey, ProvFileName
from nvflare.lighter.entity import Participant
from nvflare.lighter.spec import Builder, Project, ProvisionContext

# ---------------------------------------------------------------------------
# Helm template file contents
# These are written verbatim — they contain Helm template syntax ({{ }}) that
# must not be passed through yaml.dump or any other YAML parser.
# ---------------------------------------------------------------------------

_HELPERS_TPL = """\
{{/*
NVFlare client chart helpers
*/}}
{{- define "nvflare-client.name" -}}
{{- default .Chart.Name .Values.name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "nvflare-client.labels" -}}
app.kubernetes.io/name: {{ include "nvflare-client.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "nvflare-client.selectorLabels" -}}
app.kubernetes.io/name: {{ include "nvflare-client.name" . }}
{{- end }}

{{- define "nvflare-client.image" -}}
{{- if .Values.image.tag }}
{{- printf "%s:%s" .Values.image.repository .Values.image.tag }}
{{- else }}
{{- .Values.image.repository }}
{{- end }}
{{- end }}
"""

_CLIENT_POD_YAML = """\
apiVersion: v1
kind: Pod
metadata:
  name: {{ include "nvflare-client.name" . }}
  labels:
    {{- include "nvflare-client.labels" . | nindent 4 }}
spec:
  restartPolicy: {{ .Values.restartPolicy }}
  volumes:
    - name: {{ .Values.persistence.etc.friendlyName }}
      persistentVolumeClaim:
        claimName: {{ .Values.persistence.etc.claimName }}
    - name: {{ .Values.persistence.workspace.friendlyName }}
      persistentVolumeClaim:
        claimName: {{ .Values.persistence.workspace.claimName }}
  containers:
    - name: {{ include "nvflare-client.name" . }}
      imagePullPolicy: {{ .Values.image.pullPolicy }}
      image: {{ include "nvflare-client.image" . }}
      ports:
        - containerPort: {{ .Values.port }}
      command:
        {{- toYaml .Values.command | nindent 8 }}
      args:
        {{- toYaml .Values.args | nindent 8 }}
        - uid={{ .Values.name }}
      volumeMounts:
        - name: {{ .Values.persistence.etc.friendlyName }}
          mountPath: {{ .Values.persistence.etc.mountPath }}
        - name: {{ .Values.persistence.workspace.friendlyName }}
          mountPath: {{ .Values.persistence.workspace.mountPath }}
"""

_SERVICE_YAML = """\
apiVersion: v1
kind: Service
metadata:
  name: {{ include "nvflare-client.name" . }}
  labels:
    {{- include "nvflare-client.labels" . | nindent 4 }}
spec:
  selector:
    {{- include "nvflare-client.selectorLabels" . | nindent 4 }}
  ports:
    - protocol: TCP
      port: {{ .Values.port }}
      targetPort: {{ .Values.port }}
"""


def _split_image(docker_image: str):
    """Split 'repo:tag' into (repo, tag). If no tag is present returns ('repo', '')."""
    if ":" in docker_image:
        repo, tag = docker_image.rsplit(":", 1)
        return repo, tag
    return docker_image, ""


class ClientHelmChartBuilder(Builder):
    def __init__(
        self,
        docker_image: str,
        client_port: int = 18002,
        workspace_pvc: str = "nvflws",
        etc_pvc: str = "nvfletc",
        workspace_mount_path: str = "/var/tmp/nvflare/workspace",
        etc_mount_path: str = "/var/tmp/nvflare/etc",
    ):
        """Generate one Helm chart per client during provisioning.

        Each client in the project receives its own chart directory under
        ``<wip>/nvflare_hc_clients/<client-name>/``.  The charts use the same
        structure as the hand-authored ``helm/nvflare-client/`` chart but are
        populated with client-specific values derived from the project
        definition.

        The ``uid=`` argument that identifies the client to the FL server is
        rendered via ``{{ .Values.name }}`` in the pod template instead of
        being hard-coded in ``values.yaml``, so a single ``--set name=<site>``
        override is sufficient to re-target the chart to a different site.

        ``hostAliases`` is not supported by the generated chart or the
        hand-authored chart.  The NVFlare client connects *outbound* to the
        FL server; the cluster's DNS resolves the server hostname.

        The ``containerPort`` declared in the pod spec is the port that
        job pods use to communicate *back* to this client process after they
        are launched by ``K8sJobLauncher``.  It is not used for
        client-to-server communication.

        The Kubernetes Service is named after the client (``<client.name>``),
        not ``<client.name>-svc``, so that the service DNS name exactly matches
        ``internal.resources.host`` in ``comm_config.json``.  During
        ``build()``, ``COMM_CONFIG_ARGS`` on the participant is updated with
        ``host=client.name`` and ``port=client_port`` so that
        ``StaticFileBuilder.finalize()`` emits a ``comm_config.json`` whose
        ``internal.resources.host`` and ``internal.resources.port`` are
        identical to the Service name and ``containerPort`` / ``targetPort``
        in the Helm chart.  This ensures job pods can reach the client process
        at ``<client.name>:<client_port>`` within the cluster namespace.

        Args:
            docker_image: container image for the NVFlare client, e.g.
                ``myregistry/nvflare:2.7.0``.
            client_port: port job pods use to talk back to this client
                process (default 18002).
            workspace_pvc: name of the PVC used as the runtime workspace.
            etc_pvc: name of the PVC that holds the provisioned startup kit
                (certificates and ``fed_client.json``).
            workspace_mount_path: mount path for the workspace PVC inside the
                container.
            etc_mount_path: mount path for the etc PVC inside the container.
        """
        self.docker_image = docker_image
        self.client_port = client_port
        self.workspace_pvc = workspace_pvc
        self.etc_pvc = etc_pvc
        self.workspace_mount_path = workspace_mount_path
        self.etc_mount_path = etc_mount_path

    # ------------------------------------------------------------------
    # Builder lifecycle
    # ------------------------------------------------------------------

    def build(self, project: Project, ctx: ProvisionContext):
        server = project.get_server()
        if not server:
            raise ValueError("project has no server; cannot build client Helm charts")

        fed_learn_port = ctx.get(CtxKey.FED_LEARN_PORT, 8002)

        for client in project.get_clients():
            self._build_client_chart(client, server, fed_learn_port, ctx)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_client_chart(self, client: Participant, server: Participant, fed_learn_port: int, ctx: ProvisionContext):
        # Place the chart inside the client's own wip directory so that after
        # WorkspaceBuilder.finalize() moves wip/ → prod_NN/ it lands at:
        #   prod_NN/<client-name>/nvflare_hc_clients/
        chart_dir = os.path.join(ctx.get_ws_dir(client), ProvFileName.HELM_CHART_CLIENT_DIR)
        templates_dir = os.path.join(chart_dir, "templates")
        os.makedirs(templates_dir, exist_ok=True)

        # Align comm_config.json with the Helm chart so that job pods launched by
        # K8sJobLauncher can reach the client process at <client.name>:<client_port>.
        # The service name equals client.name (no -svc suffix), so the DNS name
        # job pods use is exactly client.name within the same namespace.
        # StaticFileBuilder.initialize() pre-populates COMM_CONFIG_ARGS = {} for every
        # participant before any build() runs; we update it here so StaticFileBuilder
        # .finalize() emits the correct comm_config.json.
        comm_config_args = client.get_prop(PropKey.COMM_CONFIG_ARGS)
        if comm_config_args is not None:
            comm_config_args.update(
                {
                    CommConfigArg.HOST: client.name,
                    CommConfigArg.PORT: self.client_port,
                    CommConfigArg.SCHEME: "tcp",
                    CommConfigArg.CONN_SEC: ConnSecurity.CLEAR,
                }
            )

        self._write_chart_yaml(chart_dir, client)
        self._write_values_yaml(chart_dir, client, server, fed_learn_port)
        self._write_template_files(templates_dir)

    def _write_chart_yaml(self, chart_dir: str, client: Participant):
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

    def _write_values_yaml(
        self,
        chart_dir: str,
        client: Participant,
        server: Participant,
        fed_learn_port: int,
    ):
        repo, tag = _split_image(self.docker_image)

        # The args list intentionally omits uid= — it is appended by the pod
        # template via {{ .Values.name }} to eliminate the duplication between
        # the top-level 'name' field and the --set argument list.
        #
        # hostAliases is intentionally omitted. The client connects outbound
        # to the server; DNS is responsible for server name resolution.
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
            "port": self.client_port,
            "command": ["/usr/local/bin/python3"],
            "args": args,
            "restartPolicy": "Never",
        }

        with open(os.path.join(chart_dir, ProvFileName.VALUES_YAML), "wt") as f:
            yaml.dump(values, f, default_flow_style=False)

    def _write_template_files(self, templates_dir: str):
        _write(os.path.join(templates_dir, "_helpers.tpl"), _HELPERS_TPL)
        _write(os.path.join(templates_dir, "client-pod.yaml"), _CLIENT_POD_YAML)
        _write(os.path.join(templates_dir, "service.yaml"), _SERVICE_YAML)


def _write(path: str, content: str):
    with open(path, "wt") as f:
        f.write(content)
