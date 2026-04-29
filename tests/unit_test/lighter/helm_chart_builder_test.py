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
import logging
import os
import tempfile

import yaml

from nvflare.lighter.constants import CommConfigArg, PropKey, ProvFileName
from nvflare.lighter.ctx import ProvisionContext
from nvflare.lighter.entity import Participant, Project
from nvflare.lighter.impl.helm_chart import HelmChartBuilder, _split_image


def _make_project(num_clients=2, fed_learn_port=8002):
    server = Participant(type="server", name="server1", org="nvidia", props={"fed_learn_port": fed_learn_port})
    clients = [Participant(type="client", name=f"site-{i}", org="nvidia") for i in range(1, num_clients + 1)]
    p = Project(name="test_project", description="test")
    p.add_participant(server)
    for c in clients:
        p.add_participant(c)
    return p


def _make_ctx(root_dir, project):
    return ProvisionContext(root_dir, project)


def _seed_comm_config_args(project):
    """Mimic StaticFileBuilder.initialize(): pre-set COMM_CONFIG_ARGS = {} on every participant."""
    for p in project.get_all_participants():
        p.set_prop(PropKey.COMM_CONFIG_ARGS, {})


def _client_chart_dir(ctx, client_name):
    """Return the wip path for a client's Helm chart directory."""
    return os.path.join(ctx.get_wip_dir(), client_name, ProvFileName.HELM_CHART)


def _server_chart_dir(ctx, project):
    """Return the wip path for the server Helm chart directory."""
    return os.path.join(ctx.get_wip_dir(), project.get_server().name, ProvFileName.HELM_CHART)


def _run(builder, project, ctx):
    """Run the full initialize → build lifecycle."""
    builder.initialize(project, ctx)
    builder.build(project, ctx)


# ---------------------------------------------------------------------------
# _split_image
# ---------------------------------------------------------------------------


class TestSplitImage:
    def test_with_tag(self):
        repo, tag = _split_image("myregistry/nvflare:2.7.0")
        assert repo == "myregistry/nvflare"
        assert tag == "2.7.0"

    def test_without_tag(self):
        repo, tag = _split_image("myregistry/nvflare")
        assert repo == "myregistry/nvflare"
        assert tag == ""

    def test_registry_with_port(self):
        repo, tag = _split_image("localhost:32000/nvflare:3.5.10")
        assert repo == "localhost:32000/nvflare"
        assert tag == "3.5.10"


# ---------------------------------------------------------------------------
# Client chart tests
# ---------------------------------------------------------------------------


class TestClientChart:
    def test_output_directories_created(self):
        project = _make_project(num_clients=2)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0"), project, ctx)

            for i in (1, 2):
                chart_dir = _client_chart_dir(ctx, f"site-{i}")
                assert os.path.isdir(chart_dir), f"chart dir missing for site-{i}"
                assert os.path.isdir(os.path.join(chart_dir, "templates"))

    def test_required_files_present(self):
        project = _make_project(num_clients=1)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0"), project, ctx)

            chart_dir = _client_chart_dir(ctx, "site-1")
            templates_dir = os.path.join(chart_dir, "templates")

            assert os.path.isfile(os.path.join(chart_dir, ProvFileName.CHART_YAML))
            assert os.path.isfile(os.path.join(chart_dir, ProvFileName.VALUES_YAML))
            assert os.path.isfile(os.path.join(templates_dir, "_helpers.tpl"))
            assert os.path.isfile(os.path.join(templates_dir, "client-deployment.yaml"))
            assert os.path.isfile(os.path.join(templates_dir, "service.yaml"))

    def test_values_name_matches_client(self):
        project = _make_project(num_clients=1)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0"), project, ctx)

            values_path = os.path.join(_client_chart_dir(ctx, "site-1"), ProvFileName.VALUES_YAML)
            with open(values_path) as f:
                values = yaml.safe_load(f)

            assert values["name"] == "site-1"

    def test_values_uid_not_in_args(self):
        """uid= must not appear in values.yaml args — it is injected by the template."""
        project = _make_project(num_clients=1)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0"), project, ctx)

            values_path = os.path.join(_client_chart_dir(ctx, "site-1"), ProvFileName.VALUES_YAML)
            with open(values_path) as f:
                values = yaml.safe_load(f)

            for arg in values.get("args", []):
                assert not str(arg).startswith("uid="), "uid= must not be in values.yaml args"

    def test_uid_in_deployment_template(self):
        """client-deployment.yaml must contain the uid={{ .Values.name }} expression."""
        project = _make_project(num_clients=1)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0"), project, ctx)

            tmpl_path = os.path.join(_client_chart_dir(ctx, "site-1"), "templates", "client-deployment.yaml")
            with open(tmpl_path) as f:
                content = f.read()

            assert "uid={{ .Values.name }}" in content

    def test_image_split_into_repo_and_tag(self):
        project = _make_project(num_clients=1)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="localhost:32000/nvflare:3.5.10"), project, ctx)

            values_path = os.path.join(_client_chart_dir(ctx, "site-1"), ProvFileName.VALUES_YAML)
            with open(values_path) as f:
                values = yaml.safe_load(f)

            assert values["image"]["repository"] == "localhost:32000/nvflare"
            assert values["image"]["tag"] == "3.5.10"

    def test_host_aliases_absent_from_generated_chart(self):
        """hostAliases must not be generated — client connects outbound; DNS handles server resolution."""
        project = _make_project(num_clients=1)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0"), project, ctx)

            values_path = os.path.join(_client_chart_dir(ctx, "site-1"), ProvFileName.VALUES_YAML)
            with open(values_path) as f:
                values = yaml.safe_load(f)

            assert "hostAliases" not in values

    def test_chart_app_version_matches_image_tag(self):
        project = _make_project(num_clients=1)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0"), project, ctx)

            chart_path = os.path.join(_client_chart_dir(ctx, "site-1"), ProvFileName.CHART_YAML)
            with open(chart_path) as f:
                chart = yaml.safe_load(f)

            assert chart["appVersion"] == "2.7.0"

    def test_comm_config_args_set_when_seeded(self):
        """COMM_CONFIG_ARGS host/port must match the Helm Service name and containerPort."""
        project = _make_project(num_clients=1)
        _seed_comm_config_args(project)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0", parent_port=18002), project, ctx)

        client = project.get_clients()[0]
        args = client.get_prop(PropKey.COMM_CONFIG_ARGS)
        assert args[CommConfigArg.HOST] == "site-1", "host must equal client.name (= Kubernetes service name)"
        assert args[CommConfigArg.PORT] == 18002, "port must equal parent_port (= containerPort / targetPort)"

    def test_comm_config_args_not_set_when_not_seeded(self):
        """When StaticFileBuilder has not run, COMM_CONFIG_ARGS is None; build() must not raise."""
        project = _make_project(num_clients=1)
        # deliberately do NOT call _seed_comm_config_args
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0"), project, ctx)

        client = project.get_clients()[0]
        assert client.get_prop(PropKey.COMM_CONFIG_ARGS) is None

    def test_service_name_equals_client_name(self):
        """service.yaml must name the Service after the client (no -svc suffix)."""
        project = _make_project(num_clients=1)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0"), project, ctx)

            svc_path = os.path.join(_client_chart_dir(ctx, "site-1"), "templates", "service.yaml")
            with open(svc_path) as f:
                content = f.read()

        assert 'name: {{ include "nvflare-client.name" . }}' in content
        assert "-svc" not in content

    def test_values_has_no_restart_policy(self):
        """Client runs as a Deployment; restartPolicy is fixed to Always by k8s and not in values."""
        project = _make_project(num_clients=1)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0"), project, ctx)

            values_path = os.path.join(_client_chart_dir(ctx, "site-1"), ProvFileName.VALUES_YAML)
            with open(values_path) as f:
                values = yaml.safe_load(f)

        assert "restartPolicy" not in values

    def test_values_port_equals_parent_port(self):
        """port in client values.yaml must equal the parent_port constructor arg."""
        project = _make_project(num_clients=1)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0", parent_port=9900), project, ctx)

            values_path = os.path.join(_client_chart_dir(ctx, "site-1"), ProvFileName.VALUES_YAML)
            with open(values_path) as f:
                values = yaml.safe_load(f)

        assert values["port"] == 9900

    def test_chart_app_version_defaults_to_latest_when_no_tag(self):
        """appVersion must be 'latest' when docker_image has no tag."""
        project = _make_project(num_clients=1)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare"), project, ctx)

            chart_path = os.path.join(_client_chart_dir(ctx, "site-1"), ProvFileName.CHART_YAML)
            with open(chart_path) as f:
                chart = yaml.safe_load(f)

        assert chart["appVersion"] == "latest"

    def test_client_values_workspace_persistence_matches_constructor(self):
        """client values.yaml should only expose the workspace PVC settings."""
        project = _make_project(num_clients=1)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(
                HelmChartBuilder(
                    docker_image="myregistry/nvflare:2.7.0",
                    workspace_pvc="my-ws-pvc",
                    workspace_mount_path="/mnt/workspace",
                ),
                project,
                ctx,
            )

            values_path = os.path.join(_client_chart_dir(ctx, "site-1"), ProvFileName.VALUES_YAML)
            with open(values_path) as f:
                values = yaml.safe_load(f)

        assert values["persistence"]["workspace"]["claimName"] == "my-ws-pvc"
        assert values["persistence"]["workspace"]["friendlyName"] == "my-ws-pvc"
        assert values["persistence"]["workspace"]["mountPath"] == "/mnt/workspace"
        assert "etc" not in values["persistence"]


# ---------------------------------------------------------------------------
# Server chart tests
# ---------------------------------------------------------------------------


class TestServerChart:
    def test_server_chart_directory_created(self):
        project = _make_project()
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0"), project, ctx)

            assert os.path.isdir(_server_chart_dir(ctx, project))
            assert os.path.isdir(os.path.join(_server_chart_dir(ctx, project), "templates"))

    def test_server_chart_yaml_present(self):
        project = _make_project()
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0"), project, ctx)

            assert os.path.isfile(os.path.join(_server_chart_dir(ctx, project), ProvFileName.CHART_YAML))
            assert os.path.isfile(os.path.join(_server_chart_dir(ctx, project), ProvFileName.VALUES_YAML))

    def test_server_deployment_file_present(self):
        project = _make_project()
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0"), project, ctx)

            templates_dir = os.path.join(_server_chart_dir(ctx, project), "templates")
            assert os.path.isfile(os.path.join(templates_dir, "_helpers.tpl"))
            assert os.path.isfile(os.path.join(templates_dir, "server-deployment.yaml"))
            assert os.path.isfile(os.path.join(templates_dir, "server-service.yaml"))
            assert os.path.isfile(os.path.join(templates_dir, "server-tcp-services.yaml"))

    def test_server_deployment_uses_provided_image(self):
        """Image repo and tag are stored in values.yaml; the deployment template references them via .Values.image."""
        project = _make_project()
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0"), project, ctx)

            values_path = os.path.join(_server_chart_dir(ctx, project), ProvFileName.VALUES_YAML)
            with open(values_path) as f:
                values = yaml.safe_load(f)

            assert values["image"]["repository"] == "myregistry/nvflare"
            assert values["image"]["tag"] == "2.7.0"

    def test_server_deployment_name_matches_server_participant(self):
        """The server name is stored in values.yaml and referenced by the deployment template."""
        project = _make_project()
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0"), project, ctx)

            values_path = os.path.join(_server_chart_dir(ctx, project), ProvFileName.VALUES_YAML)
            with open(values_path) as f:
                values = yaml.safe_load(f)

            assert values["name"] == "server1"

    def test_server_deployment_args_contain_workspace_mount_path(self):
        """The workspace mount path must appear in values.yaml args."""
        project = _make_project()
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(
                HelmChartBuilder(
                    docker_image="myregistry/nvflare:2.7.0",
                    workspace_mount_path="/mnt/ws",
                ),
                project,
                ctx,
            )

            values_path = os.path.join(_server_chart_dir(ctx, project), ProvFileName.VALUES_YAML)
            with open(values_path) as f:
                values = yaml.safe_load(f)

            assert "/mnt/ws" in values["args"], "workspace_mount_path must appear in server args"

    def test_server_service_ports_match_fed_learn_port(self):
        """Ports from the project are stored in values.yaml and referenced by the service template."""
        project = _make_project(fed_learn_port=8888)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0"), project, ctx)

            values_path = os.path.join(_server_chart_dir(ctx, project), ProvFileName.VALUES_YAML)
            with open(values_path) as f:
                values = yaml.safe_load(f)

            assert values["fedLearnPort"] == 8888

    def test_server_parent_port_in_values(self):
        """parentPort in values.yaml must match the parent_port constructor arg."""
        project = _make_project()
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0", parent_port=9000), project, ctx)

            values_path = os.path.join(_server_chart_dir(ctx, project), ProvFileName.VALUES_YAML)
            with open(values_path) as f:
                values = yaml.safe_load(f)

            assert values["parentPort"] == 9000

    def test_server_comm_config_args_set_when_seeded(self):
        """COMM_CONFIG_ARGS host must be 'server' and port must equal parent_port."""
        project = _make_project()
        _seed_comm_config_args(project)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0", parent_port=18102), project, ctx)

        server = project.get_server()
        args = server.get_prop(PropKey.COMM_CONFIG_ARGS)
        assert args[CommConfigArg.HOST] == "nvflare-server"
        assert args[CommConfigArg.PORT] == 18102, "port must equal parent_port"

    def test_server_comm_config_args_not_set_when_not_seeded(self):
        """When StaticFileBuilder has not run, COMM_CONFIG_ARGS is None; build() must not raise."""
        project = _make_project()
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0"), project, ctx)

        server = project.get_server()
        assert server.get_prop(PropKey.COMM_CONFIG_ARGS) is None

    def test_server_tcp_services_file_present(self):
        project = _make_project()
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0"), project, ctx)

            templates_dir = os.path.join(_server_chart_dir(ctx, project), "templates")
            assert os.path.isfile(os.path.join(templates_dir, "server-tcp-services.yaml"))

    def test_server_tcp_services_maps_fed_learn_port(self):
        """tcp-services ConfigMap must map fedLearnPort → nvflare-server:<fedLearnPort>."""
        project = _make_project(fed_learn_port=8002)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0"), project, ctx)

            tcp_path = os.path.join(_server_chart_dir(ctx, project), "templates", "server-tcp-services.yaml")
            with open(tcp_path) as f:
                content = f.read()

        assert "nvflare-server" in content
        assert ".Values.fedLearnPort" in content

    def test_server_values_workspace_persistence_matches_constructor(self):
        """server values.yaml should only expose the workspace PVC settings."""
        project = _make_project()
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(
                HelmChartBuilder(
                    docker_image="myregistry/nvflare:2.7.0",
                    workspace_pvc="my-ws-pvc",
                    workspace_mount_path="/mnt/workspace",
                ),
                project,
                ctx,
            )

            values_path = os.path.join(_server_chart_dir(ctx, project), ProvFileName.VALUES_YAML)
            with open(values_path) as f:
                values = yaml.safe_load(f)

            assert values["persistence"]["workspace"]["claimName"] == "my-ws-pvc"
            assert values["persistence"]["workspace"]["friendlyName"] == "my-ws-pvc"
            assert values["persistence"]["workspace"]["mountPath"] == "/mnt/workspace"
            assert "etc" not in values["persistence"]

    def test_server_resources_store_job_and_snapshot_data_on_workspace_pvc(self):
        project = _make_project()
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            local_dir = os.path.join(ctx.get_ws_dir(project.get_server()), "local")
            os.makedirs(local_dir, exist_ok=True)
            default_resources = {
                "snapshot_persistor": {"args": {"storage": {"args": {"root_dir": "/tmp/nvflare/snapshot-storage"}}}},
                "components": [{"id": "job_manager", "args": {"uri_root": "/tmp/nvflare/jobs-storage"}}],
            }
            with open(os.path.join(local_dir, "resources.json.default"), "w") as f:
                json.dump(default_resources, f)
            _run(
                HelmChartBuilder(
                    docker_image="myregistry/nvflare:2.7.0",
                    workspace_mount_path="/mnt/workspace",
                ),
                project,
                ctx,
            )

            resources_path = os.path.join(ctx.get_ws_dir(project.get_server()), "local", "resources.json")
            with open(resources_path) as f:
                resources = json.load(f)

        assert (
            resources["snapshot_persistor"]["args"]["storage"]["args"]["root_dir"] == "/mnt/workspace/snapshot-storage"
        )
        job_manager = next(comp for comp in resources["components"] if comp["id"] == "job_manager")
        assert job_manager["args"]["uri_root"] == "/mnt/workspace/jobs-storage"

    def test_server_resources_missing_default_logs_warning(self, caplog):
        project = _make_project()
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            with caplog.at_level(logging.WARNING):
                _run(
                    HelmChartBuilder(
                        docker_image="myregistry/nvflare:2.7.0",
                        workspace_mount_path="/mnt/workspace",
                    ),
                    project,
                    ctx,
                )

            resources_path = os.path.join(ctx.get_ws_dir(project.get_server()), "local", "resources.json")
            assert not os.path.exists(resources_path)

        assert "resources.json.default not found" in caplog.text

    def test_no_overseer_files_generated(self):
        """Overseer manifests must never be produced by the builder."""
        project = _make_project()
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0"), project, ctx)

            templates_dir = os.path.join(_server_chart_dir(ctx, project), "templates")
            assert not os.path.exists(os.path.join(templates_dir, ProvFileName.DEPLOYMENT_OVERSEER_YAML))
            assert not os.path.exists(os.path.join(templates_dir, ProvFileName.SERVICE_OVERSEER_YAML))

    def test_admin_port_absent_when_equal_to_fed_learn_port(self):
        """adminPort must be None in values.yaml when it equals fedLearnPort."""
        project = _make_project(fed_learn_port=8002)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            # Do NOT set ADMIN_PORT in ctx so it defaults to fed_learn_port.
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0"), project, ctx)

            values_path = os.path.join(_server_chart_dir(ctx, project), ProvFileName.VALUES_YAML)
            with open(values_path) as f:
                values = yaml.safe_load(f)

        # yaml.safe_load converts a YAML null to Python None.
        assert values.get("adminPort") is None

    def test_admin_port_present_when_distinct_from_fed_learn_port(self):
        """adminPort must appear in values.yaml when it differs from fedLearnPort."""
        from nvflare.lighter.constants import CtxKey

        project = _make_project(fed_learn_port=8002)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            ctx[CtxKey.FED_LEARN_PORT] = 8002
            ctx[CtxKey.ADMIN_PORT] = 8003
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0"), project, ctx)

            values_path = os.path.join(_server_chart_dir(ctx, project), ProvFileName.VALUES_YAML)
            with open(values_path) as f:
                values = yaml.safe_load(f)

        assert values["adminPort"] == 8003

    def test_server_deployment_has_host_port_for_fed_learn_port(self):
        """server-deployment.yaml must bind fedLearnPort as hostPort for EC2 direct access."""
        project = _make_project()
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0"), project, ctx)

            deploy_path = os.path.join(_server_chart_dir(ctx, project), "templates", "server-deployment.yaml")
            with open(deploy_path) as f:
                content = f.read()

        assert "hostPort" in content
        assert ".Values.fedLearnPort" in content

    def test_server_service_has_fixed_name(self):
        """server-service.yaml must use the fixed name 'nvflare-server', not .Values.name."""
        project = _make_project()
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0"), project, ctx)

            svc_path = os.path.join(_server_chart_dir(ctx, project), "templates", "server-service.yaml")
            with open(svc_path) as f:
                content = f.read()

        assert "name: nvflare-server" in content

    def test_tcp_services_configmap_name_and_namespace(self):
        """tcp-services ConfigMap must be named 'nginx-ingress-microk8s-conf' in namespace 'ingress'."""
        project = _make_project()
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0"), project, ctx)

            tcp_path = os.path.join(_server_chart_dir(ctx, project), "templates", "server-tcp-services.yaml")
            with open(tcp_path) as f:
                content = f.read()

        assert "nginx-ingress-microk8s-conf" in content
        assert "namespace: ingress" in content

    def test_server_chart_app_version_matches_image_tag(self):
        """appVersion in server Chart.yaml must equal the image tag."""
        project = _make_project()
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0"), project, ctx)

            chart_path = os.path.join(_server_chart_dir(ctx, project), ProvFileName.CHART_YAML)
            with open(chart_path) as f:
                chart = yaml.safe_load(f)

        assert chart["appVersion"] == "2.7.0"

    def test_server_chart_app_version_defaults_to_latest_when_no_tag(self):
        """appVersion must be 'latest' when docker_image has no tag."""
        project = _make_project()
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare"), project, ctx)

            chart_path = os.path.join(_server_chart_dir(ctx, project), ProvFileName.CHART_YAML)
            with open(chart_path) as f:
                chart = yaml.safe_load(f)

        assert chart["appVersion"] == "latest"


# ---------------------------------------------------------------------------
# Combined build produces both chart types
# ---------------------------------------------------------------------------


class TestCombinedBuild:
    def test_both_server_and_client_charts_generated(self):
        project = _make_project(num_clients=2)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            _run(HelmChartBuilder(docker_image="myregistry/nvflare:2.7.0"), project, ctx)

            # server chart
            assert os.path.isdir(_server_chart_dir(ctx, project))

            # client charts
            for i in (1, 2):
                assert os.path.isdir(_client_chart_dir(ctx, f"site-{i}"))
