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
import tempfile

import yaml

from nvflare.lighter.constants import CommConfigArg, PropKey, ProvFileName
from nvflare.lighter.ctx import ProvisionContext
from nvflare.lighter.entity import Participant, Project
from nvflare.lighter.impl.client_helm_chart import ClientHelmChartBuilder, _split_image


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


def _chart_dir(ctx, client_name):
    """Return the wip path for a client's Helm chart directory.

    After WorkspaceBuilder.finalize() this becomes:
        prod_NN/<client_name>/nvflare_hc_clients/
    """
    return os.path.join(ctx.get_wip_dir(), client_name, ProvFileName.HELM_CHART_CLIENT_DIR)


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


class TestClientHelmChartBuilder:
    def test_output_directories_created(self):
        project = _make_project(num_clients=2)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            builder = ClientHelmChartBuilder(docker_image="myregistry/nvflare:2.7.0")
            builder.build(project, ctx)

            for i in (1, 2):
                chart_dir = _chart_dir(ctx, f"site-{i}")
                assert os.path.isdir(chart_dir), f"chart dir missing for site-{i}"
                assert os.path.isdir(os.path.join(chart_dir, "templates"))

    def test_required_files_present(self):
        project = _make_project(num_clients=1)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            builder = ClientHelmChartBuilder(docker_image="myregistry/nvflare:2.7.0")
            builder.build(project, ctx)

            chart_dir = _chart_dir(ctx, "site-1")
            templates_dir = os.path.join(chart_dir, "templates")

            assert os.path.isfile(os.path.join(chart_dir, ProvFileName.CHART_YAML))
            assert os.path.isfile(os.path.join(chart_dir, ProvFileName.VALUES_YAML))
            assert os.path.isfile(os.path.join(templates_dir, "_helpers.tpl"))
            assert os.path.isfile(os.path.join(templates_dir, "client-pod.yaml"))
            assert os.path.isfile(os.path.join(templates_dir, "service.yaml"))

    def test_values_name_matches_client(self):
        project = _make_project(num_clients=1)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            builder = ClientHelmChartBuilder(docker_image="myregistry/nvflare:2.7.0")
            builder.build(project, ctx)

            values_path = os.path.join(_chart_dir(ctx, "site-1"), ProvFileName.VALUES_YAML)
            with open(values_path) as f:
                values = yaml.safe_load(f)

            assert values["name"] == "site-1"

    def test_values_uid_not_in_args(self):
        """uid= must not appear in values.yaml args — it is injected by the template."""
        project = _make_project(num_clients=1)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            builder = ClientHelmChartBuilder(docker_image="myregistry/nvflare:2.7.0")
            builder.build(project, ctx)

            values_path = os.path.join(_chart_dir(ctx, "site-1"), ProvFileName.VALUES_YAML)
            with open(values_path) as f:
                values = yaml.safe_load(f)

            for arg in values.get("args", []):
                assert not str(arg).startswith("uid="), "uid= must not be in values.yaml args"

    def test_uid_in_pod_template(self):
        """client-pod.yaml must contain the uid={{ .Values.name }} expression."""
        project = _make_project(num_clients=1)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            builder = ClientHelmChartBuilder(docker_image="myregistry/nvflare:2.7.0")
            builder.build(project, ctx)

            pod_path = os.path.join(_chart_dir(ctx, "site-1"), "templates", "client-pod.yaml")
            with open(pod_path) as f:
                content = f.read()

            assert "uid={{ .Values.name }}" in content

    def test_image_split_into_repo_and_tag(self):
        project = _make_project(num_clients=1)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            builder = ClientHelmChartBuilder(docker_image="localhost:32000/nvflare:3.5.10")
            builder.build(project, ctx)

            values_path = os.path.join(_chart_dir(ctx, "site-1"), ProvFileName.VALUES_YAML)
            with open(values_path) as f:
                values = yaml.safe_load(f)

            assert values["image"]["repository"] == "localhost:32000/nvflare"
            assert values["image"]["tag"] == "3.5.10"

    def test_host_aliases_absent_from_generated_chart(self):
        """hostAliases must not be generated — client connects outbound; DNS handles server resolution."""
        project = _make_project(num_clients=1)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            builder = ClientHelmChartBuilder(docker_image="myregistry/nvflare:2.7.0")
            builder.build(project, ctx)

            values_path = os.path.join(_chart_dir(ctx, "site-1"), ProvFileName.VALUES_YAML)
            with open(values_path) as f:
                values = yaml.safe_load(f)

            assert "hostAliases" not in values

    def test_chart_app_version_matches_image_tag(self):
        project = _make_project(num_clients=1)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            builder = ClientHelmChartBuilder(docker_image="myregistry/nvflare:2.7.0")
            builder.build(project, ctx)

            chart_path = os.path.join(_chart_dir(ctx, "site-1"), ProvFileName.CHART_YAML)
            with open(chart_path) as f:
                chart = yaml.safe_load(f)

            assert chart["appVersion"] == "2.7.0"

    def test_no_server_raises(self):
        # ProvisionContext.__init__ itself raises when the project has no server.
        import pytest

        p = Project(name="no_server", description="test")
        p.add_participant(Participant(type="client", name="site-1", org="nvidia"))
        with pytest.raises(Exception):
            with tempfile.TemporaryDirectory() as root:
                _make_ctx(root, p)

    def test_comm_config_args_set_when_seeded(self):
        """COMM_CONFIG_ARGS host/port must match the Helm chart service name and containerPort."""
        project = _make_project(num_clients=1)
        _seed_comm_config_args(project)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            builder = ClientHelmChartBuilder(docker_image="myregistry/nvflare:2.7.0", client_port=18002)
            builder.build(project, ctx)

        client = project.get_clients()[0]
        args = client.get_prop(PropKey.COMM_CONFIG_ARGS)
        assert args[CommConfigArg.HOST] == "site-1", "host must equal client.name (= Kubernetes service name)"
        assert args[CommConfigArg.PORT] == 18002, "port must equal client_port (= containerPort / targetPort)"

    def test_comm_config_args_not_set_when_not_seeded(self):
        """When StaticFileBuilder has not run, COMM_CONFIG_ARGS is None; build() must not raise."""
        project = _make_project(num_clients=1)
        # deliberately do NOT call _seed_comm_config_args
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            builder = ClientHelmChartBuilder(docker_image="myregistry/nvflare:2.7.0")
            builder.build(project, ctx)  # must not raise

        client = project.get_clients()[0]
        assert client.get_prop(PropKey.COMM_CONFIG_ARGS) is None

    def test_service_name_equals_client_name(self):
        """service.yaml must name the Service after the client (no -svc suffix)."""
        project = _make_project(num_clients=1)
        with tempfile.TemporaryDirectory() as root:
            ctx = _make_ctx(root, project)
            builder = ClientHelmChartBuilder(docker_image="myregistry/nvflare:2.7.0")
            builder.build(project, ctx)

            svc_path = os.path.join(_chart_dir(ctx, "site-1"), "templates", "service.yaml")
            with open(svc_path) as f:
                content = f.read()

        assert 'name: {{ include "nvflare-client.name" . }}' in content
        assert "-svc" not in content
