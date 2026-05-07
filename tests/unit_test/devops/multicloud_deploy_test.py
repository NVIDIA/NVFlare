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

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

DEPLOY_PY = Path(__file__).resolve().parents[3] / "devops" / "multicloud" / "deploy.py"
REPO_ROOT = DEPLOY_PY.parents[2]
DRY_RUN_CONFIG_DIR = Path(__file__).resolve().parent / "multicloud_dry_run_configs"
DRY_RUN_GOLD_DIR = Path(__file__).resolve().parent / "multicloud_dry_run_gold"
sys.path.insert(0, str(DEPLOY_PY.parent))
SPEC = importlib.util.spec_from_file_location("multicloud_deploy", DEPLOY_PY)
DEPLOY_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DEPLOY_MODULE
SPEC.loader.exec_module(DEPLOY_MODULE)


class TestPrepareRuntimeKits:
    def test_writes_study_data_config_and_invokes_deploy_prepare(self, monkeypatch, tmp_path):
        prod_dir = tmp_path / "prod"
        kit_dir = prod_dir / "site-1"
        (kit_dir / "local").mkdir(parents=True)
        work_dir = tmp_path / ".work"
        study_data = {"default": {"data": {"source": "nvfldata", "mode": "ro"}}}
        participant = DEPLOY_MODULE.Participant(
            name="site-1",
            namespace="ns-1",
            kubeconfig="/tmp/kubeconfig",
            role="client",
            cloud="gcp",
            prepare={
                "runtime": "k8s",
                "namespace": "ns-1",
                "parent": {
                    "docker_image": "repo/image:tag",
                    "workspace_pvc": "nvflws",
                    "workspace_mount_path": "/var/tmp/nvflare/workspace",
                    "pod_security_context": {"runAsUser": 1000},
                },
                "job_launcher": {
                    "default_python_path": "/usr/local/bin/python3",
                    "job_pod_security_context": {"runAsUser": 1000},
                    "pending_timeout": 7,
                },
            },
            study_data=study_data,
        )
        calls = []

        monkeypatch.setattr(DEPLOY_MODULE, "WORK_DIR", work_dir)
        monkeypatch.setattr(DEPLOY_MODULE, "nvflare_cmd", lambda: "nvflare")
        monkeypatch.setattr(DEPLOY_MODULE, "run", lambda cmd, **kwargs: calls.append(cmd) or DEPLOY_MODULE.FakeProc(0))

        prepared = DEPLOY_MODULE.prepare_runtime_kits(prod_dir, [participant])

        output_dir = kit_dir / "prepared" / "k8s"
        config_path = work_dir / "prepare-configs" / "site-1.yaml"
        assert prepared == {"site-1": output_dir}
        assert yaml.safe_load((kit_dir / "local" / "study_data.yaml").read_text()) == study_data
        assert not (kit_dir / "local" / "comm_config.json").exists()
        assert yaml.safe_load(config_path.read_text()) == {
            "runtime": "k8s",
            "namespace": "ns-1",
            "parent": {
                "docker_image": "repo/image:tag",
                "workspace_pvc": "nvflws",
                "workspace_mount_path": "/var/tmp/nvflare/workspace",
                "pod_security_context": {"runAsUser": 1000},
            },
            "job_launcher": {
                "default_python_path": "/usr/local/bin/python3",
                "job_pod_security_context": {"runAsUser": 1000},
                "pending_timeout": 7,
            },
        }
        assert calls == [
            [
                "nvflare",
                "deploy",
                "prepare",
                "--kit",
                str(kit_dir),
                "--output",
                str(output_dir),
                "--config",
                str(config_path),
            ]
        ]

    def test_injects_server_forwarded_system_monitoring_after_prepare(self, monkeypatch, tmp_path):
        prod_dir = tmp_path / "prod"
        work_dir = tmp_path / ".work"
        for site in ("gcp-server", "aws-client-2"):
            (prod_dir / site / "local").mkdir(parents=True)

        common_prepare = {
            "runtime": "k8s",
            "parent": {
                "docker_image": "repo/image:tag",
                "workspace_pvc": "nvflws",
                "workspace_mount_path": "/var/tmp/nvflare/workspace",
            },
            "job_launcher": {"default_python_path": "/usr/local/bin/python3"},
        }
        participants = [
            DEPLOY_MODULE.Participant(
                name="gcp-server",
                namespace="nvflare-server",
                kubeconfig="/tmp/gcp",
                role="server",
                cloud="gcp",
                prepare=common_prepare,
            ),
            DEPLOY_MODULE.Participant(
                name="aws-client-2",
                namespace="nvflare-client-2",
                kubeconfig="/tmp/aws",
                role="client",
                cloud="aws",
                prepare=common_prepare,
            ),
        ]
        monitoring = DEPLOY_MODULE.MonitoringConfig(
            enabled=True,
            namespace="nvflare-all-clouds-monitoring",
            statsd_host="statsd-exporter.nvflare-all-clouds-monitoring.svc.cluster.local",
            env="all-clouds",
        )

        def _run(cmd, **kwargs):
            output_dir = Path(cmd[cmd.index("--output") + 1])
            (output_dir / "local").mkdir(parents=True)
            (output_dir / "local" / "resources.json.default").write_text(
                json.dumps({"components": [{"id": "k8s_launcher", "path": "launcher", "args": {}}]})
            )
            return DEPLOY_MODULE.FakeProc(0)

        monkeypatch.setattr(DEPLOY_MODULE, "WORK_DIR", work_dir)
        monkeypatch.setattr(DEPLOY_MODULE, "nvflare_cmd", lambda: "nvflare")
        monkeypatch.setattr(DEPLOY_MODULE, "run", _run)

        prepared = DEPLOY_MODULE.prepare_runtime_kits(prod_dir, participants, monitoring=monitoring)

        server_resources = json.loads((prepared["gcp-server"] / "local" / "resources.json.default").read_text())
        client_resources = json.loads((prepared["aws-client-2"] / "local" / "resources.json.default").read_text())
        server_components = {c["id"]: c for c in server_resources["components"]}
        client_components = {c["id"]: c for c in client_resources["components"]}

        assert {"sys_metrics_collector", "remote_metrics_receiver", "statsd_reporter", "k8s_launcher"} <= set(
            server_components
        )
        assert server_components["statsd_reporter"]["args"] == {
            "site": "server",
            "host": "statsd-exporter.nvflare-all-clouds-monitoring.svc.cluster.local",
            "port": 9125,
        }
        assert {"sys_metrics_collector", "event_to_fed", "k8s_launcher"} <= set(client_components)
        assert "statsd_reporter" not in client_components
        assert client_components["sys_metrics_collector"]["args"]["streaming_to_server"] is True
        assert client_components["event_to_fed"]["args"] == {"events_to_convert": ["metrics_event"]}


class TestLoadConfig:
    def test_defaults_kubeconfig_to_repo_tmp_by_cloud(self, tmp_path):
        config_path = tmp_path / "deploy.yaml"
        config_path.write_text(
            """
clouds:
  gcp:
    prepare:
      runtime: k8s
      parent:
        docker_image: repo/image:tag
participants:
  - {name: gcp-server, cloud: gcp, namespace: nvflare-server, role: server}
"""
        )

        config = DEPLOY_MODULE.load_config(config_path)

        assert config.participants[0].kubeconfig == str(REPO_ROOT / ".tmp" / "kubeconfigs" / "gcp.yaml")

    def test_deployment_state_uses_deterministic_ip_name_from_config_name(self, tmp_path):
        config_path = tmp_path / "deploy.yaml"
        config_path.write_text(
            """
name: Test Cluster 01
clouds:
  gcp:
    prepare:
      runtime: k8s
      parent:
        docker_image: repo/image:tag
participants:
  - {name: gcp-server, cloud: gcp, namespace: nvflare-server, role: server}
"""
        )

        config = DEPLOY_MODULE.load_config(config_path)
        state = DEPLOY_MODULE.deployment_state(config, gcp_project="test-project")

        assert state["ip_name"] == "nvflare-test-cluster-01"
        assert state["participants"]["gcp-server"]["role"] == "server"

    def test_translates_study_data_pvc_to_runtime_source(self, tmp_path):
        config_path = tmp_path / "deploy.yaml"
        config_path.write_text(
            """
clouds:
  gcp:
    prepare:
      runtime: k8s
      parent:
        docker_image: repo/image:tag
    pvc_config:
      nvflws: {sc: standard-rwo, access: ReadWriteOnce, size: 1Gi}
    study_data:
      default:
        data: {pvc: nvfldata, mode: ro}
participants:
  - {name: gcp-server, cloud: gcp, namespace: nvflare-server, role: server}
"""
        )

        config = DEPLOY_MODULE.load_config(config_path)

        assert config.participants[0].study_data == {"default": {"data": {"source": "nvfldata", "mode": "ro"}}}
        state = DEPLOY_MODULE.deployment_state(config)
        assert state["participants"]["gcp-server"]["pvc_names"] == ["nvflws"]
        assert state["participants"]["gcp-server"]["cleanup_pvc_names"] == ["nvflws", "nvfldata"]

    @pytest.mark.parametrize("entry", [{"mode": "ro"}, {"pvc": "nvfldata"}])
    def test_rejects_study_data_without_pvc_or_mode(self, tmp_path, entry):
        config_path = tmp_path / "deploy.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "clouds": {
                        "gcp": {
                            "prepare": {
                                "runtime": "k8s",
                                "parent": {"docker_image": "repo/image:tag"},
                            },
                            "study_data": {"default": {"data": entry}},
                        }
                    },
                    "participants": [
                        {"name": "gcp-server", "cloud": "gcp", "namespace": "nvflare-server", "role": "server"}
                    ],
                }
            )
        )

        with pytest.raises(ValueError, match="must define pvc and mode"):
            DEPLOY_MODULE.load_config(config_path)

    def test_loads_enabled_monitoring(self, tmp_path):
        config_path = tmp_path / "deploy.yaml"
        config_path.write_text(
            """
name: all-clouds
monitoring:
  enabled: true
clouds:
  gcp:
    prepare:
      runtime: k8s
      parent:
        docker_image: repo/image:tag
participants:
  - {name: gcp-server, cloud: gcp, namespace: nvflare-server, role: server}
"""
        )

        config = DEPLOY_MODULE.load_config(config_path)

        assert config.monitoring.enabled is True
        assert config.monitoring.namespace == "nvflare-all-clouds-monitoring"
        assert config.monitoring.statsd_host == "statsd-exporter.nvflare-all-clouds-monitoring.svc.cluster.local"
        assert config.monitoring.statsd_port == 9125
        assert config.monitoring.env == "all-clouds"

    def test_render_project_keeps_project_participants_and_templates_server_ip(self, tmp_path):
        project_path = tmp_path / "project.yml"
        project_path.write_text(
            """
api_version: 3
name: explicit-project
participants:
  - name: site-server
    type: server
    org: nvidia
    default_host: __SERVER_IP__
    fed_learn_port: 8002
    admin_port: 8003
  - name: site-client
    type: client
    org: nvidia
  - name: admin@nvidia.com
    type: admin
    org: nvidia
    role: project_admin
builders: []
"""
        )
        config_path = tmp_path / "deploy.yaml"
        config_path.write_text(
            """
name: explicit-project
project_file: project.yml
clouds:
  gcp:
    prepare:
      runtime: k8s
      parent:
        docker_image: repo/image:tag
participants:
  - {name: site-server, cloud: gcp, namespace: nvflare-server, role: server}
  - {name: site-client, cloud: gcp, namespace: nvflare-client, role: client}
"""
        )

        config = DEPLOY_MODULE.load_config(config_path)
        rendered = yaml.safe_load(DEPLOY_MODULE.render_project("10.1.2.3", config))

        assert config.project_file == project_path
        assert [p["name"] for p in rendered["participants"]] == ["site-server", "site-client", "admin@nvidia.com"]
        assert rendered["participants"][0]["default_host"] == "10.1.2.3"

    def test_render_project_rejects_participant_mismatch(self, tmp_path):
        project_path = tmp_path / "project.yml"
        project_path.write_text(
            """
api_version: 3
name: explicit-project
participants:
  - {name: site-server, type: server, org: nvidia, default_host: __SERVER_IP__}
  - {name: other-client, type: client, org: nvidia}
  - {name: admin@nvidia.com, type: admin, org: nvidia, role: project_admin}
builders: []
"""
        )
        config_path = tmp_path / "deploy.yaml"
        config_path.write_text(
            """
name: explicit-project
project_file: project.yml
clouds:
  gcp:
    prepare:
      runtime: k8s
      parent:
        docker_image: repo/image:tag
participants:
  - {name: site-server, cloud: gcp, namespace: nvflare-server, role: server}
  - {name: site-client, cloud: gcp, namespace: nvflare-client, role: client}
"""
        )
        config = DEPLOY_MODULE.load_config(config_path)

        with pytest.raises(ValueError, match="deploy.py no longer generates participants"):
            DEPLOY_MODULE.render_project("10.1.2.3", config)


class TestMonitoringStack:
    def test_dry_run_applies_monitoring_stack(self, monkeypatch, capsys):
        monkeypatch.setattr(DEPLOY_MODULE, "DRY_RUN", True)
        config = DEPLOY_MODULE.DeployConfig(
            name="all-clouds",
            participants=[
                DEPLOY_MODULE.Participant(
                    name="gcp-server",
                    namespace="nvflare-server",
                    kubeconfig="/tmp/gcp.yaml",
                    role="server",
                    cloud="gcp",
                    prepare={},
                )
            ],
            server_cloud="gcp",
            gcp_project=None,
            gcp_region=None,
            aws_region=None,
            aws_eks_cluster_name=None,
            azure_resource_group=None,
            azure_location=None,
            monitoring=DEPLOY_MODULE.MonitoringConfig(
                enabled=True,
                namespace="nvflare-all-clouds-monitoring",
                statsd_host="statsd-exporter.nvflare-all-clouds-monitoring.svc.cluster.local",
            ),
        )

        DEPLOY_MODULE.deploy_monitoring_stack(config)

        output = capsys.readouterr().out
        assert "apply -f -" in output
        assert "name: nvflare-all-clouds-monitoring" in output

    def test_teardown_monitoring_uses_provider_not_cloud_alias(self, monkeypatch):
        config = DEPLOY_MODULE.DeployConfig(
            name="alias-cloud",
            participants=[
                DEPLOY_MODULE.Participant(
                    name="site-server",
                    namespace="nvflare",
                    kubeconfig="/tmp/kubeconfig",
                    role="server",
                    cloud="local-cluster",
                    provider="kubernetes",
                    prepare={},
                )
            ],
            server_cloud="local-cluster",
            server_provider="kubernetes",
            gcp_project=None,
            gcp_region=None,
            aws_region=None,
            aws_eks_cluster_name=None,
            azure_resource_group=None,
            azure_location=None,
            monitoring=DEPLOY_MODULE.MonitoringConfig(enabled=True, namespace="nvflare-monitoring"),
        )
        auth_checks = []
        run_calls = []

        monkeypatch.setattr(DEPLOY_MODULE, "check_auth_for", lambda provider: auth_checks.append(provider))
        monkeypatch.setattr(
            DEPLOY_MODULE, "run", lambda cmd, **kwargs: run_calls.append((cmd, kwargs)) or DEPLOY_MODULE.FakeProc(0)
        )

        assert DEPLOY_MODULE.teardown_monitoring_stack(config)
        assert auth_checks == ["kubernetes"]
        assert run_calls[0][0][:5] == ["kubectl", "--kubeconfig", "/tmp/kubeconfig", "delete", "ns"]


class TestDryRunGoldenOutput:
    @pytest.mark.parametrize(
        ("config_name", "command"),
        [
            ("gcp-server", "up"),
            ("aws-server", "up"),
            ("azure-server", "up"),
            ("all-clouds", "up"),
            ("gcp-server", "down"),
            ("aws-server", "down"),
            ("azure-server", "down"),
            ("all-clouds", "down"),
        ],
    )
    def test_dry_run_output_matches_gold(self, config_name, command, monkeypatch, capsys):
        monkeypatch.setattr(DEPLOY_MODULE, "DEFAULT_KUBECONFIG_DIR", DRY_RUN_CONFIG_DIR / "kubeconfigs")
        monkeypatch.setattr(DEPLOY_MODULE, "DRY_RUN", True)

        getattr(DEPLOY_MODULE, f"cmd_{command}")(
            SimpleNamespace(config=str(DRY_RUN_CONFIG_DIR / f"{config_name}.yaml"))
        )

        assert capsys.readouterr().out == (DRY_RUN_GOLD_DIR / f"{config_name}.{command}.txt").read_text()


class TestStatus:
    def test_kubernetes_status_shows_service_address_not_ip_name(self, monkeypatch, tmp_path, capsys):
        config_path = tmp_path / "local-cluster.yaml"
        config_path.write_text(
            """
name: local-cluster
server_cloud: local-cluster
clouds:
  local-cluster:
    provider: kubernetes
    kubeconfig: /tmp/kubeconfig
    prepare:
      runtime: k8s
      parent:
        docker_image: repo/image:tag
    server:
      service_type: ClusterIP
participants:
  - {name: local-server, cloud: local-cluster, namespace: nvflare, role: server}
"""
        )

        monkeypatch.setattr(DEPLOY_MODULE, "run_quiet", lambda cmd: DEPLOY_MODULE.FakeProc(0, stdout="pod-row\n"))

        DEPLOY_MODULE.cmd_status(SimpleNamespace(config=str(config_path)))

        out = capsys.readouterr().out
        assert "Server address:   nvflare-server.nvflare.svc.cluster.local" in out
        assert "IP name:" not in out


class TestKubernetesAdminEndpoint:
    def test_kubernetes_clusterip_defaults_admin_endpoint_to_local_port_forward(self):
        provider = DEPLOY_MODULE.get_provider("kubernetes")
        config = SimpleNamespace(
            server_cloud="local-cluster",
            cloud_configs={"local-cluster": {"server": {"service_type": "ClusterIP"}}},
            participants=[
                DEPLOY_MODULE.Participant(
                    name="local-server",
                    namespace="nvflare",
                    kubeconfig="/tmp/kubeconfig",
                    role="server",
                    cloud="local-cluster",
                    prepare={"runtime": "k8s", "parent": {"docker_image": "repo/image:tag"}},
                )
            ],
        )

        assert provider.admin_endpoint(config=config, server_ip="nvflare-server.nvflare.svc.cluster.local") == (
            "localhost",
            18003,
        )

    def test_configure_admin_endpoint_rewrites_admin_startup(self, tmp_path):
        startup_dir = tmp_path / "prod_00" / "admin@nvidia.com" / "startup"
        startup_dir.mkdir(parents=True)
        fed_admin = startup_dir / "fed_admin.json"
        fed_admin.write_text(json.dumps({"admin": {"host": "nvflare-server.nvflare.svc.cluster.local", "port": 8003}}))

        DEPLOY_MODULE.configure_admin_endpoint(tmp_path / "prod_00", host="localhost", port=18003)

        assert json.loads(fed_admin.read_text())["admin"] == {"host": "localhost", "port": 18003}


class TestHelmDeployFlow:
    @pytest.mark.parametrize("returncode,expected", [(0, True), (1, False)])
    def test_helm_release_exists_checks_status(self, monkeypatch, returncode, expected):
        monkeypatch.setattr(DEPLOY_MODULE, "run_quiet", lambda cmd: DEPLOY_MODULE.FakeProc(returncode))

        assert DEPLOY_MODULE.helm_release_exists("/tmp/kubeconfig", "site-1", "ns-1") is expected

    def test_deploy_recreates_release_stages_kit_and_installs(self, monkeypatch, tmp_path):
        kit_dir = tmp_path / "site-1" / "prepared" / "k8s"
        chart_dir = kit_dir / "helm_chart"
        (kit_dir / "startup").mkdir(parents=True)
        (kit_dir / "local").mkdir()
        chart_dir.mkdir()
        (kit_dir / "local" / "study_data.yaml").write_text("{}\n")

        participant = DEPLOY_MODULE.Participant(
            name="site-1",
            namespace="ns-1",
            kubeconfig="/tmp/kubeconfig",
            role="client",
            cloud="gcp",
            prepare={
                "runtime": "k8s",
                "namespace": "ns-1",
                "parent": {
                    "docker_image": "repo/image:tag",
                    "workspace_pvc": "nvflws",
                    "workspace_mount_path": "/var/tmp/nvflare/workspace",
                    "pod_security_context": {"runAsUser": 1000},
                },
                "job_launcher": {"default_python_path": "/usr/local/bin/python3"},
            },
            pvc_config={"nvflws": {"sc": "sc", "access": "ReadWriteMany", "size": "10Gi"}},
            pod_annotations={"karpenter.sh/do-not-disrupt": r"value,with=helm\chars"},
        )

        kubectl_calls = []
        helm_calls = []

        monkeypatch.setattr(DEPLOY_MODULE, "check_auth_for", lambda cloud: None)
        monkeypatch.setattr(DEPLOY_MODULE, "namespace_exists", lambda kubeconfig, ns: True)
        monkeypatch.setattr(DEPLOY_MODULE, "helm_release_exists", lambda kubeconfig, name, ns: True)
        monkeypatch.setattr(DEPLOY_MODULE, "kubectl", lambda *args: kubectl_calls.append(args))
        monkeypatch.setattr(DEPLOY_MODULE, "helm", lambda kubeconfig, *args: helm_calls.append((kubeconfig, args)))
        monkeypatch.setattr(DEPLOY_MODULE, "run", lambda *args, **kwargs: DEPLOY_MODULE.FakeProc(0))

        DEPLOY_MODULE.deploy_participant(participant, kit_dir)

        assert any("cp" in call for call in kubectl_calls)
        assert any("exec" in call and "rm" in call for call in kubectl_calls)
        assert len(helm_calls) == 2
        assert helm_calls[0][1][:4] == ("uninstall", "site-1", "-n", "ns-1")
        kubeconfig, helm_args = helm_calls[1]
        assert kubeconfig == "/tmp/kubeconfig"
        assert helm_args[:4] == ("upgrade", "--install", "site-1", str(chart_dir))
        assert "--set-string" in helm_args
        assert r"podAnnotations.karpenter\.sh\/do-not-disrupt=value\,with\=helm\\chars" in helm_args
        assert not any(str(arg).startswith("image.repository=") for arg in helm_args)
        assert not any(str(arg).startswith("image.tag=") for arg in helm_args)
        assert not any(str(arg).startswith("securityContext.") for arg in helm_args)


class TestTeardownFlow:
    def test_teardown_deletes_pods_using_configured_pvcs_before_pvcs(self, monkeypatch):
        calls = []
        pod_list = {
            "items": [
                {
                    "metadata": {"name": "job-pod"},
                    "spec": {"volumes": [{"persistentVolumeClaim": {"claimName": "client-study-data"}}]},
                },
                {
                    "metadata": {"name": "other-pod"},
                    "spec": {"volumes": [{"persistentVolumeClaim": {"claimName": "other-data"}}]},
                },
            ]
        }

        def _run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            if cmd[:7] == ["kubectl", "--kubeconfig", "/tmp/kubeconfig", "-n", "nvflare", "get", "pods"]:
                return DEPLOY_MODULE.FakeProc(0, stdout=json.dumps(pod_list))
            return DEPLOY_MODULE.FakeProc(0)

        monkeypatch.setattr(DEPLOY_MODULE, "check_auth_for", lambda provider: None)
        monkeypatch.setattr(DEPLOY_MODULE, "run", _run)

        assert DEPLOY_MODULE.teardown_participant(
            "local-client-1",
            {
                "kubeconfig": "/tmp/kubeconfig",
                "namespace": "nvflare",
                "provider": "kubernetes",
                "role": "client",
                "delete_namespace": False,
                "pvc_names": ["client-ws"],
                "cleanup_pvc_names": ["client-ws", "client-study-data"],
            },
        )

        commands = [cmd for cmd, _kwargs in calls]
        delete_pod_index = next(i for i, cmd in enumerate(commands) if cmd[5:8] == ["delete", "pod", "job-pod"])
        delete_pvc_index = next(i for i, cmd in enumerate(commands) if cmd[5:8] == ["delete", "pvc", "client-ws"])
        assert delete_pod_index < delete_pvc_index
        assert not any(cmd[5:8] == ["delete", "pod", "other-pod"] for cmd in commands)
        assert not any(cmd[5:8] == ["delete", "pvc", "client-study-data"] for cmd in commands)

    def test_cmd_down_attempts_server_after_client_failure(self, monkeypatch):
        config = DEPLOY_MODULE.DeployConfig(
            name="test-cluster",
            participants=[
                DEPLOY_MODULE.Participant(
                    name="gcp-server",
                    namespace="nvflare-server",
                    kubeconfig="/tmp/gcp",
                    role="server",
                    cloud="gcp",
                    prepare={},
                ),
                DEPLOY_MODULE.Participant(
                    name="gcp-client-1",
                    namespace="nvflare-client-1",
                    kubeconfig="/tmp/gcp",
                    role="client",
                    cloud="gcp",
                    prepare={},
                ),
            ],
            server_cloud="gcp",
            gcp_project="test-project",
            gcp_region="us-central1",
            aws_region=None,
            aws_eks_cluster_name=None,
            azure_resource_group=None,
            azure_location=None,
        )
        calls = []

        def _teardown_participants(items, parallel=True):
            calls.append([name for name, _info in items])
            return not any(info["role"] != "server" for _name, info in items)

        monkeypatch.setattr(DEPLOY_MODULE, "load_config", lambda _path: config)
        monkeypatch.setattr(DEPLOY_MODULE, "teardown_participants", _teardown_participants)

        with pytest.raises(SystemExit) as e:
            DEPLOY_MODULE.cmd_down(SimpleNamespace(config="/tmp/config.yaml"))

        assert e.value.code == 1
        assert calls == [["gcp-client-1"], ["gcp-server"]]


class TestAwsRegionResolution:
    def test_release_ip_aws_uses_configured_region_when_state_missing(self):
        provider = DEPLOY_MODULE.get_provider("aws")
        calls = []

        def _run(cmd, **kwargs):
            calls.append(cmd)
            if cmd == ["aws", "configure", "get", "region"]:
                return DEPLOY_MODULE.FakeProc(0, stdout="us-west-2\n")
            if cmd[:3] == ["aws", "ec2", "describe-addresses"]:
                return DEPLOY_MODULE.FakeProc(
                    0,
                    stdout='[{"PublicIp": "1.2.3.4", "AllocationId": "eipalloc-1"}]\n',
                )
            return DEPLOY_MODULE.FakeProc(0)

        provider.release_ip(run=_run, ip_name="nvflare-test", state={"aws_region": None})

        assert ["aws", "configure", "get", "region"] in calls
        release_cmd = next(cmd for cmd in calls if cmd[:3] == ["aws", "ec2", "release-address"])
        assert release_cmd[-2:] == ["--region", "us-west-2"]

    def test_release_ip_aws_requires_region(self):
        provider = DEPLOY_MODULE.get_provider("aws")

        with pytest.raises(RuntimeError, match="AWS region is required"):
            provider.release_ip(
                run=lambda *args, **kwargs: DEPLOY_MODULE.FakeProc(1),
                ip_name="nvflare-test",
                state={"aws_region": None},
            )

    def test_release_ip_aws_warns_on_lookup_failure(self, capsys):
        provider = DEPLOY_MODULE.get_provider("aws")
        calls = []

        def _run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            if cmd[:3] == ["aws", "ec2", "describe-addresses"]:
                return DEPLOY_MODULE.FakeProc(255, stderr="expired token")
            return DEPLOY_MODULE.FakeProc(0)

        provider.release_ip(run=_run, ip_name="nvflare-test", state={"aws_region": "us-west-2"})

        describe_call = next(call for call in calls if call[0][:3] == ["aws", "ec2", "describe-addresses"])
        assert describe_call[1]["check"] is False
        assert not any(cmd[:3] == ["aws", "ec2", "release-address"] for cmd, _kwargs in calls)
        assert "Warning: failed to describe Elastic IPs tagged Name=nvflare-test" in capsys.readouterr().out


class TestGcpProjectResolution:
    def test_release_ip_gcp_uses_active_project_when_state_missing(self):
        provider = DEPLOY_MODULE.get_provider("gcp")
        calls = []

        def _run(cmd, **kwargs):
            calls.append(cmd)
            if cmd == ["gcloud", "config", "get-value", "project"]:
                return DEPLOY_MODULE.FakeProc(0, stdout="test-project\n")
            return DEPLOY_MODULE.FakeProc(0)

        provider.release_ip(
            run=_run,
            ip_name="nvflare-test",
            state={"gcp_project": None, "gcp_region": None},
        )

        delete_cmd = calls[-1]
        assert "--project=test-project" in delete_cmd
        assert "--region=us-central1" in delete_cmd


class TestAzureIpValidation:
    def test_reserve_ip_azure_requires_resource_group_and_location(self, monkeypatch):
        called = False
        provider = DEPLOY_MODULE.get_provider("azure")

        def _run(*args, **kwargs):
            nonlocal called
            called = True
            return DEPLOY_MODULE.FakeProc(0)

        with pytest.raises(ValueError, match="resource_group"):
            provider.reserve_ip(run=_run, ip_tag="pip-name", azure_resource_group=None, azure_location="westus2")
        with pytest.raises(ValueError, match="location"):
            provider.reserve_ip(run=_run, ip_tag="pip-name", azure_resource_group="my-rg", azure_location="")

        assert called is False

    def test_release_ip_azure_requires_resource_group(self, monkeypatch):
        called = False
        provider = DEPLOY_MODULE.get_provider("azure")

        def _run(*args, **kwargs):
            nonlocal called
            called = True
            return DEPLOY_MODULE.FakeProc(0)

        with pytest.raises(ValueError, match="resource_group"):
            provider.release_ip(run=_run, ip_name="pip-name", state={"azure_resource_group": None})

        assert called is False

    def test_release_ip_azure_warns_on_delete_failure(self, monkeypatch, capsys):
        provider = DEPLOY_MODULE.get_provider("azure")

        provider.release_ip(
            run=lambda *args, **kwargs: DEPLOY_MODULE.FakeProc(1, stderr="public IP is still in use"),
            ip_name="pip-name",
            state={"azure_resource_group": "my-rg"},
        )

        out = capsys.readouterr().out
        assert "Warning: failed to delete Azure Public IP pip-name" in out
        assert "still in use" in out
