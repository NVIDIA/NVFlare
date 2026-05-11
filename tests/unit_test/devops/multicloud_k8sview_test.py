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
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace


class FakeTable:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.columns = []
        self.rows = []

    def add_column(self, *args, **kwargs):
        self.columns.append((args, kwargs))

    def add_row(self, *args):
        self.rows.append(args)


class FakeText(str):
    def __new__(cls, value="", *args, **kwargs):
        return str.__new__(cls, value)

    @classmethod
    def from_markup(cls, value):
        return cls(value)


def _install_import_stubs():
    rich_console = ModuleType("rich.console")
    rich_live = ModuleType("rich.live")
    rich_table = ModuleType("rich.table")
    rich_text = ModuleType("rich.text")

    rich_console.Console = object
    rich_console.Group = lambda *items: items
    rich_live.Live = object
    rich_table.Table = FakeTable
    rich_text.Text = FakeText

    k8s = ModuleType("kubernetes")
    k8s_client = ModuleType("kubernetes.client")
    k8s_config = ModuleType("kubernetes.config")
    k8s_exceptions = ModuleType("kubernetes.client.exceptions")
    k8s_config_exception = ModuleType("kubernetes.config.config_exception")

    class FakeApiException(Exception):
        def __init__(self, status=None, reason=None):
            self.status = status
            self.reason = reason

    class FakeConfigException(Exception):
        pass

    k8s_exceptions.ApiException = FakeApiException
    k8s_config_exception.ConfigException = FakeConfigException
    k8s_client.CoreV1Api = object

    for name, module in [
        ("rich.console", rich_console),
        ("rich.live", rich_live),
        ("rich.table", rich_table),
        ("rich.text", rich_text),
        ("kubernetes", k8s),
        ("kubernetes.client", k8s_client),
        ("kubernetes.config", k8s_config),
        ("kubernetes.client.exceptions", k8s_exceptions),
        ("kubernetes.config.config_exception", k8s_config_exception),
    ]:
        sys.modules.setdefault(name, module)


_install_import_stubs()

K8SVIEW_PY = Path(__file__).resolve().parents[3] / "devops" / "multicloud" / "k8sview.py"
SPEC = importlib.util.spec_from_file_location("multicloud_k8sview", K8SVIEW_PY)
K8SVIEW = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = K8SVIEW
SPEC.loader.exec_module(K8SVIEW)
K8SVIEW.Table = FakeTable
K8SVIEW.Text = FakeText


def _meta(name, namespace="nvflare-server", labels=None, created=None):
    return SimpleNamespace(
        name=name,
        namespace=namespace,
        labels=labels or {},
        creation_timestamp=created or datetime.now(timezone.utc),
        deletion_timestamp=None,
    )


def _pod(
    name,
    *,
    namespace="nvflare-server",
    phase="Running",
    restart_policy="Always",
    ip="10.1.1.1",
    labels=None,
    finished_at=None,
):
    terminated = SimpleNamespace(finished_at=finished_at) if finished_at else None
    state = SimpleNamespace(waiting=None, terminated=terminated)
    status = SimpleNamespace(state=state, ready=phase == "Running", restart_count=0)
    return SimpleNamespace(
        metadata=_meta(name, namespace=namespace, labels=labels),
        status=SimpleNamespace(
            phase=phase,
            pod_ip=ip,
            container_statuses=[status] if finished_at else [],
            init_container_statuses=[],
        ),
        spec=SimpleNamespace(restart_policy=restart_policy, node_name="node-1"),
    )


def _node(name, ready=True):
    return SimpleNamespace(
        metadata=_meta(name, namespace=None),
        status=SimpleNamespace(conditions=[SimpleNamespace(type="Ready", status="True" if ready else "False")]),
    )


def _pvc(name, *, namespace="nvflare-server", phase="Bound", capacity="1Gi"):
    resources = SimpleNamespace(requests={"storage": capacity})
    return SimpleNamespace(
        metadata=_meta(name, namespace=namespace),
        status=SimpleNamespace(phase=phase, capacity={"storage": capacity}),
        spec=SimpleNamespace(resources=resources, access_modes=["ReadWriteOnce"], storage_class_name="standard"),
    )


def _svc(
    name,
    *,
    namespace="nvflare-server",
    svc_type="ClusterIP",
    cluster_ip="10.0.0.1",
    external_ip=None,
    labels=None,
):
    ingress = [SimpleNamespace(ip=external_ip, hostname=None)] if external_ip else []
    return SimpleNamespace(
        metadata=_meta(name, namespace=namespace, labels=labels),
        spec=SimpleNamespace(
            type=svc_type,
            cluster_ip=cluster_ip,
            load_balancer_ip=None,
            ports=[],
        ),
        status=SimpleNamespace(load_balancer=SimpleNamespace(ingress=ingress)),
    )


def test_namespace_system_role_distinguishes_default_and_numbered_systems():
    assert K8SVIEW.namespace_system_role("nvflare-server") == ("nvflare", "server")
    assert K8SVIEW.namespace_system_role("nvflare-client-2") == ("nvflare", "client")
    assert K8SVIEW.namespace_system_role("nvflare-2908-server") == ("2908", "server")
    assert K8SVIEW.namespace_system_role("nvflare-2908-client-1") == ("2908", "client")


def test_load_participants_defaults_kubeconfig_to_repo_tmp_by_cloud(tmp_path):
    config = tmp_path / "all-clouds.yaml"
    config.write_text(
        """
clouds:
  gcp: {}
participants:
  - { name: gcp-server, cloud: gcp, namespace: nvflare-server, role: server }
"""
    )

    participants = K8SVIEW.load_participants(config)

    assert participants == [
        K8SVIEW.Participant(
            name="gcp-server",
            cloud="gcp",
            namespace="nvflare-server",
            kubeconfig=str(K8SVIEW.DEFAULT_KUBECONFIG_DIR / "gcp.yaml"),
        )
    ]


def test_build_site_rows_summarizes_job_pods_and_ips_without_listing_jobs():
    cluster = K8SVIEW.Cluster(name="gcp", cloud="gcp", kubeconfig="/tmp/gcp.yaml")
    snapshot = K8SVIEW.ClusterSnapshot(
        cluster=cluster,
        namespaces=[
            K8SVIEW.NamespaceResources(
                cluster=cluster,
                namespace="nvflare-server",
                pods=[
                    _pod("gcp-server-abc", labels={"app.kubernetes.io/name": "gcp-server"}, ip="10.1.1.10"),
                    _pod("job-1", phase="Running", restart_policy="Never", ip="10.1.1.20"),
                    _pod("job-2", phase="Succeeded", restart_policy="Never", ip="10.1.1.21"),
                ],
                pvcs=[_pvc("nvflws"), _pvc("nvfldata")],
                services=[
                    _svc(
                        "nvflare-server",
                        svc_type="LoadBalancer",
                        cluster_ip="10.0.0.10",
                        external_ip="34.1.2.3",
                        labels={"app.kubernetes.io/name": "gcp-server"},
                    )
                ],
            ),
            K8SVIEW.NamespaceResources(
                cluster=cluster,
                namespace="nvflare-2908-client-1",
                pods=[
                    _pod(
                        "gcp-flare-2908-client-1-abc",
                        namespace="nvflare-2908-client-1",
                        labels={"app.kubernetes.io/name": "gcp-flare-2908-client-1"},
                        ip="10.1.1.30",
                    )
                ],
                pvcs=[],
                services=[],
            ),
        ],
    )

    rows = K8SVIEW.build_site_rows([snapshot])
    server = next(row for row in rows if row.namespace == "nvflare-server")
    client = next(row for row in rows if row.namespace == "nvflare-2908-client-1")

    assert server.system == "nvflare"
    assert server.role == "server"
    assert server.site == "gcp-server"
    assert server.pod_ips == ["10.1.1.10"]
    assert server.service_ips == ["10.0.0.10"]
    assert server.external_ips == ["34.1.2.3"]
    assert server.job_pods_running == 1
    assert server.job_pods_total == 2
    assert server.pvc_bound == 2
    assert server.pvc_total == 2

    assert client.system == "2908"
    assert client.role == "client"
    assert client.job_pods_total == 0


def test_cluster_infra_counts_include_nodes_and_all_pods():
    cluster = K8SVIEW.Cluster(name="gcp", cloud="gcp", kubeconfig="/tmp/gcp.yaml")
    snapshot = K8SVIEW.ClusterSnapshot(
        cluster=cluster,
        namespaces=[],
        nodes=[_node("node-1", ready=True), _node("node-2", ready=False)],
        all_pods=[
            _pod("running-1", phase="Running"),
            _pod("running-2", phase="Running"),
            _pod("done", phase="Succeeded"),
        ],
    )

    assert K8SVIEW.cluster_infra_counts(snapshot) == {
        "nodes_ready": 1,
        "nodes_total": 2,
        "pods_running": 2,
        "pods_total": 3,
    }


def test_build_clusters_table_shows_nvflare_pod_counts_without_trend_column():
    cluster = K8SVIEW.Cluster(name="gcp", cloud="gcp", kubeconfig="/tmp/gcp.yaml")
    snapshot = K8SVIEW.ClusterSnapshot(
        cluster=cluster,
        namespaces=[
            K8SVIEW.NamespaceResources(
                cluster=cluster,
                namespace="nvflare-server",
                pods=[_pod("running"), _pod("done", phase="Succeeded")],
                pvcs=[],
                services=[],
            )
        ],
        nodes=[_node("node-1", ready=True)],
        all_pods=[_pod("running"), _pod("done", phase="Succeeded")],
    )

    table = K8SVIEW.build_clusters_table([snapshot])

    assert [column[0][0] for column in table.columns] == [
        "CLOUD",
        "STATUS",
        "NODES",
        "ALL PODS",
        "NVFLARE NS",
        "NVFLARE PODS",
        "ERROR",
    ]
    assert table.rows[0][5] == "1/2"


def test_prune_pods_deletes_only_old_terminal_pods():
    now = datetime.now(timezone.utc)
    old_finished = now - timedelta(seconds=K8SVIEW.PRUNE_TERMINAL_PODS_AFTER + 1)
    fresh_finished = now - timedelta(seconds=1)
    old = _pod("old", phase="Succeeded", restart_policy="Never", finished_at=old_finished)
    fresh = _pod("fresh", phase="Failed", restart_policy="Never", finished_at=fresh_finished)
    running = _pod("running", phase="Running", restart_policy="Never")
    api = SimpleNamespace(deleted=[])

    def _delete_namespaced_pod(**kwargs):
        api.deleted.append(kwargs["name"])

    api.delete_namespaced_pod = _delete_namespaced_pod

    remaining, stats = K8SVIEW.prune_pods(
        api,
        lambda: None,
        cluster_name="gcp",
        namespace="nvflare-server",
        pods=[old, fresh, running],
    )

    assert api.deleted == ["old"]
    assert stats.deleted == {("gcp", "nvflare-server", "old")}
    assert [pod.metadata.name for pod in remaining] == ["fresh", "running"]
