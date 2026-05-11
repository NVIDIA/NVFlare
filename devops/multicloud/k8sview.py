#!/usr/bin/env python3
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

"""Live multicloud pod dashboard.

By default, discovers every NVFlare namespace reachable through kubeconfigs in
.tmp/kubeconfigs and renders a top-like all-cloud summary. Pass --config for the
older focused view that reads the same YAML config as deploy.py.

Usage:
    k8sview.py [--interval 2]
    k8sview.py --config all-clouds.yaml [--interval 2]
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text
except ImportError:
    sys.exit("k8sview requires 'rich'. Install with: uv pip install rich")

try:
    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config
    from kubernetes.client.exceptions import ApiException
    from kubernetes.config.config_exception import ConfigException
except ImportError:
    sys.exit("k8sview requires 'kubernetes'. Install with: uv pip install kubernetes")


TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parents[1]
DEFAULT_KUBECONFIG_DIR = REPO_ROOT / ".tmp" / "kubeconfigs"
NVFLARE_NAMESPACE_PREFIX = "nvflare"
PRUNE_TERMINAL_PODS_AFTER = 300.0
PRUNE_TERMINAL_POD_PHASES = frozenset({"Succeeded", "Failed"})

STATUS_COLORS = {
    "Running": "green",
    "Succeeded": "dim",
    "Completed": "dim",
    "Pending": "yellow",
    "ContainerCreating": "yellow",
    "PodInitializing": "yellow",
    "NotReady": "yellow",
    "Terminating": "dim yellow",
    "Failed": "red",
    "Error": "red",
    "CrashLoopBackOff": "red",
    "ImagePullBackOff": "red",
    "ErrImagePull": "red",
    "CreateContainerConfigError": "red",
    "InvalidImageName": "red",
    "ERROR": "red",
    "Bound": "green",
    "Lost": "red",
}


@dataclass
class Participant:
    name: str
    cloud: str
    namespace: str
    kubeconfig: str


@dataclass(frozen=True)
class Cluster:
    name: str
    cloud: str
    kubeconfig: str


@dataclass
class NamespaceResources:
    cluster: Cluster
    namespace: str
    pods: list
    pvcs: list
    services: list
    error: str | None = None


@dataclass
class ClusterSnapshot:
    cluster: Cluster
    namespaces: list[NamespaceResources]
    nodes: list | None = None
    all_pods: list | None = None
    node_error: str | None = None
    error: str | None = None


@dataclass
class SiteRow:
    system: str
    cloud: str
    role: str
    site: str
    namespace: str
    site_pod: str
    site_pods_running: int
    site_pods_total: int
    pod_ips: list[str]
    service_ips: list[str]
    external_ips: list[str]
    pvc_summary: str
    pvc_bound: int
    pvc_total: int
    job_pods_running: int
    job_pods_total: int
    age: str


def load_participants(config_path: Path) -> list[Participant]:
    config_path = config_path.resolve()
    raw = yaml.safe_load(config_path.read_text())
    clouds = raw.get("clouds") or {}
    entries = raw.get("participants") or []
    if not clouds or not entries:
        raise ValueError(f"{config_path}: missing 'clouds' or 'participants' section")
    out: list[Participant] = []
    for e in entries:
        cloud_cfg = clouds.get(e["cloud"], {}) or {}
        merged = {**cloud_cfg, **e}
        kc = resolve_kubeconfig(e["cloud"])
        out.append(Participant(merged["name"], e["cloud"], merged["namespace"], str(kc)))
    return out


def resolve_kubeconfig(cloud: str) -> Path:
    return (DEFAULT_KUBECONFIG_DIR / f"{cloud}.yaml").resolve()


def load_clusters(kubeconfig_dir: Path) -> list[Cluster]:
    kubeconfig_dir = kubeconfig_dir.resolve()
    if not kubeconfig_dir.exists():
        raise ValueError(f"{kubeconfig_dir}: kubeconfig directory does not exist")
    out = []
    for path in sorted(kubeconfig_dir.glob("*.yaml")):
        cloud = path.stem
        out.append(Cluster(name=cloud, cloud=cloud, kubeconfig=str(path.resolve())))
    if not out:
        raise ValueError(f"{kubeconfig_dir}: no kubeconfig YAML files found")
    return out


# ---------------------------------------------------------------------------
# Kubernetes API cache + pod fetch
# ---------------------------------------------------------------------------
class ApiCache:
    def __init__(self):
        self._apis: dict[str, k8s_client.CoreV1Api] = {}

    def get(self, p: Participant) -> k8s_client.CoreV1Api:
        return self._get(f"participant:{p.name}:{p.kubeconfig}", p.kubeconfig)

    def get_cluster(self, c: Cluster) -> k8s_client.CoreV1Api:
        return self._get(f"cluster:{c.name}:{c.kubeconfig}", c.kubeconfig)

    def _get(self, key: str, kubeconfig: str) -> k8s_client.CoreV1Api:
        api = self._apis.get(key)
        if api is not None:
            return api
        api_client = k8s_config.new_client_from_config(config_file=kubeconfig)
        api = k8s_client.CoreV1Api(api_client)
        self._apis[key] = api
        return api

    def drop(self, p: Participant) -> None:
        self._apis.pop(f"participant:{p.name}:{p.kubeconfig}", None)

    def drop_cluster(self, c: Cluster) -> None:
        self._apis.pop(f"cluster:{c.name}:{c.kubeconfig}", None)


@dataclass
class PodFetchError:
    msg: str


@dataclass
class PruneStats:
    deleted: set[tuple[str, str, str]]
    errors: list[str]

    @classmethod
    def empty(cls) -> "PruneStats":
        return cls(deleted=set(), errors=[])

    def extend(self, other: "PruneStats") -> None:
        self.deleted.update(other.deleted)
        self.errors.extend(other.errors)


def _call(p: Participant, cache: ApiCache, method_name: str, **kwargs):
    try:
        api = cache.get(p)
        resp = getattr(api, method_name)(p.namespace, _request_timeout=8, **kwargs)
        return resp.items
    except ApiException as e:
        cache.drop(p)
        return PodFetchError(e.reason or f"HTTP {e.status}")
    except ConfigException as e:
        cache.drop(p)
        return PodFetchError(f"config: {e}")
    except Exception as e:
        cache.drop(p)
        return PodFetchError(f"{type(e).__name__}: {e}")


def fetch_pods(p: Participant, cache: ApiCache):
    return _call(p, cache, "list_namespaced_pod")


def fetch_pvcs(p: Participant, cache: ApiCache):
    return _call(p, cache, "list_namespaced_persistent_volume_claim")


def fetch_services(p: Participant, cache: ApiCache):
    return _call(p, cache, "list_namespaced_service")


def fetch_events(p: Participant, cache: ApiCache):
    return _call(p, cache, "list_namespaced_event", field_selector="type=Warning")


# ---------------------------------------------------------------------------
# Pod pruning
# ---------------------------------------------------------------------------
def _ensure_aware_utc(ts: datetime | None) -> datetime | None:
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _pod_finished_at(pod) -> datetime | None:
    finished = []
    for cs in (pod.status.init_container_statuses or []) + (pod.status.container_statuses or []):
        terminated = getattr(cs.state, "terminated", None)
        finished_at = _ensure_aware_utc(getattr(terminated, "finished_at", None))
        if finished_at is not None:
            finished.append(finished_at)
    if finished:
        return max(finished)
    return None


def pod_prune_age_start(pod) -> datetime | None:
    if pod.status.phase not in PRUNE_TERMINAL_POD_PHASES:
        return None
    return _pod_finished_at(pod) or _ensure_aware_utc(pod.metadata.creation_timestamp)


def should_prune_pod(pod, now: datetime) -> bool:
    if pod.metadata.deletion_timestamp:
        return False
    prune_age_start = pod_prune_age_start(pod)
    if prune_age_start is None:
        return False
    return (now - prune_age_start).total_seconds() >= PRUNE_TERMINAL_PODS_AFTER


def prune_pods(
    api: k8s_client.CoreV1Api,
    cache_drop,
    *,
    cluster_name: str,
    namespace: str,
    pods,
) -> tuple[list, PruneStats]:
    stats = PruneStats.empty()
    if not pods:
        return pods or [], stats
    now = datetime.now(timezone.utc)
    candidates = [pod for pod in pods if should_prune_pod(pod, now)]
    if not candidates:
        return pods, stats
    deleted_names = set()
    for pod in candidates:
        pod_name = pod.metadata.name
        try:
            api.delete_namespaced_pod(
                name=pod_name,
                namespace=namespace,
                grace_period_seconds=0,
                _request_timeout=8,
            )
            deleted_names.add(pod_name)
            stats.deleted.add((cluster_name, namespace, pod_name))
        except ApiException as e:
            if e.status == 404:
                deleted_names.add(pod_name)
                stats.deleted.add((cluster_name, namespace, pod_name))
                continue
            cache_drop()
            stats.errors.append(f"{cluster_name}/{namespace}/{pod_name}: {e.reason or f'HTTP {e.status}'}")
        except Exception as e:
            cache_drop()
            stats.errors.append(f"{cluster_name}/{namespace}/{pod_name}: {type(e).__name__}: {e}")
    return [pod for pod in pods if pod.metadata.name not in deleted_names], stats


def prune_pod_snapshots(
    snapshots: list[tuple[Participant, object]],
    cache: ApiCache,
) -> tuple[list[tuple[Participant, object]], PruneStats]:
    stats = PruneStats.empty()
    filtered = []
    for p, result in snapshots:
        if isinstance(result, list):
            try:
                api = cache.get(p)
            except Exception as e:
                stats.errors.append(f"{p.name}: {type(e).__name__}: {e}")
                filtered.append((p, result))
                continue
            result, prune_stats = prune_pods(
                api,
                lambda p=p: cache.drop(p),
                cluster_name=p.name,
                namespace=p.namespace,
                pods=result,
            )
            stats.extend(prune_stats)
        filtered.append((p, result))
    return filtered, stats


# ---------------------------------------------------------------------------
# Pod → row
# ---------------------------------------------------------------------------
def derive_status(pod) -> str:
    if pod.metadata.deletion_timestamp:
        return "Terminating"
    statuses = pod.status.container_statuses or []
    init_statuses = pod.status.init_container_statuses or []
    for cs in init_statuses + statuses:
        waiting = getattr(cs.state, "waiting", None)
        if waiting and waiting.reason:
            return waiting.reason
    for cs in statuses:
        terminated = getattr(cs.state, "terminated", None)
        if terminated and terminated.reason and terminated.reason != "Completed":
            return terminated.reason
    phase = pod.status.phase or "Unknown"
    if phase == "Running" and statuses and not all(cs.ready for cs in statuses):
        return "NotReady"
    return phase


def ready_fraction(pod) -> str:
    statuses = pod.status.container_statuses or []
    if not statuses:
        return "0/0"
    ready = sum(1 for cs in statuses if cs.ready)
    return f"{ready}/{len(statuses)}"


def restart_count(pod) -> int:
    statuses = pod.status.container_statuses or []
    return max((cs.restart_count for cs in statuses), default=0)


def age_str(obj) -> str:
    ts = obj.metadata.creation_timestamp
    if ts is None:
        return "-"
    delta = datetime.now(timezone.utc) - ts
    return age_str_from_seconds(delta.total_seconds())


def age_str_from_seconds(seconds: float) -> str:
    s = int(seconds)
    if s < 0:
        return "0s"
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        h, m = divmod(s, 3600)
        return f"{h}h{m // 60}m" if m // 60 else f"{h}h"
    d, rem = divmod(s, 86400)
    return f"{d}d{rem // 3600}h" if rem // 3600 else f"{d}d"


def status_text(status: str) -> Text:
    return Text(status, style=STATUS_COLORS.get(status, ""))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def build_pod_table(snapshots: list[tuple[Participant, object]]) -> Table:
    total_pods = sum(len(r) for _, r in snapshots if isinstance(r, list))
    table = Table(caption=f"{total_pods} pods", caption_style="dim", expand=True, title="Pods", title_style="bold")
    for col, style, width in [
        ("CLOUD", "cyan", None),
        ("PARTICIPANT", "cyan", None),
        ("NS", "cyan", None),
        ("POD", "white", None),
        ("READY", None, 6),
        ("STATUS", None, None),
        ("RESTARTS", None, 8),
        ("AGE", None, 6),
        ("NODE", "dim", None),
    ]:
        table.add_column(col, style=style, width=width, no_wrap=(col != "NODE"))

    for p, result in snapshots:
        if isinstance(result, PodFetchError):
            table.add_row(
                p.cloud,
                p.name,
                p.namespace,
                "—",
                "—",
                Text(f"ERROR: {result.msg}", style="red"),
                "—",
                "—",
                "—",
            )
            continue
        if not result:
            table.add_row(p.cloud, p.name, p.namespace, Text("(no pods)", style="dim"), "", "", "", "", "")
            continue
        for pod in sorted(result, key=lambda x: x.metadata.name):
            table.add_row(
                p.cloud,
                p.name,
                p.namespace,
                pod.metadata.name,
                ready_fraction(pod),
                status_text(derive_status(pod)),
                str(restart_count(pod)),
                age_str(pod),
                pod.spec.node_name or "—",
            )
    return table


def build_pvc_table(snapshots: list[tuple[Participant, object]]) -> Table:
    total = sum(len(r) for _, r in snapshots if isinstance(r, list))
    caption = f"{total} PVCs"
    table = Table(caption=caption, caption_style="dim", expand=True, title="PersistentVolumeClaims", title_style="bold")
    for col, style, width in [
        ("CLOUD", "cyan", None),
        ("PARTICIPANT", "cyan", None),
        ("NS", "cyan", None),
        ("NAME", "white", None),
        ("STATUS", None, None),
        ("CAPACITY", None, 10),
        ("ACCESS", None, 8),
        ("STORAGECLASS", "dim", None),
        ("AGE", None, 6),
    ]:
        table.add_column(col, style=style, width=width, no_wrap=True)

    for p, result in snapshots:
        if isinstance(result, PodFetchError):
            table.add_row(
                p.cloud, p.name, p.namespace, "—", Text(f"ERROR: {result.msg}", style="red"), "—", "—", "—", "—"
            )
            continue
        if not result:
            table.add_row(p.cloud, p.name, p.namespace, Text("(no pvcs)", style="dim"), "", "", "", "", "")
            continue
        for pvc in sorted(result, key=lambda x: x.metadata.name):
            capacity = ""
            if pvc.status.capacity and "storage" in pvc.status.capacity:
                capacity = pvc.status.capacity["storage"]
            elif pvc.spec.resources and pvc.spec.resources.requests:
                capacity = pvc.spec.resources.requests.get("storage", "")
            access = ",".join(
                m.replace("ReadWrite", "RW").replace("ReadOnly", "RO") for m in (pvc.spec.access_modes or [])
            )
            table.add_row(
                p.cloud,
                p.name,
                p.namespace,
                pvc.metadata.name,
                status_text(pvc.status.phase or "Unknown"),
                capacity,
                access,
                pvc.spec.storage_class_name or "—",
                age_str(pvc),
            )
    return table


def _lb_endpoints(svc_snapshots: list[tuple[Participant, object]]) -> list[tuple[str, str, str]]:
    """(participant, endpoint_str, state) for every type=LoadBalancer service."""
    out = []
    for p, result in svc_snapshots:
        if isinstance(result, PodFetchError) or not result:
            continue
        for svc in result:
            if svc.spec.type != "LoadBalancer":
                continue
            ingress = []
            if svc.status and svc.status.load_balancer:
                ingress = svc.status.load_balancer.ingress or []
            port = (svc.spec.ports or [None])[0]
            port_str = str(port.port) if port else "?"
            if ingress:
                ep = ingress[0].ip or ingress[0].hostname or "?"
                out.append((p.name, f"{ep}:{port_str}", "ready"))
            else:
                out.append((p.name, f":{port_str}", "pending"))
    return out


def build_summary(
    pod_snaps: list[tuple[Participant, object]],
    pvc_snaps: list[tuple[Participant, object]],
    svc_snaps: list[tuple[Participant, object]],
    config_path: Path,
    interval: float,
) -> Text:
    pod_counts: Counter = Counter()
    pod_errors = 0
    for _, r in pod_snaps:
        if isinstance(r, PodFetchError):
            pod_errors += 1
            continue
        for pod in r or []:
            pod_counts[derive_status(pod)] += 1

    pvc_counts: Counter = Counter()
    for _, r in pvc_snaps:
        if isinstance(r, list):
            for pvc in r:
                pvc_counts[pvc.status.phase or "Unknown"] += 1

    def _fmt(counts: Counter) -> str:
        if not counts:
            return "0"
        total = sum(counts.values())
        parts = [str(total)]
        for k, v in counts.most_common():
            color = STATUS_COLORS.get(k, "")
            seg = f"{v} {k}"
            parts.append(f"[{color}]{seg}[/{color}]" if color else seg)
        return f"{parts[0]} ({', '.join(parts[1:])})"

    ts = datetime.now().strftime("%H:%M:%S")
    lines = [
        f"[bold]k8sview[/] · {config_path.name} · {ts} · refresh={interval}s · Ctrl-C to quit",
        f"Pods: {_fmt(pod_counts)}" + (f"  [red]{pod_errors} fetch errors[/]" if pod_errors else ""),
        f"PVCs: {_fmt(pvc_counts)}",
    ]
    eps = _lb_endpoints(svc_snaps)
    if eps:
        for name, ep, state in eps:
            tag = "[green]ready[/]" if state == "ready" else "[yellow]pending[/]"
            lines.append(f"LB {name}: {ep} ({tag})")
    return Text.from_markup("\n".join(lines))


def build_events_table(event_snaps: list[tuple[Participant, object]], window_s: float, limit: int = 20) -> Table:
    now = datetime.now(timezone.utc)
    rows: list[tuple[float, Participant, object]] = []
    for p, result in event_snaps:
        if isinstance(result, PodFetchError) or not result:
            continue
        for ev in result:
            ts = ev.last_timestamp or ev.event_time or ev.metadata.creation_timestamp
            if ts is None:
                continue
            age = (now - ts).total_seconds()
            if age > window_s:
                continue
            rows.append((age, p, ev))
    rows.sort(key=lambda t: t[0])
    rows = rows[:limit]

    title = f"Warnings (last {int(window_s)}s, showing {len(rows)})"
    table = Table(title=title, title_style="bold yellow", expand=True, show_lines=False)
    for col, style, width in [
        ("AGE", None, 6),
        ("CLOUD", "cyan", None),
        ("PARTICIPANT", "cyan", None),
        ("NS", "cyan", None),
        ("OBJECT", "white", None),
        ("REASON", None, None),
        ("COUNT", None, 5),
        ("MESSAGE", "dim", None),
    ]:
        table.add_column(col, style=style, width=width, no_wrap=(col != "MESSAGE"))
    if not rows:
        table.add_row("—", "—", "—", "—", "—", Text("(no recent warnings)", style="dim green"), "—", "—")
        return table
    for age, p, ev in rows:
        obj = f"{ev.involved_object.kind}/{ev.involved_object.name}"
        age_s = int(age)
        age_str_ev = f"{age_s}s" if age_s < 60 else f"{age_s // 60}m"
        msg = (ev.message or "").replace("\n", " ")
        table.add_row(
            age_str_ev,
            p.cloud,
            p.name,
            p.namespace,
            obj,
            Text(ev.reason or "", style="yellow"),
            str(ev.count or 1),
            msg,
        )
    return table


def gather(participants: list[Participant], cache: ApiCache, fetch_fn) -> list[tuple[Participant, object]]:
    with ThreadPoolExecutor(max_workers=max(1, len(participants))) as ex:
        futures = {ex.submit(fetch_fn, p, cache): p for p in participants}
        by_name: dict[str, object] = {}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                by_name[p.name] = fut.result()
            except Exception as e:
                by_name[p.name] = PodFetchError(f"{type(e).__name__}: {e}")
    return [(p, by_name[p.name]) for p in participants]


# ---------------------------------------------------------------------------
# All-cloud discovery view
# ---------------------------------------------------------------------------
def is_nvflare_namespace(namespace: str) -> bool:
    return namespace == NVFLARE_NAMESPACE_PREFIX or namespace.startswith(f"{NVFLARE_NAMESPACE_PREFIX}-")


def namespace_system_role(namespace: str) -> tuple[str, str]:
    if namespace == "nvflare-server":
        return "nvflare", "server"
    if re.fullmatch(r"nvflare-client-\d+", namespace):
        return "nvflare", "client"
    m = re.fullmatch(r"nvflare-(?P<system>.+)-server", namespace)
    if m:
        return m.group("system"), "server"
    m = re.fullmatch(r"nvflare-(?P<system>.+)-client-\d+", namespace)
    if m:
        return m.group("system"), "client"
    if namespace.endswith("-server"):
        return namespace.removeprefix("nvflare-").removesuffix("-server"), "server"
    if "-client-" in namespace or namespace.endswith("-client"):
        return namespace.removeprefix("nvflare-").split("-client", 1)[0] or "nvflare", "client"
    return namespace.removeprefix("nvflare-") or "nvflare", "unknown"


def is_job_pod(pod) -> bool:
    if pod.metadata.name.startswith("kit-copy-"):
        return False
    return pod.spec.restart_policy == "Never"


def _cluster_call(cluster: Cluster, cache: ApiCache, fn):
    try:
        api = cache.get_cluster(cluster)
        return fn(api)
    except ApiException as e:
        cache.drop_cluster(cluster)
        return PodFetchError(e.reason or f"HTTP {e.status}")
    except ConfigException as e:
        cache.drop_cluster(cluster)
        return PodFetchError(f"config: {e}")
    except Exception as e:
        cache.drop_cluster(cluster)
        return PodFetchError(f"{type(e).__name__}: {e}")


def fetch_cluster_snapshot(cluster: Cluster, cache: ApiCache) -> tuple[ClusterSnapshot, PruneStats]:
    def _fetch(api):
        pods = api.list_pod_for_all_namespaces(_request_timeout=8).items
        pvcs = api.list_persistent_volume_claim_for_all_namespaces(_request_timeout=8).items
        services = api.list_service_for_all_namespaces(_request_timeout=8).items
        try:
            nodes = api.list_node(_request_timeout=8).items
            node_error = None
        except ApiException as e:
            nodes = []
            node_error = e.reason or f"HTTP {e.status}"
        except Exception as e:
            nodes = []
            node_error = f"{type(e).__name__}: {e}"
        return pods, pvcs, services, nodes, node_error

    result = _cluster_call(cluster, cache, _fetch)
    if isinstance(result, PodFetchError):
        return ClusterSnapshot(cluster=cluster, namespaces=[], error=result.msg), PruneStats.empty()

    pods, pvcs, services, nodes, node_error = result
    namespaces = sorted(
        {
            item.metadata.namespace
            for items in (pods, pvcs, services)
            for item in items
            if item.metadata.namespace and is_nvflare_namespace(item.metadata.namespace)
        }
    )
    try:
        api = cache.get_cluster(cluster)
    except Exception as e:
        return (
            ClusterSnapshot(cluster=cluster, namespaces=[], error=f"{type(e).__name__}: {e}"),
            PruneStats.empty(),
        )

    prune_stats = PruneStats.empty()
    resources = []
    for namespace in namespaces:
        ns_pods = [pod for pod in pods if pod.metadata.namespace == namespace]
        ns_pods, stats = prune_pods(
            api,
            lambda cluster=cluster: cache.drop_cluster(cluster),
            cluster_name=cluster.name,
            namespace=namespace,
            pods=ns_pods,
        )
        prune_stats.extend(stats)
        resources.append(
            NamespaceResources(
                cluster=cluster,
                namespace=namespace,
                pods=ns_pods,
                pvcs=[pvc for pvc in pvcs if pvc.metadata.namespace == namespace],
                services=[svc for svc in services if svc.metadata.namespace == namespace],
            )
        )
    deleted = prune_stats.deleted
    all_pods = [pod for pod in pods if (cluster.name, pod.metadata.namespace, pod.metadata.name) not in deleted]
    return (
        ClusterSnapshot(cluster=cluster, namespaces=resources, nodes=nodes, all_pods=all_pods, node_error=node_error),
        prune_stats,
    )


def gather_cluster_snapshots(clusters: list[Cluster], cache: ApiCache) -> tuple[list[ClusterSnapshot], PruneStats]:
    stats = PruneStats.empty()
    with ThreadPoolExecutor(max_workers=max(1, len(clusters))) as ex:
        futures = {ex.submit(fetch_cluster_snapshot, cluster, cache): cluster for cluster in clusters}
        by_name: dict[str, ClusterSnapshot] = {}
        for fut in as_completed(futures):
            cluster = futures[fut]
            try:
                snapshot, prune_stats = fut.result()
                by_name[cluster.name] = snapshot
                stats.extend(prune_stats)
            except Exception as e:
                by_name[cluster.name] = ClusterSnapshot(
                    cluster=cluster,
                    namespaces=[],
                    error=f"{type(e).__name__}: {e}",
                )
    return [by_name[cluster.name] for cluster in clusters], stats


def _site_name(ns: NamespaceResources, site_pods: list) -> str:
    for svc in sorted(ns.services, key=lambda s: s.metadata.name):
        labels = svc.metadata.labels or {}
        name = labels.get("app.kubernetes.io/name") or labels.get("app.kubernetes.io/instance")
        if name:
            return name
    for pod in sorted(site_pods, key=lambda p: p.metadata.name):
        labels = pod.metadata.labels or {}
        name = labels.get("app.kubernetes.io/name") or labels.get("app.kubernetes.io/instance")
        if name:
            return name
    return ns.namespace


def _ip_join(values: list[str], none: str = "-") -> str:
    compact = [v for v in values if v and v != "None"]
    if not compact:
        return none
    return ", ".join(sorted(dict.fromkeys(compact)))


def _service_cluster_ips(services: list) -> list[str]:
    return [svc.spec.cluster_ip for svc in services if svc.spec.cluster_ip and svc.spec.cluster_ip != "None"]


def _service_external_ips(services: list) -> list[str]:
    out = []
    for svc in services:
        if svc.spec.type != "LoadBalancer":
            continue
        if svc.spec.load_balancer_ip:
            out.append(svc.spec.load_balancer_ip)
        ingress = []
        if svc.status and svc.status.load_balancer:
            ingress = svc.status.load_balancer.ingress or []
        for item in ingress:
            endpoint = item.ip or item.hostname
            if endpoint:
                out.append(endpoint)
    return out


def _pvc_capacity(pvc) -> str:
    if pvc.status.capacity and "storage" in pvc.status.capacity:
        return pvc.status.capacity["storage"]
    if pvc.spec.resources and pvc.spec.resources.requests:
        return pvc.spec.resources.requests.get("storage", "")
    return ""


def _pvc_summary(pvcs: list) -> tuple[str, int, int]:
    if not pvcs:
        return "-", 0, 0
    bound = sum(1 for pvc in pvcs if pvc.status.phase == "Bound")
    parts = []
    for pvc in sorted(pvcs, key=lambda p: p.metadata.name):
        capacity = _pvc_capacity(pvc)
        status = "" if pvc.status.phase == "Bound" else f" {pvc.status.phase or 'Unknown'}"
        suffix = f" {capacity}" if capacity else ""
        parts.append(f"{pvc.metadata.name}{status}{suffix}")
    return f"{bound}/{len(pvcs)} Bound: {', '.join(parts)}", bound, len(pvcs)


def _status_summary(pods: list) -> tuple[str, int, int]:
    if not pods:
        return "-", 0, 0
    counts = Counter(derive_status(pod) for pod in pods)
    running = counts.get("Running", 0)
    parts = [f"{count} {status}" for status, count in counts.most_common()]
    return ", ".join(parts), running, len(pods)


def _oldest_age(objs: list) -> str:
    timestamps = [_ensure_aware_utc(obj.metadata.creation_timestamp) for obj in objs if obj.metadata.creation_timestamp]
    if not timestamps:
        return "-"
    return age_str_from_seconds((datetime.now(timezone.utc) - min(timestamps)).total_seconds())


def node_is_ready(node) -> bool:
    for condition in node.status.conditions or []:
        if condition.type == "Ready":
            return condition.status == "True"
    return False


def pod_is_running(pod) -> bool:
    return pod.status.phase == "Running" and not pod.metadata.deletion_timestamp


def cluster_infra_counts(snapshot: ClusterSnapshot) -> dict[str, int]:
    nodes = snapshot.nodes or []
    pods = snapshot.all_pods or []
    return {
        "nodes_ready": sum(1 for node in nodes if node_is_ready(node)),
        "nodes_total": len(nodes),
        "pods_running": sum(1 for pod in pods if pod_is_running(pod)),
        "pods_total": len(pods),
    }


def cluster_nvflare_pod_counts(snapshot: ClusterSnapshot) -> dict[str, int]:
    pods = [pod for ns in snapshot.namespaces for pod in ns.pods]
    return {
        "running": sum(1 for pod in pods if pod_is_running(pod)),
        "total": len(pods),
    }


def build_site_rows(snapshots: list[ClusterSnapshot]) -> list[SiteRow]:
    rows = []
    for snapshot in snapshots:
        for ns in snapshot.namespaces:
            system, role = namespace_system_role(ns.namespace)
            job_pods = [pod for pod in ns.pods if is_job_pod(pod)]
            site_pods = [pod for pod in ns.pods if not is_job_pod(pod)]
            site_pod_summary, site_pods_running, site_pods_total = _status_summary(site_pods)
            pvc_summary, pvc_bound, pvc_total = _pvc_summary(ns.pvcs)
            rows.append(
                SiteRow(
                    system=system,
                    cloud=ns.cluster.cloud,
                    role=role,
                    site=_site_name(ns, site_pods),
                    namespace=ns.namespace,
                    site_pod=site_pod_summary,
                    site_pods_running=site_pods_running,
                    site_pods_total=site_pods_total,
                    pod_ips=[pod.status.pod_ip for pod in site_pods if pod.status.pod_ip],
                    service_ips=_service_cluster_ips(ns.services),
                    external_ips=_service_external_ips(ns.services),
                    pvc_summary=pvc_summary,
                    pvc_bound=pvc_bound,
                    pvc_total=pvc_total,
                    job_pods_running=sum(1 for pod in job_pods if pod.status.phase == "Running"),
                    job_pods_total=len(job_pods),
                    age=_oldest_age(ns.pods + ns.pvcs + ns.services),
                )
            )
    return sorted(rows, key=lambda r: (r.system, r.role != "server", r.cloud, r.namespace))


def build_all_cloud_summary(
    snapshots: list[ClusterSnapshot],
    rows: list[SiteRow],
    prune_stats: PruneStats,
    kubeconfig_dir: Path,
    interval: float,
) -> Text:
    ts = datetime.now().strftime("%H:%M:%S")
    cluster_errors = [s for s in snapshots if s.error]
    systems = len({row.system for row in rows})
    clients = sum(1 for row in rows if row.role == "client")
    servers = sum(1 for row in rows if row.role == "server")
    job_running = sum(row.job_pods_running for row in rows)
    job_total = sum(row.job_pods_total for row in rows)
    reachable_clusters = sum(1 for snapshot in snapshots if not snapshot.error)
    nodes_ready = sum(cluster_infra_counts(snapshot)["nodes_ready"] for snapshot in snapshots)
    nodes_total = sum(cluster_infra_counts(snapshot)["nodes_total"] for snapshot in snapshots)
    pods_running = sum(cluster_infra_counts(snapshot)["pods_running"] for snapshot in snapshots)
    pods_total = sum(cluster_infra_counts(snapshot)["pods_total"] for snapshot in snapshots)
    lines = [
        f"[bold]k8sview[/] · all-cloud · {ts} · refresh={interval}s · Ctrl-C to quit",
        f"Kubeconfigs: {kubeconfig_dir}",
        f"Clusters: {reachable_clusters}/{len(snapshots)} reachable · nodes={nodes_ready}/{nodes_total} Ready "
        f"· all pods={pods_running}/{pods_total} Running",
        f"Systems: {systems} · servers={servers} · clients={clients} · job pods={job_running}/{job_total} Running",
    ]
    if prune_stats.deleted:
        lines.append(f"Pruned terminal pods: [green]{len(prune_stats.deleted)}[/]")
    if cluster_errors or prune_stats.errors:
        lines.append(f"[red]Errors: {len(cluster_errors) + len(prune_stats.errors)}[/]")
    return Text.from_markup("\n".join(lines))


def build_systems_table(rows: list[SiteRow], snapshots: list[ClusterSnapshot]) -> Table:
    table = Table(title="Systems", title_style="bold", expand=True)
    for col, style, width in [
        ("SYSTEM", "cyan", None),
        ("SERVER", None, 8),
        ("CLIENTS", None, 8),
        ("CLOUDS", None, None),
        ("EXTERNAL", None, None),
        ("SITE PODS", None, 10),
        ("JOB PODS", None, 9),
        ("PVC", None, 8),
        ("NS", None, 4),
    ]:
        table.add_column(col, style=style, width=width, no_wrap=(col not in {"CLOUDS", "EXTERNAL"}))

    systems = sorted({row.system for row in rows})
    if not systems:
        errors = [s for s in snapshots if s.error]
        if errors:
            for snapshot in errors:
                table.add_row(
                    snapshot.cluster.name,
                    "-",
                    "-",
                    snapshot.cluster.cloud,
                    Text(f"ERROR: {snapshot.error}", style="red"),
                    "-",
                    "-",
                    "-",
                    "-",
                )
        else:
            table.add_row("-", "-", "-", "-", Text("(no NVFlare namespaces found)", style="dim"), "-", "-", "-", "-")
        return table

    for system in systems:
        sys_rows = [row for row in rows if row.system == system]
        servers = [row for row in sys_rows if row.role == "server"]
        clients = [row for row in sys_rows if row.role == "client"]
        site_running = sum(row.site_pods_running for row in sys_rows)
        site_total = sum(row.site_pods_total for row in sys_rows)
        job_running = sum(row.job_pods_running for row in sys_rows)
        job_total = sum(row.job_pods_total for row in sys_rows)
        pvc_bound = sum(row.pvc_bound for row in sys_rows)
        pvc_total = sum(row.pvc_total for row in sys_rows)
        external = _ip_join([ip for row in servers for ip in row.external_ips])
        table.add_row(
            system,
            str(len(servers)),
            str(len(clients)),
            ", ".join(sorted({row.cloud for row in sys_rows})),
            external,
            f"{site_running}/{site_total}",
            f"{job_running}/{job_total}",
            f"{pvc_bound}/{pvc_total}",
            str(len(sys_rows)),
        )
    return table


def build_clusters_table(snapshots: list[ClusterSnapshot]) -> Table:
    table = Table(title="Clusters", title_style="bold", expand=True)
    for col, style, width in [
        ("CLOUD", "cyan", None),
        ("STATUS", None, 10),
        ("NODES", None, 9),
        ("ALL PODS", None, 10),
        ("NVFLARE NS", None, 10),
        ("NVFLARE PODS", None, 12),
        ("ERROR", "red", None),
    ]:
        table.add_column(col, style=style, width=width, no_wrap=(col != "ERROR"))

    for snapshot in snapshots:
        counts = cluster_infra_counts(snapshot)
        nvflare_counts = cluster_nvflare_pod_counts(snapshot)
        if snapshot.error:
            status = Text("ERROR", style="red")
            error = snapshot.error
        elif snapshot.node_error:
            status = Text("PARTIAL", style="yellow")
            error = f"nodes: {snapshot.node_error}"
        else:
            status = Text("OK", style="green")
            error = "-"
        table.add_row(
            snapshot.cluster.cloud,
            status,
            f"{counts['nodes_ready']}/{counts['nodes_total']}",
            f"{counts['pods_running']}/{counts['pods_total']}",
            str(len(snapshot.namespaces)),
            f"{nvflare_counts['running']}/{nvflare_counts['total']}",
            error,
        )
    return table


def build_sites_table(rows: list[SiteRow]) -> Table:
    table = Table(title="Sites", title_style="bold", expand=True)
    for col, style, width in [
        ("SYSTEM", "cyan", None),
        ("ROLE", None, 7),
        ("CLOUD", "cyan", None),
        ("SITE", "white", None),
        ("NAMESPACE", "cyan", None),
        ("POD IP", None, None),
        ("SERVICE IP", None, None),
        ("EXTERNAL", None, None),
        ("SITE PODS", None, 12),
        ("JOB PODS", None, 9),
        ("PVC", None, None),
        ("AGE", None, 6),
    ]:
        table.add_column(col, style=style, width=width, no_wrap=True)
    if not rows:
        table.add_row("-", "-", "-", "-", "-", "-", "-", "-", "-", "-", Text("(none)", style="dim"), "-")
        return table
    for row in rows:
        role = Text(row.role, style="bold green" if row.role == "server" else "")
        table.add_row(
            row.system,
            role,
            row.cloud,
            row.site,
            row.namespace,
            _ip_join(row.pod_ips),
            _ip_join(row.service_ips),
            _ip_join(row.external_ips),
            row.site_pod,
            f"{row.job_pods_running}/{row.job_pods_total}",
            row.pvc_summary,
            row.age,
        )
    return table


def build_errors_table(snapshots: list[ClusterSnapshot], prune_stats: PruneStats) -> Table | None:
    rows = []
    for snapshot in snapshots:
        if snapshot.error:
            rows.append((snapshot.cluster.name, snapshot.error))
        if snapshot.node_error:
            rows.append((f"{snapshot.cluster.name}/nodes", snapshot.node_error))
    for error in prune_stats.errors:
        rows.append(("prune", error))
    if not rows:
        return None
    table = Table(title="Errors", title_style="bold red", expand=True)
    table.add_column("SOURCE", style="cyan", no_wrap=True)
    table.add_column("MESSAGE", style="red")
    for source, message in rows[:12]:
        table.add_row(source, message)
    return table


def build_all_cloud_view(
    snapshots: list[ClusterSnapshot],
    prune_stats: PruneStats,
    kubeconfig_dir: Path,
    interval: float,
    rows: list[SiteRow] | None = None,
) -> Group:
    if rows is None:
        rows = build_site_rows(snapshots)
    parts = [
        build_all_cloud_summary(snapshots, rows, prune_stats, kubeconfig_dir, interval),
        build_systems_table(rows, snapshots),
        build_sites_table(rows),
    ]
    parts.append(build_clusters_table(snapshots))
    errors = build_errors_table(snapshots, prune_stats)
    if errors is not None:
        parts.append(errors)
    return Group(*parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Live multicloud dashboard.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to deploy config YAML. When omitted, discover all NVFlare namespaces in .tmp/kubeconfigs.",
    )
    parser.add_argument(
        "--kubeconfig-dir",
        default=str(DEFAULT_KUBECONFIG_DIR),
        help="Directory of kubeconfig YAML files for the all-cloud view",
    )
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds")
    parser.add_argument("--events-window", type=float, default=300.0, help="Show Warning events newer than N seconds")
    args = parser.parse_args()

    cache = ApiCache()
    console = Console()
    console.clear()

    if args.config:
        config_path = Path(args.config).resolve()
        participants = load_participants(config_path)
        with Live(console=console, refresh_per_second=4, screen=False) as live:
            try:
                while True:
                    pod_snap = gather(participants, cache, fetch_pods)
                    pod_snap, _ = prune_pod_snapshots(pod_snap, cache)
                    pvc_snap = gather(participants, cache, fetch_pvcs)
                    svc_snap = gather(participants, cache, fetch_services)
                    evt_snap = gather(participants, cache, fetch_events)
                    live.update(
                        Group(
                            build_summary(pod_snap, pvc_snap, svc_snap, config_path, args.interval),
                            build_pod_table(pod_snap),
                            build_pvc_table(pvc_snap),
                            build_events_table(evt_snap, args.events_window),
                        )
                    )
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                pass
        return

    kubeconfig_dir = Path(args.kubeconfig_dir).resolve()
    clusters = load_clusters(kubeconfig_dir)
    with Live(console=console, refresh_per_second=4, screen=False) as live:
        try:
            while True:
                snapshots, prune_stats = gather_cluster_snapshots(clusters, cache)
                rows = build_site_rows(snapshots)
                live.update(build_all_cloud_view(snapshots, prune_stats, kubeconfig_dir, args.interval, rows))
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
