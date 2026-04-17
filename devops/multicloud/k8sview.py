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

Reads the same YAML config as deploy.py and renders a top-like live table of
pods across every participating cluster. Read-only.

Usage:
    k8sview.py --config aws-server.yaml [--interval 2]
"""

from __future__ import annotations

import argparse
import sys
import time
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
DEFAULT_CONFIG = TOOL_DIR / "gcp-server.yaml"

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


def load_participants(config_path: Path) -> list[Participant]:
    config_path = config_path.resolve()
    raw = yaml.safe_load(config_path.read_text())
    clouds = raw.get("clouds") or {}
    entries = raw.get("participants") or []
    if not clouds or not entries:
        raise ValueError(f"{config_path}: missing 'clouds' or 'participants' section")
    out: list[Participant] = []
    for e in entries:
        merged = {**clouds.get(e["cloud"], {}), **e}
        kc = (config_path.parent / merged["kubeconfig"]).resolve()
        out.append(Participant(merged["name"], e["cloud"], merged["namespace"], str(kc)))
    return out


# ---------------------------------------------------------------------------
# Kubernetes API cache + pod fetch
# ---------------------------------------------------------------------------
class ApiCache:
    def __init__(self):
        self._apis: dict[str, k8s_client.CoreV1Api] = {}

    def get(self, p: Participant) -> k8s_client.CoreV1Api:
        api = self._apis.get(p.name)
        if api is not None:
            return api
        api_client = k8s_config.new_client_from_config(config_file=p.kubeconfig)
        api = k8s_client.CoreV1Api(api_client)
        self._apis[p.name] = api
        return api

    def drop(self, p: Participant) -> None:
        self._apis.pop(p.name, None)


@dataclass
class PodFetchError:
    msg: str


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
    s = int(delta.total_seconds())
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
    from collections import Counter

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
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Live multicloud pod dashboard (read-only).")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to deploy config YAML")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds")
    parser.add_argument("--events-window", type=float, default=300.0, help="Show Warning events newer than N seconds")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    participants = load_participants(config_path)

    cache = ApiCache()
    console = Console()

    with Live(console=console, refresh_per_second=4, screen=False) as live:
        try:
            while True:
                pod_snap = gather(participants, cache, fetch_pods)
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


if __name__ == "__main__":
    main()
