#!/usr/bin/env python3
"""Rich live pod table for the NVFlare OpenShift deployment scripts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text
except ImportError:
    sys.exit("openshift_k8s_watch requires the Python package 'rich'. Install it with: python3 -m pip install rich")


STATUS_STYLES = {
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
    "Unknown": "red",
}


@dataclass
class WatchConfig:
    kube_cmd: str
    namespace: str
    work_dir: Path
    last_job_id_file: Path
    interval: float
    once: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show a Rich live pod table for NVFlare OpenShift pods.")
    parser.add_argument("--once", action="store_true", help="Render one snapshot and exit.")
    parser.add_argument("--interval", type=float, default=3.0, help="Refresh interval in seconds. Default: 3.")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> WatchConfig:
    work_dir = Path(os.environ.get("WORK_DIR", "/tmp/nvflare/openshift-e2e"))
    return WatchConfig(
        kube_cmd=os.environ.get("KUBE_CMD", "oc"),
        namespace=os.environ.get("NAMESPACE", "nvflare-e2e"),
        work_dir=work_dir,
        last_job_id_file=Path(os.environ.get("LAST_JOB_ID_FILE", str(work_dir / "last_job_id"))),
        interval=args.interval,
        once=args.once,
    )


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def last_job_id(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def age_str(created_at: str | None) -> str:
    ts = parse_timestamp(created_at)
    if ts is None:
        return "-"
    seconds = max(0, int((datetime.now(timezone.utc) - ts).total_seconds()))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        hours, rem = divmod(seconds, 3600)
        minutes = rem // 60
        return f"{hours}h{minutes}m" if minutes else f"{hours}h"
    days, rem = divmod(seconds, 86400)
    hours = rem // 3600
    return f"{days}d{hours}h" if hours else f"{days}d"


def pod_status(pod: dict[str, Any]) -> str:
    metadata = pod.get("metadata") or {}
    status = pod.get("status") or {}
    if metadata.get("deletionTimestamp"):
        return "Terminating"

    statuses = (status.get("initContainerStatuses") or []) + (status.get("containerStatuses") or [])
    for container_status in statuses:
        waiting = ((container_status.get("state") or {}).get("waiting") or {})
        if waiting.get("reason"):
            return waiting["reason"]

    for container_status in status.get("containerStatuses") or []:
        terminated = ((container_status.get("state") or {}).get("terminated") or {})
        reason = terminated.get("reason")
        if reason and reason != "Completed":
            return reason

    phase = status.get("phase") or "Unknown"
    container_statuses = status.get("containerStatuses") or []
    if phase == "Running" and container_statuses and not all(c.get("ready") for c in container_statuses):
        return "NotReady"
    if phase == "Succeeded":
        return "Completed"
    return phase


def ready_fraction(pod: dict[str, Any]) -> str:
    statuses = (pod.get("status") or {}).get("containerStatuses") or []
    if not statuses:
        return "0/0"
    ready = sum(1 for container_status in statuses if container_status.get("ready"))
    return f"{ready}/{len(statuses)}"


def restart_count(pod: dict[str, Any]) -> str:
    statuses = (pod.get("status") or {}).get("containerStatuses") or []
    return str(sum(int(container_status.get("restartCount") or 0) for container_status in statuses))


def status_text(status: str) -> Text:
    return Text(status, style=STATUS_STYLES.get(status, ""))


def get_pods(config: WatchConfig) -> tuple[list[dict[str, Any]], str | None]:
    ns_result = run_cmd([config.kube_cmd, "get", "namespace", config.namespace, "-o", "json"])
    if ns_result.returncode != 0:
        msg = (ns_result.stderr or ns_result.stdout).strip() or f"namespace {config.namespace} not found"
        return [], msg

    pod_result = run_cmd(
        [
            config.kube_cmd,
            "-n",
            config.namespace,
            "get",
            "pods",
            "--sort-by=.metadata.creationTimestamp",
            "-o",
            "json",
        ]
    )
    if pod_result.returncode != 0:
        msg = (pod_result.stderr or pod_result.stdout).strip() or "failed to list pods"
        return [], msg

    try:
        payload = json.loads(pod_result.stdout)
    except json.JSONDecodeError as e:
        return [], f"failed to parse pod JSON: {e}"
    return payload.get("items") or [], None


def build_table(pods: list[dict[str, Any]], error: str | None) -> Table:
    table = Table(title="Pods", caption=f"{len(pods)} pods", caption_style="dim", expand=True)
    table.add_column("NAME", no_wrap=True)
    table.add_column("READY", justify="right", no_wrap=True)
    table.add_column("STATUS", no_wrap=True)
    table.add_column("RESTARTS", justify="right", no_wrap=True)
    table.add_column("AGE", justify="right", no_wrap=True)

    if error:
        table.add_row("-", "-", Text(error, style="red"), "-", "-")
        return table

    if not pods:
        table.add_row(Text("(no pods)", style="dim"), "", "", "", "")
        return table

    for pod in pods:
        metadata = pod.get("metadata") or {}
        status = pod_status(pod)
        table.add_row(
            metadata.get("name") or "-",
            ready_fraction(pod),
            status_text(status),
            restart_count(pod),
            age_str(metadata.get("creationTimestamp")),
        )
    return table


def build_view(config: WatchConfig) -> Group:
    pods, error = get_pods(config)
    job_id = last_job_id(config.last_job_id_file)
    header = Text("NVFlare OpenShift pods", style="bold")
    details = [
        f"Namespace: {config.namespace}",
        f"Work dir:  {config.work_dir}",
    ]
    if job_id:
        details.append(f"Last job:  {job_id}")
    details.append(datetime.now(timezone.utc).strftime("UTC:       %Y-%m-%dT%H:%M:%SZ"))
    return Group(header, Text("\n".join(details), style="dim"), build_table(pods, error))


def main() -> int:
    args = parse_args()
    if args.interval <= 0:
        sys.exit("--interval must be greater than 0")
    config = load_config(args)
    console = Console()

    if config.once:
        console.print(build_view(config))
        return 0

    with Live(build_view(config), console=console, refresh_per_second=4, transient=False) as live:
        while True:
            time.sleep(config.interval)
            live.update(build_view(config))


if __name__ == "__main__":
    raise SystemExit(main())
