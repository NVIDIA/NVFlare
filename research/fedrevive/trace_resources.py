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

"""Sample a user systemd scope and its NVFlare simulator processes."""

import argparse
import csv
import datetime
import subprocess
import time
from pathlib import Path

FIELDS = (
    "elapsed_s",
    "timestamp",
    "cgroup_memory_bytes",
    "cgroup_peak_bytes",
    "pids_current",
    "process_count",
    "total_rss_bytes",
    "total_threads",
    "launcher_rss_bytes",
    "simulator_rss_bytes",
    "simulator_threads",
    "worker_count",
    "worker_rss_bytes",
    "max_worker_rss_bytes",
    "worker_threads",
    "max_worker_threads",
    "other_rss_bytes",
    "other_threads",
)


def _read_int(path: Path) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return 0


def _process_stats(pid: int):
    proc = Path("/proc") / str(pid)
    try:
        cmdline = proc.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        status = proc.joinpath("status").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None

    rss_kib = 0
    threads = 0
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            rss_kib = int(line.split()[1])
        elif line.startswith("Threads:"):
            threads = int(line.split()[1])
    if "simulator_worker" in cmdline:
        role = "worker"
    elif "job.py" in cmdline:
        role = "launcher"
    elif "nvflare.private.fed.app.simulator.simulator" in cmdline:
        role = "simulator"
    else:
        role = "other"
    return role, rss_kib * 1024, threads


def _control_group(unit: str) -> Path:
    result = subprocess.run(
        ["systemctl", "--user", "show", unit, "-p", "ControlGroup", "--value"],
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if not value:
        raise RuntimeError(f"systemd unit has no control group: {unit}")
    return Path("/sys/fs/cgroup") / value.lstrip("/")


def _sample(cgroup: Path, elapsed: float) -> dict:
    try:
        pids = [int(value) for value in cgroup.joinpath("cgroup.procs").read_text().split()]
    except FileNotFoundError:
        pids = []

    role_rss = {role: 0 for role in ("launcher", "simulator", "worker", "other")}
    role_threads = {role: 0 for role in ("launcher", "simulator", "worker", "other")}
    worker_rss = []
    worker_threads = []
    process_count = 0
    for pid in pids:
        stats = _process_stats(pid)
        if stats is None:
            continue
        role, rss, threads = stats
        process_count += 1
        role_rss[role] += rss
        role_threads[role] += threads
        if role == "worker":
            worker_rss.append(rss)
            worker_threads.append(threads)

    return {
        "elapsed_s": f"{elapsed:.3f}",
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "cgroup_memory_bytes": _read_int(cgroup / "memory.current"),
        "cgroup_peak_bytes": _read_int(cgroup / "memory.peak"),
        "pids_current": _read_int(cgroup / "pids.current"),
        "process_count": process_count,
        "total_rss_bytes": sum(role_rss.values()),
        "total_threads": sum(role_threads.values()),
        "launcher_rss_bytes": role_rss["launcher"],
        "simulator_rss_bytes": role_rss["simulator"],
        "simulator_threads": role_threads["simulator"],
        "worker_count": len(worker_rss),
        "worker_rss_bytes": sum(worker_rss),
        "max_worker_rss_bytes": max(worker_rss, default=0),
        "worker_threads": sum(worker_threads),
        "max_worker_threads": max(worker_threads, default=0),
        "other_rss_bytes": role_rss["other"],
        "other_threads": role_threads["other"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", required=True, help="User systemd scope name")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be positive")

    cgroup = _control_group(args.unit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    empty_samples = 0
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        while cgroup.exists():
            row = _sample(cgroup, time.monotonic() - started)
            writer.writerow(row)
            stream.flush()
            empty_samples = empty_samples + 1 if row["process_count"] == 0 else 0
            if empty_samples >= 4:
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
