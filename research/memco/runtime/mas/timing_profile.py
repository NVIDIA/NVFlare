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

"""
Lightweight profiling helpers: append one JSON object per line (JSONL).

Enable via:
- CLI: --profile_timing (eval_collab_domain_adaptation sets NV_DAMAS_PROFILE=1 and mem_config["profile_timing"])
- Env: NV_DAMAS_PROFILE=1 / true / yes / on
- mem_config / task_config: profile_timing=True
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _is_timing_metric_key(key: str) -> bool:
    """Heuristic: only aggregate fields that look like durations (seconds)."""
    return key.endswith("_s") or key == "wall_s"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return rows


def _accumulate_timing_keys(
    target: dict[str, list[float]],
    row: dict[str, Any],
    *,
    key_prefix: str = "",
) -> None:
    for k, v in row.items():
        if not _is_timing_metric_key(k):
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            target[f"{key_prefix}{k}"].append(float(v))
    steps = row.get("step_timings")
    if isinstance(steps, list):
        for st in steps:
            if isinstance(st, dict):
                _accumulate_timing_keys(target, st, key_prefix="step.")


def _format_stats(vals: list[float]) -> tuple[float, float, float, float]:
    if not vals:
        return 0.0, 0.0, 0.0, 0.0
    return float(sum(vals)), float(sum(vals) / len(vals)), float(max(vals)), float(min(vals))


def print_timing_report(
    log_base: str,
    global_dir: str | None = None,
    *,
    banner: str = "",
) -> None:
    """
    Scan ``log_base/**/timing_*.jsonl`` and optional ``global_dir/timing_merge_gg.jsonl``,
    then print aggregated timing stats to stdout.
    """
    sep = "=" * 88
    print(f"\n{sep}", flush=True)
    print(banner or "NV_DAMAS 时间剖析汇总 (--profile_timing)", flush=True)
    print(sep, flush=True)

    base = Path(log_base)
    files: list[Path] = []
    if base.is_dir():
        files = sorted(base.rglob("timing_*.jsonl"))
    gpath: Path | None = None
    if global_dir:
        gp = Path(global_dir) / "timing_merge_gg.jsonl"
        if gp.is_file():
            gpath = gp

    if gpath is not None:
        rows = _load_jsonl(gpath)
        print("\n[全局] merge GG", flush=True)
        print(f"  文件: {gpath}", flush=True)
        if not rows:
            print("  (无记录)", flush=True)
        else:
            by_kind: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            kcnt = Counter(str(r.get("kind", "_default")) for r in rows)
            for r in rows:
                kind = str(r.get("kind", "_default"))
                _accumulate_timing_keys(by_kind[kind], r)
            for kind, acc in sorted(by_kind.items()):
                print(f"  kind={kind!r}  JSONL 行数={kcnt[kind]}", flush=True)
                for metric in sorted(acc.keys()):
                    s, mean, mx, mn = _format_stats(acc[metric])
                    n = len(acc[metric])
                    print(
                        f"    {metric:40s}  n={n:5d}  sum={s:10.3f}s  mean={mean:8.3f}s  max={mx:8.3f}s  min={mn:8.3f}s",
                        flush=True,
                    )

    if not files:
        print(f"\n未在 log_base 下找到 timing_*.jsonl: {log_base}", flush=True)
        print(sep + "\n", flush=True)
        return

    print(f"\n共 {len(files)} 个剖析文件 (log_base={log_base})", flush=True)

    rollup: dict[str, list[float]] = defaultdict(list)
    for fp in files:
        try:
            rel = fp.relative_to(base.resolve())
        except ValueError:
            rel = fp
        rows = _load_jsonl(fp)
        print(f"\n--- {rel} ---", flush=True)
        if not rows:
            print("  (空文件)", flush=True)
            continue
        by_kind: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        kcnt = Counter(str(r.get("kind", "_default")) for r in rows)
        for r in rows:
            kind = str(r.get("kind", "_default"))
            _accumulate_timing_keys(by_kind[kind], r)
            bucket = defaultdict(list)
            _accumulate_timing_keys(bucket, r)
            for mk, vs in bucket.items():
                rollup[mk].extend(vs)
        for kind, acc in sorted(by_kind.items()):
            print(f"  [kind={kind}]  JSONL 行数={kcnt[kind]}", flush=True)
            for metric in sorted(acc.keys()):
                s, mean, mx, mn = _format_stats(acc[metric])
                n = len(acc[metric])
                print(
                    f"    {metric:44s}  n={n:5d}  sum={s:12.3f}s  mean={mean:9.3f}s  max={mx:9.3f}s  min={mn:9.3f}s",
                    flush=True,
                )

    print("\n[全 run 合计] 所有 log_base 下 timing_*.jsonl 中耗时字段横向累加（n 为样本条数）", flush=True)
    for metric in sorted(rollup.keys()):
        s, mean, mx, mn = _format_stats(rollup[metric])
        n = len(rollup[metric])
        print(
            f"  {metric:46s}  n={n:6d}  sum={s:14.3f}s  mean={mean:9.3f}s  max={mx:9.3f}s  min={mn:9.3f}s",
            flush=True,
        )

    print(f"\n{sep}\n", flush=True)


def enabled(*, global_config: dict[str, Any] | None = None, task_config: dict[str, Any] | None = None) -> bool:
    ev = os.environ.get("NV_DAMAS_PROFILE", "").strip().lower()
    if ev in ("1", "true", "yes", "on"):
        return True
    if task_config and bool(task_config.get("profile_timing")):
        return True
    if global_config and bool(global_config.get("profile_timing")):
        return True
    return False


def append(log_dir: str, filename: str, record: dict[str, Any]) -> None:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, filename)
    row = dict(record)
    row.setdefault("ts_wall", time.time())
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
