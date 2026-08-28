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

"""Local training-metric exports for visualization."""

from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_METRICS_JSONL = "agenticfl_training_metrics.jsonl"
DEFAULT_TENSORBOARD_DIR = "tensorboard"


def export_tensorboard_metrics(
    workspace: str | Path,
    *,
    output_dir: str | Path | None = None,
    metrics_name: str = DEFAULT_METRICS_JSONL,
) -> dict[str, Any]:
    """Convert client-local aggregate JSONL metrics into TensorBoard event files."""

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:  # pragma: no cover - depends on optional runtime packaging
        raise RuntimeError("TensorBoard export requires the 'tensorboard' package") from exc

    workspace_path = Path(workspace)
    if not workspace_path.is_dir():
        raise ValueError(f"training workspace does not exist: {workspace_path}")

    destination = Path(output_dir) if output_dir is not None else workspace_path / DEFAULT_TENSORBOARD_DIR
    metric_paths = sorted(
        path
        for path in workspace_path.glob(f"*/{metrics_name}")
        if path.is_file() and path.parent.name not in {"server", DEFAULT_TENSORBOARD_DIR}
    )
    if not metric_paths:
        raise ValueError(f"no client metric files named {metrics_name!r} found under {workspace_path}")

    _prepare_destination(destination)
    aggregate_values: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    clients: list[dict[str, Any]] = []
    all_tags: set[str] = set()

    for metrics_path in metric_paths:
        client_id = metrics_path.parent.name
        records = _read_metric_records(metrics_path)
        writer = SummaryWriter(log_dir=str(destination / client_id))
        client_tags: set[str] = set()
        try:
            for index, record in enumerate(records):
                step = _metric_step(record, index=index)
                for name, value in _numeric_metrics(record).items():
                    tag = f"metrics/{name}"
                    writer.add_scalar(tag, value, step)
                    aggregate_values[step][name].append(value)
                    client_tags.add(tag)
        finally:
            writer.close()
        all_tags.update(client_tags)
        clients.append(
            {
                "client_id": client_id,
                "record_count": len(records),
                "tags": sorted(client_tags),
            }
        )

    mean_writer = SummaryWriter(log_dir=str(destination / "_client_mean"))
    try:
        for step, metrics in sorted(aggregate_values.items()):
            for name, values in sorted(metrics.items()):
                mean_writer.add_scalar(f"metrics/{name}", sum(values) / len(values), step)
                mean_writer.add_scalar(f"contributors/{name}", len(values), step)
    finally:
        mean_writer.close()

    manifest = {
        "schema_version": "agenticfl.tensorboard_export.v1",
        "status": "completed",
        "workspace": str(workspace_path),
        "output_dir": str(destination),
        "metrics_name": metrics_name,
        "client_count": len(clients),
        "clients": clients,
        "tags": sorted(all_tags),
        "aggregate_run": {
            "name": "_client_mean",
            "method": "unweighted_mean_of_available_client_scalars",
        },
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _prepare_destination(destination: Path) -> None:
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"TensorBoard output path is not a directory: {destination}")
    if destination.is_dir() and any(destination.iterdir()):
        manifest_path = destination / "manifest.json"
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ValueError(f"refusing to replace non-AgenticFL TensorBoard directory: {destination}") from exc
        if existing.get("schema_version") != "agenticfl.tensorboard_export.v1":
            raise ValueError(f"refusing to replace unknown TensorBoard export: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)


def _read_metric_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path} at line {line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"metric record in {path} at line {line_number} must be an object")
        records.append(record)
    if not records:
        raise ValueError(f"metric file is empty: {path}")
    return records


def _metric_step(record: dict[str, Any], *, index: int) -> int:
    value = record.get("round", index)
    if isinstance(value, bool):
        return index
    try:
        step = int(value)
    except (TypeError, ValueError):
        return index
    return step if step >= 0 else index


def _numeric_metrics(record: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, value in record.items():
        if name in {"round", "client_id"} or isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            metrics[str(name)] = numeric
    return metrics
