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

"""Timing finalization for benchmark records and summaries."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import flatten_numbers, load_json


@dataclass(frozen=True)
class LifecycleEpochs:
    script_start: int
    skill_availability_start: int
    skill_availability_end: int
    input_copy_start: int
    input_copy_end: int
    prompt_prep_start: int
    prompt_prep_end: int
    agent_start: int
    agent_end: int
    post_process_start: int
    post_process_end: int
    report_outcome_start: int
    report_outcome_end: int
    script_end: int

    @classmethod
    def from_sequence(cls, values: list[int]) -> "LifecycleEpochs":
        if len(values) != 14:
            raise ValueError(f"expected 14 lifecycle epoch values; got {len(values)}")
        return cls(*values)


def _write_json_files_atomic(payloads: dict[Path, Any]) -> None:
    staged_paths: list[tuple[Path, Path]] = []
    try:
        for path, value in payloads.items():
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
                temp_path = Path(stream.name)
                staged_paths.append((temp_path, path))
                json.dump(value, stream, indent=2, sort_keys=True)
        for temp_path, path in staged_paths:
            os.replace(temp_path, path)
        staged_paths.clear()
    finally:
        for temp_path, _path in staged_paths:
            try:
                temp_path.unlink()
            except OSError:
                pass


def finalize_timing(
    summary_path: Path,
    record_path: Path,
    timing_path: Path,
    activity_path: Path,
    epochs: LifecycleEpochs,
) -> None:
    setup_elapsed = epochs.agent_start - epochs.script_start
    agent_elapsed = epochs.agent_end - epochs.agent_start
    phase_seconds = {
        "container_elapsed_seconds": epochs.script_end - epochs.script_start,
        "setup_elapsed_seconds": setup_elapsed,
        "skill_exposure_elapsed_seconds": epochs.skill_availability_end - epochs.skill_availability_start,
        "input_copy_elapsed_seconds": epochs.input_copy_end - epochs.input_copy_start,
        "prompt_prepare_elapsed_seconds": epochs.prompt_prep_end - epochs.prompt_prep_start,
        "agent_elapsed_seconds": agent_elapsed,
        "post_process_elapsed_seconds": epochs.post_process_end - epochs.post_process_start,
        "report_elapsed_seconds": epochs.report_outcome_end - epochs.report_outcome_start,
    }
    timing = {
        "epoch_seconds": {
            "script_start": epochs.script_start,
            "agent_start": epochs.agent_start,
            "agent_end": epochs.agent_end,
            "script_end": epochs.script_end,
        },
        "phase_seconds": phase_seconds,
    }
    summary = load_json(summary_path, {}) or {}
    record = load_json(record_path, {}) or {}
    activity = load_json(activity_path, {}) or {}
    summary["phase_seconds"] = phase_seconds
    if summary.get("agent_elapsed_seconds") is None:
        summary["agent_elapsed_seconds"] = agent_elapsed
    summary["activity"] = {
        "event_count": activity.get("event_count"),
        "first_event_timestamp": activity.get("first_event_timestamp"),
        "last_event_timestamp": activity.get("last_event_timestamp"),
        "event_span_seconds": activity.get("event_span_seconds"),
        "max_inter_event_gap_seconds": activity.get("max_inter_event_gap_seconds"),
        "command_count": activity.get("command_count"),
        "unique_command_count": activity.get("unique_command_count"),
        "command_prefix_counts": activity.get("command_prefix_counts"),
        "hint_counts": activity.get("hint_counts"),
    }
    metrics = record.setdefault("process_metrics", {})
    if isinstance(metrics, dict):
        metrics["phase_seconds"] = phase_seconds
        metrics["activity"] = summary["activity"]
        if metrics.get("agent_elapsed_seconds") is None:
            metrics["agent_elapsed_seconds"] = agent_elapsed
    summary["process_metrics"] = (
        record.get("process_metrics")
        if isinstance(record.get("process_metrics"), dict)
        else summary.get("process_metrics", {})
    )
    summary["all_metrics"] = flatten_numbers(summary)
    _write_json_files_atomic({timing_path: timing, summary_path: summary, record_path: record})
