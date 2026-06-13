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

"""Agent event parsing for usage and activity metrics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

MAX_ACTIVITY_COMMANDS = 200
MAX_TRACKED_EVENT_TYPES = 500
MAX_TRACKED_COMMAND_PREFIXES = 500
MAX_TRACKED_UNIQUE_COMMANDS = 10000


def walk(obj: Any, depth: int = 20) -> Iterable[dict[str, Any]]:
    if depth < 0:
        return
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from walk(value, depth - 1)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk(value, depth - 1)


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def normalize_command(value: Any) -> str | None:
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    if isinstance(value, str):
        return value
    return None


def increment_bounded_counter(counter: Counter[str], key: str, max_keys: int) -> bool:
    # Returns True when the key was rejected because the bounded counter is full.
    if key in counter or len(counter) < max_keys:
        counter[key] += 1
        return False
    return True


def parse_usage_and_activity_data(events_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    # Usage and activity are derived in one pass so large event streams are only
    # decoded once; keep output-specific normalization below clearly separated.
    last_total_tokens: float | None = None
    max_total_tokens: float | None = None
    last_input_tokens: float | None = None
    last_output_tokens: float | None = None
    max_input_tokens: float | None = None
    max_output_tokens: float | None = None
    raw_usage_object_count = 0
    token_parser_warnings: list[str] = []
    event_count = 0
    decode_errors = 0
    event_types: Counter[str] = Counter()
    command_prefixes: Counter[str] = Counter()
    commands: list[str] = []
    unique_commands_seen: set[str] = set()
    command_count = 0
    commands_truncated = False
    unique_commands_truncated = False
    event_types_truncated = False
    command_prefixes_truncated = False
    hint_counts: Counter[str] = Counter()
    first_event_dt = None
    first_event_timestamp = None
    last_event_dt = None
    last_event_timestamp = None
    previous_event_dt = None
    max_inter_event_gap_seconds = None

    hints = {
        "skill_md": ["skill.md"],
        "skill_references": ["/references/", " references/"],
        "skill_metadata": ["evals/evals.json", "/evals/"],
        "benchmark_md": ["benchmark.md"],
        "agent_inspect": ["agent inspect"],
        "agent_skill_setup": ["skills install", "skills list"],
        "py_compile": ["py_compile"],
        "python_job_py": ["python job.py", "python3 job.py"],
        "simulation": ["simulator", "simulate", "--workspace-root"],
        "shell_find": ["find "],
        "shell_search": ["rg ", "grep "],
        "shell_cat_or_sed": ["cat ", "sed ", "nl -ba"],
    }

    def maybe_add_command(key: str, value: Any) -> None:
        nonlocal command_count, command_prefixes_truncated, commands_truncated, unique_commands_truncated
        if str(key).lower() not in {"cmd", "command", "shell_command"}:
            return
        command = normalize_command(value)
        if not command:
            return
        command = command.strip()
        if not command:
            return
        command_count += 1
        if command in unique_commands_seen or len(unique_commands_seen) < MAX_TRACKED_UNIQUE_COMMANDS:
            unique_commands_seen.add(command)
        else:
            unique_commands_truncated = True
        if len(commands) < MAX_ACTIVITY_COMMANDS:
            commands.append(command)
        else:
            commands_truncated = True
        if increment_bounded_counter(command_prefixes, command.split()[0], MAX_TRACKED_COMMAND_PREFIXES):
            command_prefixes_truncated = True

    def add_text_hints(text: str) -> None:
        lowered = text.lower()
        for name, needles in hints.items():
            if any(needle in lowered for needle in needles):
                hint_counts[name] += 1

    if events_path.exists():
        with events_path.open("r", encoding="utf-8", errors="replace") as events:
            for line in events:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                add_text_hints(line)
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    decode_errors += 1
                    continue
                event_count += 1
                if isinstance(event, dict):
                    event_timestamp = event.get("harness_timestamp") or event.get("timestamp")
                    event_dt = parse_timestamp(event_timestamp)
                    if event_dt is not None:
                        if first_event_dt is None:
                            first_event_dt = event_dt
                            first_event_timestamp = event_timestamp
                        if previous_event_dt is not None:
                            gap = (event_dt - previous_event_dt).total_seconds()
                            if gap >= 0 and (max_inter_event_gap_seconds is None or gap > max_inter_event_gap_seconds):
                                max_inter_event_gap_seconds = gap
                        previous_event_dt = event_dt
                        last_event_dt = event_dt
                        last_event_timestamp = event_timestamp
                    for key in ("type", "event", "name"):
                        value = event.get(key)
                        if isinstance(value, str):
                            if increment_bounded_counter(event_types, value, MAX_TRACKED_EVENT_TYPES):
                                event_types_truncated = True
                    for container_key in ("msg", "item", "delta"):
                        container = event.get(container_key)
                        if isinstance(container, dict):
                            value = container.get("type") or container.get("name")
                            if isinstance(value, str):
                                if increment_bounded_counter(event_types, value, MAX_TRACKED_EVENT_TYPES):
                                    event_types_truncated = True
                for item in walk(event):
                    lowered = {str(k).lower(): v for k, v in item.items()}
                    if any(k in lowered for k in ("usage", "token_usage")):
                        raw_usage_object_count += 1
                    for key, value in item.items():
                        maybe_add_command(key, value)
                    for key, value in lowered.items():
                        if not isinstance(value, (int, float)) or isinstance(value, bool):
                            continue
                        normalized = key.replace("-", "_")
                        if normalized in {"total_tokens", "total_token_count", "total_used_tokens"}:
                            last_total_tokens = value
                            max_total_tokens = value if max_total_tokens is None else max(max_total_tokens, value)
                        elif normalized in {"input_tokens", "prompt_tokens", "input_token_count"}:
                            max_input_tokens = value if max_input_tokens is None else max(max_input_tokens, value)
                            last_input_tokens = value
                        elif normalized in {"output_tokens", "completion_tokens", "output_token_count"}:
                            max_output_tokens = value if max_output_tokens is None else max(max_output_tokens, value)
                            last_output_tokens = value

    total_tokens = last_total_tokens
    fallback_total = None
    if last_input_tokens is not None or last_output_tokens is not None:
        fallback_total = (last_input_tokens or 0) + (last_output_tokens or 0)
    if total_tokens is None and fallback_total is not None:
        total_tokens = fallback_total
        token_parser_warnings.append(
            "No cumulative total_tokens field was found; total_tokens uses last input_tokens plus last output_tokens."
        )
    if total_tokens is not None and fallback_total is not None and total_tokens < fallback_total:
        token_parser_warnings.append(
            "Last cumulative total_tokens is smaller than last input_tokens plus last output_tokens; token fields may not share the same semantics."
        )

    usage = {
        "total_tokens": total_tokens,
        "max_input_tokens": max_input_tokens,
        "max_output_tokens": max_output_tokens,
        "last_input_tokens": last_input_tokens,
        "last_output_tokens": last_output_tokens,
        "max_total_tokens_seen": max_total_tokens,
        "token_parser": "last cumulative total_tokens from agent JSON events; fallback is last input_tokens plus last output_tokens",
        "token_parser_warnings": token_parser_warnings,
        "usage_objects_seen": raw_usage_object_count,
    }
    activity = {
        "event_count": event_count,
        "json_decode_errors": decode_errors,
        "timestamp_field": "harness_timestamp",
        "first_event_timestamp": first_event_timestamp,
        "last_event_timestamp": last_event_timestamp,
        "event_span_seconds": (
            round((last_event_dt - first_event_dt).total_seconds(), 3)
            if first_event_dt is not None and last_event_dt is not None
            else None
        ),
        "max_inter_event_gap_seconds": (
            round(max_inter_event_gap_seconds, 3) if max_inter_event_gap_seconds is not None else None
        ),
        "event_types": dict(event_types.most_common()),
        "event_types_truncated": event_types_truncated,
        "command_count": command_count,
        "unique_command_count": len(unique_commands_seen),
        "unique_commands_truncated": unique_commands_truncated,
        "command_prefix_counts": dict(command_prefixes.most_common()),
        "command_prefix_counts_truncated": command_prefixes_truncated,
        "hint_counts": dict(hint_counts.most_common()),
        "commands": commands,
        "commands_truncated": commands_truncated,
    }
    return usage, activity


def parse_usage_and_activity(events_path: Path, usage_out: Path, activity_out: Path) -> None:
    usage, activity = parse_usage_and_activity_data(events_path)
    usage_out.write_text(json.dumps(usage, indent=2, sort_keys=True), encoding="utf-8")
    activity_out.write_text(json.dumps(activity, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("parse",))
    parser.add_argument("events_path", type=Path)
    parser.add_argument("usage_out", type=Path)
    parser.add_argument("activity_out", type=Path)
    args = parser.parse_args()
    parse_usage_and_activity(args.events_path, args.usage_out, args.activity_out)


if __name__ == "__main__":
    main()
