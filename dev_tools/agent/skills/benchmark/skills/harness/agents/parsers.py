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

"""Parser registries for YAML-driven benchmark agent adapters."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

MAX_ACTIVITY_COMMANDS = 200
CLAUDE_SHELL_TOOL_NAMES = {"bash"}


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def parse_usage_and_activity_data(events_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from ..events import parse_usage_and_activity_data as runtime_parse_usage_and_activity_data

    return runtime_parse_usage_and_activity_data(events_path)


def event_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def normalize_jsonl_event(raw_line: str) -> dict[str, Any] | None:
    stripped = raw_line.rstrip("\n")
    if not stripped:
        return None
    timestamp = event_timestamp()
    try:
        event = json.loads(stripped)
    except json.JSONDecodeError:
        event = {"type": "harness.unparsed_event", "raw": stripped}
    if isinstance(event, dict):
        event.setdefault("timestamp", timestamp)
        event["harness_timestamp"] = timestamp
        return event
    return {
        "type": "harness.non_object_event",
        "timestamp": timestamp,
        "harness_timestamp": timestamp,
        "value": event,
    }


def normalize_claude_stream_event(raw_line: str) -> dict[str, Any] | None:
    event = normalize_jsonl_event(raw_line)
    if event is None:
        return None
    if event.get("type", "").startswith("harness."):
        event.setdefault("event_type", event.get("type"))
        return event

    event_type = str(event.get("type") or "unknown")
    subtype = event.get("subtype")
    event["event_type"] = f"{event_type}.{subtype}" if subtype else event_type
    if event_type == "result" and isinstance(event.get("result"), str):
        event["final_message"] = event["result"]
    # Neutral events expose one primary tool per raw event. Keep the first
    # shell-tool command as the stable activity signal for report aggregation.
    for tool_use in claude_tool_uses(event):
        event.setdefault("tool_kind", tool_use.get("name"))
        command = claude_tool_command(tool_use)
        if command:
            event.setdefault("command_text", command)
            break
    return event


def claude_message_content(event: dict[str, Any]) -> list[Any]:
    message = event.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    return content if isinstance(content, list) else []


def claude_tool_uses(event: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in claude_message_content(event) if isinstance(item, dict) and item.get("type") == "tool_use"]


def claude_tool_command(tool_use: dict[str, Any]) -> str | None:
    tool_name = str(tool_use.get("name") or "").lower()
    if tool_name not in CLAUDE_SHELL_TOOL_NAMES:
        return None
    tool_input = tool_use.get("input")
    if not isinstance(tool_input, dict):
        return None
    for key in ("command", "cmd", "shell_command"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def numeric_token_field(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def claude_usage_total(usage: dict[str, Any]) -> float:
    # Claude cache write/read token fields are included in the headline total
    # because they contribute to run cost; the neutral cache_tokens field also
    # exposes them separately for report layers that split cache cost.
    return sum(
        numeric_token_field(usage, key)
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
    )


def claude_usage_has_tokens(usage: dict[str, Any]) -> bool:
    return any(
        numeric_token_field(usage, key) > 0
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
    )


def claude_usage_objects(event: dict[str, Any]) -> list[dict[str, Any]]:
    usage_objects = []
    usage = event.get("usage")
    if isinstance(usage, dict):
        usage_objects.append(usage)
    message = event.get("message")
    if isinstance(message, dict) and isinstance(message.get("usage"), dict):
        usage_objects.append(message["usage"])
    return usage_objects


def iter_json_events(events_path: Path) -> tuple[list[dict[str, Any]], int]:
    events = []
    decode_errors = 0
    if not events_path.exists():
        return events, decode_errors
    with events_path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                decode_errors += 1
                continue
            if isinstance(event, dict):
                events.append(event)
    return events, decode_errors


def parse_claude_stream_usage(events_path: Path) -> dict[str, Any]:
    events, decode_errors = iter_json_events(events_path)
    result_usage: dict[str, Any] | None = None
    result_usage_count = 0
    summed = {
        "input_tokens": 0.0,
        "output_tokens": 0.0,
        "cache_creation_input_tokens": 0.0,
        "cache_read_input_tokens": 0.0,
    }
    usage_objects_seen = 0
    total_cost_usd = None
    for event in events:
        if event.get("type") == "result" and isinstance(event.get("usage"), dict):
            usage_objects_seen += 1
            result_usage_count += 1
            if result_usage is not None:
                for key in summed:
                    summed[key] += numeric_token_field(result_usage, key)
            result_usage = event["usage"]
            if isinstance(event.get("message"), dict) and isinstance(event["message"].get("usage"), dict):
                usage_objects_seen += 1
            if isinstance(event.get("total_cost_usd"), (int, float)):
                total_cost_usd = event.get("total_cost_usd")
            continue
        if event.get("type") == "result" and isinstance(event.get("total_cost_usd"), (int, float)):
            total_cost_usd = event.get("total_cost_usd")
        for usage in claude_usage_objects(event):
            usage_objects_seen += 1
            for key in summed:
                summed[key] += numeric_token_field(usage, key)

    if result_usage is not None and claude_usage_has_tokens(result_usage):
        selected = result_usage
    else:
        selected = summed
    total_tokens = claude_usage_total(selected)
    parser_warnings = []
    if usage_objects_seen == 0:
        parser_warnings.append("No Claude usage objects were found in the stream-json events.")
        total_tokens = None
    elif result_usage is None:
        parser_warnings.append(
            "No Claude result usage object was found; token fields are summed from message usage objects."
        )
    elif not claude_usage_has_tokens(result_usage) and claude_usage_has_tokens(summed):
        parser_warnings.append(
            "Claude result usage object had no nonzero token fields; token fields are summed from message usage objects."
        )
    elif result_usage_count > 1:
        parser_warnings.append(
            "Multiple Claude result usage objects were found; final cumulative result usage was used."
        )
    cache_creation_tokens = selected.get("cache_creation_input_tokens")
    cache_read_tokens = selected.get("cache_read_input_tokens")
    cache_tokens = numeric_token_field(selected, "cache_creation_input_tokens") + numeric_token_field(
        selected, "cache_read_input_tokens"
    )
    return {
        "total_tokens": total_tokens,
        "input_tokens": selected.get("input_tokens"),
        "output_tokens": selected.get("output_tokens"),
        "cache_tokens": cache_tokens,
        "cost": total_cost_usd,
        "parser_warnings": parser_warnings,
        "cache_creation_input_tokens": selected.get("cache_creation_input_tokens"),
        "cache_read_input_tokens": selected.get("cache_read_input_tokens"),
        "raw_cache_creation_input_tokens": cache_creation_tokens,
        "raw_cache_read_input_tokens": cache_read_tokens,
        "total_cost_usd": total_cost_usd,
        "json_decode_errors": decode_errors,
        "usage_objects_seen": usage_objects_seen,
        "result_usage_objects_seen": result_usage_count,
        "token_parser": "Claude stream-json result usage; fallback sums message usage objects",
    }


ACTIVITY_HINTS = {
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


def parse_claude_stream_activity(events_path: Path) -> dict[str, Any]:
    events, decode_errors = iter_json_events(events_path)
    event_types: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    command_prefixes: Counter[str] = Counter()
    hint_counts: Counter[str] = Counter()
    commands: list[str] = []
    unique_commands_seen: set[str] = set()
    first_event_dt = None
    first_event_timestamp = None
    last_event_dt = None
    last_event_timestamp = None
    previous_event_dt = None
    max_inter_event_gap_seconds = None
    for event in events:
        event_type = str(event.get("event_type") or event.get("type") or "unknown")
        event_types[event_type] += 1
        timestamp = event.get("harness_timestamp") or event.get("timestamp")
        event_dt = parse_timestamp(timestamp)
        if event_dt is not None:
            if first_event_dt is None:
                first_event_dt = event_dt
                first_event_timestamp = timestamp
            if previous_event_dt is not None:
                gap = (event_dt - previous_event_dt).total_seconds()
                if gap >= 0 and (max_inter_event_gap_seconds is None or gap > max_inter_event_gap_seconds):
                    max_inter_event_gap_seconds = gap
            previous_event_dt = event_dt
            last_event_dt = event_dt
            last_event_timestamp = timestamp
        command = event.get("command_text")
        tool_kind = event.get("tool_kind")
        if isinstance(tool_kind, str) and tool_kind:
            tool_counts[tool_kind] += 1
        if isinstance(command, str) and command.strip():
            command = command.strip()
            if len(commands) < MAX_ACTIVITY_COMMANDS:
                commands.append(command)
            unique_commands_seen.add(command)
            command_prefixes[command.split()[0]] += 1
            lowered = command.lower()
            for name, needles in ACTIVITY_HINTS.items():
                if any(needle in lowered for needle in needles):
                    hint_counts[name] += 1

    # Augment shell-pattern hints with structured tool call counts so that
    # agents using tool APIs (e.g. Claude Read/Skill/Agent tools) are reflected
    # in the same Activity Insights rows as their shell equivalents.
    hint_counts["shell_cat_or_sed"] += tool_counts.get("Read", 0)
    hint_counts["skill_references"] += tool_counts.get("Skill", 0)
    hint_counts["agent_inspect"] += tool_counts.get("Agent", 0)

    return {
        "event_count": len(events),
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
        "tool_counts": dict(tool_counts.most_common()),
        "hint_counts": dict(hint_counts.most_common()),
        "command_count": sum(command_prefixes.values()),
        "unique_command_count": len(unique_commands_seen),
        "command_prefix_counts": dict(command_prefixes.most_common()),
        "commands": commands,
        "commands_truncated": sum(command_prefixes.values()) > len(commands),
        "max_recorded_commands": MAX_ACTIVITY_COMMANDS,
    }


@lru_cache(maxsize=32)
def parse_cached_usage_and_activity(events_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return parse_usage_and_activity_data(Path(events_path))


def cached_usage(events_path: Path) -> dict[str, Any]:
    usage, _activity = parse_cached_usage_and_activity(str(events_path))
    return dict(usage)


def cached_activity(events_path: Path) -> dict[str, Any]:
    _usage, activity = parse_cached_usage_and_activity(str(events_path))
    return dict(activity)


EVENT_PARSERS = {
    "claude_stream_json": normalize_claude_stream_event,
    "codex_jsonl": normalize_jsonl_event,
    "generic_jsonl": normalize_jsonl_event,
}

USAGE_PARSERS = {
    "claude_stream_usage": parse_claude_stream_usage,
    "codex_cumulative_usage": cached_usage,
    "generic_cli_usage": cached_usage,
}

ACTIVITY_PARSERS = {
    "claude_stream_activity": parse_claude_stream_activity,
    "codex_jsonl_activity": cached_activity,
    "generic_jsonl_activity": cached_activity,
}

FINAL_MESSAGE_SOURCE_TYPES = {"file", "structured_event", "stdout_tail", "not_available"}
VALID_FINAL_MESSAGE_PARSER_IDS = {
    "generic_stdout_last_message",
    "generic_structured_event_message",
}


def validate_event_parser(parser_id: str) -> None:
    if parser_id not in EVENT_PARSERS:
        raise ValueError(f"Unknown agent event parser: {parser_id}")


def validate_usage_parser(parser_id: str) -> None:
    if parser_id not in USAGE_PARSERS:
        raise ValueError(f"Unknown agent usage parser: {parser_id}")


def validate_activity_parser(parser_id: str) -> None:
    if parser_id not in ACTIVITY_PARSERS:
        raise ValueError(f"Unknown agent activity parser: {parser_id}")


def validate_final_message_config(source_type: str, parser_id: str | None = None) -> None:
    if source_type not in FINAL_MESSAGE_SOURCE_TYPES:
        raise ValueError(
            f"Unknown final message source_type: {source_type}. "
            f"Valid source types: {', '.join(sorted(FINAL_MESSAGE_SOURCE_TYPES))}"
        )
    if parser_id and parser_id not in VALID_FINAL_MESSAGE_PARSER_IDS:
        raise ValueError(f"Unknown final message parser: {parser_id}")


def normalize_event_with_parser(raw_line: str, parser_id: str) -> dict[str, Any] | None:
    validate_event_parser(parser_id)
    parser = EVENT_PARSERS[parser_id]
    return parser(raw_line)


def parse_usage_from_events(events_path: Path, usage_config: Any) -> dict[str, Any]:
    parser_id = getattr(usage_config, "parser", None) or "generic_cli_usage"
    validate_usage_parser(parser_id)
    parser = USAGE_PARSERS[parser_id]
    usage = parser(events_path)
    usage.setdefault("parser_id", parser_id)
    return usage


def parse_activity_from_events(events_path: Path, activity_config: Any) -> dict[str, Any]:
    parser_id = getattr(activity_config, "parser", None) or "generic_jsonl_activity"
    validate_activity_parser(parser_id)
    parser = ACTIVITY_PARSERS[parser_id]
    activity = parser(events_path)
    activity.setdefault("parser_id", parser_id)
    return activity
