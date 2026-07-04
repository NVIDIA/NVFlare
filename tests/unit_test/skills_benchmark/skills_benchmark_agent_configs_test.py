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

import json
import re
from pathlib import Path

import pytest

BENCHMARK_ROOT = Path(__file__).resolve().parents[3] / "dev_tools" / "agent" / "skills" / "benchmark"


def test_codex_event_normalizer_returns_agent_event():
    from skills.harness.agents.registry import load_agent_adapter

    event = load_agent_adapter("codex").normalize_event('{"type": "turn", "message": "ok"}')

    assert event["type"] == "turn"
    assert event["message"] == "ok"
    assert re.match(r"\d{4}-\d{2}-\d{2}T", event["harness_timestamp"])


def test_known_pending_agent_event_normalizer_fails_fast():
    from skills.harness.agents.registry import load_agent_adapter

    try:
        load_agent_adapter("hermes")
    except ValueError as exc:
        assert "BENCHMARK_AGENT='hermes'" in str(exc)
        assert "known but not implemented" in str(exc)
    else:
        raise AssertionError("unsupported benchmark agent should fail before event parsing")


def test_codex_agent_config_loads_parser_and_classifier_ids():
    from skills.harness.agents.config import AgentConfig

    config_path = BENCHMARK_ROOT / "config" / "agents" / "codex.yaml"

    config = AgentConfig.load(config_path)

    assert config.name == "codex"
    assert config.events.parser == "codex_jsonl"
    assert config.usage.parser == "codex_cumulative_usage"
    assert config.activity.parser == "codex_jsonl_activity"
    assert config.exit_classifier == "stderr_patterns"
    assert config.exit_config["rules"]
    assert "{prompt_text}" not in json.dumps(config.raw)


def test_claude_agent_config_uses_config_dir_and_valid_final_message_source():
    from skills.harness.agents.config import AgentConfig

    config_path = BENCHMARK_ROOT / "config" / "agents" / "claude.yaml"

    config = AgentConfig.load(config_path)

    assert config.agent_home_env == "CLAUDE_CONFIG_DIR"
    assert config.requires_explicit_model is False
    assert config.final_message["source_type"] == "structured_event"
    assert config.events.parser == "claude_stream_json"
    assert config.usage.parser == "claude_stream_usage"
    assert config.activity.parser == "claude_stream_activity"
    assert config.exit_classifier == "stderr_patterns"
    assert config.exit_config["rules"]
    assert config.launch["model_argv"] == ["--model", "{model}"]
    assert config.launch["model_argv_position"] == "before_final_arg"


def test_adapter_template_rejects_positional_placeholders():
    from skills.harness.agents.config import render_string

    try:
        render_string("agent {}", {"agent": "codex"})
    except ValueError as exc:
        assert "positional" in str(exc)
    else:
        raise AssertionError("adapter templates should reject positional placeholders")


def test_adapter_template_rejects_attribute_and_index_access():
    from skills.harness.agents.config import render_string

    for template in ("{workspace_dir.parent}", "{argv[0]}"):
        try:
            render_string(template, {"workspace_dir": "workspace", "argv": ["agent"]})
        except ValueError as exc:
            assert "attribute or index access" in str(exc)
        else:
            raise AssertionError("adapter templates should reject attribute and index access")


def test_claude_adapter_launch_spec_uses_stream_json_without_prompt_text(tmp_path):
    from skills.harness.agents.base import AgentLaunchContext
    from skills.harness.agents.registry import load_agent_adapter

    result_dir = tmp_path / "results"
    workspace_dir = tmp_path / "workspace"
    prompt_file = tmp_path / "prompt.txt"
    result_dir.mkdir()
    workspace_dir.mkdir()
    prompt_file.write_text("Convert this job.\n", encoding="utf-8")
    config = AgentLaunchContext(
        model="claude-test",
        model_was_explicit=True,
        result_dir=result_dir,
        workspace_dir=workspace_dir,
        prompt_file=prompt_file,
        events_dest=result_dir / "agent_events.jsonl",
        stderr_dest=result_dir / "agent_stderr.txt",
        final_message_dest=result_dir / "agent_last_message.txt",
    )

    spec = load_agent_adapter("claude").launch_spec(config)

    rendered_argv = " ".join(spec.argv)
    assert spec.prompt_input_mode == "stdin"
    assert "--output-format" in spec.argv
    assert "stream-json" in spec.argv
    assert "--dangerously-skip-permissions" in spec.sandbox_flags
    assert "Convert this job" not in rendered_argv
    assert "claude-test" in spec.argv
    assert spec.argv[spec.argv.index("--model") + 1] == "claude-test"
    assert spec.argv.index("--model") < spec.argv.index("--print")
    assert spec.argv[-1] == "--print"


def test_claude_adapter_uses_cli_default_model_when_unspecified():
    from skills.harness.agents.registry import load_agent_adapter

    adapter = load_agent_adapter("claude")

    assert adapter.model_from_env({}) == "unspecified_default"
    assert adapter.model_was_explicit({}) is False


def test_runtime_env_uses_agent_run_config_model_explicit_field():
    from skills.harness.agents.registry import load_agent_adapter

    class Config:
        agent_model = "claude-test"
        agent_model_was_explicit = True

    env = load_agent_adapter("claude").runtime_env(Config())

    assert env["BENCHMARK_AGENT_MODEL"] == "claude-test"


def test_runtime_env_omits_agent_model_when_not_explicit():
    from skills.harness.agents.registry import load_agent_adapter

    class Config:
        agent_model = "unspecified_default"
        agent_model_was_explicit = False

    env = load_agent_adapter("claude").runtime_env(Config())

    assert "BENCHMARK_AGENT_MODEL" not in env


def test_claude_launch_spec_omits_model_flag_when_not_explicit(tmp_path):
    from skills.harness.agents.base import AgentLaunchContext
    from skills.harness.agents.registry import load_agent_adapter

    result_dir = tmp_path / "results"
    workspace_dir = tmp_path / "workspace"
    prompt_file = tmp_path / "prompt.txt"
    result_dir.mkdir()
    workspace_dir.mkdir()
    prompt_file.write_text("Convert this job.\n", encoding="utf-8")
    context = AgentLaunchContext(
        model="unspecified_default",
        model_was_explicit=False,
        result_dir=result_dir,
        workspace_dir=workspace_dir,
        prompt_file=prompt_file,
        events_dest=result_dir / "agent_events.jsonl",
        stderr_dest=result_dir / "agent_stderr.txt",
        final_message_dest=result_dir / "agent_last_message.txt",
    )

    spec = load_agent_adapter("claude").launch_spec(context)

    assert "--model" not in spec.argv
    assert spec.argv[-1] == "--print"


def test_claude_stream_parser_normalizes_event_usage_and_activity(tmp_path):
    from skills.harness.agents.registry import load_agent_adapter

    adapter = load_agent_adapter("claude")
    events_path = tmp_path / "agent_events.jsonl"
    raw_events = [
        {
            "type": "system",
            "subtype": "init",
            "session_id": "session-1",
        },
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "running"},
                    {"type": "tool_use", "name": "Bash", "input": {"command": "python job.py --export"}},
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "result": "Final AUROC 0.75",
            "total_cost_usd": 0.01,
            "message": {
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                }
            },
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 7,
            },
        },
    ]
    with events_path.open("w", encoding="utf-8") as stream:
        for raw_event in raw_events:
            normalized = adapter.normalize_event(json.dumps(raw_event))
            stream.write(json.dumps(normalized) + "\n")

    usage = adapter.parse_usage(events_path)
    activity = adapter.parse_activity(events_path)

    assert usage["parser_id"] == "claude_stream_usage"
    assert usage["total_tokens"] == 130
    assert usage["cache_tokens"] == 10
    assert usage["cost"] == 0.01
    assert usage["parser_warnings"] == []
    assert usage["total_cost_usd"] == 0.01
    assert activity["parser_id"] == "claude_stream_activity"
    assert activity["event_types"]["result.success"] == 1
    assert activity["tool_counts"]["Bash"] == 1
    assert activity["commands"] == ["python job.py --export"]
    result_event = json.loads(events_path.read_text(encoding="utf-8").splitlines()[-1])
    assert result_event["final_message"] == "Final AUROC 0.75"
    assistant_event = json.loads(events_path.read_text(encoding="utf-8").splitlines()[1])
    assert assistant_event.get("command_text") == "python job.py --export"
    assert "command" not in assistant_event


def test_generic_event_usage_and_activity_share_cached_parse(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from skills.harness.agents import parsers

    events_path = tmp_path / "agent_events.jsonl"
    events_path.write_text("{}\n", encoding="utf-8")
    calls = {"count": 0}

    def counted_parse(path):
        calls["count"] += 1
        assert path == events_path
        return {"total_tokens": 7}, {"event_count": 1}

    parsers.parse_cached_usage_and_activity.cache_clear()
    monkeypatch.setattr(parsers, "parse_usage_and_activity_data", counted_parse)

    usage = parsers.parse_usage_from_events(events_path, SimpleNamespace(parser="codex_cumulative_usage"))
    activity = parsers.parse_activity_from_events(events_path, SimpleNamespace(parser="codex_jsonl_activity"))

    assert usage == {"total_tokens": 7, "parser_id": "codex_cumulative_usage"}
    assert activity == {"event_count": 1, "parser_id": "codex_jsonl_activity"}
    assert calls["count"] == 1
    parsers.parse_cached_usage_and_activity.cache_clear()


def test_claude_stream_usage_falls_back_when_result_usage_is_zero(tmp_path):
    from skills.harness.agents.registry import load_agent_adapter

    adapter = load_agent_adapter("claude")
    events_path = tmp_path / "agent_events.jsonl"
    raw_events = [
        {
            "type": "assistant",
            "message": {
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_creation_input_tokens": 2,
                    "cache_read_input_tokens": 3,
                }
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        },
    ]
    with events_path.open("w", encoding="utf-8") as stream:
        for raw_event in raw_events:
            stream.write(json.dumps(adapter.normalize_event(json.dumps(raw_event))) + "\n")

    usage = adapter.parse_usage(events_path)

    assert usage["total_tokens"] == 20
    assert usage["cache_tokens"] == 5
    assert "no nonzero token fields" in usage["parser_warnings"][0]


def test_claude_stream_usage_accumulates_multiple_result_events(tmp_path):
    from skills.harness.agents.registry import load_agent_adapter

    adapter = load_agent_adapter("claude")
    events_path = tmp_path / "agent_events.jsonl"
    raw_events = [
        {
            "type": "assistant",
            "message": {
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_creation_input_tokens": 10,
                    "cache_read_input_tokens": 20,
                }
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 1,
                "cache_read_input_tokens": 2,
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "usage": {
                "input_tokens": 20,
                "output_tokens": 7,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 4,
            },
        },
    ]
    with events_path.open("w", encoding="utf-8") as stream:
        for raw_event in raw_events:
            stream.write(json.dumps(adapter.normalize_event(json.dumps(raw_event))) + "\n")

    usage = adapter.parse_usage(events_path)

    assert usage["total_tokens"] == 34
    assert usage["cache_tokens"] == 7
    assert usage["result_usage_objects_seen"] == 2
    assert "final cumulative result usage was used" in usage["parser_warnings"][0]


def test_claude_stream_parser_ignores_non_shell_tool_command_fields(tmp_path):
    from skills.harness.agents.registry import load_agent_adapter

    adapter = load_agent_adapter("claude")
    raw_event = {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "tool_use", "name": "Notebook", "input": {"command": "not a shell command"}},
            ],
        },
    }

    normalized = adapter.normalize_event(json.dumps(raw_event))

    assert "command_text" not in normalized


def test_claude_stream_parser_uses_exact_shell_tool_allowlist():
    from skills.harness.agents.registry import load_agent_adapter

    adapter = load_agent_adapter("claude")
    raw_event = {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "tool_use", "name": "BashRunner", "input": {"command": "python job.py"}},
            ],
        },
    }

    normalized = adapter.normalize_event(json.dumps(raw_event))

    assert "command_text" not in normalized


def test_claude_final_message_source_materializes_structured_result_event(tmp_path):
    from collections import deque

    from skills.harness.agents.registry import load_agent_adapter

    agent_run = pytest.importorskip("skills.harness.container.agent_run")
    AgentRunConfig = agent_run.AgentRunConfig
    materialize_final_message = agent_run.materialize_final_message

    adapter = load_agent_adapter("claude")
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    config = AgentRunConfig(
        mode="with_skills",
        use_preinstalled_skills=True,
        job_input_dir=tmp_path / "job",
        result_dir=result_dir,
        records_dir=result_dir / "records",
        run_root=tmp_path / "run",
        prompt_source=tmp_path / "prompt.txt",
        progress_interval_seconds=0,
        sdk_image_kind="test-skills",
        agent="claude",
        agent_model="claude-test",
        agent_home=tmp_path / ".claude",
        agent_model_was_explicit=True,
    )
    with config.agent_events_path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(adapter.normalize_event(json.dumps({"type": "assistant", "message": {}}))) + "\n")
        stream.write(
            json.dumps(
                adapter.normalize_event(
                    json.dumps({"type": "result", "subtype": "success", "result": "Final Claude response"})
                )
            )
            + "\n"
        )

    materialize_final_message(config, adapter, deque())

    assert config.agent_last_message_path.read_text(encoding="utf-8") == "Final Claude response"
    metadata = json.loads((result_dir / "final_message_source.json").read_text(encoding="utf-8"))
    assert metadata["source_type"] == "structured_event"
    assert metadata["status"] == "materialized"
    assert metadata["parser_warnings"]


def test_structured_final_message_not_read_when_stdout_reader_active(tmp_path):
    from collections import deque

    from skills.harness.agents.registry import load_agent_adapter

    agent_run = pytest.importorskip("skills.harness.container.agent_run")
    AgentRunConfig = agent_run.AgentRunConfig
    materialize_final_message = agent_run.materialize_final_message

    adapter = load_agent_adapter("claude")
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    config = AgentRunConfig(
        mode="with_skills",
        use_preinstalled_skills=True,
        job_input_dir=tmp_path / "job",
        result_dir=result_dir,
        records_dir=result_dir / "records",
        run_root=tmp_path / "run",
        prompt_source=tmp_path / "prompt.txt",
        progress_interval_seconds=0,
        sdk_image_kind="test-skills",
        agent="claude",
        agent_model="claude-test",
        agent_home=tmp_path / ".claude",
        agent_model_was_explicit=True,
    )
    with config.agent_events_path.open("w", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                adapter.normalize_event(
                    json.dumps({"type": "result", "subtype": "success", "result": "Final Claude response"})
                )
            )
            + "\n"
        )

    materialize_final_message(config, adapter, deque(), stdout_tail_truncated=True)

    assert config.agent_last_message_path.read_text(encoding="utf-8") == ""
    metadata = json.loads((result_dir / "final_message_source.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "missing"
    assert metadata["stdout_tail_truncated"] is True
    assert "still active" in metadata["message"]


def test_launch_spec_metadata_records_sandbox_flags_and_bypass_reason(tmp_path):
    from skills.harness.agents.base import AgentLaunchSpec, SkillExposureResult

    agent_run = pytest.importorskip("skills.harness.container.agent_run")
    AgentRunConfig = agent_run.AgentRunConfig
    write_launch_spec_metadata = agent_run.write_launch_spec_metadata

    result_dir = tmp_path / "results"
    result_dir.mkdir()
    config = AgentRunConfig(
        mode="with_skills",
        use_preinstalled_skills=True,
        job_input_dir=tmp_path / "job",
        result_dir=result_dir,
        records_dir=result_dir / "records",
        run_root=tmp_path / "run",
        prompt_source=tmp_path / "prompt.txt",
        progress_interval_seconds=0,
        sdk_image_kind="test-skills",
        agent="claude",
        agent_model="claude-test",
        agent_home=tmp_path / ".claude",
        agent_model_was_explicit=True,
    )
    launch = AgentLaunchSpec(
        argv=["claude", "--dangerously-skip-permissions", "--print"],
        cwd=tmp_path,
        prompt_file=tmp_path / "prompt.txt",
        prompt_input_mode="stdin",
        stdout_events_dest=result_dir / "agent_events.jsonl",
        stderr_dest=result_dir / "agent_stderr.txt",
        final_message_dest=result_dir / "agent_last_message.txt",
        environment={"CLAUDE_CONFIG_DIR": "/workspace/.claude"},
        sandbox_flags=["--dangerously-skip-permissions"],
        bypass_reason="isolated benchmark container",
        launch_timeout=10,
    )
    skill_exposure = SkillExposureResult(
        status="prepared",
        mechanism_type="launch_flag",
        launch_args=["--add-dir", "/workspace/.claude/skills"],
        environment={"CLAUDE_CONFIG_DIR": "/workspace/.claude"},
    )

    write_launch_spec_metadata(
        config,
        [*launch.argv, *skill_exposure.launch_args],
        {**launch.environment, **skill_exposure.environment},
        launch,
        skill_exposure,
    )

    metadata = json.loads((result_dir / "launch_spec_metadata.json").read_text(encoding="utf-8"))
    assert metadata["sandbox_flags"] == ["--dangerously-skip-permissions"]
    assert metadata["bypass_reason"] == "isolated benchmark container"
    assert metadata["skill_launch_args"] == ["--add-dir", "/workspace/.claude/skills"]
    assert metadata["environment_keys"] == ["CLAUDE_CONFIG_DIR"]


def test_claude_exit_classifier_detects_auth_and_model_failures(tmp_path):
    from skills.harness.agents.registry import load_agent_adapter

    adapter = load_agent_adapter("claude")
    stderr = tmp_path / "stderr.txt"
    stderr.write_text("Authentication failed: please login to Claude Code.\n", encoding="utf-8")

    auth_summary = adapter.exit_summary(1, stderr)
    stderr.write_text("", encoding="utf-8")
    last_message = tmp_path / "agent_last_message.txt"
    last_message.write_text("Not logged in - Please run /login\n", encoding="utf-8")
    auth_from_last_message = adapter.exit_summary(1, stderr, evidence_paths=(last_message,))
    stderr.write_text("Model is not supported in this account.\n", encoding="utf-8")
    model_summary = adapter.exit_summary(1, stderr)
    stderr.write_text("Permission approval is required.\n", encoding="utf-8")
    permission_summary = adapter.exit_summary(1, stderr)
    stderr.write_text(
        "Sandbox disabled: sandbox is enabled but dependencies are missing: "
        "bubblewrap (bwrap) not installed, socat not installed.\n",
        encoding="utf-8",
    )
    sandbox_dependency_summary = adapter.exit_summary(1, stderr)

    assert auth_summary["classifier"] == "stderr_patterns"
    assert auth_summary["failure_category"] == "agent_auth_failure"
    assert auth_from_last_message["failure_category"] == "agent_auth_failure"
    assert "Not logged in" in auth_from_last_message["classification_excerpt"]
    assert model_summary["failure_category"] == "agent_model_unsupported"
    assert permission_summary["failure_category"] == "agent_sandbox_or_approval_failure"
    assert sandbox_dependency_summary["failure_category"] == "agent_sandbox_or_approval_failure"


def test_codex_exit_classifier_prioritizes_missing_cli_over_stderr_text(tmp_path):
    from skills.harness.agents.registry import load_agent_adapter

    adapter = load_agent_adapter("codex")
    stderr = tmp_path / "stderr.txt"
    stderr.write_text("Authentication failed because command was not found.\n", encoding="utf-8")

    summary = adapter.exit_summary(127, stderr)

    assert summary["failure_category"] == "agent_cli_missing"


def test_agent_config_rejects_unknown_parser_id(tmp_path):
    from skills.harness.agents.config import AgentConfig

    source_path = BENCHMARK_ROOT / "config" / "agents" / "codex.yaml"
    config_path = tmp_path / "bad_parser.yaml"
    config_path.write_text(source_path.read_text(encoding="utf-8").replace("codex_jsonl", "missing_parser", 1))

    try:
        AgentConfig.load(config_path)
    except ValueError as exc:
        assert "Unknown agent event parser: missing_parser" in str(exc)
    else:
        raise AssertionError("unknown adapter event parser should fail during config load")


def test_agent_config_rejects_unknown_exit_classifier(tmp_path):
    from skills.harness.agents.config import AgentConfig

    source_path = BENCHMARK_ROOT / "config" / "agents" / "codex.yaml"
    config_path = tmp_path / "bad_exit.yaml"
    config_path.write_text(
        source_path.read_text(encoding="utf-8").replace("classifier: stderr_patterns", "classifier: bad")
    )

    try:
        AgentConfig.load(config_path)
    except ValueError as exc:
        assert "Unknown agent exit classifier: bad" in str(exc)
    else:
        raise AssertionError("unknown adapter exit classifier should fail during config load")


def test_agent_config_rejects_unknown_final_message_source_type(tmp_path):
    from skills.harness.agents.config import AgentConfig

    source_path = BENCHMARK_ROOT / "config" / "agents" / "codex.yaml"
    config_path = tmp_path / "bad_final_source.yaml"
    config_path.write_text(source_path.read_text(encoding="utf-8").replace("source_type: file", "source_type: bad"))

    try:
        AgentConfig.load(config_path)
    except ValueError as exc:
        assert "Unknown final message source_type: bad" in str(exc)
    else:
        raise AssertionError("unknown final message source type should fail during config load")


def test_agent_config_rejects_unknown_final_message_parser(tmp_path):
    from skills.harness.agents.config import AgentConfig

    source_path = BENCHMARK_ROOT / "config" / "agents" / "claude.yaml"
    config_path = tmp_path / "bad_final_parser.yaml"
    config_path.write_text(
        source_path.read_text(encoding="utf-8").replace(
            "parser: generic_structured_event_message", "parser: missing_final_parser"
        )
    )

    try:
        AgentConfig.load(config_path)
    except ValueError as exc:
        assert "Unknown final message parser: missing_final_parser" in str(exc)
    else:
        raise AssertionError("unknown final message parser should fail during config load")


def test_agent_config_rejects_model_argv_without_explicit_position(tmp_path):
    from skills.harness.agents.config import AgentConfig

    source_path = BENCHMARK_ROOT / "config" / "agents" / "codex.yaml"
    config_path = tmp_path / "bad_model_argv.yaml"
    config_path.write_text(
        source_path.read_text(encoding="utf-8").replace("  model_argv_position: before_stdin_sentinel\n", ""),
        encoding="utf-8",
    )

    try:
        AgentConfig.load(config_path)
    except ValueError as exc:
        assert "launch.model_argv_position" in str(exc)
    else:
        raise AssertionError("model_argv must declare an explicit placement policy")


def test_agent_config_rejects_append_model_argv_for_stdin_prompt(tmp_path):
    from skills.harness.agents.config import AgentConfig

    source_path = BENCHMARK_ROOT / "config" / "agents" / "codex.yaml"
    config_path = tmp_path / "bad_model_argv_position.yaml"
    config_path.write_text(
        source_path.read_text(encoding="utf-8").replace(
            "model_argv_position: before_stdin_sentinel", "model_argv_position: append"
        ),
        encoding="utf-8",
    )

    try:
        AgentConfig.load(config_path)
    except ValueError as exc:
        assert "launch.model_argv_position=append is not valid with stdin prompt delivery" in str(exc)
    else:
        raise AssertionError("stdin model_argv must not append after the prompt sentinel")


def test_agent_config_rejects_unknown_when_condition(tmp_path):
    from skills.harness.agents.config import AgentConfig

    source_path = BENCHMARK_ROOT / "config" / "agents" / "codex.yaml"
    config_path = tmp_path / "bad_when.yaml"
    config_path.write_text(
        source_path.read_text(encoding="utf-8").replace(
            "  argv:\n",
            "  argv:\n" "    - when: use_stict_mode\n" "      args:\n" "        - --strict\n",
        ),
        encoding="utf-8",
    )

    try:
        AgentConfig.load(config_path)
    except ValueError as exc:
        assert "unknown condition 'use_stict_mode'" in str(exc)
    else:
        raise AssertionError("unknown when conditions should fail during adapter config load")


def test_agent_adapter_cache_can_be_cleared_for_tests():
    from skills.harness.agents.registry import clear_agent_adapter_cache, load_agent_adapter

    first = load_agent_adapter("codex")
    clear_agent_adapter_cache()
    second = load_agent_adapter("codex")

    assert first is not second


def test_codex_adapter_build_args_come_from_profile():
    from skills.harness.agents.registry import load_agent_adapter

    adapter = load_agent_adapter("codex")
    build_args = adapter.build_args()
    assert build_args["BENCHMARK_DOCKER_AGENT"] == "codex"
    assert build_args["BENCHMARK_AGENT_HOME"] == "/workspace/.codex"
    assert build_args["AGENT_CLI_NAME"] == "codex"
    assert build_args["AGENT_INSTALL_COMMAND"] == 'npm install -g "@openai/codex@0.137.0"'
    assert build_args["AGENT_VERSION_COMMAND"] == "codex --version"


def test_claude_adapter_build_auth_and_skill_exposure_contract(tmp_path):
    from types import SimpleNamespace

    from skills.harness.agents.base import SkillExposureContext
    from skills.harness.agents.registry import load_agent_adapter

    adapter = load_agent_adapter("claude")
    build_args = adapter.build_args()

    assert build_args["BENCHMARK_DOCKER_AGENT"] == "claude"
    assert build_args["BENCHMARK_AGENT_HOME"] == "/workspace/.claude"
    assert build_args["AGENT_CLI_NAME"] == "claude"
    assert build_args["AGENT_INSTALL_COMMAND"] == 'npm install -g "@anthropic-ai/claude-code@2.1.170"'
    assert build_args["AGENT_VERSION_COMMAND"] == "claude --version"
    assert "ANTHROPIC_API_KEY" in adapter.passthrough_env_names()

    host_home = tmp_path / ".claude"
    host_home.mkdir()
    (host_home / ".credentials.json").write_text("{}\n", encoding="utf-8")
    mounts = adapter.auth_mounts(SimpleNamespace(host_agent_home=host_home))
    assert any(mount.container_path == "/workspace/.claude/.credentials.json" for mount in mounts)

    spec = adapter.skill_exposure(
        SkillExposureContext(
            result_dir=tmp_path / "results",
            container_home=tmp_path / ".claude-container",
            mode="with_skills",
            skills_enabled=True,
            sdk_image_kind="test-skills",
        )
    )
    assert spec.mechanism_type == "launch_flag"
    assert spec.launch_args == ["--add-dir", str(tmp_path / ".claude-container" / "skills")]
    assert spec.metadata_files == []


def test_shared_dockerfile_uses_profile_agent_install_commands():
    dockerfile = (BENCHMARK_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")

    assert "AGENT_INSTALL_COMMAND" in dockerfile
    assert "AGENT_VERSION_COMMAND" in dockerfile
    assert "bubblewrap" in dockerfile
    assert "socat" in dockerfile
    assert 'BENCHMARK_DOCKER_AGENT}" = "codex"' not in dockerfile
    assert "@openai/codex" not in dockerfile
    assert "@anthropic-ai/claude-code" not in dockerfile
    assert "/workspace/.codex /workspace/.claude" not in dockerfile


def test_sdk_skills_setup_default_agent_home_is_neutral(monkeypatch):
    from skills.harness.container import sdk_skills_setup

    monkeypatch.delenv("BENCHMARK_AGENT_HOME", raising=False)

    assert sdk_skills_setup.agent_home() == Path("/workspace/agent-home")


def test_adapter_auth_mounts_reject_path_components(tmp_path):
    from types import SimpleNamespace

    import yaml
    from skills.harness.agents.config import ConfigurableAgentAdapter

    config_path = tmp_path / "bad-agent.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "bad",
                "display_name": "Bad",
                "agent_home_env": "BAD_HOME",
                "container_home": "/workspace/.bad",
                "launch": {
                    "argv": ["bad", "{prompt_file}"],
                    "prompt_input_mode": "file_arg",
                },
                "skill_exposure": {"mechanism_type": "none"},
                "final_message": {"source_type": "not_available"},
                "events": {"parser": "generic_jsonl"},
                "usage": {"parser": "generic_cli_usage"},
                "activity": {"parser": "generic_jsonl_activity"},
                "exit": {"classifier": "generic_cli"},
                "auth": {"files": [{"source": "../token.json", "target": "token.json"}]},
            }
        ),
        encoding="utf-8",
    )
    adapter = ConfigurableAgentAdapter(config_path)

    try:
        adapter.auth_mounts(SimpleNamespace(host_agent_home=tmp_path))
    except ValueError as exc:
        assert "auth file source must be a file name" in str(exc)
    else:
        raise AssertionError("auth mount source paths should not escape the configured agent home")


def test_agent_config_rejects_unknown_skill_exposure_mechanism(tmp_path):
    import yaml
    from skills.harness.agents.config import AgentConfig

    config_path = tmp_path / "bad-agent.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "bad",
                "display_name": "Bad",
                "agent_home_env": "BAD_HOME",
                "container_home": "/workspace/.bad",
                "launch": {
                    "argv": ["bad", "{prompt_file}"],
                    "prompt_input_mode": "file_arg",
                },
                "skill_exposure": {"mechanism_type": "magic_marketplace"},
                "final_message": {"source_type": "not_available"},
                "events": {"parser": "generic_jsonl"},
                "usage": {"parser": "generic_cli_usage"},
                "activity": {"parser": "generic_jsonl_activity"},
                "exit": {"classifier": "generic_cli"},
            }
        ),
        encoding="utf-8",
    )

    try:
        AgentConfig.load(config_path)
    except ValueError as exc:
        assert "skill_exposure.mechanism_type" in str(exc)
        assert "preinstalled_home" in str(exc)
        assert "launch_flag" in str(exc)
    else:
        raise AssertionError("adapter configs should reject unsupported skill exposure mechanisms")


def test_agent_config_rejects_unsafe_legacy_artifact_prefix(tmp_path):
    import yaml
    from skills.harness.agents.config import AgentConfig

    config_path = tmp_path / "bad-agent.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "bad",
                "display_name": "Bad",
                "agent_home_env": "BAD_HOME",
                "container_home": "/workspace/.bad",
                "legacy_artifact_prefixes": ["../../escape"],
                "launch": {
                    "argv": ["bad", "{prompt_file}"],
                    "prompt_input_mode": "file_arg",
                },
                "skill_exposure": {"mechanism_type": "none"},
                "final_message": {"source_type": "not_available"},
                "events": {"parser": "generic_jsonl"},
                "usage": {"parser": "generic_cli_usage"},
                "activity": {"parser": "generic_jsonl_activity"},
                "exit": {"classifier": "generic_cli"},
            }
        ),
        encoding="utf-8",
    )

    try:
        AgentConfig.load(config_path)
    except ValueError as exc:
        assert "legacy_artifact_prefixes" in str(exc)
        assert "bare file prefixes" in str(exc)
    else:
        raise AssertionError("legacy artifact prefixes must not contain path separators")
