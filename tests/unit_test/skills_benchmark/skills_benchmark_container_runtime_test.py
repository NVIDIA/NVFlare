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
import sys
from pathlib import Path


def test_agent_config_rejects_prompt_text_placeholder(tmp_path):
    from skills.harness.agents.config import AgentConfig

    config_path = tmp_path / "unsafe.yaml"
    config_path.write_text(
        json.dumps(
            {
                "name": "unsafe",
                "display_name": "Unsafe Agent",
                "default_model": "default",
                "agent_home_env": "UNSAFE_HOME",
                "container_home": "/workspace/.unsafe",
                "launch": {"argv": ["unsafe", "run", "{prompt_text}"]},
            }
        ),
        encoding="utf-8",
    )

    try:
        AgentConfig.load(config_path)
    except ValueError as exc:
        assert "must not use {prompt_text}" in str(exc)
    else:
        raise AssertionError("agent adapter config must reject prompt_text injection paths")


def test_agent_config_rejects_file_arg_without_prompt_file_placeholder(tmp_path):
    from skills.harness.agents.config import AgentConfig

    config_path = tmp_path / "missing_prompt_file.yaml"
    config_path.write_text(
        json.dumps(
            {
                "name": "unsafe",
                "display_name": "Unsafe Agent",
                "default_model": "default",
                "agent_home_env": "UNSAFE_HOME",
                "container_home": "/workspace/.unsafe",
                "launch": {"argv": ["unsafe", "run"], "prompt_input_mode": "file_arg"},
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
        assert "must include {prompt_file}" in str(exc)
    else:
        raise AssertionError("file_arg prompt delivery must include the prompt file in argv")


def test_codex_adapter_launch_spec_uses_prompt_file_without_prompt_text(tmp_path):
    from skills.harness.agents.base import AgentLaunchContext
    from skills.harness.agents.registry import load_agent_adapter

    result_dir = tmp_path / "results"
    workspace_dir = tmp_path / "workspace"
    prompt_file = tmp_path / "prompt.txt"
    result_dir.mkdir()
    workspace_dir.mkdir()
    prompt_file.write_text("Convert this job. Do not leak prompt text into argv.\n", encoding="utf-8")
    config = AgentLaunchContext(
        model="test-model",
        model_was_explicit=True,
        result_dir=result_dir,
        workspace_dir=workspace_dir,
        prompt_file=prompt_file,
        events_dest=result_dir / "agent_events.jsonl",
        stderr_dest=result_dir / "agent_stderr.txt",
        final_message_dest=result_dir / "agent_last_message.txt",
    )

    spec = load_agent_adapter("codex").launch_spec(config)

    rendered_argv = " ".join(spec.argv)
    assert spec.prompt_file == prompt_file
    assert spec.prompt_input_mode == "stdin"
    assert spec.final_message_dest == result_dir / "agent_last_message.txt"
    assert "{prompt_text}" not in rendered_argv
    assert "Convert this job" not in rendered_argv
    assert spec.argv[-1] == "-"
    assert spec.argv[-3:] == ["-m", "test-model", "-"]
    assert "--dangerously-bypass-approvals-and-sandbox" in spec.sandbox_flags
    assert spec.bypass_reason


def test_codex_adapter_runtime_env_sets_generic_agent_model_and_home(tmp_path):
    from types import SimpleNamespace

    from skills.harness.agents.registry import load_agent_adapter

    agent_home = tmp_path / ".codex"
    env = load_agent_adapter("codex").runtime_env(
        SimpleNamespace(
            agent_model="test-model",
            agent_home=agent_home,
            model_was_explicit=True,
        )
    )

    assert env["BENCHMARK_AGENT"] == "codex"
    assert env["BENCHMARK_AGENT_MODEL"] == "test-model"
    assert env["CODEX_HOME"] == "/workspace/.codex"


def test_container_config_uses_generic_agent_model_and_home(monkeypatch):
    from skills.harness.container.agent_run import AgentRunConfig

    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setenv("BENCHMARK_AGENT", "codex")
    monkeypatch.setenv("BENCHMARK_AGENT_MODEL", "generic-model")
    monkeypatch.setenv("BENCHMARK_AGENT_HOME", "/workspace/agent-home")

    config = AgentRunConfig.from_env()

    assert config.agent_model == "generic-model"
    assert config.agent_home == Path("/workspace/agent-home")
    assert not hasattr(config, "codex_home")


def test_container_config_requires_benchmark_agent(monkeypatch):
    from skills.harness.container.agent_run import AgentRunConfig

    monkeypatch.delenv("BENCHMARK_AGENT", raising=False)

    try:
        AgentRunConfig.from_env()
    except SystemExit as exc:
        assert "BENCHMARK_AGENT is required" in str(exc)
    else:
        raise AssertionError("in-container config should require explicit BENCHMARK_AGENT")


def test_container_config_rejects_invalid_progress_interval(monkeypatch):
    from skills.harness.container.agent_run import AgentRunConfig

    monkeypatch.setenv("BENCHMARK_AGENT", "codex")
    monkeypatch.setenv("PROGRESS_INTERVAL_SECONDS", "fast")

    try:
        AgentRunConfig.from_env()
    except SystemExit as exc:
        assert "PROGRESS_INTERVAL_SECONDS must be an integer" in str(exc)
    else:
        raise AssertionError("invalid progress interval should fail with a clean config error")


def test_agent_subprocess_env_hides_harness_controls_and_adapter_model_env(monkeypatch):
    from types import SimpleNamespace

    from skills.harness.container.agent_run import agent_subprocess_env

    adapter = SimpleNamespace(model_env_names=lambda: ("CUSTOM_AGENT_MODEL",))
    monkeypatch.setenv("MODE", "with_skills")
    monkeypatch.setenv("JOB_INPUT_DIR", "/workspace/input")
    monkeypatch.setenv("BENCHMARK_AGENT", "codex")
    monkeypatch.setenv("BENCHMARK_AGENT_MODEL", "generic-model")
    monkeypatch.setenv("CUSTOM_AGENT_MODEL", "custom-model")
    monkeypatch.setenv("AGENT_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("OPENAI_API_KEY", "kept-for-agent-auth")

    env = agent_subprocess_env({"CODEX_HOME": "/workspace/.codex"}, adapter)

    assert "MODE" not in env
    assert "JOB_INPUT_DIR" not in env
    assert "BENCHMARK_AGENT" not in env
    assert "BENCHMARK_AGENT_MODEL" not in env
    assert "CUSTOM_AGENT_MODEL" not in env
    assert "AGENT_TIMEOUT_SECONDS" not in env
    assert env["OPENAI_API_KEY"] == "kept-for-agent-auth"
    assert env["CODEX_HOME"] == "/workspace/.codex"


def test_progress_writer_serializes_concurrent_file_writes(tmp_path):
    import threading

    from skills.harness.container.progress import ProgressWriter

    writer = ProgressWriter("with_skills", 0, tmp_path / "progress.jsonl")
    threads = [threading.Thread(target=writer.write, args=(f"phase-{index}", "running", index)) for index in range(20)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    rows = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 20
    assert {row["phase"] for row in rows} == {f"phase-{index}" for index in range(20)}


def test_launch_subprocess_argv_wraps_login_shell_command():
    from skills.harness.container.agent_run import launch_subprocess_argv

    argv = launch_subprocess_argv(["agent", "run", "prompt with spaces"], login_shell=True)

    assert argv[:3] == ["/bin/bash", "--login", "-c"]
    assert argv[3].startswith("exec agent run")
    assert "'prompt with spaces'" in argv[3]


def test_run_agent_enforces_launch_timeout(tmp_path, monkeypatch):
    from skills.harness.agents.base import AgentLaunchSpec, FinalMessageSource
    from skills.harness.container import agent_run
    from skills.harness.container.agent_run import AGENT_TIMEOUT_EXIT_CODE, AgentRunConfig, ProgressWriter, run_agent

    class TimeoutAdapter:
        def launch_spec(self, config):
            return AgentLaunchSpec(
                argv=[sys.executable, "-c", "import time; time.sleep(5)"],
                cwd=config.workspace_dir,
                prompt_file=config.prompt_file,
                prompt_input_mode="stdin",
                stdout_events_dest=config.events_dest,
                stderr_dest=config.stderr_dest,
                final_message_dest=config.final_message_dest,
                launch_timeout=1,
            )

        def normalize_event(self, raw_line):
            return None

        def final_message_source(self, result_dir):
            return FinalMessageSource(source_type="not_available")

        def model_env_names(self):
            return ()

    result_dir = tmp_path / "results"
    run_root = tmp_path / "run"
    workspace = run_root / "workspace"
    result_dir.mkdir()
    workspace.mkdir(parents=True)
    prompt = result_dir / "prompt.txt"
    prompt.write_text("prompt\n", encoding="utf-8")
    config = AgentRunConfig(
        mode="with_skills",
        use_preinstalled_skills=True,
        job_input_dir=tmp_path / "job",
        result_dir=result_dir,
        records_dir=result_dir / "records",
        run_root=run_root,
        prompt_source=prompt,
        progress_interval_seconds=0,
        sdk_image_kind="test-skills",
        agent="test",
        agent_model="test-model",
        agent_home=tmp_path / ".agent",
        agent_model_was_explicit=False,
    )
    monkeypatch.setattr(agent_run, "load_agent_adapter", lambda _agent: TimeoutAdapter())

    _start, _end, exit_code = run_agent(config, ProgressWriter(config.mode, 0, config.progress_log_path))

    assert exit_code == AGENT_TIMEOUT_EXIT_CODE
    assert "timed out after 1 seconds" in config.agent_stderr_path.read_text(encoding="utf-8")


def test_run_agent_stops_stdout_reader_before_events_file_closes(tmp_path, monkeypatch):
    import threading
    import time

    from skills.harness.agents.base import AgentLaunchSpec, FinalMessageSource
    from skills.harness.container import agent_run
    from skills.harness.container.agent_run import AgentRunConfig, ProgressWriter, run_agent

    class GuardedEventsDest:
        def __init__(self):
            self.closed = False
            self.write_after_close = False
            self.writes: list[str] = []

        def __str__(self):
            return "guarded-agent-events.jsonl"

        def open(self, *args, **kwargs):
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.closed = True
            return False

        def write(self, value):
            if self.closed:
                self.write_after_close = True
            self.writes.append(value)

        def flush(self):
            return None

    normalize_started = threading.Event()
    release_normalize = threading.Event()
    guarded_events = GuardedEventsDest()

    class SlowNormalizeAdapter:
        def launch_spec(self, config):
            return AgentLaunchSpec(
                argv=[sys.executable, "-c", 'print(\'{"type":"event"}\')'],
                cwd=config.workspace_dir,
                prompt_file=config.prompt_file,
                prompt_input_mode="stdin",
                stdout_events_dest=guarded_events,
                stderr_dest=config.stderr_dest,
                final_message_dest=config.final_message_dest,
            )

        def normalize_event(self, raw_line):
            normalize_started.set()
            release_normalize.wait(timeout=2)
            return {"type": "event"}

        def final_message_source(self, result_dir):
            return FinalMessageSource(source_type="not_available")

        def model_env_names(self):
            return ()

    result_dir = tmp_path / "results"
    run_root = tmp_path / "run"
    result_dir.mkdir()
    (run_root / "workspace").mkdir(parents=True)
    prompt = result_dir / "prompt.txt"
    prompt.write_text("prompt\n", encoding="utf-8")
    config = AgentRunConfig(
        mode="with_skills",
        use_preinstalled_skills=True,
        job_input_dir=tmp_path / "job",
        result_dir=result_dir,
        records_dir=result_dir / "records",
        run_root=run_root,
        prompt_source=prompt,
        progress_interval_seconds=0,
        sdk_image_kind="test-skills",
        agent="test",
        agent_model="test-model",
        agent_home=tmp_path / ".agent",
        agent_model_was_explicit=False,
    )
    monkeypatch.setattr(agent_run, "AGENT_TERMINATE_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(agent_run, "load_agent_adapter", lambda _agent: SlowNormalizeAdapter())

    _start, _end, exit_code = run_agent(config, ProgressWriter(config.mode, 0, config.progress_log_path))
    assert normalize_started.is_set()
    assert exit_code == 0

    release_normalize.set()
    time.sleep(0.1)

    assert guarded_events.closed is True
    assert guarded_events.write_after_close is False


def test_run_agent_closes_stdout_pipe_when_reader_is_blocked(tmp_path, monkeypatch):
    import threading

    from skills.harness.agents.base import AgentLaunchSpec, FinalMessageSource
    from skills.harness.container import agent_run
    from skills.harness.container.agent_run import AgentRunConfig, ProgressWriter, run_agent

    class BlockingStdout:
        def __init__(self):
            self.closed = False
            self.close_event = threading.Event()

        def __iter__(self):
            return self

        def __next__(self):
            self.close_event.wait(timeout=2)
            raise StopIteration

        def close(self):
            self.closed = True
            self.close_event.set()

    blocked_stdout = BlockingStdout()

    class FakeProcess:
        stdout = blocked_stdout

        def wait(self, timeout=None):
            return 0

    class BlockedReaderAdapter:
        def launch_spec(self, config):
            return AgentLaunchSpec(
                argv=[sys.executable, "-c", "pass"],
                cwd=config.workspace_dir,
                prompt_file=config.prompt_file,
                prompt_input_mode="stdin",
                stdout_events_dest=config.events_dest,
                stderr_dest=config.stderr_dest,
                final_message_dest=config.final_message_dest,
            )

        def normalize_event(self, raw_line):
            return {"type": "event"}

        def final_message_source(self, result_dir):
            return FinalMessageSource(source_type="not_available")

        def model_env_names(self):
            return ()

    result_dir = tmp_path / "results"
    run_root = tmp_path / "run"
    result_dir.mkdir()
    (run_root / "workspace").mkdir(parents=True)
    prompt = result_dir / "prompt.txt"
    prompt.write_text("prompt\n", encoding="utf-8")
    config = AgentRunConfig(
        mode="with_skills",
        use_preinstalled_skills=True,
        job_input_dir=tmp_path / "job",
        result_dir=result_dir,
        records_dir=result_dir / "records",
        run_root=run_root,
        prompt_source=prompt,
        progress_interval_seconds=0,
        sdk_image_kind="test-skills",
        agent="test",
        agent_model="test-model",
        agent_home=tmp_path / ".agent",
        agent_model_was_explicit=False,
    )
    monkeypatch.setattr(agent_run, "AGENT_TERMINATE_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(agent_run, "load_agent_adapter", lambda _agent: BlockedReaderAdapter())
    monkeypatch.setattr(agent_run.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

    _start, _end, exit_code = run_agent(config, ProgressWriter(config.mode, 0, config.progress_log_path))

    assert exit_code == 0
    assert blocked_stdout.closed is True


def test_run_agent_materializes_final_message_before_reader_error(tmp_path, monkeypatch):
    from skills.harness.agents.base import AgentLaunchSpec, FinalMessageSource
    from skills.harness.container import agent_run
    from skills.harness.container.agent_run import AgentRunConfig, ProgressWriter, run_agent

    class ReaderErrorAdapter:
        def launch_spec(self, config):
            return AgentLaunchSpec(
                argv=[sys.executable, "-c", "print('partial final message')"],
                cwd=config.workspace_dir,
                prompt_file=config.prompt_file,
                prompt_input_mode="stdin",
                stdout_events_dest=config.events_dest,
                stderr_dest=config.stderr_dest,
                final_message_dest=config.final_message_dest,
            )

        def normalize_event(self, raw_line):
            raise ValueError("bad event")

        def final_message_source(self, result_dir):
            return FinalMessageSource(source_type="stdout_tail", tail_bytes=100, parser="generic_stdout_last_message")

        def model_env_names(self):
            return ()

    result_dir = tmp_path / "results"
    run_root = tmp_path / "run"
    result_dir.mkdir()
    (run_root / "workspace").mkdir(parents=True)
    prompt = result_dir / "prompt.txt"
    prompt.write_text("prompt\n", encoding="utf-8")
    config = AgentRunConfig(
        mode="with_skills",
        use_preinstalled_skills=True,
        job_input_dir=tmp_path / "job",
        result_dir=result_dir,
        records_dir=result_dir / "records",
        run_root=run_root,
        prompt_source=prompt,
        progress_interval_seconds=0,
        sdk_image_kind="test-skills",
        agent="test",
        agent_model="test-model",
        agent_home=tmp_path / ".agent",
        agent_model_was_explicit=False,
    )
    monkeypatch.setattr(agent_run, "load_agent_adapter", lambda _agent: ReaderErrorAdapter())

    try:
        run_agent(config, ProgressWriter(config.mode, 0, config.progress_log_path))
    except RuntimeError as exc:
        assert "Failed to read agent stdout" in str(exc)
    else:
        raise AssertionError("reader errors should still propagate after final message materialization")

    assert config.agent_last_message_path.read_text(encoding="utf-8") == "partial final message\n"


def test_materialize_final_message_from_stdout_tail(tmp_path):
    from collections import deque

    from skills.harness.agents.base import FinalMessageSource
    from skills.harness.container.agent_run import AgentRunConfig, materialize_final_message

    class StdoutAdapter:
        def final_message_source(self, result_dir):
            return FinalMessageSource(source_type="stdout_tail", tail_bytes=12, parser="generic_stdout_last_message")

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
        agent="test",
        agent_model="test-model",
        agent_home=tmp_path / ".agent",
        agent_model_was_explicit=False,
    )

    materialize_final_message(
        config,
        StdoutAdapter(),
        deque(["first line\n", "final message\n"]),
        stdout_tail_truncated=True,
    )

    assert config.agent_last_message_path.read_text(encoding="utf-8") == "nal message\n"
    metadata = json.loads((result_dir / "final_message_source.json").read_text(encoding="utf-8"))
    assert metadata["source_type"] == "stdout_tail"
    assert metadata["status"] == "materialized"
    assert metadata["stdout_tail_truncated"] is True


def test_stdout_tail_line_is_bounded_by_bytes():
    from skills.harness.container.agent_run import (
        MAX_STDOUT_TAIL_LINE_BYTES,
        STDOUT_TAIL_TRUNCATED_MARKER,
        truncate_stdout_tail_line,
    )

    line = ("a" * (MAX_STDOUT_TAIL_LINE_BYTES + 100)) + "\n"

    truncated = truncate_stdout_tail_line(line)

    assert len(truncated.encode("utf-8")) <= MAX_STDOUT_TAIL_LINE_BYTES
    assert STDOUT_TAIL_TRUNCATED_MARKER in truncated
    assert truncated.endswith("\n")
