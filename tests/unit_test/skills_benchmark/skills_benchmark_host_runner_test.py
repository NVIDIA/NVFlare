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
import os
import stat
import subprocess
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[3] / "dev_tools" / "agent" / "skills" / "benchmark"


def test_run_sh_defaults_to_pair_when_first_arg_is_option(tmp_path):
    script = BENCHMARK_ROOT / "bin" / "run.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    recorder = tmp_path / "python_args.txt"
    fake_python = fake_bin / "python3"
    fake_python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$RECORD_PATH"\n', encoding="utf-8")
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["RECORD_PATH"] = str(recorder)

    result = subprocess.run(
        [str(script), "--prompt", "prompt.txt", "job"],
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0
    assert recorder.read_text(encoding="utf-8").splitlines() == [
        "-m",
        "skills.harness.host.runner",
        "pair",
        "--prompt",
        "prompt.txt",
        "job",
    ]


def test_expand_home_path_uses_pathlib_expanduser():
    from skills.harness.host.common import expand_home_path

    assert expand_home_path("~/nvflare") == str(Path.home() / "nvflare")
    assert expand_home_path("/workspace/input") == "/workspace/input"


def test_agent_availability_probe_records_missing_cli(tmp_path, monkeypatch):
    from skills.harness.container import agent_run
    from skills.harness.container.agent_run import AgentRunConfig, run_agent_availability_probe

    class MissingProbeAdapter:
        def availability_probe(self):
            return ["/definitely/missing/agent-cli"]

        def runtime_env(self, config):
            return {}

        def model_env_names(self):
            return ()

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
    monkeypatch.setattr(agent_run, "load_agent_adapter", lambda _agent: MissingProbeAdapter())

    try:
        run_agent_availability_probe(config)
    except RuntimeError as exc:
        assert "Agent availability probe failed to start" in str(exc)
    else:
        raise AssertionError("missing agent CLI should fail availability probe")

    probe = json.loads((result_dir / "agent_availability_probe.json").read_text(encoding="utf-8"))
    assert probe["status"] == "failed"
    assert probe["exit_code"] == 127


def test_host_image_config_rejects_unsupported_agent(monkeypatch):
    from skills.harness.host.common import ImageConfig

    monkeypatch.setenv("BENCHMARK_AGENT", "hermes")

    try:
        ImageConfig.from_env()
    except SystemExit as exc:
        assert "BENCHMARK_AGENT='hermes'" in str(exc)
        assert "known but not implemented" in str(exc)
    else:
        raise AssertionError("unsupported benchmark agent should fail before image selection")


def test_host_docker_args_use_migrated_container_entrypoint(tmp_path):
    from skills.harness.agents.registry import load_agent_adapter
    from skills.harness.host.common import CONTAINER_PROMPT_PATH, CaseConfig, ImageConfig, docker_args_for_case

    job_input = tmp_path / "job"
    prompt_dir = tmp_path / "prompts"
    result_dir = tmp_path / "results"
    agent_home = tmp_path / ".codex"
    job_input.mkdir()
    prompt_dir.mkdir()
    prompt_path = prompt_dir / "benchmark_prompt.txt"
    prompt_path.write_text("convert this job\n", encoding="utf-8")

    config = CaseConfig(
        mode="with_skills",
        use_preinstalled_skills=True,
        job_input_dir=job_input,
        result_dir=result_dir,
        prompt_path=prompt_path,
        images=ImageConfig(
            image_name="agent-skills-benchmark:codex-skills",
            baseline_image_name="agent-skills-benchmark:codex-baseline",
            report_image_name="agent-skills-benchmark:codex-skills",
        ),
        progress_interval_seconds="0",
        agent="codex",
        agent_model="unspecified_default",
        model_was_explicit=False,
        adapter=load_agent_adapter("codex"),
        host_agent_home=agent_home,
        mount_host_agent_auth=False,
    )

    args = docker_args_for_case(config)

    assert "-m" in args
    module_index = args.index("-m") + 1
    assert args[module_index] == "skills.harness.container.agent_run"
    assert f"{prompt_path}:{CONTAINER_PROMPT_PATH}:ro" in args
    assert f"PROMPT_SOURCE={CONTAINER_PROMPT_PATH}" in args
    assert "RECORDS_DIR=/workspace/results/records" in args


def test_prepare_result_mount_allows_non_root_container_writes(tmp_path):
    from skills.harness.host.common import prepare_result_mount

    result_dir = tmp_path / "results"

    prepare_result_mount(result_dir)

    assert result_dir.is_dir()
    assert (result_dir / "records").is_dir()
    result_mode = result_dir.stat().st_mode
    records_mode = (result_dir / "records").stat().st_mode
    assert result_mode & stat.S_IWOTH
    assert result_mode & stat.S_IXOTH
    assert records_mode & stat.S_IWOTH
    assert records_mode & stat.S_IXOTH


def test_add_agent_auth_mounts_rejects_symlinked_optional_file(tmp_path):
    if not hasattr(os, "symlink"):
        return
    from skills.harness.agents.base import DockerMount
    from skills.harness.host.common import add_agent_auth_mounts

    real_auth = tmp_path / "real-auth.json"
    real_auth.write_text("{}\n", encoding="utf-8")
    linked_auth = tmp_path / "linked-auth.json"
    linked_auth.symlink_to(real_auth)
    args = []

    add_agent_auth_mounts(
        args,
        mounts=[DockerMount(host_path=linked_auth, container_path="/workspace/.agent/auth.json")],
    )

    assert args == []


def test_add_agent_auth_mounts_stages_read_only_file_for_container_user(tmp_path):
    from skills.harness.agents.base import DockerMount
    from skills.harness.host.common import add_agent_auth_mounts

    auth = tmp_path / "auth.json"
    auth.write_text('{"token": "secret"}\n', encoding="utf-8")
    auth.chmod(stat.S_IRUSR | stat.S_IWUSR)
    staging_dir = tmp_path / "staged"
    args = []

    add_agent_auth_mounts(
        args,
        mounts=[DockerMount(host_path=auth, container_path="/workspace/.codex/auth.json", read_only=True)],
        staging_dir=staging_dir,
    )

    volume = args[args.index("-v") + 1]
    staged_source = Path(volume.split(":", 1)[0])
    assert staged_source != auth
    assert staged_source.read_text(encoding="utf-8") == auth.read_text(encoding="utf-8")
    assert stat.S_IMODE(staged_source.stat().st_mode) & stat.S_IROTH
    assert volume.endswith(":/workspace/.codex/auth.json:ro")


def test_add_agent_auth_mounts_rejects_symlinked_required_file(tmp_path):
    if not hasattr(os, "symlink"):
        return
    from skills.harness.agents.base import DockerMount
    from skills.harness.host.common import add_agent_auth_mounts

    real_auth = tmp_path / "real-auth.json"
    real_auth.write_text("{}\n", encoding="utf-8")
    linked_auth = tmp_path / "linked-auth.json"
    linked_auth.symlink_to(real_auth)

    try:
        add_agent_auth_mounts(
            [],
            mounts=[DockerMount(host_path=linked_auth, container_path="/workspace/.agent/auth.json", required=True)],
        )
    except SystemExit as exc:
        assert "must not be a symlink" in str(exc)
    else:
        raise AssertionError("required symlinked auth file should be rejected")


def test_write_benchmark_reports_clears_agent_parser_cache(tmp_path, monkeypatch):
    from skills.harness.host import runner

    cleared = {"count": 0}

    class CachedParser:
        @staticmethod
        def cache_clear():
            cleared["count"] += 1

    monkeypatch.setattr(runner, "parse_cached_usage_and_activity", CachedParser)
    monkeypatch.setattr(runner.metrics_report, "write_reports", lambda _root, _title: None)
    monkeypatch.setattr(runner.benchmark_insights, "collect_benchmark_runs", lambda _root: {})
    monkeypatch.setattr(runner.benchmark_insights, "benchmark_report", lambda _root, _runs: "report\n")

    statuses = runner.write_benchmark_reports(tmp_path)

    assert statuses == {"metrics_report": 0, "benchmark_insights": 0}
    assert cleared["count"] == 1


def test_enforce_result_size_budget_reports_oversized_results(tmp_path):
    from skills.harness.agents.registry import load_agent_adapter
    from skills.harness.host.common import CaseConfig, ImageConfig
    from skills.harness.host.runner import enforce_result_size_budget

    result_dir = tmp_path / "results"
    result_dir.mkdir()
    result_dir.joinpath("large.txt").write_text("too large\n", encoding="utf-8")
    config = CaseConfig(
        mode="with_skills",
        use_preinstalled_skills=True,
        job_input_dir=tmp_path / "job",
        result_dir=result_dir,
        prompt_path=tmp_path / "prompt.txt",
        images=ImageConfig(
            image_name="agent-skills-benchmark:codex-skills",
            baseline_image_name="agent-skills-benchmark:codex-baseline",
            report_image_name="agent-skills-benchmark:codex-skills",
        ),
        progress_interval_seconds="0",
        agent="codex",
        agent_model="unspecified_default",
        model_was_explicit=False,
        adapter=load_agent_adapter("codex"),
        host_agent_home=tmp_path / ".codex",
        mount_host_agent_auth=False,
        result_size_budget_bytes=1,
    )

    assert enforce_result_size_budget(config) is True
    payload = json.loads((result_dir / "result_size_budget.json").read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert payload["budget_bytes"] == 1


def test_emit_case_failure_summary_prints_actionable_error(tmp_path):
    from types import SimpleNamespace

    from skills.harness.host import runner

    result_dir = tmp_path / "results"
    result_dir.mkdir()
    result_dir.joinpath("run_summary.json").write_text(
        json.dumps(
            {
                "agent_process_exit_code": 1,
                "final_container_exit_code": 1,
                "failure_root_cause": "agent_auth_failure",
            }
        ),
        encoding="utf-8",
    )
    result_dir.joinpath("agent_exit_summary.json").write_text(
        json.dumps(
            {
                "failure_category": "agent_auth_failure",
                "stderr_excerpt": "failed to read auth token\nsecond detail\nthird detail\nfourth detail\nfifth detail\n",
            }
        ),
        encoding="utf-8",
    )
    log = tmp_path / "console.log"

    runner.emit_case_failure_summary(
        SimpleNamespace(mode="with_skills", result_dir=result_dir),
        1,
        logs=(log,),
        prefix="run_00002",
    )

    text = log.read_text(encoding="utf-8")
    assert "[run_00002] Run failed: mode=with_skills; final_status=1" in text
    assert "Failure exit codes: agent_process_exit=1; final_container_exit=1" in text
    assert "Failure category: agent_auth_failure" in text
    assert "failed to read auth token" in text
    assert "fifth detail" not in text
    assert "Failure artifacts:" in text


def test_directory_size_bytes_does_not_traverse_symlinked_directories(tmp_path):
    from skills.harness.host.runner import directory_size_bytes

    result_dir = tmp_path / "results"
    outside = tmp_path / "outside"
    result_dir.mkdir()
    outside.mkdir()
    (result_dir / "small.txt").write_text("ok", encoding="utf-8")
    (outside / "large.txt").write_text("x" * 1000, encoding="utf-8")
    os.symlink(outside, result_dir / "outside_link")

    assert directory_size_bytes(result_dir) == 2


def test_case_config_for_entry_applies_resource_policy(tmp_path, monkeypatch):
    from skills.harness.host import runner

    entry = {
        "agent": "codex",
        "mode": "with_skills",
        "skills_enabled": True,
        "job_path": str(tmp_path / "job"),
        "record_dir": "records/run",
        "prompt_source": str(tmp_path / "prompt.txt"),
        "agent_model": "unspecified_default",
        "model_source": "adapter_default",
        "resource_policy": {
            "agent_timeout_seconds": 11,
            "container_timeout_seconds": 22,
            "result_size_budget_bytes": 33,
        },
    }
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex"))

    config = runner.case_config_for_entry(entry, tmp_path / "results")

    assert config.agent_timeout_seconds == 11
    assert config.container_timeout_seconds == 22
    assert config.result_size_budget_bytes == 33


def test_positive_int_resource_value_accepts_float_values():
    from skills.harness.host.runner import positive_int_resource_value

    assert positive_int_resource_value(1800.0) == 1800
    assert positive_int_resource_value(0.5) is None
    assert positive_int_resource_value(0.0) is None
    assert positive_int_resource_value(True) is None


def test_stream_command_warns_when_reader_thread_stays_alive(tmp_path, monkeypatch):
    from skills.harness.host import common

    class FakeStdout:
        def close(self):
            return None

    class FakeProcess:
        stdout = FakeStdout()

        def wait(self, timeout=None):
            return 0

    class StuckThread:
        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon

        def start(self):
            return None

        def join(self, timeout=None):
            return None

        def is_alive(self):
            return True

    monkeypatch.setattr(common.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(common.threading, "Thread", StuckThread)
    log = tmp_path / "console.log"

    status = common.stream_command(["docker", "run"], logs=(log,), prefix="run_00001")

    assert status == 0
    assert "[run_00001] Output reader thread did not stop within 2 seconds." in log.read_text(encoding="utf-8")


def test_host_cli_accepts_results_root(tmp_path):
    from skills.harness.host.common import parse_host_cli_options

    job_input = tmp_path / "job"
    prompt = tmp_path / "prompt.txt"
    results_root = tmp_path / "bench-results"
    job_input.mkdir()
    prompt.write_text("convert this job\n", encoding="utf-8")

    options = parse_host_cli_options(
        ["--prompt", str(prompt), "--results-root", str(results_root), "--training-code", str(job_input)],
        "pair",
    )

    assert options.job_input == job_input
    assert options.prompt_path == prompt
    assert options.results_root == results_root
    assert options.result_root is None
    assert options.result_dir is None


def test_host_cli_accepts_direct_run_selection_flags(tmp_path):
    from skills.harness.host.common import parse_host_cli_options

    job_input = tmp_path / "job"
    prompt = tmp_path / "prompt.txt"
    agent_home = tmp_path / "agent-home"
    job_input.mkdir()
    agent_home.mkdir()
    prompt.write_text("convert this job\n", encoding="utf-8")

    options = parse_host_cli_options(
        [
            "--prompt",
            str(prompt),
            "--agent",
            "claude",
            "--model",
            "claude-test",
            "--mode",
            "without_skills",
            "--workflow",
            "fedavg",
            "--job-scale",
            "medium",
            "--agent-home",
            str(agent_home),
            "--no-agent-auth-mount",
            str(job_input),
        ],
        "run-one",
    )

    assert options.agent == "claude"
    assert options.model == "claude-test"
    assert options.mode == "without_skills"
    assert options.workflow == "fedavg"
    assert options.job_scale == "medium"
    assert options.agent_home == agent_home
    assert options.mount_agent_auth is False


def test_scenario_cli_accepts_generic_auth_flags(tmp_path):
    from skills.harness.host.runner import parse_scenario_cli_options

    scenario = tmp_path / "scenario.yaml"
    output_dir = tmp_path / "results"
    agent_home = tmp_path / "agent-home"
    scenario.write_text("name: test\n", encoding="utf-8")
    agent_home.mkdir()

    options = parse_scenario_cli_options(
        [
            str(scenario),
            "--output-dir",
            str(output_dir),
            "--agent-home",
            str(agent_home),
            "--no-agent-auth-mount",
        ]
    )

    assert options.scenario_path == scenario
    assert options.result_root == output_dir
    assert options.agent_home == agent_home
    assert options.mount_agent_auth is False


def test_run_scenario_passes_generic_auth_flags_to_execution(tmp_path, monkeypatch):
    from skills.harness.host import runner

    job_input = tmp_path / "job"
    prompt = tmp_path / "prompt.txt"
    scenario = tmp_path / "scenario.yaml"
    output_dir = tmp_path / "results"
    agent_home = tmp_path / "agent-home"
    job_input.mkdir()
    agent_home.mkdir()
    prompt.write_text("convert this job\n", encoding="utf-8")
    scenario.write_text(
        "\n".join(
            [
                "name: scenario auth",
                f"prompt: {prompt}",
                "agents:",
                "  - name: codex",
                "    models: [codex-test]",
                "workflows:",
                "  - name: default",
                "jobs:",
                "  - name: job",
                f"    path: {job_input}",
                "    scale: small",
                "comparison:",
                "  type: one",
                "  mode: with_skills",
                "",
            ]
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_execute(compilation, *, result_root, logs=(), runtime_auth_options=None):
        captured["result_root"] = result_root
        captured["runtime_auth_options"] = runtime_auth_options
        return {"run_00001": 0}, {"status": "passed"}

    monkeypatch.setattr(runner, "execute_run_plan", fake_execute)

    status = runner.run_scenario(
        [
            str(scenario),
            "--output-dir",
            str(output_dir),
            "--agent-home",
            str(agent_home),
            "--no-agent-auth-mount",
        ]
    )

    assert status == 0
    assert captured["result_root"] == output_dir
    assert captured["runtime_auth_options"].agent_home == agent_home
    assert captured["runtime_auth_options"].mount_agent_auth is False


def test_host_cli_rejects_mode_for_pair(tmp_path):
    from skills.harness.host.common import parse_host_cli_options

    job_input = tmp_path / "job"
    prompt = tmp_path / "prompt.txt"
    job_input.mkdir()
    prompt.write_text("convert this job\n", encoding="utf-8")

    try:
        parse_host_cli_options(["--prompt", str(prompt), "--mode", "with_skills", str(job_input)], "pair")
    except SystemExit as exc:
        assert "--mode is only supported for run-one" in str(exc)
    else:
        raise AssertionError("--mode should be rejected for pair")


def test_default_results_root_uses_codex_compat_alias(monkeypatch, tmp_path):
    from skills.harness.host.common import default_results_root

    compat_root = tmp_path / "compat-results"
    monkeypatch.delenv("AGENT_BENCHMARK_RESULTS_ROOT", raising=False)
    monkeypatch.setenv("CODEX_DOCKER_RESULTS_ROOT", str(compat_root))

    assert default_results_root() == compat_root


def test_pair_compilation_accepts_explicit_absolute_prompt_path(tmp_path):
    from skills.harness.host.common import parse_host_cli_options
    from skills.harness.host.runner import pair_compilation_from_options

    job_input = tmp_path / "job"
    prompt = tmp_path / "prompt.txt"
    results_root = tmp_path / "bench-results"
    job_input.mkdir()
    prompt.write_text("convert this job\n", encoding="utf-8")
    options = parse_host_cli_options(
        ["--prompt", str(prompt), "--results-root", str(results_root), "--training-code", str(job_input)],
        "pair",
    )

    compilation = pair_compilation_from_options(options)

    assert compilation.scenario["prompt"]["path"] == str(prompt.resolve())


def test_pair_compilation_uses_direct_run_selection_flags(tmp_path, monkeypatch):
    from skills.harness.host.common import parse_host_cli_options
    from skills.harness.host.runner import pair_compilation_from_options

    job_input = tmp_path / "job"
    prompt = tmp_path / "prompt.txt"
    job_input.mkdir()
    prompt.write_text("convert this job\n", encoding="utf-8")
    monkeypatch.delenv("BENCHMARK_AGENT", raising=False)
    monkeypatch.delenv("BENCHMARK_AGENT_MODEL", raising=False)

    options = parse_host_cli_options(
        [
            "--prompt",
            str(prompt),
            "--agent",
            "codex",
            "--model",
            "codex-test",
            "--workflow",
            "fedavg",
            "--job-scale",
            "medium",
            str(job_input),
        ],
        "pair",
    )

    compilation = pair_compilation_from_options(options)
    entries = compilation.run_plan["entries"]

    assert entries
    assert {entry["agent"] for entry in entries} == {"codex"}
    assert {entry["agent_model"] for entry in entries} == {"codex-test"}
    assert {entry["workflow"] for entry in entries} == {"fedavg"}
    assert {entry["job_scale"] for entry in entries} == {"medium"}


def test_case_config_uses_generic_runtime_auth_overrides(tmp_path, monkeypatch):
    from skills.harness.host.common import parse_host_cli_options
    from skills.harness.host.runner import RuntimeAuthOptions, case_config_for_entry, pair_compilation_from_options

    job_input = tmp_path / "job"
    prompt = tmp_path / "prompt.txt"
    agent_home = tmp_path / "custom-agent-home"
    result_root = tmp_path / "results"
    job_input.mkdir()
    agent_home.mkdir()
    prompt.write_text("convert this job\n", encoding="utf-8")
    monkeypatch.delenv("BENCHMARK_AGENT", raising=False)
    monkeypatch.delenv("BENCHMARK_AGENT_MODEL", raising=False)

    options = parse_host_cli_options(
        ["--prompt", str(prompt), "--agent", "codex", "--model", "codex-test", str(job_input)],
        "pair",
    )
    compilation = pair_compilation_from_options(options)
    entry = compilation.run_plan["entries"][0]

    config = case_config_for_entry(
        entry,
        result_root,
        RuntimeAuthOptions(agent_home=agent_home, mount_agent_auth=False),
    )

    assert config.host_agent_home == agent_home
    assert config.mount_host_agent_auth is False


def test_run_one_executes_compiled_one_scenario(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from skills.harness.host import runner

    job_input = tmp_path / "job"
    prompt = tmp_path / "prompt.txt"
    result_root = tmp_path / "results"
    job_input.mkdir()
    prompt.write_text("convert this job\n", encoding="utf-8")
    captured = {}
    monkeypatch.setenv("MODE", "with_skills")
    monkeypatch.setattr(
        runner.ImageConfig,
        "for_adapter",
        classmethod(
            lambda cls, _adapter: SimpleNamespace(
                image_name="skills-image",
                baseline_image_name="baseline-image",
                report_image_name="report-image",
            )
        ),
    )

    def fake_execute(compilation, *, result_root, logs=()):
        captured["compilation"] = compilation
        captured["result_root"] = result_root
        return {"run_00001": 0}, {"status": "passed"}

    monkeypatch.setattr(runner, "execute_run_plan", fake_execute)

    status = runner.run_one(["--prompt", str(prompt), "--output-dir", str(result_root), str(job_input)])

    assert status == 0
    assert captured["result_root"] == result_root
    run_plan = captured["compilation"].run_plan
    assert run_plan["comparison_type"] == "one"
    assert run_plan["run_count"] == 1
    assert run_plan["entries"][0]["mode"] == "with_skills"
    assert run_plan["entries"][0]["record_dir"].startswith("records/")


def test_run_one_reports_image_config_validation_without_traceback(tmp_path, monkeypatch, capsys):
    from skills.harness.host import runner

    job_input = tmp_path / "job"
    prompt = tmp_path / "prompt.txt"
    result_root = tmp_path / "results"
    job_input.mkdir()
    prompt.write_text("convert this job\n", encoding="utf-8")
    monkeypatch.setenv("MODE", "with_skills")
    monkeypatch.setattr(
        runner.ImageConfig,
        "for_adapter",
        classmethod(lambda cls, _adapter: (_ for _ in ()).throw(ValueError("bad image template"))),
    )

    status = runner.run_one(["--prompt", str(prompt), "--output-dir", str(result_root), str(job_input)])

    assert status == 1
    captured = capsys.readouterr()
    assert "Scenario validation failed: bad image template" in captured.err
    assert "Traceback" not in captured.err


def test_run_pair_returns_failure_for_any_run_id_status(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from skills.harness.host import runner

    job_input = tmp_path / "job"
    prompt = tmp_path / "prompt.txt"
    result_root = tmp_path / "results"
    job_input.mkdir()
    prompt.write_text("convert this job\n", encoding="utf-8")
    monkeypatch.setattr(
        runner.ImageConfig,
        "for_adapter",
        classmethod(
            lambda cls, _adapter: SimpleNamespace(
                image_name="skills-image",
                baseline_image_name="baseline-image",
                report_image_name="report-image",
            )
        ),
    )
    monkeypatch.setattr(
        runner,
        "execute_run_plan",
        lambda *args, **kwargs: ({"run_00001": 1, "run_00002": 0}, {"status": "ok"}),
    )

    status = runner.run_pair(["--prompt", str(prompt), "--output-dir", str(result_root), str(job_input)])

    assert status == 1


def test_run_pair_reports_scenario_validation_without_traceback(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    from skills.harness.host import runner
    from skills.harness.scenarios import ScenarioValidationError

    job_input = tmp_path / "job"
    prompt = tmp_path / "prompt.txt"
    result_root = tmp_path / "results"
    job_input.mkdir()
    prompt.write_text("convert this job\n", encoding="utf-8")
    monkeypatch.setattr(
        runner.ImageConfig,
        "for_adapter",
        classmethod(
            lambda cls, _adapter: SimpleNamespace(
                image_name="skills-image",
                baseline_image_name="baseline-image",
                report_image_name="report-image",
            )
        ),
    )
    monkeypatch.setattr(
        runner,
        "execute_run_plan",
        lambda *args, **kwargs: (_ for _ in ()).throw(ScenarioValidationError("expanded path too long")),
    )

    status = runner.run_pair(["--prompt", str(prompt), "--output-dir", str(result_root), str(job_input)])

    assert status == 1
    captured = capsys.readouterr()
    assert "Scenario validation failed: expanded path too long" in captured.err
    assert "Traceback" not in captured.err


def test_run_pair_writes_host_report_status_after_preflight_report(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from skills.harness.host import runner
    from skills.harness.scenarios import ScenarioValidationError

    job_input = tmp_path / "job"
    prompt = tmp_path / "prompt.txt"
    result_root = tmp_path / "results"
    job_input.mkdir()
    prompt.write_text("convert this job\n", encoding="utf-8")
    monkeypatch.setattr(
        runner.ImageConfig,
        "for_adapter",
        classmethod(
            lambda cls, _adapter: SimpleNamespace(
                image_name="skills-image",
                baseline_image_name="baseline-image",
                report_image_name="report-image",
            )
        ),
    )

    def fail_after_report(_compilation, *, result_root, logs=()):
        reports_dir = result_root / "reports"
        reports_dir.mkdir(parents=True)
        reports_dir.joinpath("scenario_report.md").write_text("preflight report\n", encoding="utf-8")
        raise ScenarioValidationError("preflight failed")

    monkeypatch.setattr(runner, "execute_run_plan", fail_after_report)

    status = runner.run_pair(["--prompt", str(prompt), "--output-dir", str(result_root), str(job_input)])

    assert status == 1
    host_status = json.loads((result_root / "host_report_status.json").read_text(encoding="utf-8"))
    assert host_status["status"] == "ok"
    assert host_status["scenario_report"] == str(result_root / "reports" / "scenario_report.md")


def test_host_cli_output_dir_maps_to_exact_result_location(tmp_path):
    from skills.harness.host.common import parse_host_cli_options

    job_input = tmp_path / "job"
    prompt = tmp_path / "prompt.txt"
    output_dir = tmp_path / "exact-output"
    job_input.mkdir()
    prompt.write_text("convert this job\n", encoding="utf-8")

    comparison_options = parse_host_cli_options(
        ["--prompt", str(prompt), "--output-dir", str(output_dir), str(job_input)],
        "pair",
    )
    single_options = parse_host_cli_options(
        ["--prompt", str(prompt), "--output-dir", str(output_dir), str(job_input)],
        "run-one",
    )

    assert comparison_options.result_root == output_dir
    assert comparison_options.result_dir is None
    assert single_options.result_dir == output_dir
    assert single_options.result_root is None


def test_host_cli_requires_prompt_path(tmp_path):
    from skills.harness.host.common import parse_host_cli_options

    job_input = tmp_path / "job"
    job_input.mkdir()

    try:
        parse_host_cli_options([str(job_input)], "pair")
    except SystemExit as exc:
        assert "Prompt file is required" in str(exc)
    else:
        raise AssertionError("parse_host_cli_options should require --prompt")


def test_container_config_rejects_unknown_mode(monkeypatch):
    from skills.harness.container.agent_run import AgentRunConfig

    monkeypatch.setenv("MODE", "with_skill_typo")

    try:
        AgentRunConfig.from_env()
    except SystemExit as exc:
        assert "Unknown MODE with_skill_typo" in str(exc)
        assert "without_skills" in str(exc)
        assert "with_skills" in str(exc)
    else:
        raise AssertionError("unknown MODE should fail before skill defaulting")


def test_container_config_rejects_mode_skill_flag_conflict(monkeypatch):
    from skills.harness.container.agent_run import AgentRunConfig

    monkeypatch.setenv("MODE", "without_skills")
    monkeypatch.setenv("USE_PREINSTALLED_SKILLS", "true")

    try:
        AgentRunConfig.from_env()
    except SystemExit as exc:
        assert "conflicts with MODE=without_skills" in str(exc)
        assert "expected false" in str(exc)
    else:
        raise AssertionError("MODE and USE_PREINSTALLED_SKILLS disagreement should fail fast")
