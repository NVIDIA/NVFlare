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
import stat
import sys


def test_benchmark_insights_explains_docker_image_failures(tmp_path):
    from skills.harness.modes import NO_SKILLS_MODE
    from skills.harness.reports.benchmark_insights import (
        collect_benchmark_runs,
        failure_root_cause,
        human_readable_status,
    )

    mode_dir = tmp_path / NO_SKILLS_MODE
    mode_dir.mkdir()
    (mode_dir / "container_exit_code.json").write_text(json.dumps({"exit_code": 1}) + "\n", encoding="utf-8")
    (tmp_path / "console_output.log").write_text(
        "[without_skills] Unable to find image 'agent-skills-benchmark:codex-baseline' locally\n"
        "[without_skills] docker: Error response from daemon: pull access denied for agent-skills-benchmark\n",
        encoding="utf-8",
    )

    run = collect_benchmark_runs(tmp_path)[NO_SKILLS_MODE]

    assert run["available"] is True
    assert "Docker image unavailable" in failure_root_cause(run)
    assert "container exit 1" in human_readable_status(run)


def test_benchmark_insights_scopes_shared_console_evidence_by_mode(tmp_path):
    from skills.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from skills.harness.reports.benchmark_insights import collect_benchmark_runs, dependency_reference_notes

    for mode in (NO_SKILLS_MODE, WITH_SKILLS_MODE):
        records_dir = tmp_path / mode / "records"
        records_dir.mkdir(parents=True)
        (records_dir / f"{mode}_record.json").write_text(
            json.dumps(
                {
                    "source_input_delta": {"final_files": [{"path": "requirements-train.txt"}]},
                    "workspace_delta": {
                        "workspace_added_files": (
                            [{"path": "requirements-federated.txt"}] if mode == NO_SKILLS_MODE else []
                        )
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
    (tmp_path / "console_output.log").write_text(
        "[without_skills] python3 -m pip install -r requirements-federated.txt failed\n"
        "[with_skills] completed without dependency install errors\n",
        encoding="utf-8",
    )

    runs = collect_benchmark_runs(tmp_path)

    assert dependency_reference_notes(runs[NO_SKILLS_MODE]) == [
        "`requirements-federated.txt` provenance: agent-generated file.",
    ]
    assert dependency_reference_notes(runs[WITH_SKILLS_MODE]) == []


def test_benchmark_insights_caps_agent_events_text(tmp_path, monkeypatch):
    from skills.harness.modes import NO_SKILLS_MODE
    from skills.harness.reports import benchmark_insights

    mode_dir = tmp_path / NO_SKILLS_MODE
    mode_dir.mkdir()
    (mode_dir / "agent_events.jsonl").write_text("0123456789", encoding="utf-8")
    monkeypatch.setattr(benchmark_insights, "MAX_AGENT_EVENTS_TEXT_BYTES", 8)

    run = benchmark_insights.collect_benchmark_runs(tmp_path)[NO_SKILLS_MODE]

    assert run["agent_events_text"] == "01234567"


def test_benchmark_reports_read_canonical_record_layout(tmp_path):
    from skills.harness.common import write_json
    from skills.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from skills.harness.reports import metrics_report
    from skills.harness.reports.benchmark_insights import benchmark_report, collect_benchmark_runs

    entries = []
    for index, mode in enumerate((NO_SKILLS_MODE, WITH_SKILLS_MODE), start=1):
        record_dir = (
            tmp_path
            / "records"
            / "agent=codex"
            / "model=default"
            / "workflow=default"
            / "job=ames"
            / "repeat=01"
            / f"mode={mode}"
        )
        record_dir.mkdir(parents=True)
        entries.append(
            {
                "run_id": f"run_{index:05d}",
                "mode": mode,
                "agent": "codex",
                "agent_model": "default",
                "model_source": "scenario",
                "record_dir": str(record_dir.relative_to(tmp_path)),
            }
        )
        write_json(
            record_dir / "run_summary.json",
            {
                "mode": mode,
                "elapsed_seconds": 10 + index,
                "token_count": 100 + index,
                "agent_exit_code": 0,
                "final_container_exit_code": 0,
            },
        )
        write_json(record_dir / "container_exit_code.json", {"exit_code": 0})
        write_json(record_dir / "agent_activity.json", {"command_count": index})
        write_json(
            record_dir / "benchmark_record.json",
            {
                "mode": mode,
                "reported_validation_metric": {"name": "AUROC", "value": 0.7 + index / 100},
            },
        )
    write_json(tmp_path / "run_plan.json", {"entries": entries})

    runs = collect_benchmark_runs(tmp_path)
    assert runs[NO_SKILLS_MODE]["available"] is True
    assert runs[NO_SKILLS_MODE]["agent"] == "codex"
    assert runs[NO_SKILLS_MODE]["agent_model"] == "default"
    assert runs[WITH_SKILLS_MODE]["record"]["reported_validation_metric"]["name"] == "AUROC"
    insights = benchmark_report(tmp_path, runs)
    assert "## Run Identity" in insights
    assert "| No skills baseline | codex | default | scenario | without_skills |" in insights
    assert "## Cost And Work Comparison" in insights
    assert "| Dependency install seconds |" in insights
    assert "| Run | Elapsed seconds | Tokens | Commands |" not in insights

    metrics_report.write_reports(tmp_path, "Synthetic Metrics")

    assert (tmp_path / "metrics_report.json").is_file()
    metrics_markdown = (tmp_path / "metrics_report.md").read_text(encoding="utf-8")
    assert "Metrics (AUROC)" in metrics_markdown
    assert "| No skills baseline | codex | default |" in metrics_markdown


def test_fl_algorithm_section_reads_captured_server_workflow_config(tmp_path):
    from skills.harness.modes import WITH_SKILLS_MODE
    from skills.harness.reports.benchmark_insights import fl_algorithm_section, fl_algorithm_summary

    mode_dir = tmp_path / WITH_SKILLS_MODE
    config_path = (
        mode_dir
        / "workspace_delta"
        / "runtime_artifacts"
        / "runtime_workspaces"
        / "job"
        / "server"
        / "simulate_job"
        / "app_server"
        / "config"
        / "config_fed_server.json"
    )
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "workflows": [
                    {
                        "id": "controller",
                        "path": "nvflare.app_common.workflows.scaffold.Scaffold",
                        "args": {"num_rounds": 3},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    run = {
        "available": True,
        "label": "With skills",
        "mode_dir": mode_dir,
        "agent_last_message": "- **Recipe:** `scaffold-pt` -> `ScaffoldRecipe`",
        "workspace_delta": {
            "runtime_artifacts": [
                {
                    "artifact_path": (
                        "runtime_artifacts/runtime_workspaces/job/server/simulate_job/app_server/config/"
                        "config_fed_server.json"
                    ),
                    "path": "runtime_workspaces/job/server/simulate_job/app_server/config/config_fed_server.json",
                }
            ]
        },
    }
    runs = {WITH_SKILLS_MODE: run}

    assert fl_algorithm_summary(runs, [WITH_SKILLS_MODE]) == "With skills: SCAFFOLD (3 rounds)"
    section = fl_algorithm_section(runs, [WITH_SKILLS_MODE])
    assert "## FL Algorithm / Workflow" in section
    assert "| With skills | SCAFFOLD | scaffold-pt | 3 |" in section
    assert "nvflare.app_common.workflows.scaffold.Scaffold" in section


def test_fl_algorithm_prefers_training_workflow_over_initialization(tmp_path):
    from skills.harness.modes import WITH_SKILLS_MODE
    from skills.harness.reports.benchmark_insights import fl_algorithm_info

    mode_dir = tmp_path / WITH_SKILLS_MODE
    config_path = (
        mode_dir
        / "workspace_delta"
        / "runtime_artifacts"
        / "runtime_workspaces"
        / "job"
        / "server"
        / "simulate_job"
        / "app_server"
        / "config"
        / "config_fed_server.json"
    )
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "workflows": [
                    {
                        "id": "init",
                        "path": "nvflare.app_common.workflows.initialize_global_weights.InitializeGlobalWeights",
                        "args": {},
                    },
                    {
                        "id": "train",
                        "path": "nvflare.app_common.workflows.scatter_and_gather.ScatterAndGather",
                        "args": {"num_rounds": 5, "train_task_name": "train"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    run = {
        "available": True,
        "mode_dir": mode_dir,
        "workspace_delta": {
            "runtime_artifacts": [
                {
                    "artifact_path": (
                        "runtime_artifacts/runtime_workspaces/job/server/simulate_job/app_server/config/"
                        "config_fed_server.json"
                    ),
                    "path": "runtime_workspaces/job/server/simulate_job/app_server/config/config_fed_server.json",
                }
            ]
        },
    }

    info = fl_algorithm_info(run)

    assert info["algorithm"] == "ScatterAndGather"
    assert info["num_rounds"] == 5
    assert info["workflow_id"] == "train"


def test_fl_algorithm_recipe_prefers_generated_source_over_recipe_list_catalog(tmp_path):
    from skills.harness.modes import WITH_SKILLS_MODE
    from skills.harness.reports.benchmark_insights import fl_algorithm_info

    mode_dir = tmp_path / WITH_SKILLS_MODE
    config_path = (
        mode_dir
        / "workspace_delta"
        / "runtime_artifacts"
        / "runtime_workspaces"
        / "job"
        / "server"
        / "simulate_job"
        / "app_server"
        / "config"
        / "config_fed_server.json"
    )
    job_path = mode_dir / "workspace_delta" / "changed_files" / "job.py"
    config_path.parent.mkdir(parents=True)
    job_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "workflows": [
                    {
                        "id": "controller",
                        "path": "nvflare.app_common.workflows.fedavg.FedAvg",
                        "args": {"num_rounds": 3},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    job_path.write_text(
        "from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe\n\n"
        "recipe = FedAvgRecipe(name='job', min_clients=3, num_rounds=3)\n",
        encoding="utf-8",
    )
    run = {
        "available": True,
        "mode_dir": mode_dir,
        "agent_last_message": ("Recipe: `cyclic-pt`\n" '{"data": [{"name": "cyclic-pt"}, {"name": "fedavg-pt"}]}'),
        "workspace_delta": {
            "changed_files": [{"artifact_path": "changed_files/job.py", "path": "job.py"}],
            "runtime_artifacts": [
                {
                    "artifact_path": (
                        "runtime_artifacts/runtime_workspaces/job/server/simulate_job/app_server/config/"
                        "config_fed_server.json"
                    ),
                    "path": "runtime_workspaces/job/server/simulate_job/app_server/config/config_fed_server.json",
                }
            ],
        },
    }

    info = fl_algorithm_info(run)

    assert info["algorithm"] == "FedAvg"
    assert info["recipe"] == "fedavg-pt"
    assert "recipe fedavg-pt" in info["evidence"]


def test_fl_algorithm_recipe_mismatch_is_quality_issue(tmp_path):
    from skills.harness.modes import WITH_SKILLS_MODE
    from skills.harness.reports.benchmark_insights import run_quality_issues, run_status_kind

    mode_dir = tmp_path / WITH_SKILLS_MODE
    config_path = (
        mode_dir
        / "workspace_delta"
        / "runtime_artifacts"
        / "runtime_workspaces"
        / "job"
        / "server"
        / "simulate_job"
        / "app_server"
        / "config"
        / "config_fed_server.json"
    )
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "workflows": [
                    {
                        "id": "controller",
                        "path": "nvflare.app_common.workflows.fedavg.FedAvg",
                        "args": {"num_rounds": 3},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    run = {
        "available": True,
        "mode_dir": mode_dir,
        "agent_last_message": "**Recipe:** `cyclic-pt`",
        "container_exit": {"exit_code": 0},
        "record": {},
        "run": {"final_container_exit_code": 0},
        "workspace_delta": {
            "runtime_artifacts": [
                {
                    "artifact_path": (
                        "runtime_artifacts/runtime_workspaces/job/server/simulate_job/app_server/config/"
                        "config_fed_server.json"
                    ),
                    "path": "runtime_workspaces/job/server/simulate_job/app_server/config/config_fed_server.json",
                }
            ]
        },
    }

    issues = run_quality_issues(run)

    assert run_status_kind(run) == "needs review"
    assert issues == [
        "Failed check `fl_algorithm_recipe_match`: runtime workflow `FedAvg` does not match selected recipe "
        "`cyclic-pt` (expected one of: Cyclic)."
    ]


def test_mode_dir_for_benchmark_does_not_guess_ambiguous_canonical_layout(tmp_path):
    from skills.harness.modes import NO_SKILLS_MODE
    from skills.harness.reports.benchmark_insights import mode_dir_for_benchmark

    for repeat in ("01", "02"):
        (
            tmp_path
            / "records"
            / "agent=codex"
            / "model=default"
            / "workflow=default"
            / "job=ames"
            / f"repeat={repeat}"
            / f"mode={NO_SKILLS_MODE}"
        ).mkdir(parents=True)

    assert mode_dir_for_benchmark(tmp_path, NO_SKILLS_MODE) == tmp_path / NO_SKILLS_MODE


def test_numeric_comparison_rejects_bool_values():
    from skills.harness.reports.metrics_report import numeric_comparison

    rows = [
        {"summary": {"elapsed_seconds": 10, "token_count": 100}},
        {"summary": {"elapsed_seconds": True, "token_count": False}},
    ]

    assert numeric_comparison(rows) == {}


def test_numeric_comparison_uses_mode_names_not_row_order():
    from skills.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from skills.harness.reports.metrics_report import numeric_comparison

    rows = [
        {"mode": WITH_SKILLS_MODE, "summary": {"elapsed_seconds": 13, "token_count": 150}},
        {"mode": NO_SKILLS_MODE, "summary": {"elapsed_seconds": 10, "token_count": 100}},
    ]

    assert numeric_comparison(rows) == {
        "elapsed_seconds_with_skills_minus_without_skills": 3,
        "token_count_with_skills_minus_without_skills": 50,
    }


def test_cost_comparison_separates_dependency_install_time():
    from skills.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from skills.harness.reports.benchmark_insights import chart_value_display, cost_comparison_section

    def event_lines(command: str, start: str, end: str, item_id: str) -> list[str]:
        return [
            json.dumps(
                {
                    "timestamp": start,
                    "type": "item.started",
                    "item": {"command": command, "id": item_id, "type": "command_execution"},
                }
            ),
            json.dumps(
                {
                    "timestamp": end,
                    "type": "item.completed",
                    "item": {
                        "aggregated_output": "ok",
                        "command": command,
                        "exit_code": 0,
                        "id": item_id,
                        "status": "completed",
                        "type": "command_execution",
                    },
                }
            ),
        ]

    runs = {
        NO_SKILLS_MODE: {
            "label": "No skills baseline",
            "run": {"elapsed_seconds": 100, "token_count": 10},
            "activity": {"command_count": 2, "unique_command_count": 2},
            "workspace_delta": {},
            "agent_events_text": "\n".join(
                event_lines("uv pip install -r requirements.txt", "2026-06-13T00:00:00Z", "2026-06-13T00:00:15Z", "a")
                + event_lines("python job.py", "2026-06-13T00:00:20Z", "2026-06-13T00:01:30Z", "b")
            ),
        },
        WITH_SKILLS_MODE: {
            "label": "With skills",
            "run": {"elapsed_seconds": 300, "token_count": 12},
            "activity": {"command_count": 2, "unique_command_count": 2},
            "workspace_delta": {},
            "agent_events_text": "\n".join(
                event_lines(
                    "uv pip install -r requirements-train.txt",
                    "2026-06-13T00:00:00Z",
                    "2026-06-13T00:02:00Z",
                    "a",
                )
                + event_lines("python job.py", "2026-06-13T00:02:10Z", "2026-06-13T00:04:50Z", "b")
            ),
        },
    }

    section = cost_comparison_section(runs, [NO_SKILLS_MODE, WITH_SKILLS_MODE])

    assert "`Runtime seconds` is total elapsed time minus captured dependency-install command time" in section
    assert "`Dependency install seconds` is captured dependency-install command time" in section
    assert "Command span timing is operation-level evidence, not a strict wall-clock partition" in section
    assert "| Total time seconds | 100 | 300 | 200 |" in section
    assert "| Runtime seconds | 85 | 180 | 95 |" in section
    assert "| Dependency install seconds | 15 | 120 | 105 |" in section
    assert "| Non-install command seconds | 70 | 160 | 90 |" in section
    assert chart_value_display(1009.055, "seconds") == "1009"


def test_cost_comparison_treats_missing_dependency_install_spans_as_zero():
    from skills.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from skills.harness.reports.benchmark_insights import cost_comparison_section

    runs = {
        NO_SKILLS_MODE: {
            "label": "No skills baseline",
            "run": {"elapsed_seconds": 100, "token_count": 10},
            "activity": {},
            "workspace_delta": {},
            "agent_events_text": "",
        },
        WITH_SKILLS_MODE: {
            "label": "With skills",
            "run": {"elapsed_seconds": 120, "token_count": 12},
            "activity": {},
            "workspace_delta": {},
            "agent_events_text": "",
        },
    }

    section = cost_comparison_section(runs, [NO_SKILLS_MODE, WITH_SKILLS_MODE])

    assert "| Runtime seconds | 100 | 120 | 20 |" in section
    assert "| Dependency install seconds | 0 | 0 | 0 |" in section


def test_cost_comparison_keeps_attempted_install_with_missing_timing_unknown():
    from skills.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from skills.harness.reports.benchmark_insights import _elapsed_time_accounting_note, cost_comparison_section

    runs = {
        NO_SKILLS_MODE: {
            "label": "No skills baseline",
            "run": {"elapsed_seconds": 100, "token_count": 10},
            "activity": {},
            "workspace_delta": {},
            "agent_events_text": "",
        },
        WITH_SKILLS_MODE: {
            "label": "With skills",
            "run": {"elapsed_seconds": 120, "token_count": 12},
            "activity": {"commands": ["uv pip install -r requirements.txt"]},
            "workspace_delta": {},
            "agent_events_text": "",
        },
    }

    section = cost_comparison_section(runs, [NO_SKILLS_MODE, WITH_SKILLS_MODE])

    assert "| Runtime seconds | 100 | NA | NA |" in section
    assert "| Dependency install seconds | 0 | NA | NA |" in section
    accounting = _elapsed_time_accounting_note(runs[WITH_SKILLS_MODE], runs[NO_SKILLS_MODE])
    assert "| With skills | 120s | NA | NA | NA |" in accounting
    assert "NAs" not in accounting


def test_why_section_surfaces_repeated_successful_job_executions():
    from skills.harness.reports.benchmark_insights import _why_slower, job_run_status_reason, job_run_status_section

    def bash_pair(
        tool_id: str,
        command: str,
        description: str,
        start: str,
        end: str,
        output: str = "Finished FedAvg.",
    ) -> list[str]:
        return [
            json.dumps(
                {
                    "harness_timestamp": start,
                    "message": {
                        "content": [
                            {
                                "id": tool_id,
                                "input": {"command": command, "description": description},
                                "name": "Bash",
                                "type": "tool_use",
                            }
                        ]
                    },
                }
            ),
            json.dumps(
                {
                    "harness_timestamp": end,
                    "message": {
                        "content": [
                            {
                                "content": output,
                                "is_error": False,
                                "tool_use_id": tool_id,
                                "type": "tool_result",
                            }
                        ]
                    },
                    "tool_use_result": {"stdout": output, "stderr": ""},
                }
            ),
        ]

    with_run = {
        "available": True,
        "label": "With skills",
        "run": {"elapsed_seconds": 300},
        "agent_events_text": "\n".join(
            bash_pair(
                "run-1",
                "python3 job.py --num-sites 3 --num-rounds 3",
                "Run 3-site simulation",
                "2026-06-13T20:00:00Z",
                "2026-06-13T20:01:20Z",
            )
            + bash_pair(
                "run-2",
                "rm -rf /tmp/nvflare/workspaces/ames && python3 job.py --num-sites 3 --num-rounds 3",
                "Re-run simulation with aligned metric names",
                "2026-06-13T20:02:00Z",
                "2026-06-13T20:04:00Z",
            )
        ),
    }
    base_run = {
        "available": True,
        "label": "No skills baseline",
        "run": {"elapsed_seconds": 100},
        "agent_events_text": "\n".join(
            bash_pair(
                "base-run",
                "python3 job.py --num-sites 3 --num-rounds 3",
                "Run baseline simulation",
                "2026-06-13T20:00:00Z",
                "2026-06-13T20:01:00Z",
            )
        ),
    }

    reason = job_run_status_reason(with_run)
    status_section = job_run_status_section({"with": with_run}, ["with"])
    why_text = "\n".join(_why_slower(with_run, base_run))

    assert "2 successful job/simulator executions captured" in reason
    assert "total job time 200s" in reason
    assert "Re-run simulation with aligned metric names" in reason
    assert "### Repeated Job/Simulation Executions" not in status_section
    assert "### Repeated Job/Simulation Executions" in why_text
    assert "| With skills | 2 | 200s |" in why_text
    assert "runtime workspace was cleared before rerun" in why_text
    assert (
        "Baseline comparison: No skills baseline had 1 command classified successful job/simulator execution totaling 60s."
        in why_text
    )


def test_structure_tree_falls_back_to_final_workspace_when_changed_python_is_empty():
    from skills.harness.modes import WITH_SKILLS_MODE
    from skills.harness.reports.benchmark_insights import structure_trees_section

    report = structure_trees_section(
        {
            WITH_SKILLS_MODE: {
                "available": True,
                "label": "With skills",
                "workspace_delta": {
                    "changed_files": [
                        {"path": "nvflare_jobs/ames_fedavg/README.md"},
                        {"path": "nvflare_jobs/ames_fedavg/requirements.txt"},
                    ],
                    "final_structure_files": [
                        {"path": "download_data.py"},
                        {"path": "model.py"},
                        {"path": "nvflare_jobs/ames_fedavg/client.py"},
                        {"path": "nvflare_jobs/ames_fedavg/job.py"},
                        {"path": "nvflare_jobs/ames_fedavg/model.py"},
                    ],
                },
            }
        },
        [WITH_SKILLS_MODE],
    )

    assert "Final workspace:" in report
    assert "Changed/generated files:" in report
    assert "none" not in report
    assert "nvflare_jobs" in report
    assert "client.py" in report
    assert "job.py" in report
    assert "README.md" in report
    assert "requirements.txt" in report


def test_structure_score_does_not_count_nested_job_source_as_current_structure():
    from skills.harness.reports.benchmark_insights import (
        nested_generated_structure_display,
        structure_required_display,
        structure_score,
    )

    run = {
        "available": True,
        "workspace_delta": {
            "final_structure_files": [
                {"path": "model.py"},
                {"path": "nvflare_jobs/ames_fedavg/client.py"},
                {"path": "nvflare_jobs/ames_fedavg/job.py"},
                {"path": "nvflare_jobs/ames_fedavg/model.py"},
            ],
            "changed_files": [
                {"path": "nvflare_jobs/ames_fedavg/client.py"},
                {"path": "nvflare_jobs/ames_fedavg/job.py"},
                {"path": "nvflare_jobs/ames_fedavg/model.py"},
            ],
        },
    }

    assert structure_score(run) == 1 / 3
    assert structure_required_display(run).startswith("1/3 present; missing client.py, job.py")
    assert "nested copies ignored" in structure_required_display(run)
    assert nested_generated_structure_display(run) == "nvflare_jobs/ames_fedavg (client.py, job.py, model.py)"


def test_generated_code_quality_section_reports_evidence_without_gate_language(tmp_path):
    from skills.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from skills.harness.reports.benchmark_insights import generated_code_quality_section

    def command_event(command: str, output: str, exit_code: int = 0) -> str:
        status = "completed" if exit_code == 0 else "failed"
        return json.dumps(
            {
                "item": {
                    "aggregated_output": output,
                    "command": command,
                    "exit_code": exit_code,
                    "id": "item_1",
                    "status": status,
                    "type": "command_execution",
                },
                "type": "item.completed",
            }
        )

    def run_with_client(
        mode: str, client_source: str, install_command: str, install_output: str, rel_path: str = "client.py"
    ) -> dict:
        mode_dir = tmp_path / mode
        client_path = mode_dir / "workspace_delta" / "changed_files" / rel_path
        log_path = mode_dir / "workspace_delta" / "runtime_artifacts" / "site-1" / "log.txt"
        client_path.parent.mkdir(parents=True)
        log_path.parent.mkdir(parents=True)
        client_path.write_text(client_source, encoding="utf-8")
        log_path.write_text("[site-1] round 1 epoch 01 train_loss=0.4 device=cpu\n", encoding="utf-8")
        return {
            "available": True,
            "label": mode,
            "skills": "with skills" if mode == WITH_SKILLS_MODE else "without skills",
            "mode_dir": mode_dir,
            "workspace_delta": {
                "changed_files": [
                    {
                        "artifact_path": f"changed_files/{rel_path}",
                        "path": rel_path,
                    }
                ],
                "runtime_artifacts": [
                    {
                        "artifact_path": "runtime_artifacts/site-1/log.txt",
                        "path": "site-1/log.txt",
                        "source_path": "/tmp/nvflare/workspaces/job/site-1/log.txt",
                    }
                ],
            },
            "agent_events_text": command_event(install_command, install_output),
        }

    repeated_setup_client = """
import nvflare.client as flare
def partition_frame(frame, site_index, num_clients): return frame
while flare.is_running():
    input_model = flare.receive()
    train_frame, valid_frame, test_frame = load_data_frames(args.data_dir)
    train_frame = partition_frame(train_frame, site_index, args.num_clients)
    train_loader = make_loader(train_frame)
    criterion, optimizer, _ = build_loss_and_optimizer(model, train_frame, args, device)
    valid_metrics = evaluate(model, valid_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)
    append_record(results_path, {"metrics": test_metrics})
"""
    lean_client = """
import nvflare.client as flare
train_frame = load_split(args.data_dir, "train")
local_train = site_shard(train_frame, site_name)
train_loader = DataLoader(local_train)
criterion, optimizer, pos_weight_value = build_loss_and_optimizer(model, local_train, args, device)
while flare.is_running():
    input_model = flare.receive()
    for epoch in range(1, args.local_epochs + 1):
        print(f"[{site_name}] round {round_num} epoch {epoch:02d}")
    global_metrics = evaluate(model, valid_loader, criterion, device)
    local_metrics = evaluate(model, valid_loader, criterion, device)
"""
    runs = {
        NO_SKILLS_MODE: run_with_client(
            NO_SKILLS_MODE,
            lean_client,
            "python -m pip install torch --index-url https://download.pytorch.org/whl/cpu",
            "Successfully installed torch-2.12.0+cpu",
            rel_path="run_nvflare_fedavg.py",
        ),
        WITH_SKILLS_MODE: run_with_client(
            WITH_SKILLS_MODE,
            repeated_setup_client,
            "uv pip install -r requirements-train.txt",
            "Successfully installed nvidia-cublas-13.1 nvidia-cudnn-cu13-9.20 triton-3.7 torch-2.12",
        ),
    }

    section = generated_code_quality_section(runs, [NO_SKILLS_MODE, WITH_SKILLS_MODE])

    assert "These are evidence signals" in section
    assert "They do not change pass/fail quality gates" in section
    assert "Overall code quality signal" in section
    assert "/7 evidence points" in section
    assert "explicit sharding" in section
    assert "API pattern" in section
    assert "context: Client API loop pattern" in section
    assert "Loss/optimizer lifecycle" in section
    assert "Data/DataLoader lifecycle" in section
    assert "poor: loss/optimizer rebuilt inside FL loop" in section
    assert "good: loss/optimizer built outside FL loop" in section
    assert "good: data loaded before FL loop, DataLoader built before FL loop" in section
    assert "poor: data loaded inside FL loop, DataLoader built inside FL loop" in section
    assert "good: 2 evaluate call(s) in FL loop, test evaluation inside FL loop" in section
    assert "runtime logs show per-epoch progress" in section
    assert "runtime artifacts captured separately from temp/runtime paths" in section
    assert "CPU-only framework wheel" in section
    assert "accelerator-capable dependency stack" in section
    assert "skill requirements install not followed" not in section
    assert "CPU-only framework installs are faster, but they should only be treated as comparable" in section


def test_runtime_output_locality_scores_workspace_changes_as_caution():
    from skills.harness.reports.benchmark_insights import _assessment_from_locality, _runtime_output_locality_signal

    run = {
        "workspace_delta": {
            "changed_files": [{"path": "fl_workspace/ames_fedavg/server/simulate_job/app_server/custom/client.py"}],
            "runtime_artifacts": [
                {
                    "path": "server/log.txt",
                    "source_path": "/tmp/nvflare/workspaces/ames/server/log.txt",
                }
            ],
        }
    }

    evidence = _runtime_output_locality_signal(run)

    assert "runtime artifacts captured separately from temp/runtime paths" in evidence
    assert "runtime output appears in workspace changes" in evidence
    assert _assessment_from_locality(evidence) == "caution"


def test_generated_code_quality_does_not_claim_loop_placement_when_loop_missing(tmp_path):
    from skills.harness.modes import NO_SKILLS_MODE
    from skills.harness.reports.benchmark_insights import generated_code_quality_section

    mode_dir = tmp_path / NO_SKILLS_MODE
    source_path = mode_dir / "workspace_delta" / "changed_files" / "run_nvflare_fedavg.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        """
from torch.utils.data import DataLoader

train_frame = load_split(args.data_dir, "train")
train_loader = DataLoader(train_frame)
test_frame = load_split(args.data_dir, "test")
test_loader = DataLoader(test_frame)
criterion, optimizer = build_loss_and_optimizer(model, train_frame, args, device)
metric = evaluate(model, test_loader, criterion, device)
""",
        encoding="utf-8",
    )
    run = {
        "available": True,
        "label": NO_SKILLS_MODE,
        "mode_dir": mode_dir,
        "workspace_delta": {
            "changed_files": [
                {
                    "artifact_path": "changed_files/run_nvflare_fedavg.py",
                    "path": "run_nvflare_fedavg.py",
                }
            ]
        },
    }

    section = generated_code_quality_section({NO_SKILLS_MODE: run}, [NO_SKILLS_MODE])

    assert "caution: loss/optimizer setup present; FL loop not captured" in section
    assert "caution: data loading present, DataLoader construction present; FL loop not captured" in section
    assert "good: 1 evaluate call(s) in generated code, test evaluation present, FL loop not captured" in section
    assert "data loaded before FL loop; FL loop not captured" not in section
    assert "evaluate call(s) in FL loop" not in section
    assert "test evaluation inside FL loop" not in section


def test_generated_code_quality_detects_model_learner_api_pattern_without_filename_bias(tmp_path):
    from skills.harness.modes import NO_SKILLS_MODE
    from skills.harness.reports.benchmark_insights import generated_code_quality_section

    mode_dir = tmp_path / NO_SKILLS_MODE
    source_path = mode_dir / "workspace_delta" / "changed_files" / "custom_training_component.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        """
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.abstract.model_learner import ModelLearner

class CustomTrainingComponent(ModelLearner):
    def initialize(self):
        train_frame, valid_frame, test_frame = load_data_frames(self.data_dir)
        self._train_loader, self._valid_loader, self._test_loader = build_data_loaders(
            train_frame, valid_frame, test_frame, self.vocab, self._args
        )
        self._criterion, _, _ = build_loss_and_optimizer(self._model, train_frame, self._args, self._device)

    def train(self, model: FLModel) -> FLModel:
        optimizer = torch.optim.AdamW(self._model.parameters())
        train_one_epoch(self._model, self._train_loader, self._criterion, optimizer, self._device)
        valid_metrics = evaluate(self._model, self._valid_loader, self._criterion, self._device)
        return FLModel(metrics=valid_metrics)
""",
        encoding="utf-8",
    )
    run = {
        "available": True,
        "label": NO_SKILLS_MODE,
        "mode_dir": mode_dir,
        "workspace_delta": {
            "changed_files": [
                {
                    "artifact_path": "changed_files/custom_training_component.py",
                    "path": "custom_training_component.py",
                }
            ]
        },
    }

    section = generated_code_quality_section({NO_SKILLS_MODE: run}, [NO_SKILLS_MODE])

    assert "context: ModelLearner pattern" in section
    assert "poor: loss/optimizer rebuilt inside FL loop" in section
    assert "good: data loaded before FL loop, DataLoader built before FL loop" in section


def test_dependency_strategy_scores_with_skills_cpu_shortcut_as_instruction_failure():
    from skills.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from skills.harness.reports.benchmark_insights import generated_code_quality_section

    def run(mode: str, command: str, output: str) -> dict:
        return {
            "available": True,
            "label": mode,
            "skills": "with skills" if mode == WITH_SKILLS_MODE else "without skills",
            "agent_events_text": json.dumps(
                {
                    "item": {
                        "aggregated_output": output,
                        "command": command,
                        "exit_code": 0,
                        "id": "item_1",
                        "status": "completed",
                        "type": "command_execution",
                    },
                    "type": "item.completed",
                }
            ),
            "workspace_delta": {},
        }

    section = generated_code_quality_section(
        {
            NO_SKILLS_MODE: run(
                NO_SKILLS_MODE,
                "python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu",
                "Successfully installed torch-2.12.0+cpu",
            ),
            WITH_SKILLS_MODE: run(
                WITH_SKILLS_MODE,
                "python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu",
                "Successfully installed torch-2.12.0+cpu",
            ),
        },
        [NO_SKILLS_MODE, WITH_SKILLS_MODE],
    )

    assert "caution: targeted package install, CPU-only framework wheel, succeeded" in section
    assert (
        "poor: targeted package install, CPU-only framework wheel, succeeded, skill requirements install not followed"
        in section
    )


def test_workspace_delta_issue_allows_final_structure_and_runtime_artifacts():
    from skills.harness.reports.benchmark_insights import run_quality_issues

    run = {
        "available": True,
        "record": {
            "workspace_delta": {
                "changed_file_count": 0,
                "runtime_artifact_count": 37,
                "final_structure_files": [
                    {"path": "nvflare_jobs/ames_fedavg/client.py"},
                    {"path": "nvflare_jobs/ames_fedavg/job.py"},
                ],
            }
        },
    }

    issues = run_quality_issues(run)

    assert not any("workspace_delta" in issue for issue in issues)


def test_workspace_delta_issue_allows_manifest_counts_without_file_lists():
    from skills.harness.reports.benchmark_insights import run_quality_issues

    run = {
        "available": True,
        "record": {
            "workspace_delta": {
                "changed_file_count": 0,
                "copied_file_count": 37,
                "final_structure_file_count": 5,
                "runtime_artifact_count": 37,
            }
        },
    }

    issues = run_quality_issues(run)

    assert not any("workspace_delta" in issue for issue in issues)


def test_status_summary_is_human_readable_for_failures():
    from skills.harness.modes import NO_SKILLS_MODE
    from skills.harness.reports.benchmark_insights import status_summary

    runs = {
        NO_SKILLS_MODE: {
            "available": True,
            "container_exit": {"exit_code": 1},
            "console_text": "docker: Error response from daemon: pull access denied for agent-skills-benchmark",
            "run": {},
            "status": "missing",
            "validation_metric": {},
        }
    }

    summary = status_summary(runs, [NO_SKILLS_MODE])

    assert "No skills baseline: failed" in summary
    assert "container exit 1" in summary
    assert "Docker image unavailable" in summary
    assert "exit=1" not in summary


def test_failure_analysis_extracts_unsupported_model_message():
    from skills.harness.reports.benchmark_insights import failure_evidence, failure_root_cause

    run = {
        "available": True,
        "agent_events_text": "The 'gpt-5.3-codex' model is not supported when using Codex with a ChatGPT account.",
        "container_exit": {"exit_code": 1},
        "run": {"agent_exit_code": 1},
        "status": "missing",
        "validation_metric": {},
    }

    assert failure_root_cause(run) == (
        "Agent model selection failed: The 'gpt-5.3-codex' model is not supported when using Codex with a ChatGPT account."
    )
    assert failure_evidence(run) == (
        "The 'gpt-5.3-codex' model is not supported when using Codex with a ChatGPT account."
    )


def test_failure_root_cause_prefers_agent_exit_classifier():
    from skills.harness.reports.benchmark_insights import failure_root_cause

    run = {
        "available": True,
        "agent_events_text": "unstructured error text",
        "record": {"agent_exit_summary": {"failure_category": "agent_auth_failure"}},
        "run": {"agent_exit_code": 1},
    }

    assert failure_root_cause(run) == "Agent failure category: agent_auth_failure"


def test_failure_root_cause_infers_auth_from_agent_last_message():
    from skills.harness.modes import WITH_SKILLS_MODE
    from skills.harness.reports.benchmark_insights import (
        failure_analysis_section,
        failure_root_cause,
        job_run_action,
        job_run_status_reason,
    )

    run = {
        "available": True,
        "agent_events_text": '{"error":"authentication_failed","message":{"content":[{"text":"Not logged in"}]}}',
        "agent_last_message": "Not logged in - Please run /login",
        "container_exit": {"exit_code": 1},
        "record": {"agent_exit_summary": {"failure_category": "agent_unknown_failure"}},
        "run": {"final_container_exit_code": 1},
    }

    assert failure_root_cause(run) == "Agent failure category: agent_auth_failure"
    assert "agent_auth_failure" in job_run_status_reason(run)
    assert "Not logged in" in job_run_status_reason(run)
    assert '{"error"' not in job_run_status_reason(run)
    assert "Authenticate the selected agent" in job_run_action(run)

    section = failure_analysis_section({WITH_SKILLS_MODE: run}, [WITH_SKILLS_MODE])
    assert "Job run status: not_started" in section
    assert "agent_auth_failure" in section
    assert "Not logged in" in section


def test_failure_analysis_identifies_agent_generated_requirements_file():
    from skills.harness.reports.benchmark_insights import dependency_reference_notes

    run = {
        "agent_last_message": "Install with python3 -m pip install -r requirements-federated.txt.",
        "record": {
            "source_input_delta": {
                "final_files": [
                    {"path": "requirements-train.txt"},
                ]
            },
            "workspace_delta": {
                "workspace_added_files": [
                    {"path": "requirements-federated.txt"},
                ]
            },
        },
    }

    assert dependency_reference_notes(run) == [
        "`requirements-federated.txt` provenance: agent-generated file.",
    ]


def test_readme_metric_alignment_uses_aggregated_validation_metric_scalar():
    from skills.harness.quality_signals import metric_signal

    signal = metric_signal(
        None,
        "AUROC is the main metric.\n",
        """
Round 2 validation AUROC by site:
- `site-1`: `0.7659574468`
- `site-2`: `0.7554566645`
- `site-3`: `0.7373779931`
- aggregated best validation metric: `0.7529307015`
""",
    )

    metric = signal["reported_validation_metric"]
    assert signal["status"] == "pass"
    assert signal["aligned_with_readme"] is True
    assert signal["metric_value_available"] is True
    assert signal["metric_scalar_available"] is True
    assert metric["name"] == "AUROC"
    assert metric["value"] == 0.7529307015
    assert metric["value_scope"] == "fl_summary_metric"
    assert metric["site_value_count"] == 3
    assert metric["summary_value_label"] == "aggregated best validation metric"


def test_readme_metric_alignment_uses_named_aggregated_metric_scalar():
    from skills.harness.quality_signals import metric_signal

    signal = metric_signal(
        None,
        "AUROC is the main metric.\n",
        """
Validation:
- Local training AUROC: 0.7531
- Best aggregated validation AUROC: 0.7623334631865992
- Final site metrics: site-1 valid AUROC 0.767293, site-2 valid AUROC 0.757374
""",
    )

    metric = signal["reported_validation_metric"]
    assert signal["status"] == "pass"
    assert signal["metric_scalar_available"] is True
    assert metric["name"] == "AUROC"
    assert metric["value"] == 0.7623334631865992
    assert metric["value_scope"] == "fl_summary_metric"
    assert metric["summary_value_label"] == "Best aggregated validation AUROC"


def test_metric_alignment_rejects_out_of_range_auroc_from_dependency_version():
    from skills.harness.quality_signals import metric_signal

    signal = metric_signal(
        None,
        "Primary validation metric: AUROC.\n",
        """
The job uses NVFLARE 2.8 recipe APIs.
The job config uses AUROC as key_metric.
Requirements:
- nvflare[PT]>=2.8.0

No simulation was run.
""",
    )

    metric = signal["reported_validation_metric"]
    assert signal["status"] == "missing"
    assert signal["metric_value_available"] is False
    assert metric["name"] == "AUROC"
    assert metric["value"] is None
    assert metric["reported_values"] == []


def test_benchmark_report_sanitizes_stale_out_of_range_metric_record():
    from skills.harness.reports.benchmark_insights import quality_signal, validation_metric_from_record

    record = {
        "reported_validation_metric": {
            "name": "AUROC",
            "reported_value_entries": [{"value": 2.8}],
            "reported_values": [2.8],
            "source": "agent_last_message",
            "value": 2.8,
            "value_scope": "reported_scalar",
        },
        "quality_signals": {
            "job_guidance_primary_validation_metric": {
                "expected_primary_metric": "AUROC",
                "evidence": "Job guidance declares AUROC as the primary metric, and the final response reported AUROC 2.8000.",
                "metric_value_available": True,
                "reported_validation_metric": {
                    "name": "AUROC",
                    "reported_value_entries": [{"value": 2.8}],
                    "reported_values": [2.8],
                    "value": 2.8,
                    "value_scope": "reported_scalar",
                },
                "status": "pass",
            }
        },
    }

    metric = validation_metric_from_record(record)
    signal = quality_signal(record)

    assert metric["name"] == "AUROC"
    assert metric["value"] is None
    assert metric["reported_values"] == []
    assert signal["status"] == "missing"
    assert signal["metric_value_available"] is False
    assert "did not report a plausible numeric value" in signal["evidence"]


def test_job_guidance_metric_alignment_uses_non_readme_docs(tmp_path):
    from skills.harness.quality_signals import metric_signal
    from skills.harness.records import discover_job_guidance

    job = tmp_path / "job"
    docs = job / "docs"
    docs.mkdir(parents=True)
    docs.joinpath("metrics.md").write_text("Target validation metric: accuracy.\n", encoding="utf-8")

    sources, guidance_text = discover_job_guidance(job)
    signal = metric_signal(
        sources,
        guidance_text,
        "Server best validation metric at round 3: 0.8123 accuracy",
    )

    assert signal["expected_primary_metric"] == "accuracy"
    assert signal["aligned_with_job_guidance"] is True
    assert signal["sources"][0]["path"].endswith("metrics.md")
    assert signal["reported_validation_metric"]["name"] == "accuracy"


def test_job_guidance_skips_symlink_guidance_files(tmp_path):
    from skills.harness import records

    job = tmp_path / "job"
    job.mkdir()
    job.joinpath("target.md").write_text("Target validation metric: accuracy.\n", encoding="utf-8")
    left = job / "readme-left.md"
    right = job / "readme-right.md"
    try:
        left.symlink_to("target.md")
        right.symlink_to("target.md")
    except (OSError, NotImplementedError):
        return

    sources, guidance_text = records.discover_job_guidance(job)

    assert sources == []
    assert guidance_text == ""


def test_job_guidance_skips_symlinked_docs_directory(tmp_path):
    from skills.harness import records

    job = tmp_path / "job"
    outside_docs = tmp_path / "outside_docs"
    job.mkdir()
    outside_docs.mkdir()
    outside_docs.joinpath("README.md").write_text("Target validation metric: accuracy.\n", encoding="utf-8")
    try:
        job.joinpath("docs").symlink_to(outside_docs, target_is_directory=True)
    except (OSError, NotImplementedError):
        return

    sources, guidance_text = records.discover_job_guidance(job)

    assert sources == []
    assert guidance_text == ""


def test_job_guidance_skips_oversized_guidance_files(tmp_path):
    from skills.harness.records import MAX_GUIDANCE_FILE_BYTES, discover_job_guidance

    job = tmp_path / "job"
    job.mkdir()
    with job.joinpath("README.md").open("wb") as stream:
        stream.truncate(MAX_GUIDANCE_FILE_BYTES + 1)

    sources, guidance_text = discover_job_guidance(job)

    assert sources == []
    assert guidance_text == ""


def test_job_guidance_stops_collecting_after_guidance_file_cap(tmp_path, monkeypatch):
    from skills.harness import records

    job = tmp_path / "job"
    job.mkdir()
    for index in range(20):
        job.joinpath(f"readme-{index:02d}.md").write_text("Target validation metric: accuracy.\n", encoding="utf-8")
    calls = {"count": 0}

    def counted_is_guidance_file(path):
        calls["count"] += 1
        return True

    monkeypatch.setattr(records, "MAX_GUIDANCE_FILES", 3)
    monkeypatch.setattr(records, "is_guidance_file", counted_is_guidance_file)

    sources, _guidance_text = records.discover_job_guidance(job)

    assert len(sources) == 3
    assert calls["count"] == 3


def test_job_guidance_metric_alignment_includes_prompt(tmp_path):
    from skills.harness.quality_signals import metric_signal
    from skills.harness.records import discover_job_guidance

    job = tmp_path / "job"
    job.mkdir()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Convert this job. Primary validation metric: AUROC.\n", encoding="utf-8")

    sources, guidance_text = discover_job_guidance(job, prompt)
    signal = metric_signal(
        sources,
        guidance_text,
        "Aggregated best validation metric: 0.7529 AUROC",
    )

    assert signal["expected_primary_metric"] == "AUROC"
    assert signal["aligned_with_job_guidance"] is True
    assert signal["sources"][0]["source_type"] == "prompt"


def test_job_guidance_metric_alignment_uses_source_priority(tmp_path):
    from skills.harness.quality_signals import metric_signal
    from skills.harness.records import discover_job_guidance

    job = tmp_path / "job"
    job.mkdir()
    job.joinpath("README.md").write_text("AUROC is the main metric.\n", encoding="utf-8")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Convert this job. Primary validation metric: accuracy.\n", encoding="utf-8")

    sources, guidance_text = discover_job_guidance(job, prompt)
    signal = metric_signal(
        sources,
        guidance_text,
        "Server best validation metric at round 3: 0.8123 accuracy",
    )

    assert signal["expected_primary_metric"] == "accuracy"
    assert signal["source"] == str(prompt)
    assert signal["matched_source"] == {"path": str(prompt), "source_type": "prompt"}
    assert signal["aligned_with_job_guidance"] is True


def test_job_guidance_metric_alignment_reports_matched_doc_source(tmp_path):
    from skills.harness.quality_signals import metric_signal
    from skills.harness.records import discover_job_guidance

    job = tmp_path / "job"
    job.mkdir()
    readme = job / "README.md"
    readme.write_text("AUROC is the main metric.\n", encoding="utf-8")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Convert this job with NVFLARE.\n", encoding="utf-8")

    sources, guidance_text = discover_job_guidance(job, prompt)
    signal = metric_signal(
        sources,
        guidance_text,
        "Aggregated best validation metric: 0.7529 AUROC",
    )

    assert signal["expected_primary_metric"] == "AUROC"
    assert signal["source"] == str(readme)
    assert signal["matched_source"] == {"path": str(readme), "source_type": "job_documentation"}
    assert signal["sources"][0]["source_type"] == "prompt"
    assert signal["aligned_with_job_guidance"] is True


def test_metric_mismatch_reports_actual_metric_without_marking_missing():
    from skills.harness.modes import NO_SKILLS_MODE
    from skills.harness.quality_signals import metric_signal
    from skills.harness.reports.benchmark_insights import (
        benchmark_outcome,
        human_readable_status,
        missing_result_metrics_section,
        outcome_metrics_table,
    )

    signal = metric_signal(
        None,
        "AUROC is the main metric.\n",
        "Best validation accuracy: 0.8123",
    )
    run = {
        "available": True,
        "label": "No skills baseline",
        "container_exit": {"exit_code": 0},
        "run": {"final_container_exit_code": 0},
        "record": {"quality_signals": {"job_guidance_primary_validation_metric": signal}},
        "validation_metric": signal["reported_validation_metric"],
    }
    runs = {NO_SKILLS_MODE: run}

    assert signal["mismatch"] is True
    assert signal["reported_validation_metric"]["name"] == "accuracy"
    assert "completed with metric mismatch" in human_readable_status(run)
    assert benchmark_outcome(run).startswith("warn:")
    assert "accuracy 0.8123" in missing_result_metrics_section(runs, [NO_SKILLS_MODE])
    assert "no parseable validation metric" not in missing_result_metrics_section(runs, [NO_SKILLS_MODE])
    assert "| Metrics (accuracy) | accuracy 0.8123 |" in outcome_metrics_table(runs, [NO_SKILLS_MODE])


def test_artifact_metric_satisfies_result_gate_when_final_response_metric_is_incomplete():
    from skills.harness.modes import WITH_SKILLS_MODE
    from skills.harness.reports.benchmark_insights import (
        benchmark_outcome,
        failure_analysis_section,
        human_readable_status,
        missing_result_metrics_section,
        quality_signal_table,
        run_quality_issues,
    )

    run = {
        "available": True,
        "label": "With skills",
        "container_exit": {"exit_code": 0},
        "run": {"final_container_exit_code": 0},
        "record": {
            "quality_signals": {
                "job_guidance_primary_validation_metric": {
                    "status": "missing",
                    "expected_primary_metric": "AUROC",
                    "evidence": (
                        "Job guidance declares AUROC as the primary metric, and the final response mentioned "
                        "AUROC but did not report a plausible numeric value."
                    ),
                    "reported_validation_metric": {
                        "name": "AUROC",
                        "value": None,
                        "reported_values": [],
                        "reported_value_entries": [],
                    },
                }
            }
        },
        "validation_metric": {
            "name": "AUROC",
            "source": "metrics_artifact",
            "summary_value_label": "artifact aggregated validation metric final_aggregated_metrics.[2].value",
            "value": 0.7816101804960395,
            "value_scope": "fl_summary_metric",
        },
    }
    runs = {WITH_SKILLS_MODE: run}

    assert run_quality_issues(run) == []
    assert human_readable_status(run) == "passed"
    assert benchmark_outcome(run) == "pass: scalar FL result metric available"
    assert missing_result_metrics_section(runs, [WITH_SKILLS_MODE]) == ""
    failure_analysis = failure_analysis_section(runs, [WITH_SKILLS_MODE])
    assert "Outcome: passed. AUROC 0.7816" in failure_analysis
    assert "Reporting note: Final response reporting gap" in failure_analysis
    quality_table = quality_signal_table(runs, [WITH_SKILLS_MODE])
    assert "artifact metric present; final response gap" in quality_table


def test_metric_mismatch_evidence_includes_integer_metric_value():
    from skills.harness.quality_signals import format_metric_value

    assert format_metric_value(1) == " 1."
    assert format_metric_value(1.0) == " 1.0000."
    assert format_metric_value(None) == "."


def test_missing_target_metric_section_reports_observed_alternate_metrics():
    from skills.harness.modes import NO_SKILLS_MODE
    from skills.harness.reports.benchmark_insights import (
        additional_or_observed_metric_values_display,
        missing_result_metrics_section,
        outcome_details_table,
    )

    run = {
        "available": True,
        "label": "No skills baseline",
        "container_exit": {"exit_code": 0},
        "run": {"final_container_exit_code": 0},
        "record": {
            "quality_signals": {
                "job_guidance_primary_validation_metric": {
                    "status": "missing",
                    "expected_primary_metric": "AUROC",
                    "evidence": "Job guidance declares AUROC as the primary metric, but the final response did not report it.",
                    "reported_validation_metric": {
                        "name": None,
                        "value": None,
                        "reported_values": [],
                        "reported_value_entries": [],
                    },
                }
            }
        },
        "validation_metric": {"name": None, "value": None, "reported_values": [], "reported_value_entries": []},
        "agent_last_message": "Validation accuracy: 0.8123\nValidation loss: 0.421",
    }

    section = missing_result_metrics_section({NO_SKILLS_MODE: run}, [NO_SKILLS_MODE])

    assert "accuracy 0.8123" in section
    assert "loss 0.4210" in section
    assert "no parseable validation metric" not in section
    assert additional_or_observed_metric_values_display(run, "AUROC") == "accuracy 0.8123; loss 0.4210"
    assert "Additional/other validation metric values" in outcome_details_table({NO_SKILLS_MODE: run}, [NO_SKILLS_MODE])


def test_failure_analysis_reports_recovered_job_failure_and_metric_gap():
    from skills.harness.modes import NO_SKILLS_MODE
    from skills.harness.reports.benchmark_insights import (
        additional_or_observed_metric_values_display,
        failure_analysis_section,
        outcome_details_table,
    )

    failed_output = (
        "TypeError: SmilesCNN.__init__() missing 4 required positional arguments: "
        "'vocab_size', 'embed_dim', 'num_filters', and 'dropout'\n"
        "RuntimeError: Simulator run failed with exit code 2.\n"
    )
    success_output = (
        "Finished FedAvg.\n"
        "site-1: round=0 train_loss=0.6275 valid_auroc=0.7049\n"
        "site-2: round=0 train_loss=0.6259 valid_auroc=0.7342\n"
        "Result workspace: /tmp/agent_benchmark/ames-smoke\n"
    )
    events = [
        {
            "item": {
                "type": "command_execution",
                "id": "item_1",
                "command": "python3 fedavg_job.py --n-clients 2",
                "status": "failed",
                "exit_code": 1,
                "aggregated_output": failed_output,
            }
        },
        {
            "item": {
                "type": "command_execution",
                "id": "item_2",
                "command": "python3 fedavg_job.py --n-clients 2",
                "status": "completed",
                "exit_code": 0,
                "aggregated_output": success_output,
            }
        },
    ]
    run = {
        "available": True,
        "label": "No skills baseline",
        "container_exit": {"exit_code": 0},
        "run": {"final_container_exit_code": 0},
        "record": {
            "quality_signals": {
                "job_guidance_primary_validation_metric": {
                    "status": "missing",
                    "expected_primary_metric": "AUROC",
                    "evidence": "Job guidance declares AUROC as the primary metric, but the final response did not report it.",
                    "reported_validation_metric": {
                        "name": None,
                        "value": None,
                        "reported_values": [],
                        "reported_value_entries": [],
                    },
                }
            }
        },
        "validation_metric": {"name": None, "value": None, "reported_values": [], "reported_value_entries": []},
        "agent_events_text": "\n".join(json.dumps(event) for event in events),
    }

    section = failure_analysis_section({NO_SKILLS_MODE: run}, [NO_SKILLS_MODE])

    assert "Command Evidence" in section
    assert "recovered by a later successful similar command" in section
    assert "SmilesCNN.__init__() missing 4 required positional arguments" in section
    assert "Recovery evidence" in section
    assert "a later simulator/job command exited 0" in section
    assert "valid_auroc=0.7049" in section
    assert "Metric reporting gap" in section
    assert "aggregate `AUROC` scalar" in section
    assert additional_or_observed_metric_values_display(run, "AUROC") == (
        "Final site metrics=NA; log/per-site evidence: site-1: round=0 train_loss=0.6275 valid_auroc=0.7049; "
        "site-2: round=0 train_loss=0.6259 valid_auroc=0.7342"
    )
    details = outcome_details_table({NO_SKILLS_MODE: run}, [NO_SKILLS_MODE])
    assert "Reported validation metric | AUROC NA" in details
    assert "log/per-site evidence" in details


def test_job_run_status_section_reports_bash_blocked_not_started(tmp_path):
    from skills.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from skills.harness.reports.benchmark_insights import (
        bash_blocked_diagnostic,
        benchmark_report,
        job_run_status,
        job_run_status_section,
    )

    permission_result = {
        "message": {
            "content": [
                {
                    "content": "Claude requested permissions to use Bash, but you haven't granted it yet.",
                    "type": "tool_result",
                }
            ]
        },
        "type": "user",
    }
    final_result = {
        "final_message": "Usage:\n```bash\ncd nvflare_jobs/ames_fedavg\npython job.py\n```",
        "permission_denials": [{"tool_name": "Bash", "tool_input": {"command": "python job.py"}}],
        "subtype": "success",
        "type": "result",
    }
    run = {
        "available": True,
        "label": "With skills",
        "activity": {
            "commands": ["find /workspace/run/with_skills/workspace -type f"],
            "hint_counts": {"python_job_py": 0, "simulation": 0},
        },
        "agent_events_text": "\n".join(json.dumps(event) for event in (permission_result, final_result)),
        "agent_last_message": final_result["final_message"],
        "container_exit": {"exit_code": 0},
        "record": {},
        "run": {"final_container_exit_code": 0},
    }
    runs = {
        NO_SKILLS_MODE: {"available": False, "label": "No skills baseline"},
        WITH_SKILLS_MODE: run,
    }

    assert job_run_status(run) == "not_started"

    section = job_run_status_section(runs, [NO_SKILLS_MODE, WITH_SKILLS_MODE])
    report = benchmark_report(tmp_path, runs)
    assert "## Job Run Status" in report
    assert "| Job execution | No skills baseline: unknown" in report
    assert "With skills: not_started (Bash blocked 1 time(s)" in report
    assert "| With skills | not_started | Bash blocked 1 time(s)" in section
    assert "Fix agent Bash/tool permissions and rerun" in section
    diagnostic = bash_blocked_diagnostic(run)
    assert "--tools" in diagnostic
    assert "--allowedTools" not in diagnostic
    assert "python job.py" not in section


def test_job_run_status_uses_claude_bash_output_to_detect_completed_simulation():
    from skills.harness.reports.benchmark_insights import job_run_status, job_run_status_reason

    tool_id = "toolu_job"
    command_event = {
        "event_type": "assistant",
        "message": {
            "content": [
                {
                    "id": tool_id,
                    "input": {"command": "timeout 300 python job.py --num-rounds 1"},
                    "name": "Bash",
                    "type": "tool_use",
                }
            ]
        },
    }
    result_event = {
        "event_type": "user",
        "message": {
            "content": [
                {
                    "content": "site-1: round=0 train_loss=0.5440 valid_auroc=0.7254\nFinished FedAvg.\nSimulation workspace: /tmp/nvflare/workspaces/ames_fedavg",
                    "is_error": False,
                    "tool_use_id": tool_id,
                    "type": "tool_result",
                }
            ]
        },
        "tool_use_result": {
            "interrupted": False,
            "stderr": "",
            "stdout": "site-1: round=0 train_loss=0.5440 valid_auroc=0.7254\nFinished FedAvg.\nSimulation workspace: /tmp/nvflare/workspaces/ames_fedavg",
        },
    }
    run = {
        "available": True,
        "activity": {"commands": ["timeout 300 python job.py --num-rounds 1"]},
        "agent_events_text": "\n".join(json.dumps(event) for event in (command_event, result_event)),
    }

    assert job_run_status(run) == "completed"
    assert job_run_status_reason(run) == "simulation completed — FL workflow reached Finished state"


def test_agent_command_spans_pair_claude_bash_tool_use_and_result():
    from skills.harness.reports.benchmark_insights import agent_command_spans

    tool_id = "toolu_install"
    command_event = {
        "event_type": "assistant",
        "harness_timestamp": "2026-06-13T00:00:00Z",
        "message": {
            "content": [
                {
                    "id": tool_id,
                    "input": {"command": "uv pip install -r requirements.txt"},
                    "name": "Bash",
                    "type": "tool_use",
                }
            ]
        },
    }
    result_event = {
        "event_type": "user",
        "harness_timestamp": "2026-06-13T00:00:25Z",
        "message": {
            "content": [
                {
                    "content": "Successfully installed dependencies",
                    "is_error": False,
                    "tool_use_id": tool_id,
                    "type": "tool_result",
                }
            ]
        },
        "tool_use_result": {
            "interrupted": False,
            "stderr": "",
            "stdout": "Successfully installed dependencies",
        },
    }
    run = {"agent_events_text": "\n".join(json.dumps(event) for event in (command_event, result_event))}

    spans = agent_command_spans(run)

    assert spans == [
        {
            "command": "uv pip install -r requirements.txt",
            "description": "",
            "duration_seconds": 25.0,
            "exit_code": 0,
            "id": tool_id,
            "index": 0,
            "output": "Successfully installed dependencies",
            "status": "completed",
        }
    ]


def test_job_run_status_detects_generated_simulation_entrypoint():
    from skills.harness.reports.benchmark_insights import job_run_status, job_run_status_reason

    event = {
        "item": {
            "aggregated_output": (
                "Finished FedAvg.\n"
                "Simulation workspace: outputs/nvflare_workspace/ames_fedavg\n"
                "Final weighted validation metrics: AUROC 0.7592"
            ),
            "command": "python3 run_nvflare_simulation.py",
            "exit_code": 0,
            "id": "item_1",
            "status": "completed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ["python3 run_nvflare_simulation.py"]},
        "agent_events_text": json.dumps(event),
    }

    assert job_run_status(run) == "completed"
    assert job_run_status_reason(run) == "simulation completed — FL workflow reached Finished state"


def test_job_run_status_detects_wrapper_that_invokes_nvflare_simulator():
    from skills.harness.reports.benchmark_insights import job_run_status, job_run_status_reason

    event = {
        "item": {
            "aggregated_output": (
                "Running: /workspace/venv/bin/python3 -m nvflare.cli simulator "
                "/workspace/fl_job/ames_fedavg -w /workspace/fl_workspace -n 3 -t 3\n"
                "Finished FedAvg.\n"
                "Simulation workspace: /workspace/fl_workspace\n"
            ),
            "command": "python3 run_nvflare_fedavg.py",
            "exit_code": 0,
            "id": "item_1",
            "status": "completed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ["python3 run_nvflare_fedavg.py"]},
        "agent_events_text": json.dumps(event),
    }

    assert job_run_status(run) == "completed"
    assert job_run_status_reason(run) == "simulation completed — FL workflow reached Finished state"


def test_job_run_status_uses_metric_artifact_to_avoid_not_started_contradiction():
    from skills.harness.reports.benchmark_insights import job_run_status, job_run_status_reason

    run = {
        "available": True,
        "activity": {"commands": ["/bin/bash -lc 'rg --files'"]},
        "validation_metric": {
            "name": "AUROC",
            "reported_values": [0.7652],
            "source": "metrics_artifact",
            "source_path": "/workspace/results/workspace_delta/runtime_artifacts/server/metrics/summary.json",
        },
    }

    assert job_run_status(run) == "completed"
    reason = job_run_status_reason(run)
    assert "job execution inferred from captured runtime metric artifact" in reason
    assert "summary.json" in reason
    assert "command detector did not identify" in reason


def test_job_run_status_does_not_infer_completion_from_changed_file_metric_artifact():
    from skills.harness.reports.benchmark_insights import job_run_status

    run = {
        "available": True,
        "activity": {"commands": ["/bin/bash -lc 'rg --files'"]},
        "validation_metric": {
            "name": "AUROC",
            "reported_values": [0.7652],
            "source": "metrics_artifact",
            "source_path": (
                "/workspace/results/workspace_delta/changed_files/fl_workspace/ames_fedavg/"
                "server/simulate_job/metrics/metrics_summary.json"
            ),
        },
    }

    assert job_run_status(run) == "not_started"


def test_job_run_status_ignores_successful_simulation_helper_scripts():
    from skills.harness.reports.benchmark_insights import job_run_status

    events = [
        {
            "item": {
                "aggregated_output": "simulation config ok",
                "command": "python check_simulation_config.py",
                "exit_code": 0,
                "id": "item_1",
                "status": "completed",
                "type": "command_execution",
            }
        },
        {
            "item": {
                "aggregated_output": "simulator import ok",
                "command": "python validate_simulator_install.py",
                "exit_code": 0,
                "id": "item_2",
                "status": "completed",
                "type": "command_execution",
            }
        },
        {
            "item": {
                "aggregated_output": "nvflare import ok",
                "command": "python check_nvflare_install.py",
                "exit_code": 0,
                "id": "item_3",
                "status": "completed",
                "type": "command_execution",
            }
        },
        {
            "item": {
                "aggregated_output": "job config ok",
                "command": "python validate_job_config.py",
                "exit_code": 0,
                "id": "item_4",
                "status": "completed",
                "type": "command_execution",
            }
        },
        {
            "item": {
                "aggregated_output": "job setup ok",
                "command": "python check_job_setup.py",
                "exit_code": 0,
                "id": "item_5",
                "status": "completed",
                "type": "command_execution",
            }
        },
        {
            "item": {
                "aggregated_output": "job helper ok",
                "command": "python validate_job.py",
                "exit_code": 0,
                "id": "item_6",
                "status": "completed",
                "type": "command_execution",
            }
        },
        {
            "item": {
                "aggregated_output": "job helper ok",
                "command": "python check_job.py",
                "exit_code": 0,
                "id": "item_7",
                "status": "completed",
                "type": "command_execution",
            }
        },
        {
            "item": {
                "aggregated_output": "job tests passed",
                "command": "python run_job_tests.py",
                "exit_code": 0,
                "id": "item_8",
                "status": "completed",
                "type": "command_execution",
            }
        },
    ]
    run = {
        "available": True,
        "activity": {
            "commands": [
                "python check_simulation_config.py",
                "python validate_simulator_install.py",
                "python check_nvflare_install.py",
                "python validate_job_config.py",
                "python check_job_setup.py",
                "python validate_job.py",
                "python check_job.py",
                "python run_job_tests.py",
            ]
        },
        "agent_events_text": "\n".join(json.dumps(event) for event in events),
    }

    assert job_run_status(run) == "not_started"


def test_why_slower_reports_long_running_command_spans(tmp_path):
    from skills.harness.reports.benchmark_insights import _why_slower

    def command_events(command, start, end, item_id="item_1", output="ok", exit_code=0):
        status = "completed" if exit_code == 0 else "failed"
        return [
            {
                "timestamp": start,
                "type": "item.started",
                "item": {
                    "command": command,
                    "id": item_id,
                    "status": "in_progress",
                    "type": "command_execution",
                },
            },
            {
                "timestamp": end,
                "type": "item.completed",
                "item": {
                    "aggregated_output": output,
                    "command": command,
                    "exit_code": exit_code,
                    "id": item_id,
                    "status": status,
                    "type": "command_execution",
                },
            },
        ]

    def source_fields(mode: str, source: str) -> dict:
        mode_dir = tmp_path / mode
        source_path = mode_dir / "workspace_delta" / "changed_files" / "client.py"
        source_path.parent.mkdir(parents=True)
        source_path.write_text(source, encoding="utf-8")
        return {
            "mode_dir": mode_dir,
            "workspace_delta": {
                "changed_files": [
                    {
                        "artifact_path": "changed_files/client.py",
                        "path": "client.py",
                    }
                ]
            },
        }

    cuda_install_output = (
        "Downloading nvidia-cublas (517.7MiB)\n"
        "WARNING: Connection timed out while downloading.\n"
        "WARNING: Attempting to resume incomplete download (211.0 MB/426.4 MB, attempt 1)\n"
        "WARNING: Retrying after connection broken by "
        "NameResolutionError(\"Failed to resolve 'files.pythonhosted.org'\")\n"
        "Downloading nvidia-cudnn-cu13 (424.0MiB)\n"
        "Downloading triton (179.7MiB)\n"
        "Installed torch==2.12.0\n"
    )
    in_process_output = (
        "2026-06-13 08:50:40,563 - INFO - Round 1 started.\n"
        "2026-06-13 09:06:12,642 - INFO - [server] download tx T1 done: status=finished elapsed=905.27s\n"
        "2026-06-13 09:06:28,198 - INFO - Aggregated 3/3 results\n"
        "PTInProcessClientAPIExecutor - INFO - Waiting for result from peer\n"
        "Finished FedAvg.\n"
    )
    simulator_output = (
        "Running: /workspace/venv/bin/python3 -m nvflare.cli simulator /workspace/fl_job -w /workspace/fl_workspace -n 3 -t 3\n"
        "2026-06-13 08:01:40,056 - INFO - Round 2 started.\n"
        "2026-06-13 08:01:45,239 - INFO - [server] download tx T2 done: status=finished elapsed=8.53s\n"
        "2026-06-13 08:02:01,678 - INFO - Aggregated 3/3 results\n"
        "PTClientAPILauncherExecutor - INFO - received result\n"
        "Finished FedAvg.\n"
        "Simulation workspace: /workspace/fl_workspace\n"
    )
    with_events = command_events(
        "uv pip install -r requirements-train.txt",
        "2026-06-13T08:00:00Z",
        "2026-06-13T08:20:00Z",
        output=cuda_install_output,
    ) + command_events(
        "python job.py",
        "2026-06-13T08:21:00Z",
        "2026-06-13T08:51:00Z",
        item_id="item_2",
        output=in_process_output,
    )
    base_events = (
        command_events(
            "python3 -m pip install -r requirements-train.txt",
            "2026-06-13T08:00:00Z",
            "2026-06-13T08:03:22Z",
            output="Downloading torch and nvidia-cudnn-cu13\n",
            exit_code=-1,
        )
        + command_events(
            "python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu",
            "2026-06-13T08:03:30Z",
            "2026-06-13T08:03:57Z",
            item_id="item_2",
            output="Successfully installed torch-2.12.0+cpu\n",
        )
        + command_events(
            "python3 run_nvflare_fedavg.py",
            "2026-06-13T08:04:00Z",
            "2026-06-13T08:06:00Z",
            item_id="item_3",
            output=simulator_output,
        )
    )
    with_run = {
        "label": "With skills",
        "run": {"elapsed_seconds": 3200},
        "activity": {},
        "agent_events_text": "\n".join(json.dumps(event) for event in with_events),
        **source_fields(
            "with_skills",
            """
import nvflare.client as flare
train_frame = load_split(args.data_dir, "train")
train_loader = DataLoader(train_frame)
while flare.is_running():
    criterion, optimizer = build_loss_and_optimizer(model, train_frame, args, device)
    test_metrics = evaluate(model, test_loader, criterion, device)
    append_record(results_path, {"metrics": test_metrics})
""",
        ),
    }
    base_run = {
        "label": "No skills baseline",
        "run": {"elapsed_seconds": 300},
        "activity": {},
        "agent_events_text": "\n".join(json.dumps(event) for event in base_events),
        **source_fields(
            "without_skills",
            """
import nvflare.client as flare
train_frame = load_split(args.data_dir, "train")
train_loader = DataLoader(train_frame)
criterion, optimizer = build_loss_and_optimizer(model, train_frame, args, device)
while flare.is_running():
    train_one_epoch(model, train_loader, criterion, optimizer, device)
""",
        ),
    }

    explanation = "\n".join(_why_slower(with_run, base_run))

    assert "Slowdown driver comparison" in explanation
    assert "captured command time contributing to wall-clock slowdown" in explanation
    assert "| Captured command time | 3000s | 349s | +2651s |" in explanation
    assert "Elapsed time accounting" in explanation
    assert "| Run | Total | Dependency install | Runtime after install | Captured non-install commands |" in explanation
    assert "| With skills | 3200s | 1200s | 2000s | 1800s |" in explanation
    assert "| No skills baseline | 300s | 229s | 71s | 120s |" in explanation
    assert "Captured command spans identify slow operations but are not guaranteed to add up exactly" in explanation
    assert "Longest command comparison" in explanation
    assert "| Rank | With skills | No skills baseline |" in explanation
    assert "| 1 | `python job.py` (1800s, exit 0) | `python3 -m pip install -r requirements-train.txt`" in explanation
    assert "| 2 | `uv pip install -r requirements-train.txt`" in explanation
    assert "`python3 run_nvflare_fedavg.py` (120s, exit 0)" in explanation
    assert "uv pip install -r requirements-train.txt" in explanation
    assert "python job.py" in explanation
    assert "Dependency install path differed" in explanation
    assert "requirements-file install" in explanation
    assert "downloaded packages included" in explanation
    assert "nvidia-cublas" in explanation
    assert "Installer form differed" in explanation
    assert "Network/download evidence" in explanation
    assert "connection timeout" in explanation
    assert "resumed incomplete download" in explanation
    assert "DNS resolution failure" in explanation
    assert "baseline longest install log showed no captured network retry/timeout markers" in explanation
    assert "targeted package install" in explanation
    assert "NVFLARE runtime path diverged" in explanation
    assert "| Run | Runtime path | Successful runs | Total captured time | Representative command |" in explanation
    assert (
        "| With skills | `recipe.execute(SimEnv(...))` with `PTInProcessClientAPIExecutor` | 1 command | 1800s |"
        in explanation
    )
    assert (
        "| No skills baseline | exported job + `nvflare.cli simulator ... -t 3` with external client processes | 1 command | 120s |"
        in explanation
    )
    assert "recipe.execute(SimEnv(...))" in explanation
    assert "PTInProcessClientAPIExecutor" in explanation
    assert "nvflare.cli simulator ... -t 3" in explanation
    assert "external client processes" in explanation
    assert "Slow FL round evidence" in explanation
    assert "Transfer/wait evidence" in explanation
    assert "training/validation work, NVFLARE result transfer, synchronization wait" in explanation
    assert "should be investigated separately from generated-code efficiency" in explanation
    assert "Generated-code efficiency issue aligns with slower non-install runtime" in explanation
    assert "loss/optimizer lifecycle" in explanation
    assert "Quality-versus-speed tradeoff: useful validation work also adds per-round workload" in explanation
    assert "may explain part of the long per-round wait" in explanation
    assert "Dependency cost is separate from code efficiency" in explanation


def test_why_section_reports_runtime_regression_when_total_time_is_not_slower():
    from skills.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from skills.harness.reports.benchmark_insights import why_section

    def command_events(command, start, end, item_id, output="ok"):
        return [
            {
                "timestamp": start,
                "type": "item.started",
                "item": {
                    "command": command,
                    "id": item_id,
                    "status": "in_progress",
                    "type": "command_execution",
                },
            },
            {
                "timestamp": end,
                "type": "item.completed",
                "item": {
                    "aggregated_output": output,
                    "command": command,
                    "exit_code": 0,
                    "id": item_id,
                    "status": "completed",
                    "type": "command_execution",
                },
            },
        ]

    with_events = command_events(
        "uv pip install -r requirements-train.txt",
        "2026-06-13T00:00:00Z",
        "2026-06-13T00:00:50Z",
        "with_install",
    ) + command_events(
        "python job.py",
        "2026-06-13T00:01:00Z",
        "2026-06-13T00:07:40Z",
        "with_job",
        output="Finished FedAvg.\n",
    )
    base_events = command_events(
        "python -m pip install -r requirements-train.txt",
        "2026-06-13T00:00:00Z",
        "2026-06-13T00:05:00Z",
        "base_install",
    ) + command_events(
        "python run_job.py",
        "2026-06-13T00:05:05Z",
        "2026-06-13T00:05:45Z",
        "base_job",
        output="Finished FedAvg.\n",
    )
    runs = {
        NO_SKILLS_MODE: {
            "label": "No skills baseline",
            "run": {"elapsed_seconds": 600, "token_count": 1000},
            "activity": {"event_types": {"assistant": 2}},
            "agent_events_text": "\n".join(json.dumps(event) for event in base_events),
        },
        WITH_SKILLS_MODE: {
            "label": "With skills",
            "run": {"elapsed_seconds": 500, "token_count": 1000},
            "activity": {"event_types": {"assistant": 5}},
            "agent_events_text": "\n".join(json.dumps(event) for event in with_events),
        },
    }

    section = why_section(runs, [NO_SKILLS_MODE, WITH_SKILLS_MODE])

    assert "## Why" in section
    assert "Why With skills has longer runtime after install" in section
    assert "| With skills | 500s | 50s | 450s | 400s |" in section
    assert "| No skills baseline | 600s | 300s | 300s | 40s |" in section
    assert "Slowdown driver comparison" in section
    assert "| Captured non-install command time | 400s | 40s | +360s |" in section
    assert "450s vs 340s Captured command time" not in section
    assert "captured non-install command time contributing to runtime-after-install regression" in section
    assert "extra wall time came from tools" not in section
    assert "| Assistant turns | 5 | 2 | +3 | extra model round-trips |" in section
    assert "wall-clock overhead;" not in section


def test_runtime_path_note_includes_baseline_fallback_command_time():
    from skills.harness.reports.benchmark_insights import _runtime_path_slowdown_note

    def command_events(command, start, end, item_id, output="ok"):
        return [
            {
                "timestamp": start,
                "type": "item.started",
                "item": {
                    "command": command,
                    "id": item_id,
                    "status": "in_progress",
                    "type": "command_execution",
                },
            },
            {
                "timestamp": end,
                "type": "item.completed",
                "item": {
                    "aggregated_output": output,
                    "command": command,
                    "exit_code": 0,
                    "id": item_id,
                    "status": "completed",
                    "type": "command_execution",
                },
            },
        ]

    with_run = {
        "label": "With skills",
        "agent_events_text": "\n".join(
            json.dumps(event)
            for event in command_events(
                "python job.py",
                "2026-06-13T08:00:00Z",
                "2026-06-13T08:14:02Z",
                "with_job",
                output="PTInProcessClientAPIExecutor - INFO - result received\nFinished FedAvg.\n",
            )
        ),
    }
    base_run = {
        "label": "No skills baseline",
        "agent_events_text": "\n".join(
            json.dumps(event)
            for event in (
                command_events(
                    "python run_experiment.py",
                    "2026-06-13T08:00:00Z",
                    "2026-06-13T08:03:00Z",
                    "base_cmd",
                )
                + command_events(
                    "nl -ba nvflare_scaffold_job.py | sed -n '1,240p'",
                    "2026-06-13T08:04:00Z",
                    "2026-06-13T08:04:01Z",
                    "inspect_job_source",
                    output='print("Finished FedAvg.")\ncmd = "python -m nvflare.cli simulator job -w workspace"\n',
                )
            )
        ),
    }

    note = "\n".join(_runtime_path_slowdown_note(with_run, base_run))

    assert "NVFLARE runtime path diverged" in note
    assert "| Run | Runtime path | Successful runs | Total captured time | Representative command |" in note
    assert (
        "| With skills | `recipe.execute(SimEnv(...))` with `PTInProcessClientAPIExecutor` | 1 command | 842s |" in note
    )
    assert "| No skills baseline | no classified successful job/simulator command | 0 commands | NA |" in note
    assert "`python job.py` (842s, exit 0)" in note
    assert "no classified successful job/simulator command" in note
    assert "`python run_experiment.py` (180s, exit 0)" in note
    assert "nl -ba nvflare_scaffold_job.py" not in note


def test_longest_command_table_empty_cells_preserve_threshold():
    from skills.harness.reports.benchmark_insights import _longest_command_comparison_note

    def command_events(command, start, end, item_id):
        return [
            {
                "timestamp": start,
                "type": "item.started",
                "item": {
                    "command": command,
                    "id": item_id,
                    "status": "in_progress",
                    "type": "command_execution",
                },
            },
            {
                "timestamp": end,
                "type": "item.completed",
                "item": {
                    "aggregated_output": "ok",
                    "command": command,
                    "exit_code": 0,
                    "id": item_id,
                    "status": "completed",
                    "type": "command_execution",
                },
            },
        ]

    with_events = command_events("python long_one.py", "2026-06-13T08:00:00Z", "2026-06-13T08:02:00Z", "w1")
    with_events += command_events("python long_two.py", "2026-06-13T08:03:00Z", "2026-06-13T08:04:00Z", "w2")
    base_events = command_events("python base.py", "2026-06-13T08:00:00Z", "2026-06-13T08:01:00Z", "b1")

    note = _longest_command_comparison_note(
        {
            "label": "With skills",
            "agent_events_text": "\n".join(json.dumps(event) for event in with_events),
        },
        {
            "label": "No skills baseline",
            "agent_events_text": "\n".join(json.dumps(event) for event in base_events),
        },
    )

    assert "| 2 | `python long_two.py` (60s, exit 0) | no timed command span >=30s captured |" in note


def test_fewer_turns_note_does_not_invent_command_runtime_cause():
    from skills.harness.reports.benchmark_insights import _why_slower

    with_run = {
        "label": "With skills",
        "run": {"elapsed_seconds": 120},
        "activity": {"event_types": {"assistant": 2}},
        "agent_events_text": "",
    }
    base_run = {
        "label": "No skills baseline",
        "run": {"elapsed_seconds": 100},
        "activity": {"event_types": {"assistant": 4}},
        "agent_events_text": "",
    }

    explanation = "\n".join(_why_slower(with_run, base_run))

    assert "Assistant turns" not in explanation
    assert "better explained by captured command/runtime duration" not in explanation


def test_why_slower_does_not_blame_code_quality_when_runtime_excluding_install_is_faster(tmp_path):
    from skills.harness.reports.benchmark_insights import _why_slower

    def command_events(command, start, end, item_id, output="ok"):
        return [
            {
                "timestamp": start,
                "type": "item.started",
                "item": {
                    "command": command,
                    "id": item_id,
                    "status": "in_progress",
                    "type": "command_execution",
                },
            },
            {
                "timestamp": end,
                "type": "item.completed",
                "item": {
                    "aggregated_output": output,
                    "command": command,
                    "exit_code": 0,
                    "id": item_id,
                    "status": "completed",
                    "type": "command_execution",
                },
            },
        ]

    def run_with_client(mode: str, source: str, events: list[dict], elapsed_seconds: int) -> dict:
        mode_dir = tmp_path / mode
        source_path = mode_dir / "workspace_delta" / "changed_files" / "client.py"
        source_path.parent.mkdir(parents=True)
        source_path.write_text(source, encoding="utf-8")
        return {
            "label": mode,
            "run": {"elapsed_seconds": elapsed_seconds},
            "activity": {},
            "agent_events_text": "\n".join(json.dumps(event) for event in events),
            "mode_dir": mode_dir,
            "workspace_delta": {
                "changed_files": [
                    {
                        "artifact_path": "changed_files/client.py",
                        "path": "client.py",
                    }
                ]
            },
        }

    with_source = """
import nvflare.client as flare
train_frame = load_split(args.data_dir, "train")
train_loader = DataLoader(train_frame)
while flare.is_running():
    criterion, optimizer = build_loss_and_optimizer(model, train_frame, args, device)
    test_metrics = evaluate(model, test_loader, criterion, device)
"""
    base_source = """
import nvflare.client as flare
train_frame = load_split(args.data_dir, "train")
train_loader = DataLoader(train_frame)
criterion, optimizer = build_loss_and_optimizer(model, train_frame, args, device)
while flare.is_running():
    train_one_epoch(model, train_loader, criterion, optimizer, device)
"""
    with_events = command_events(
        "pip install -r requirements-train.txt",
        "2026-06-13T00:00:00Z",
        "2026-06-13T00:21:58Z",
        "item_1",
        output="Downloading nvidia-cublas\nSuccessfully installed torch\n",
    ) + command_events(
        "python job.py",
        "2026-06-13T00:22:00Z",
        "2026-06-13T00:24:00Z",
        "item_2",
        output="Finished FedAvg.\n",
    )
    base_events = command_events(
        "python -m pip install -r requirements-train.txt",
        "2026-06-13T00:00:00Z",
        "2026-06-13T00:06:30Z",
        "item_1",
        output="Successfully installed torch\n",
    ) + command_events(
        "python run_nvflare_fedavg.py",
        "2026-06-13T00:07:00Z",
        "2026-06-13T00:23:40Z",
        "item_2",
        output="Finished FedAvg.\n",
    )

    explanation = "\n".join(
        _why_slower(
            run_with_client("with_skills", with_source, with_events, elapsed_seconds=1855),
            run_with_client("without_skills", base_source, base_events, elapsed_seconds=1399),
        )
    )

    assert "Dependency install path differed" in explanation
    assert "Generated-code efficiency issue is not the measured slowdown driver" in explanation
    assert "runtime excluding dependency install is 537s vs 1009s" in explanation
    assert "Quality evidence did not make non-install runtime slower in this run" in explanation
    assert "Generated-code efficiency issue aligns with slower non-install runtime" not in explanation
    assert "may explain part of the long per-round wait" not in explanation


def test_dependency_install_detection_ignores_process_grep():
    from skills.harness.reports.benchmark_insights import is_dependency_install_command

    assert is_dependency_install_command("uv pip install -r requirements-train.txt")
    assert is_dependency_install_command("python3 -m pip install torch")
    assert not is_dependency_install_command("python -m pip show nvflare torch")
    assert not is_dependency_install_command(
        "for p in /proc/[0-9]*; do tr '\\0' ' ' < \"$p/cmdline\" | grep -E 'python3 -m pip|pip install'; done"
    )


def test_job_run_status_requires_success_evidence_for_simulation_script():
    from skills.harness.reports.benchmark_insights import job_run_status

    event = {
        "item": {
            "aggregated_output": "configuration loaded successfully",
            "command": "python run_nvflare_simulation.py",
            "exit_code": 0,
            "id": "item_1",
            "status": "completed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ["python run_nvflare_simulation.py"]},
        "agent_events_text": json.dumps(event),
    }

    assert job_run_status(run) == "started_failed"


def test_job_run_status_detects_leading_job_entrypoint():
    from skills.harness.reports.benchmark_insights import job_run_status

    event = {
        "item": {
            "aggregated_output": "Job Status: FINISHED:COMPLETED\nResult can be found in: /tmp/nvflare/fedxgb",
            "command": "python job_vertical.py",
            "exit_code": 0,
            "id": "item_1",
            "status": "completed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ["python job_vertical.py"]},
        "agent_events_text": json.dumps(event),
    }

    assert job_run_status(run) == "completed"


def test_job_success_ignores_file_inspection_matching_python_job_text():
    from skills.harness.reports.benchmark_insights import job_command_succeeded, job_run_status

    event = {
        "command": "/bin/bash -lc 'rg \"python job.py|Finished FedAvg\" -n .'",
        "exit_code": 0,
        "output": 'README.md:Run with python job.py\njob.py:print("Finished FedAvg.")',
        "status": "completed",
    }
    run = {
        "available": True,
        "activity": {"commands": [event["command"]]},
        "agent_events_text": json.dumps(
            {"item": {"type": "command_execution", "aggregated_output": event["output"], **event}}
        ),
    }

    assert job_command_succeeded(event) is False
    assert job_run_status(run) == "not_started"


def test_job_success_ignores_wrapped_file_inspection_matching_python_job_text():
    from skills.harness.reports.benchmark_insights import job_command_succeeded, job_run_status

    event = {
        "command": "/bin/bash -lc 'cd /work && rg \"python job.py|Finished FedAvg\" -n .'",
        "exit_code": 0,
        "output": 'README.md:Run with python job.py\njob.py:print("Finished FedAvg.")',
        "status": "completed",
    }
    run = {
        "available": True,
        "activity": {"commands": [event["command"]]},
        "agent_events_text": json.dumps(
            {"item": {"type": "command_execution", "aggregated_output": event["output"], **event}}
        ),
    }

    assert job_command_succeeded(event) is False
    assert job_run_status(run) == "not_started"


def test_job_success_detects_execution_after_wrapped_file_inspection():
    from skills.harness.reports.benchmark_insights import job_command_succeeded, job_run_status

    event = {
        "command": "/bin/bash -lc 'cd /work && rg \"python job.py|Finished FedAvg\" -n . && python job.py'",
        "exit_code": 0,
        "output": 'README.md:Run with python job.py\njob.py:print("Finished FedAvg.")\nFinished FedAvg.',
        "status": "completed",
    }
    run = {
        "available": True,
        "activity": {"commands": [event["command"]]},
        "agent_events_text": json.dumps(
            {"item": {"type": "command_execution", "aggregated_output": event["output"], **event}}
        ),
    }

    assert job_command_succeeded(event) is True
    assert job_run_status(run) == "completed"


def test_job_success_ignores_grep_pattern_with_inline_semicolon():
    from skills.harness.reports.benchmark_insights import job_command_succeeded, job_run_status

    # The ';' lives inside the rg search pattern; quote-aware splitting must keep it as a single
    # inspection segment rather than exposing "python job.py" as an executed job.
    event = {
        "command": 'rg "foo; python job.py" -n .',
        "exit_code": 0,
        "output": 'job.py:print("Finished FedAvg.")',
        "status": "completed",
    }
    run = {
        "available": True,
        "activity": {"commands": [event["command"]]},
        "agent_events_text": json.dumps(
            {"item": {"type": "command_execution", "aggregated_output": event["output"], **event}}
        ),
    }

    assert job_command_succeeded(event) is False
    assert job_run_status(run) == "not_started"


def test_job_success_ignores_cd_prefix_before_grep_inspection():
    from skills.harness.reports.benchmark_insights import job_command_succeeded, job_run_status

    # A benign 'cd' prefix must not push python_script_name onto the broad regex and treat the
    # grep pattern as a real python job.py run.
    event = {
        "command": "cd /work && rg 'python job.py|Finished FedAvg' -n .",
        "exit_code": 0,
        "output": 'README.md:Run with python job.py\njob.py:print("Finished FedAvg.")',
        "status": "completed",
    }
    run = {
        "available": True,
        "activity": {"commands": [event["command"]]},
        "agent_events_text": json.dumps(
            {"item": {"type": "command_execution", "aggregated_output": event["output"], **event}}
        ),
    }

    assert job_command_succeeded(event) is False
    assert job_run_status(run) == "not_started"


def test_job_success_requires_evidence_when_inspection_follows_job_with_semicolon():
    from skills.harness.reports.benchmark_insights import job_command_succeeded

    # The job failed but a trailing ';' inspection segment makes the aggregate exit code 0; the
    # direct-job exit code can no longer be trusted, so success evidence is required.
    event = {
        "command": "python job.py ; cat results.txt",
        "exit_code": 0,
        "output": "Traceback (most recent call last):\nRuntimeError: job crashed",
        "status": "completed",
    }

    assert job_command_succeeded(event) is False


def test_job_success_trusts_direct_job_exit_in_and_chain():
    from skills.harness.reports.benchmark_insights import job_command_succeeded

    # An '&&' chain reaching exit 0 implies the job itself succeeded, so the aggregate exit code
    # is trustworthy even without explicit success output.
    event = {
        "command": "python job.py && python validate.py",
        "exit_code": 0,
        "output": "validation ok",
        "status": "completed",
    }

    assert job_command_succeeded(event) is True


def test_simulator_wrapper_detected_after_non_runtime_first_segment():
    from skills.harness.reports.benchmark_insights import job_run_status, job_run_status_reason

    # The runtime wrapper is the second segment; per-segment classification must not stop at the
    # non-runtime first script (prepare_data.py).
    event = {
        "item": {
            "aggregated_output": (
                "Running: python3 -m nvflare.cli simulator /workspace/fl_job -w /workspace/ws -n 2 -t 2\n"
                "Finished FedAvg.\n"
            ),
            "command": "python prepare_data.py && python run_nvflare_fedavg.py",
            "exit_code": 0,
            "id": "item_1",
            "status": "completed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ["python prepare_data.py && python run_nvflare_fedavg.py"]},
        "agent_events_text": json.dumps(event),
    }

    assert job_run_status(run) == "completed"
    assert "simulation completed" in job_run_status_reason(run)


def test_simulator_wrapper_detected_for_make_target():
    from skills.harness.reports.benchmark_insights import job_run_status

    event = {
        "item": {
            "aggregated_output": (
                "python3 -m nvflare.cli simulator /workspace/fl_job -w /workspace/ws -n 2 -t 2\n" "Finished FedAvg.\n"
            ),
            "command": "make simulate",
            "exit_code": 0,
            "id": "item_1",
            "status": "completed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ["make simulate"]},
        "agent_events_text": json.dumps(event),
    }

    assert job_run_status(run) == "completed"


def test_job_run_status_requires_success_evidence_for_leading_job_entrypoint():
    from skills.harness.reports.benchmark_insights import job_run_status

    event = {
        "item": {
            "aggregated_output": "usage: job_vertical.py [--run_psi] [--run_training]\nerror: select a run mode",
            "command": "python job_vertical.py",
            "exit_code": 0,
            "id": "item_1",
            "status": "completed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ["python job_vertical.py"]},
        "agent_events_text": json.dumps(event),
    }

    assert job_run_status(run) == "started_failed"


def test_job_run_status_detects_ambiguous_job_suffix_with_success_evidence():
    from skills.harness.reports.benchmark_insights import job_run_status

    event = {
        "item": {
            "aggregated_output": "Job Status is: FINISHED:COMPLETED\nResult location: /tmp/nvflare/results",
            "command": "python fedavg_job.py",
            "exit_code": 0,
            "id": "item_1",
            "status": "completed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ["python fedavg_job.py"]},
        "agent_events_text": json.dumps(event),
    }

    assert job_run_status(run) == "completed"


def test_job_run_status_requires_success_evidence_for_ambiguous_job_suffix_script():
    from skills.harness.reports.benchmark_insights import job_run_status

    event = {
        "item": {
            "aggregated_output": "configuration loaded successfully",
            "command": "python fedavg_job.py",
            "exit_code": 0,
            "id": "item_1",
            "status": "completed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ["python fedavg_job.py"]},
        "agent_events_text": json.dumps(event),
    }

    assert job_run_status(run) == "started_failed"


def test_job_run_status_rejects_result_path_with_failed_status():
    from skills.harness.reports.benchmark_insights import job_run_status

    event = {
        "item": {
            "aggregated_output": (
                "Result can be found in: /tmp/nvflare/fedxgb\n" "Job Status is: FINISHED:EXECUTION_EXCEPTION"
            ),
            "command": "python fedavg_job.py",
            "exit_code": 0,
            "id": "item_1",
            "status": "completed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ["python fedavg_job.py"]},
        "agent_events_text": json.dumps(event),
    }

    assert job_run_status(run) == "started_failed"


def test_job_output_failure_status_vetoes_result_location():
    from skills.harness.reports.benchmark_insights import job_output_succeeded

    assert not job_output_succeeded("Result location: /tmp/r\nJob Status: FINISHED:EXECUTION_EXCEPTION")
    assert not job_output_succeeded("Result can be found in: /tmp/r\nStatus is: FINISHED:ABORTED")
    assert job_output_succeeded("Result location: /tmp/r\nJob Status: FINISHED:COMPLETED")


def test_job_output_failure_status_vetoes_legacy_terminal_statuses():
    from skills.harness.reports.benchmark_insights import job_output_succeeded

    assert not job_output_succeeded("Result location: /tmp/r\nJob Status: FAILED")
    assert not job_output_succeeded("Result can be found in: /tmp/r\nJob Status is: FINISHED_EXCEPTION")
    assert not job_output_succeeded("Result location: /tmp/r\nStatus: ABANDONED")
    assert not job_output_succeeded("Result location: /tmp/r\nStatus: ABORTED")
    # Legacy success form must still pass.
    assert job_output_succeeded("Result location: /tmp/r\nJob Status: FINISHED_OK")


def test_recovered_by_later_success_requires_simulation_success_evidence():
    from skills.harness.reports.benchmark_insights import command_failure_diagnostics

    failed_event = {
        "item": {
            "aggregated_output": "Traceback (most recent call last):\nRuntimeError: simulation failed",
            "command": "python run_nvflare_simulation.py",
            "exit_code": 1,
            "id": "item_1",
            "status": "failed",
            "type": "command_execution",
        }
    }
    incomplete_success_event = {
        "item": {
            "aggregated_output": "configuration loaded successfully",
            "command": "python run_nvflare_simulation.py",
            "exit_code": 0,
            "id": "item_2",
            "status": "completed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ["python run_nvflare_simulation.py", "python run_nvflare_simulation.py"]},
        "agent_events_text": "\n".join(json.dumps(event) for event in (failed_event, incomplete_success_event)),
    }

    diagnostics = command_failure_diagnostics(run)

    assert diagnostics
    assert "not recovered in this run" in diagnostics[0]


def test_failure_analysis_formats_multiline_recovered_command_as_single_line():
    from skills.harness.modes import WITH_SKILLS_MODE
    from skills.harness.reports.benchmark_insights import failure_analysis_section

    failed_event = {
        "item": {
            "aggregated_output": "Traceback (most recent call last):\nTypeError: unsupported operand type(s)",
            "command": "python3 -m py_compile client.py job.py && python3 - <<'EOF'\nprint('check')\nEOF",
            "exit_code": 1,
            "id": "item_1",
            "status": "failed",
            "type": "command_execution",
        }
    }
    recovered_event = {
        "item": {
            "aggregated_output": "Finished FedAvg.",
            "command": "python3 job.py",
            "exit_code": 0,
            "id": "item_2",
            "status": "completed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "label": "With skills",
        "agent_events_text": "\n".join(json.dumps(event) for event in (failed_event, recovered_event)),
        "container_exit": {"exit_code": 0},
        "record": {},
        "run": {"final_container_exit_code": 0},
    }

    section = failure_analysis_section({WITH_SKILLS_MODE: run}, [WITH_SKILLS_MODE])

    assert "Recovered Command Evidence" in section
    assert "| Command | Exit | Recovery | Root cause | Dependency evidence |" in section
    assert "python3 -m py_compile client.py job.py && python3 - <<'EOF' ... EOF" in section
    assert "print('check')\nEOF" not in section


def test_job_run_status_reason_includes_failed_job_command_error():
    from skills.harness.reports.benchmark_insights import job_run_action, job_run_status, job_run_status_reason

    event = {
        "item": {
            "aggregated_output": "Traceback (most recent call last):\nModuleNotFoundError: No module named 'torch'",
            "command": "python job.py",
            "exit_code": 1,
            "id": "item_1",
            "status": "failed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ["python job.py"]},
        "agent_events_text": json.dumps(event),
    }

    assert job_run_status(run) == "started_failed"
    reason = job_run_status_reason(run)
    assert "missing Python dependency `torch`" in reason
    assert "no dependency install command was captured" in reason
    assert "Install the job requirements" in job_run_action(run)


def test_completed_job_run_status_reason_includes_recovered_dependency_failure():
    from skills.harness.reports.benchmark_insights import job_run_action, job_run_status, job_run_status_reason

    failed_probe = {
        "item": {
            "aggregated_output": "Traceback (most recent call last):\nModuleNotFoundError: No module named 'torch'",
            "command": 'python -c "import torch"',
            "exit_code": 1,
            "id": "probe",
            "status": "failed",
            "type": "command_execution",
        }
    }
    install = {
        "item": {
            "aggregated_output": "Successfully installed torch",
            "command": "pip install -r requirements.txt",
            "exit_code": 0,
            "id": "install",
            "status": "completed",
            "type": "command_execution",
        }
    }
    successful_job = {
        "item": {
            "aggregated_output": "site-1: round=0 train_loss=0.5 valid_auroc=0.7\nFinished FedAvg.",
            "command": "python job.py",
            "exit_code": 0,
            "id": "job",
            "status": "completed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ['python -c "import torch"', "pip install -r requirements.txt", "python job.py"]},
        "agent_events_text": "\n".join(json.dumps(event) for event in (failed_probe, install, successful_job)),
    }

    reason = job_run_status_reason(run)

    assert job_run_status(run) == "completed"
    assert "simulation completed" in reason
    assert "earlier missing Python dependency `torch` was recovered" in reason
    assert "a dependency install command later succeeded" in reason
    assert "inspect recovered command failures" in job_run_action(run)


def test_failure_analysis_keeps_recovered_bash_issue_for_passed_run():
    from skills.harness.modes import WITH_SKILLS_MODE
    from skills.harness.reports.benchmark_insights import failure_analysis_section

    permission_result = {
        "message": {
            "content": [
                {
                    "content": "Claude requested permissions to use Bash, but you haven't granted it yet.",
                    "type": "tool_result",
                }
            ]
        },
        "type": "user",
    }
    final_result = {
        "final_message": "Completed simulation.",
        "permission_denials": [
            {"tool_name": "Bash", "tool_input": {"command": "rm -rf /tmp/workspace && python job.py"}}
        ],
        "subtype": "success",
        "type": "result",
    }
    run = {
        "available": True,
        "label": "With skills",
        "activity": {"hint_counts": {"python_job_py": 1, "simulation": 1}},
        "agent_events_text": "\n".join(json.dumps(event) for event in (permission_result, final_result)),
        "container_exit": {"exit_code": 0},
        "record": {},
        "run": {"final_container_exit_code": 0},
    }

    section = failure_analysis_section({WITH_SKILLS_MODE: run}, [WITH_SKILLS_MODE])

    assert "Outcome: passed" in section
    assert "Recovered Bash/tool issue" in section
    assert "Bash tool was blocked 1 time(s)" in section
    assert "Denied command: `rm -rf /tmp/workspace && python job.py`" in section
    assert "costs extra tool turns, tokens, and elapsed time" in section


def test_failure_analysis_reports_dependency_install_evidence_for_missing_module():
    from skills.harness.modes import NO_SKILLS_MODE
    from skills.harness.reports.benchmark_insights import failure_analysis_section

    event = {
        "item": {
            "aggregated_output": "Traceback (most recent call last):\nModuleNotFoundError: No module named 'torch'",
            "command": 'python -c "import torch"',
            "exit_code": 1,
            "id": "probe",
            "status": "failed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "label": "No skills baseline",
        "container_exit": {"exit_code": 1},
        "activity": {"commands": ['python -c "import torch"']},
        "agent_events_text": json.dumps(event),
    }

    section = failure_analysis_section({NO_SKILLS_MODE: run}, [NO_SKILLS_MODE])

    assert "ModuleNotFoundError: No module named 'torch'" in section
    assert "no dependency install command was captured before the failed job run" in section


def test_job_run_status_reason_reports_failed_dependency_install():
    from skills.harness.reports.benchmark_insights import job_run_status_reason

    install_event = {
        "item": {
            "aggregated_output": "ERROR: Could not find a version that satisfies the requirement torch",
            "command": "python -m pip install -r requirements.txt",
            "exit_code": 1,
            "id": "install",
            "status": "failed",
            "type": "command_execution",
        }
    }
    job_event = {
        "item": {
            "aggregated_output": "Traceback (most recent call last):\nModuleNotFoundError: No module named 'torch'",
            "command": "python job.py",
            "exit_code": 1,
            "id": "job",
            "status": "failed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ["python -m pip install -r requirements.txt", "python job.py"]},
        "agent_events_text": "\n".join(json.dumps(event) for event in (install_event, job_event)),
    }

    reason = job_run_status_reason(run)

    assert "missing Python dependency `torch`" in reason
    assert "dependency install attempted and failed" in reason
    assert "requirements.txt" in reason


def test_job_run_status_reason_reports_successful_install_with_wrong_runtime():
    from skills.harness.reports.benchmark_insights import job_run_action, job_run_status_reason

    install_event = {
        "item": {
            "aggregated_output": "Successfully installed torch",
            "command": "python -m pip install -r requirements.txt",
            "exit_code": 0,
            "id": "install",
            "status": "completed",
            "type": "command_execution",
        }
    }
    job_event = {
        "item": {
            "aggregated_output": "Traceback (most recent call last):\nModuleNotFoundError: No module named 'torch'",
            "command": "python job.py",
            "exit_code": 1,
            "id": "job",
            "status": "failed",
            "type": "command_execution",
        }
    }
    run = {
        "available": True,
        "activity": {"commands": ["python -m pip install -r requirements.txt", "python job.py"]},
        "agent_events_text": "\n".join(json.dumps(event) for event in (install_event, job_event)),
    }

    reason = job_run_status_reason(run)

    assert "dependency install command succeeded" in reason
    assert "simulator uses the environment where requirements were installed" in job_run_action(run)


def test_readme_metric_alignment_uses_server_best_validation_metric_scalar():
    from skills.harness.quality_signals import metric_signal

    signal = metric_signal(
        None,
        "Primary validation metric: AUROC.\n",
        """
Final round metrics:
- `site-1`: valid AUROC `0.7696`, test AUROC `0.7331`
- `site-2`: valid AUROC `0.7148`, test AUROC `0.7771`
- `site-3`: valid AUROC `0.7708`, test AUROC `0.7352`
- Server best validation metric at round 2: `0.7517306189541327`
""",
    )

    metric = signal["reported_validation_metric"]
    assert signal["status"] == "pass"
    assert signal["aligned_with_readme"] is True
    assert metric["name"] == "AUROC"
    assert metric["value"] == 0.7517306189541327
    assert metric["value_scope"] == "fl_summary_metric"
    assert metric["site_value_count"] == 6
    assert metric["summary_value_label"] == "Server best validation metric at round 2"


def test_readme_metric_alignment_passes_for_site_level_values_without_scalar():
    from skills.harness.quality_signals import metric_signal

    signal = metric_signal(
        None,
        "Primary validation metric: AUROC.\n",
        """
Final round metrics:
- `site-1`: valid AUROC `0.7696`
- `site-2`: valid AUROC `0.7148`
- `site-3`: valid AUROC `0.7708`
""",
    )

    metric = signal["reported_validation_metric"]
    assert signal["status"] == "pass"
    assert signal["aligned_with_readme"] is True
    assert signal["metric_value_available"] is True
    assert signal["metric_scalar_available"] is False
    assert signal["mismatch"] is False
    assert metric["name"] == "AUROC"
    assert metric["value"] is None
    assert metric["value_scope"] == "site_values_only"


def test_metrics_chart_names_metric_once_in_panel_title():
    from skills.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from skills.harness.reports.benchmark_insights import embedded_bar_chart, outcome_metrics_table

    def run(label: str, value: float) -> dict:
        return {
            "label": label,
            "available": True,
            "status": "0",
            "run": {"elapsed_seconds": 1, "token_count": 1, "agent_exit_code": 0, "final_container_exit_code": 0},
            "activity": {"command_count": 1},
            "record": {},
            "workspace_delta": {},
            "validation_metric": {"name": "AUROC", "value": value},
        }

    chart = embedded_bar_chart(
        {
            NO_SKILLS_MODE: run("No skills baseline", 0.7562),
            WITH_SKILLS_MODE: run("With skills", 0.7529),
        }
    )
    table = outcome_metrics_table(
        {
            NO_SKILLS_MODE: run("No skills baseline", 0.7562),
            WITH_SKILLS_MODE: run("With skills", 0.7529),
        },
        [NO_SKILLS_MODE, WITH_SKILLS_MODE],
    )

    assert "Metrics (AUROC)" in chart
    assert "Code quality" in chart
    assert "FL scalar result" not in chart
    assert "AUROC 0." not in chart
    assert chart.count("AUROC") == 1
    assert ">0.7529<" in chart
    assert "| Metrics (AUROC) | AUROC 0.7562 | AUROC 0.7529 |" in table
    assert "FL scalar result" not in table


def test_metrics_chart_uses_labeled_aggregated_metric_from_legacy_record():
    from skills.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from skills.harness.reports.benchmark_insights import embedded_bar_chart, outcome_metrics_table

    def run(label: str, metric: dict) -> dict:
        return {
            "label": label,
            "available": True,
            "run": {"final_container_exit_code": 0},
            "activity": {},
            "validation_metric": metric,
        }

    runs = {
        NO_SKILLS_MODE: run("No skills baseline", {"name": "AUROC", "value": None, "reported_value_entries": []}),
        WITH_SKILLS_MODE: run(
            "With skills",
            {
                "name": "AUROC",
                "value": None,
                "reported_value_entries": [
                    {"value": 0.7531},
                    {"label": "Best aggregated validation AUROC", "value": 0.7623334631865992},
                    {"label": "Final site metrics", "value": 0.767293},
                ],
            },
        ),
    }

    chart = embedded_bar_chart(runs)
    table = outcome_metrics_table(runs, [NO_SKILLS_MODE, WITH_SKILLS_MODE])

    assert ">0.7623<" in chart
    assert "| Metrics (AUROC) | AUROC NA | AUROC 0.7623 |" in table


def test_metrics_chart_marks_mixed_metric_names_non_comparable():
    from skills.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from skills.harness.reports.benchmark_insights import embedded_bar_chart, outcome_metrics_table

    def run(label: str, metric_name: str, value: float) -> dict:
        return {
            "label": label,
            "available": True,
            "run": {"final_container_exit_code": 0},
            "activity": {},
            "validation_metric": {"name": metric_name, "value": value},
        }

    runs = {
        NO_SKILLS_MODE: run("No skills baseline", "accuracy", 0.8123),
        WITH_SKILLS_MODE: run("With skills", "AUROC", 0.7529),
    }

    chart = embedded_bar_chart(runs)
    table = outcome_metrics_table(runs, [NO_SKILLS_MODE, WITH_SKILLS_MODE])

    assert "Metrics (mixed validation metrics)" in chart
    assert "Not comparable" in chart
    assert "No skills baseline: accuracy" in chart
    assert "With skills: AUROC" in chart
    assert "| Metrics (mixed validation metrics) | accuracy 0.8123 | AUROC 0.7529 |" in table


def test_structure_tree_renderer_uses_tree_format():
    from skills.harness.reports.benchmark_insights import tree_from_paths

    tree = tree_from_paths(
        [
            "client.py",
            "runtime_job_config/ames_fedavg/ames_fedavg/app/config/config_fed_client.json",
            "runtime_job_config/ames_fedavg/ames_fedavg/app/custom/model.py",
        ]
    )

    assert tree.startswith(".\n")
    assert "|-- client.py" in tree
    assert "`-- runtime_job_config" in tree
    assert "        `-- ames_fedavg" in tree
    assert "- runtime_job_config/ames_fedavg" not in tree


def test_run_summary_uses_agent_keys_without_codex_aliases(tmp_path):
    from skills.harness.records import write_json, write_run_summary

    final_record = tmp_path / "record.json"
    summary_path = tmp_path / "run_summary.json"
    write_json(
        final_record,
        {
            "mode": "with_skills",
            "agent_process_passed": True,
            "agent_process_exit_code": 0,
            "codex_process_passed": True,
            "codex_process_exit_code": 0,
            "agent_usage": {"total_tokens": 10},
            "codex_usage": {"total_tokens": 10},
            "process_metrics": {
                "agent_exit_code": 0,
                "codex_exit_code": 0,
                "elapsed_seconds": 1,
                "command_count": 0,
            },
        },
    )

    write_run_summary(final_record, summary_path, print_summary=False)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["agent_process_passed"] is True
    assert summary["agent_process_exit_code"] == 0
    assert summary["agent_exit_code"] == 0
    assert summary["command_count"] == 0
    assert summary["agent_usage"] == {"total_tokens": 10}
    assert "codex_process_passed" not in summary
    assert "codex_process_exit_code" not in summary
    assert "codex_exit_code" not in summary
    assert "codex_usage" not in summary
    assert not any(key.startswith("codex_") for key in summary["all_metrics"])


def test_run_summary_ignores_codex_usage_fallback_and_reports_prompt_hash(tmp_path):
    from skills.harness.records import write_json, write_run_summary

    final_record = tmp_path / "record.json"
    summary_path = tmp_path / "run_summary.json"
    write_json(
        final_record,
        {
            "mode": "with_skills",
            "codex_usage": {"total_tokens": 10},
            "process_metrics": {
                "elapsed_seconds": 3,
                "agent_elapsed_seconds": 2,
            },
        },
    )
    write_json(
        tmp_path / "prompt_metadata.json",
        {
            "prompt_sha256": "abc123",
            "template_path": "/workspace/prompts/prompt.txt",
        },
    )

    write_run_summary(final_record, summary_path, print_summary=False)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["agent_usage"] == {}
    assert summary["agent_elapsed_seconds"] == 2
    assert summary["elapsed_seconds"] == 2
    assert summary["prompt_hash"] == "abc123"
    assert summary["prompt_source"] == "/workspace/prompts/prompt.txt"
    assert summary["structure_quality_signal"] == {
        "status": "unavailable",
        "reason": "structure quality was not captured for this run",
    }


def test_make_tree_readable_does_not_follow_symlinked_directories(tmp_path):
    from skills.harness.common import make_tree_readable

    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "secret.txt"
    outside_file.write_text("keep private\n", encoding="utf-8")
    outside_file.chmod(0o600)
    root = tmp_path / "result"
    root.mkdir()
    (root / "outside_link").symlink_to(outside, target_is_directory=True)

    make_tree_readable(root)

    assert stat.S_IMODE(outside_file.stat().st_mode) == 0o600


def test_scenario_report_escapes_markdown_table_pipes(tmp_path):
    from skills.harness.reports.scenario_report import write_scenario_report

    write_scenario_report(
        tmp_path,
        {
            "scenario_name": "pipe scenario",
            "status": "passed",
            "completed_run_count": 1,
            "expanded_case_count": 1,
            "winner_policy": "quality",
            "runs": [
                {
                    "run_id": "run_pipe",
                    "label": "with|skills",
                    "agent": "claude",
                    "agent_model": "default|model",
                    "model_source": "adapter_default",
                    "mode": "with_skills",
                }
            ],
            "aggregate_results": {
                "by_label": {
                    "with|skills": {
                        "run_count": 1,
                        "quality_pass_count": 1,
                        "agent_elapsed_seconds": {"median": 2.5},
                        "token_count": {"median": 10},
                    }
                },
                "winner": {"label": "with|skills"},
            },
        },
    )

    report = (tmp_path / "reports" / "scenario_report.md").read_text(encoding="utf-8")
    assert "with\\|skills" in report
    assert "## Run Identity" in report
    assert "default\\|model" in report


def test_report_generators_write_two_mode_outputs(tmp_path, monkeypatch):
    from skills.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from skills.harness.reports import metrics_report
    from skills.harness.reports.benchmark_insights import main as insights_main

    for mode, value in ((NO_SKILLS_MODE, 0.7562), (WITH_SKILLS_MODE, 0.7529)):
        mode_dir = tmp_path / mode
        records_dir = mode_dir / "records"
        records_dir.mkdir(parents=True)
        (mode_dir / "container_exit_code.json").write_text(json.dumps({"exit_code": 0}) + "\n", encoding="utf-8")
        (mode_dir / "run_summary.json").write_text(
            json.dumps(
                {
                    "mode": mode,
                    "elapsed_seconds": 10,
                    "token_count": 100,
                    "agent_exit_code": 0,
                    "final_container_exit_code": 0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (records_dir / f"{mode}_record.json").write_text(
            json.dumps(
                {
                    "mode": mode,
                    "reported_validation_metric": {"name": "AUROC", "value": value},
                    "process_metrics": {"elapsed_seconds": 10, "token_count": 100},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (mode_dir / "agent_activity.json").write_text(json.dumps({"command_count": 3}) + "\n", encoding="utf-8")
        (mode_dir / "agent_usage.json").write_text(json.dumps({"total_tokens": 100}) + "\n", encoding="utf-8")

    original_load_json = metrics_report.load_json
    common_file_reads = {
        "run_summary.json": 0,
        "record": 0,
        "agent_activity.json": 0,
        "agent_usage.json": 0,
    }

    def counted_load_json(path, default=None):
        if path.name in common_file_reads:
            common_file_reads[path.name] += 1
        elif path.name.endswith("_record.json") or path.name == "benchmark_record.json":
            common_file_reads["record"] += 1
        return original_load_json(path, default)

    monkeypatch.setattr(metrics_report, "load_json", counted_load_json)
    metrics_report.write_reports(tmp_path, "Synthetic Metrics")
    monkeypatch.setattr(sys, "argv", ["benchmark_insights", str(tmp_path)])
    insights_main()

    assert (tmp_path / "metrics_report.json").is_file()
    metrics_markdown = (tmp_path / "metrics_report.md").read_text(encoding="utf-8")
    insights_markdown = (tmp_path / "benchmark_insights.md").read_text(encoding="utf-8")
    assert "<svg" in metrics_markdown
    assert "<svg" in insights_markdown
    assert "Metrics (AUROC)" in metrics_markdown
    assert "Metrics (AUROC)" in insights_markdown
    assert "Benchmark Metrics Comparison" not in insights_markdown
    assert "with_skills_eval" not in insights_markdown
    assert "Evaluator" not in insights_markdown
    assert common_file_reads == {
        "run_summary.json": 2,
        "record": 2,
        "agent_activity.json": 2,
        "agent_usage.json": 2,
    }
