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
from pathlib import Path


def test_benchmark_insights_explains_docker_image_failures(tmp_path):
    from assist_tools.skills_benchmark.skills.harness.modes import NO_SKILLS_MODE
    from assist_tools.skills_benchmark.skills.harness.reports.benchmark_insights import (
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
    from assist_tools.skills_benchmark.skills.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from assist_tools.skills_benchmark.skills.harness.reports.benchmark_insights import (
        collect_benchmark_runs,
        dependency_reference_notes,
    )

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
    from assist_tools.skills_benchmark.skills.harness.modes import NO_SKILLS_MODE
    from assist_tools.skills_benchmark.skills.harness.reports import benchmark_insights

    mode_dir = tmp_path / NO_SKILLS_MODE
    mode_dir.mkdir()
    (mode_dir / "agent_events.jsonl").write_text("0123456789", encoding="utf-8")
    monkeypatch.setattr(benchmark_insights, "MAX_AGENT_EVENTS_TEXT_BYTES", 8)

    run = benchmark_insights.collect_benchmark_runs(tmp_path)[NO_SKILLS_MODE]

    assert run["agent_events_text"] == "01234567"


def test_benchmark_reports_read_canonical_record_layout(tmp_path):
    from assist_tools.skills_benchmark.skills.harness.common import write_json
    from assist_tools.skills_benchmark.skills.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from assist_tools.skills_benchmark.skills.harness.reports import metrics_report
    from assist_tools.skills_benchmark.skills.harness.reports.benchmark_insights import collect_benchmark_runs

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
            {"run_id": f"run_{index:05d}", "mode": mode, "record_dir": str(record_dir.relative_to(tmp_path))}
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
    assert runs[WITH_SKILLS_MODE]["record"]["reported_validation_metric"]["name"] == "AUROC"

    metrics_report.write_reports(tmp_path, "Synthetic Metrics")

    assert (tmp_path / "metrics_report.json").is_file()
    assert "Metrics (AUROC)" in (tmp_path / "metrics_report.md").read_text(encoding="utf-8")


def test_numeric_comparison_rejects_bool_values():
    from assist_tools.skills_benchmark.skills.harness.reports.metrics_report import numeric_comparison

    rows = [
        {"summary": {"elapsed_seconds": 10, "token_count": 100}},
        {"summary": {"elapsed_seconds": True, "token_count": False}},
    ]

    assert numeric_comparison(rows) == {}


def test_structure_tree_falls_back_to_final_workspace_when_changed_python_is_empty():
    from assist_tools.skills_benchmark.skills.harness.modes import WITH_SKILLS_MODE
    from assist_tools.skills_benchmark.skills.harness.reports.benchmark_insights import structure_trees_section

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


def test_status_summary_is_human_readable_for_failures():
    from assist_tools.skills_benchmark.skills.harness.modes import NO_SKILLS_MODE
    from assist_tools.skills_benchmark.skills.harness.reports.benchmark_insights import status_summary

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
    from assist_tools.skills_benchmark.skills.harness.reports.benchmark_insights import (
        failure_evidence,
        failure_root_cause,
    )

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
    from assist_tools.skills_benchmark.skills.harness.reports.benchmark_insights import failure_root_cause

    run = {
        "available": True,
        "agent_events_text": "unstructured error text",
        "record": {"agent_exit_summary": {"failure_category": "agent_auth_failure"}},
        "run": {"agent_exit_code": 1},
    }

    assert failure_root_cause(run) == "Agent failure category: agent_auth_failure"


def test_failure_analysis_identifies_agent_generated_requirements_file():
    from assist_tools.skills_benchmark.skills.harness.reports.benchmark_insights import dependency_reference_notes

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


def test_shared_lifecycle_requires_dependency_preflight_before_missing_dependency_blocker():
    lifecycle = (Path(__file__).resolve().parents[3] / "skills" / "_shared" / "nvflare-job-lifecycle.md").read_text(
        encoding="utf-8"
    )

    assert "all generated NVFLARE jobs" in lifecycle
    assert "recipe-, framework-, and algorithm-independent" in lifecycle
    assert "requirements-train.txt" in lifecycle
    assert "python -m pip install -r <requirements-file>" in lifecycle
    assert "dependency or import failures" in lifecycle
    assert "Report missing dependencies as blockers only when" in lifecycle


def test_readme_metric_alignment_uses_aggregated_validation_metric_scalar():
    from assist_tools.skills_benchmark.skills.harness.quality_signals import metric_signal

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
    from assist_tools.skills_benchmark.skills.harness.quality_signals import metric_signal

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


def test_job_guidance_metric_alignment_uses_non_readme_docs(tmp_path):
    from assist_tools.skills_benchmark.skills.harness.quality_signals import metric_signal
    from assist_tools.skills_benchmark.skills.harness.records import discover_job_guidance

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
    from assist_tools.skills_benchmark.skills.harness import records

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
    from assist_tools.skills_benchmark.skills.harness import records

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
    from assist_tools.skills_benchmark.skills.harness.records import MAX_GUIDANCE_FILE_BYTES, discover_job_guidance

    job = tmp_path / "job"
    job.mkdir()
    with job.joinpath("README.md").open("wb") as stream:
        stream.truncate(MAX_GUIDANCE_FILE_BYTES + 1)

    sources, guidance_text = discover_job_guidance(job)

    assert sources == []
    assert guidance_text == ""


def test_job_guidance_stops_collecting_after_guidance_file_cap(tmp_path, monkeypatch):
    from assist_tools.skills_benchmark.skills.harness import records

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
    from assist_tools.skills_benchmark.skills.harness.quality_signals import metric_signal
    from assist_tools.skills_benchmark.skills.harness.records import discover_job_guidance

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
    from assist_tools.skills_benchmark.skills.harness.quality_signals import metric_signal
    from assist_tools.skills_benchmark.skills.harness.records import discover_job_guidance

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
    from assist_tools.skills_benchmark.skills.harness.quality_signals import metric_signal
    from assist_tools.skills_benchmark.skills.harness.records import discover_job_guidance

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
    from assist_tools.skills_benchmark.skills.harness.modes import NO_SKILLS_MODE
    from assist_tools.skills_benchmark.skills.harness.quality_signals import metric_signal
    from assist_tools.skills_benchmark.skills.harness.reports.benchmark_insights import (
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


def test_metric_mismatch_evidence_includes_integer_metric_value():
    from assist_tools.skills_benchmark.skills.harness.quality_signals import format_metric_value

    assert format_metric_value(1) == " 1."
    assert format_metric_value(1.0) == " 1.0000."
    assert format_metric_value(None) == "."


def test_missing_target_metric_section_reports_observed_alternate_metrics():
    from assist_tools.skills_benchmark.skills.harness.modes import NO_SKILLS_MODE
    from assist_tools.skills_benchmark.skills.harness.reports.benchmark_insights import (
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
    from assist_tools.skills_benchmark.skills.harness.modes import NO_SKILLS_MODE
    from assist_tools.skills_benchmark.skills.harness.reports.benchmark_insights import (
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

    assert "Command evidence" in section
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


def test_readme_metric_alignment_uses_server_best_validation_metric_scalar():
    from assist_tools.skills_benchmark.skills.harness.quality_signals import metric_signal

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
    from assist_tools.skills_benchmark.skills.harness.quality_signals import metric_signal

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
    from assist_tools.skills_benchmark.skills.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from assist_tools.skills_benchmark.skills.harness.reports.benchmark_insights import (
        embedded_bar_chart,
        outcome_metrics_table,
    )

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
    assert "FL scalar result" not in chart
    assert "AUROC 0." not in chart
    assert chart.count("AUROC") == 1
    assert ">0.7529<" in chart
    assert "| Metrics (AUROC) | AUROC 0.7562 | AUROC 0.7529 |" in table
    assert "FL scalar result" not in table


def test_metrics_chart_uses_labeled_aggregated_metric_from_legacy_record():
    from assist_tools.skills_benchmark.skills.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from assist_tools.skills_benchmark.skills.harness.reports.benchmark_insights import (
        embedded_bar_chart,
        outcome_metrics_table,
    )

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
    from assist_tools.skills_benchmark.skills.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from assist_tools.skills_benchmark.skills.harness.reports.benchmark_insights import (
        embedded_bar_chart,
        outcome_metrics_table,
    )

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
    from assist_tools.skills_benchmark.skills.harness.reports.benchmark_insights import tree_from_paths

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
    from assist_tools.skills_benchmark.skills.harness.records import write_json, write_run_summary

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
            },
        },
    )

    write_run_summary(final_record, summary_path, print_summary=False)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["agent_process_passed"] is True
    assert summary["agent_process_exit_code"] == 0
    assert summary["agent_exit_code"] == 0
    assert summary["agent_usage"] == {"total_tokens": 10}
    assert "codex_process_passed" not in summary
    assert "codex_process_exit_code" not in summary
    assert "codex_exit_code" not in summary
    assert "codex_usage" not in summary
    assert not any(key.startswith("codex_") for key in summary["all_metrics"])


def test_run_summary_ignores_codex_usage_fallback_and_reports_prompt_hash(tmp_path):
    from assist_tools.skills_benchmark.skills.harness.records import write_json, write_run_summary

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
    from assist_tools.skills_benchmark.skills.harness.common import make_tree_readable

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
    from assist_tools.skills_benchmark.skills.harness.reports.scenario_report import write_scenario_report

    write_scenario_report(
        tmp_path,
        {
            "scenario_name": "pipe scenario",
            "status": "passed",
            "completed_run_count": 1,
            "expanded_case_count": 1,
            "winner_policy": "quality",
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


def test_report_generators_write_two_mode_outputs(tmp_path, monkeypatch):
    from assist_tools.skills_benchmark.skills.harness.modes import NO_SKILLS_MODE, WITH_SKILLS_MODE
    from assist_tools.skills_benchmark.skills.harness.reports import metrics_report
    from assist_tools.skills_benchmark.skills.harness.reports.benchmark_insights import main as insights_main

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

    original_collect_benchmark_runs = metrics_report.collect_benchmark_runs
    collect_calls = {"count": 0}

    def counted_collect_benchmark_runs(root):
        collect_calls["count"] += 1
        return original_collect_benchmark_runs(root)

    monkeypatch.setattr(metrics_report, "collect_benchmark_runs", counted_collect_benchmark_runs)
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
    assert collect_calls["count"] == 1
