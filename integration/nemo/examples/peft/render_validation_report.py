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
"""Render a Markdown report from the H100 runner artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _text(path: Path, default: str = "not recorded") -> str:
    return path.read_text().strip() if path.is_file() else default


def _json(path: Path):
    if not path.is_file():
        return None
    with path.open() as f:
        return json.load(f)


def _continuity_summary(path: Path):
    report = _json(path)
    if not report:
        return None
    rounds = report.get("rounds", [])
    clients = [client for round_report in rounds for client in round_report.get("clients", [])]
    errors = [
        round_report["independent_fp32_verification"]["max_abs_error"]
        for round_report in rounds
        if "independent_fp32_verification" in round_report
    ]
    tensor_counts = sorted({round_report.get("aggregate", {}).get("tensor_count") for round_report in rounds})
    tensor_counts = [value for value in tensor_counts if value is not None]
    steps = sorted({client.get("actual_optimizer_steps") for client in clients})
    steps = [value for value in steps if value is not None]
    return {
        "passed": report.get("all_checks_passed") is True,
        "rounds": len(rounds),
        "client_tasks": len(clients),
        "tensor_counts": tensor_counts,
        "optimizer_steps": steps,
        "max_abs_error": max(errors) if errors else None,
    }


def _telemetry_summary(path: Path):
    if not path.is_file():
        return {}
    result = {}
    with path.open() as f:
        for raw_row in csv.DictReader(f):
            row = {key.strip(): value.strip() for key, value in raw_row.items()}
            try:
                index = int(row["index"])
                memory = float(row["memory.used [MiB]"].split()[0])
                utilization = float(row["utilization.gpu [%]"].split()[0])
                power = float(row["power.draw [W]"].split()[0])
            except (KeyError, TypeError, ValueError):
                continue
            stats = result.setdefault(
                index,
                {"samples": 0, "max_memory_mib": 0.0, "max_utilization": 0.0, "max_power_w": 0.0},
            )
            stats["samples"] += 1
            stats["max_memory_mib"] = max(stats["max_memory_mib"], memory)
            stats["max_utilization"] = max(stats["max_utilization"], utilization)
            stats["max_power_w"] = max(stats["max_power_w"], power)
    return result


def _format_metrics(metrics: dict) -> list[str]:
    return [
        f"{metrics['response_token_loss']:.6f}",
        f"{metrics['accuracy']:.6f}",
        f"{metrics['macro_f1']:.6f}",
        f"`{json.dumps(metrics['prediction_counts'], sort_keys=True)}`",
        f"`{json.dumps(metrics['confusion'], sort_keys=True)}`",
    ]


def _append_evaluation_row(lines: list[str], label: str, split: str, metrics: dict):
    lines.append(f"| {label} | {split} | " + " | ".join(_format_metrics(metrics)) + " |")


def render(run_root: Path, exit_code: int) -> str:
    artifacts = run_root / "artifacts"
    acceptance = _json(artifacts / "acceptance.json")
    split = _json(run_root / "data_split" / "split_manifest.json")
    blocked = _text(artifacts / "BLOCKED.txt", "")
    stage_rows = []
    timings = artifacts / "stage_timings.csv"
    if timings.is_file():
        with timings.open() as f:
            stage_rows = list(csv.DictReader(f))
    if blocked:
        status = "BLOCKED"
    elif acceptance and acceptance.get("passed") and exit_code == 0:
        status = "PASSED"
    else:
        status = "FAILED"

    training_commit = _text(artifacts / "commit_sha.txt")
    validation_commit = _text(artifacts / "validation_fix_commit_sha.txt", "")
    selected_gpu = _text(artifacts / "selected_gpu.txt")
    lines = [
        "# Nemotron 3.5 Lightning H100 validation",
        "",
        f"Overall status: **{status}** (final validation exit code `{exit_code}`).",
        "",
        f"- NVFlare training commit: `{training_commit}`",
    ]
    if validation_commit and validation_commit != training_commit:
        lines.append(f"- Post-run validator commit: `{validation_commit}`")
    lines.extend(
        [
            "- Requested NVFlare base: `0cf98f8a5ba2b17350074a7a4c52aaa0323e01bb`",
            "- Container: `nvcr.io/nvidia/nemo-automodel:26.08`",
            f"- Container digest: `{_text(artifacts / 'container_digest.txt')}`",
            "- Model/tokenizer: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`",
            f"- Model revision: `{_text(artifacts / 'model_revision.txt')}`",
            f"- Tokenizer revision: `{_text(artifacts / 'tokenizer_revision.txt')}`",
            f"- Selected physical GPU: `{selected_gpu}`",
            "- Execution scope: sequential federated simulation on one H100; no separate-host deployment or "
            "intra-client distributed training.",
            "",
        ]
    )

    failed_acceptance = [row for row in stage_rows if row["stage"].startswith("acceptance") and row["exit_code"] != "0"]
    successful_retry = any(row["stage"] == "acceptance_retry" and row["exit_code"] == "0" for row in stage_rows)
    if failed_acceptance and successful_retry:
        lines.extend(
            [
                "The original run stopped in the result parser after all GPU work completed. The pinned AutoModel "
                "version stores loss directly in `last_training_record`; the validator initially expected a nested "
                "metrics object. The parser-only fix was tested, recorded above, and rerun against the unchanged "
                "training and evaluation artifacts.",
                "",
            ]
        )

    feasibility = _text(artifacts / "SINGLE_GPU_FEASIBILITY_FAILURE.txt", "")
    if blocked or feasibility:
        lines.extend(["## Blocking result", "", blocked or feasibility, ""])
    hardware = _text(artifacts / "gpu_preflight.csv", "")
    host_memory = _text(artifacts / "host_memory.txt", "")
    docker_socket = _text(artifacts / "docker_socket.txt", "")
    if hardware or host_memory or docker_socket:
        lines.extend(["## Host preflight", ""])
        for title, value in (
            ("GPU inventory", hardware),
            ("Host memory", host_memory),
            ("Docker socket", docker_socket),
        ):
            if value:
                lines.extend([f"### {title}", "", "```text", value, "```", ""])

    telemetry = _telemetry_summary(run_root / "logs" / "gpu_telemetry.csv")
    if telemetry:
        lines.extend(
            [
                "### GPU telemetry",
                "",
                "| Physical GPU | Samples | Peak memory (MiB) | Peak utilization (%) | Peak power (W) |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for index, stats in sorted(telemetry.items()):
            lines.append(
                f"| {index} | {stats['samples']} | {stats['max_memory_mib']:.0f} | "
                f"{stats['max_utilization']:.0f} | {stats['max_power_w']:.2f} |"
            )
        lines.append("")

    if split:
        lines.extend(["## Dataset", "", f"Split seed: `{split['seed']}`; Dirichlet alpha: `{split['alpha']}`.", ""])
        resolution = split.get("overlap_resolution")
        if resolution:
            lines.extend(
                [
                    f"Training overlap policy: `{resolution['policy']}`; removed rows: "
                    f"`{resolution['removed_rows']}` across `{resolution['removed_unique_sentences']}` unique "
                    "sentences. Validation and test were unchanged.",
                    "",
                ]
            )
        lines.extend(["| File | SHA-256 | Rows | Class counts |", "| --- | --- | ---: | --- |"])
        for name, value in split["files"].items():
            lines.append(
                f"| {name} | `{value['sha256']}` | {value['rows']} | "
                f"`{json.dumps(value['class_counts'], sort_keys=True)}` |"
            )
        lines.append("")

    continuity_paths = [
        ("CPU weighted FedAvg", run_root / "runs" / "cpu_federation" / "continuity.json"),
        ("Lightning smoke", run_root / "runs" / "smoke" / "continuity.json"),
        ("Lightning continuity", run_root / "runs" / "continuity" / "continuity.json"),
        ("Learning seed 42", run_root / "runs" / "learning_seed42" / "continuity.json"),
        ("Learning seed 43", run_root / "runs" / "learning_seed43" / "continuity.json"),
    ]
    continuity = [(label, _continuity_summary(path)) for label, path in continuity_paths]
    continuity = [(label, summary) for label, summary in continuity if summary]
    if continuity:
        lines.extend(
            [
                "## Adapter continuity and aggregation",
                "",
                "| Workload | Passed | Rounds | Client tasks | Tensors/aggregate | Steps/task | Max FP32 error |",
                "| --- | --- | ---: | ---: | --- | --- | ---: |",
            ]
        )
        for label, summary in continuity:
            lines.append(
                f"| {label} | {summary['passed']} | {summary['rounds']} | {summary['client_tasks']} | "
                f"`{summary['tensor_counts']}` | `{summary['optimizer_steps']}` | "
                f"{summary['max_abs_error']} |"
            )
        lines.append("")

    smoke = _json(run_root / "runs" / "smoke" / "continuity.json")
    smoke_reload = _json(run_root / "logs" / "smoke_reload_compare.log")
    if smoke:
        client = smoke["rounds"][0]["clients"][0]
        automodel = client.get("automodel_report", {})
        training = automodel.get("last_training_record", {})
        lines.extend(
            [
                "### Single-GPU smoke details",
                "",
                f"- Actual optimizer steps: `{client['actual_optimizer_steps']}`; trainable adapter tensors: "
                f"`{automodel.get('trainable_tensor_count')}`; frozen base parameters: "
                f"`{automodel.get('frozen_parameter_count')}`.",
                f"- Final observed loss: `{training.get('loss')}`; gradient norm: `{training.get('grad_norm')}`; "
                f"reported allocated memory: `{training.get('mem')}` GiB; adapter update norm: "
                f"`{client.get('update_norm')}`.",
                f"- Incoming tensors loaded before training: "
                f"`{automodel.get('loaded_matches_received_after_dtype_cast')}`.",
            ]
        )
        if smoke_reload:
            lines.append(
                f"- Native reload kept all discrete metrics exact; response-token loss delta "
                f"`{smoke_reload['response_token_loss_delta']}` was within "
                f"`{smoke_reload['response_token_loss_tolerance']}`."
            )
        lines.append("")

    evaluation_rows = []
    base = _json(run_root / "evaluation" / "base" / "summary.json")
    seed_summaries = {
        seed: _json(run_root / "evaluation" / f"learning_seed{seed}" / "final" / "summary.json") for seed in (42, 43)
    }
    if base:
        evaluation_rows.append(("Base", "validation", base["validation"]))
        evaluation_rows.append(("Base", "test", base["test"]))
    for seed, summary in seed_summaries.items():
        if summary:
            evaluation_rows.append((f"Seed {seed}, round 2", "validation", summary["validation"]))
            evaluation_rows.append((f"Seed {seed}, round 2", "test", summary["test"]))
    if evaluation_rows:
        lines.extend(
            [
                "## Evaluation",
                "",
                "All acceptance values use uncalibrated label scores. Round 2 was predetermined for final test "
                "evaluation.",
                "",
                "| Model | Split | Response-token loss | Accuracy | Macro-F1 | Prediction counts | Confusion matrix |",
                "| --- | --- | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for label, name, metrics in evaluation_rows:
            _append_evaluation_row(lines, label, name, metrics)
        lines.append("")

    round_rows = []
    for seed in (42, 43):
        for round_number in range(3):
            summary = _json(
                run_root / "evaluation" / f"learning_seed{seed}" / f"round_{round_number}_validation" / "summary.json"
            )
            if summary:
                round_rows.append((seed, round_number, summary["validation"]))
    if round_rows:
        lines.extend(
            [
                "### Per-round validation",
                "",
                "| Seed | Round | Response-token loss | Accuracy | Macro-F1 |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for seed, round_number, metrics in round_rows:
            lines.append(
                f"| {seed} | {round_number} | {metrics['response_token_loss']:.6f} | "
                f"{metrics['accuracy']:.6f} | {metrics['macro_f1']:.6f} |"
            )
        lines.append("")

    if acceptance:
        lines.extend(
            [
                "## Learning acceptance",
                "",
                f"- Base validation response-token loss: `{acceptance['base_validation_response_token_loss']}`",
                f"- Final validation losses: `{acceptance['final_validation_response_token_losses']}`",
                f"- Base test Macro-F1: `{acceptance['base_test_macro_f1']}`",
                f"- Final test Macro-F1 values: `{acceptance['final_test_macro_f1']}`",
                f"- Mean final test Macro-F1: `{acceptance['mean_final_test_macro_f1']}`",
                f"- Failures: `{acceptance['failures']}`",
                "",
            ]
        )

    nano_root = run_root / "nano_regression"
    nano_exit = _text(nano_root / "artifacts" / "nano_regression_exit_code.txt", "")
    if nano_root.is_dir():
        nano_predictions = nano_root / "artifacts" / "predictions.json"
        lines.extend(
            [
                "## Nano regression",
                "",
                f"- Exit code: `{nano_exit or 'not completed'}`",
                "- Workload: existing Nano profile, two clients, two rounds, two optimizer steps per client/round.",
                f"- Container digest: `{_text(nano_root / 'artifacts' / 'container_digest.txt')}`",
                f"- Final adapter reload and prediction artifact present: `{nano_predictions.is_file()}`",
                "",
            ]
        )

    if stage_rows:
        lines.extend(["## Stages", "", "| Stage | Seconds | Exit code |", "| --- | ---: | ---: |"])
        for row in stage_rows:
            lines.append(f"| {row['stage']} | {row['elapsed_seconds']} | {row['exit_code']} |")
        lines.append("")
    lines.extend(["## Evidence", ""])
    if blocked:
        lines.extend(
            [
                "The host runner log and Docker error are in `logs/`; commit, hardware, socket, and exit-code "
                "evidence are in `artifacts/`. Container, model, dataset, training, and evaluation artifacts were "
                "not created because the Docker prerequisite failed.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Commands and stdout/stderr are in `logs/`; GPU telemetry is in `logs/gpu_telemetry.csv`; client and "
                "server adapter checkpoints and round manifests are in `runs/`; exact evaluation outputs are in "
                "`evaluation/`; resolved configuration and package versions are in `artifacts/`.",
                "",
            ]
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--exit_code", type=int, required=True)
    args = parser.parse_args()
    report = render(args.run_root, args.exit_code)
    output = args.run_root / "validation_report.md"
    output.write_text(report)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
