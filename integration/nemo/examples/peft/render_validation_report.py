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
"""Render a concise Markdown report from the H100 runner artifacts."""

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
    lines = [
        "# Nemotron 3.5 Lightning H100 validation",
        "",
        f"Overall status: **{status}** (runner exit code `{exit_code}`).",
        "",
        f"- NVFlare commit: `{_text(artifacts / 'commit_sha.txt')}`",
        "- Requested NVFlare base: `0cf98f8a5ba2b17350074a7a4c52aaa0323e01bb`",
        "- Container: `nvcr.io/nvidia/nemo-automodel:26.08`",
        f"- Container digest: `{_text(artifacts / 'container_digest.txt')}`",
        "- Model/tokenizer: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`",
        f"- Model revision: `{_text(artifacts / 'model_revision.txt')}`",
        f"- Tokenizer revision: `{_text(artifacts / 'tokenizer_revision.txt')}`",
        "- Execution scope: sequential federated simulation on one H100; no separate-host deployment or "
        "intra-client distributed training.",
        "",
    ]
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
    if split:
        lines.extend(["## Dataset", "", f"Split seed: `{split['seed']}`; Dirichlet alpha: `{split['alpha']}`.", ""])
        lines.extend(["| File | SHA-256 | Rows | Class counts |", "| --- | --- | ---: | --- |"])
        for name, value in split["files"].items():
            lines.append(
                f"| {name} | `{value['sha256']}` | {value['rows']} | "
                f"`{json.dumps(value['class_counts'], sort_keys=True)}` |"
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
