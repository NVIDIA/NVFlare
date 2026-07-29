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

"""Compare standard NVFlare simulation with Collab direct calls for SFT."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from collab.benchmarks.prepare_data import validate_data_identity

BENCHMARK_DIR = Path(__file__).resolve().parent
ADVANCED_DIR = BENCHMARK_DIR.parents[1]
DEFAULT_CONFIG = BENCHMARK_DIR / "configs" / "pt_llm_sft.json"
SCHEMES = {
    "standard": {
        "module": "collab.benchmarks.pt_llm_sft.standard_simulator",
        "extra_args": [],
    },
    "collab": {
        "module": "collab.benchmarks.pt_llm_sft.benchmark",
        "extra_args": ["--exchange-format", "native"],
    },
}


def run_one(scheme: str, output_root: Path, source_config: Path) -> dict:
    metrics_file = output_root / "metrics" / f"pt_llm_sft_{scheme}.json"
    workspace_root = output_root / "workspace" / scheme
    effective_config = output_root / "configs" / f"pt_llm_sft_{scheme}.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)
    effective_config.parent.mkdir(parents=True, exist_ok=True)
    with source_config.open(encoding="utf-8") as stream:
        config = json.load(stream)
    validate_data_identity(Path(config["data_root"]), config)
    config["trainer_output_root"] = str(output_root / "trainer" / scheme)
    with effective_config.open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")

    spec = SCHEMES[scheme]
    command = [
        sys.executable,
        "-m",
        spec["module"],
        "--config",
        str(effective_config),
        "--metrics-file",
        str(metrics_file),
        "--workspace-root",
        str(workspace_root),
        *spec["extra_args"],
    ]
    env = os.environ.copy()
    if config.get("force_cpu"):
        env["CUDA_VISIBLE_DEVICES"] = ""
    if config.get("offline_mode"):
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
    if config.get("hf_home"):
        env["HF_HOME"] = config["hf_home"]

    print(f"\nRunning pt_llm_sft ({scheme})", flush=True)
    started = time.perf_counter()
    subprocess.run(command, cwd=ADVANCED_DIR, env=env, check=True)
    process_seconds = time.perf_counter() - started
    with metrics_file.open(encoding="utf-8") as stream:
        result = json.load(stream)
    result["scheme"] = scheme
    result["runner_process_seconds"] = process_seconds
    with metrics_file.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


def print_comparison(results: list[dict]):
    print("\nMatched simulator comparison")
    print("| Scheme | End-to-end process time (s) |")
    print("|---|---:|")
    by_scheme = {}
    for result in results:
        by_scheme[result["scheme"]] = result["runner_process_seconds"]
        print(f"| {result['scheme']} | {result['runner_process_seconds']:.6f} |")
    if {"standard", "collab"} <= set(by_scheme):
        standard = by_scheme["standard"]
        collab = by_scheme["collab"]
        difference = (collab - standard) / standard * 100
        print(f"\nCollab time difference versus standard: {difference:+.2f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheme",
        nargs="+",
        choices=tuple(SCHEMES),
        default=tuple(SCHEMES),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/tmp/nvflare/collab/benchmarks/results"),
    )
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    source_config = args.config.expanduser().resolve()
    results = [run_one(scheme, output_root, source_config) for scheme in args.scheme]
    print_comparison(results)
    summary_file = output_root / "summary.json"
    with summary_file.open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
        stream.write("\n")
    print(f"\nRaw metrics and summary: {output_root}")


if __name__ == "__main__":
    main()
