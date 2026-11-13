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

"""Qualify source fixtures and execute the extracted plugin's report helper.

This does not run an agent, convert a job, or replace official Tier 3 evaluation.
"""

import argparse
import importlib.metadata
import json
import math
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from prepare import hashes, write_json


def stop(process):
    """Release the command's process group on timeout or cancellation."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # A child can outlive an exited parent, so also clean up the group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def run(command, cwd, logs, name):
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1")
    env["MPLCONFIGDIR"] = str(logs / "matplotlib")
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    with (logs / f"{name}.out").open("w") as out, (logs / f"{name}.err").open("w") as err:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=out, stderr=err, start_new_session=True)
        try:
            return process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{name} exceeded the 120 second limit") from None
        finally:
            stop(process)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(prepared, output):
    cases = json.loads((prepared / "cases.json").read_text())
    plugin = Path(cases["plugin"])
    require(
        not output.is_relative_to(prepared) and not output.is_relative_to(plugin),
        "Check output must be outside the prepared inputs and plugin",
    )
    source_hashes = json.loads((prepared / "source-hashes.json").read_text())
    plugin_hashes = json.loads((prepared / "plugin-hashes.json").read_text())
    require(hashes(prepared / "workspaces") == source_hashes, "Prepared input fixtures changed")
    require(hashes(plugin) == plugin_hashes, "Extracted plugin changed")
    output.mkdir(parents=True, exist_ok=False)
    checks = []
    status = {
        "kind": "fixture qualification and direct packaged-helper execution",
        "agent_evaluation": False,
        "versions": {
            n: importlib.metadata.version(n) for n in ("torch", "pytorch-lightning", "transformers", "accelerate")
        },
        "checks": checks,
    }
    import nvflare

    status["nvflare"] = {"version": nvflare.__version__, "module": nvflare.__file__}
    cli = str(Path(sys.executable).parent / "nvflare")
    try:
        for framework in ("pytorch", "lightning", "huggingface"):
            target = output / f"convert-{framework}"
            shutil.copytree(prepared / "workspaces" / target.name, target)
            rc = run(
                [cli, "agent", "inspect", "source", str(target), "--format", "json"],
                target,
                output,
                f"inspect-{framework}",
            )
            require(rc == 0, f"{framework} source inspection failed")
            inspection = json.loads((output / f"inspect-{framework}.out").read_text())
            expected = f"nvflare-convert-{framework}"
            require(inspection["data"]["routing"]["recommended_skill"] == expected, f"Unexpected owner for {framework}")
            rc = run([sys.executable, "train.py"], target, output, f"train-{framework}")
            require(rc == 0, f"{framework} source training failed; inspect retained logs")
            rows = []
            for line in (output / f"train-{framework}.out").read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            metric = next((r for r in reversed(rows) if isinstance(r, dict) and r.get("framework") == framework), None)
            require(
                metric is not None and math.isfinite(metric["loss"]), f"Missing finite training metric for {framework}"
            )
            checks.append(
                {"case": target.name, "status": "source_fixture_passed", "metric": metric, "conversion_run": False}
            )

        reporter = plugin / "skills/nvflare-autofl-report/scripts/generate_report.py"
        for state in ("stopped", "active", "pending"):
            name = f"report-{state}"
            target = output / name
            shutil.copytree(prepared / "workspaces" / name, target)
            before = hashes(target)
            rc = run([sys.executable, str(reporter), str(target)], target, output, name)
            unchanged = all(
                hashes(target).get(p) == digest for p, digest in before.items() if not p.endswith("campaign.lock")
            )
            require(unchanged, f"{name} modified campaign evidence")
            summary = target / "autofl_report_summary.json"
            report = target / "autofl_final_report.md"
            if state == "stopped":
                require(rc == 0 and report.is_file() and summary.is_file(), "Stopped report artifacts missing")
                data = json.loads(summary.read_text())
                require(
                    data["baseline"]["score"] == 0.5 and data["best"]["score"] == 0.7,
                    "Report scores differ from fixture",
                )
                require(data["termination"]["state_allowed_final_response"] is True, "Stopped-state provenance lost")
                checks.append(
                    {
                        "case": name,
                        "status": "packaged_helper_passed",
                        "plot_available": data["artifacts"]["progress_plot_available"],
                    }
                )
            else:
                text = (output / f"{name}.err").read_text()
                reason = "final_response_allowed=false" if state == "active" else "candidate evidence is unfinished"
                require(
                    rc != 0 and reason in text and not summary.exists() and not report.exists(),
                    f"{name} did not reject correctly",
                )
                checks.append({"case": name, "status": "packaged_helper_rejected_as_expected"})
        status["passed"] = True
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        status["passed"] = False
        status["error"] = str(exc)
    finally:
        status["plugin_unchanged"] = hashes(plugin) == plugin_hashes
        status["input_fixtures_unchanged"] = hashes(prepared / "workspaces") == source_hashes
        status["passed"] = (
            status.get("passed", False) and status["plugin_unchanged"] and status["input_fixtures_unchanged"]
        )
        write_json(output / "results.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        status = check(args.prepared.resolve(), args.output.resolve())
    except (OSError, ValueError, KeyError, importlib.metadata.PackageNotFoundError) as exc:
        parser.exit(1, f"Workflow check could not start: {exc}\n")
    print(json.dumps(status, indent=2))
    raise SystemExit(0 if status["passed"] else 1)


if __name__ == "__main__":
    main()
