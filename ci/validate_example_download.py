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

"""Exercise an installed distribution's real example download and zero-flag run.

Run in scheduled/release CI, not normal PR CI: GitHub must contain the selected
commit and catalog. All NVFlare imports happen in the installed interpreter,
outside the source checkout. Preserve output under --output for release evidence.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_VERIFY_MODEL = """
import json
import sys
from pathlib import Path
import torch
from model import create_model

run = Path(sys.argv[1]) / "hello-pt" / "server" / "simulate_job"
summary = json.loads((run / "metrics/metrics_summary.json").read_text())
assert summary["status"] == "metrics_reported", summary
assert summary["final_round"] == 2, summary
rounds = [json.loads(line) for line in (run / "metrics/round_metrics.jsonl").read_text().splitlines()]
initial = {item["name"]: item["value"] for item in rounds[0]["aggregated_metrics"]}["accuracy"]
results = json.loads((run / "cross_site_val/cross_val_results.json").read_text())
assert set(results) == {"site-1", "site-2"}, results
accuracy = {site: values["SRV_FL_global_model.pt"]["accuracy"] for site, values in results.items()}
assert initial <= 20.0, initial
assert min(accuracy.values()) >= 60.0, accuracy
assert min(accuracy.values()) >= initial + 40.0, (initial, accuracy)
model_path = run / "app_server/FL_global_model.pt"
create_model().load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True)["model"])
print(json.dumps({"initial_accuracy": initial, "final_accuracy": accuracy, "model": str(model_path)}))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Python in the clean installed environment")
    parser.add_argument(
        "--output", required=True, type=Path, help="new directory for all downloaded files and evidence"
    )
    parser.add_argument("--expected-commit", required=True, help="commit from which the distribution was built")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["NVFLARE_EXAMPLES_CACHE_DIR"] = str(root / "download-cache")
    env["NVFLARE_SIMULATOR_WORKSPACE_ROOT"] = str(root / "simulation")

    def execute(command, label, cwd=root, timeout=180):
        with (root / f"{label}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [args.python, *command],
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        result.check_returncode()
        return (root / f"{label}.log").read_text(encoding="utf-8")

    # Reset only the installed command's cache in this dedicated validation environment.
    execute(["-m", "nvflare.cli", "examples", "cache", "clear", "--format", "json"], "clear")
    downloaded = json.loads(
        execute(["-m", "nvflare.cli", "examples", "get", "hello-pt", "--format", "json"], "download")
    )
    data = downloaded["data"]
    assert downloaded["status"] == "ok" and data["cache_status"] == "downloaded", downloaded
    assert data["commit"] == args.expected_commit, data
    example = root / "hello-pt"
    provenance = json.loads((example / ".nvflare-example.json").read_text())
    assert provenance["commit"] == args.expected_commit, provenance
    cached = json.loads(
        execute(
            ["-m", "nvflare.cli", "examples", "get", "hello-pt", "--dest", "cached-copy", "--format", "json"],
            "cache-hit",
        )
    )
    assert cached["data"]["cache_status"] == "cache hit", cached
    execute(["job.py"], "training", cwd=example, timeout=300)
    metrics = json.loads(execute(["-c", _VERIFY_MODEL, str(root / "simulation")], "model-check", cwd=example))
    evidence = {"download": data, "metrics": metrics}
    (root / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
