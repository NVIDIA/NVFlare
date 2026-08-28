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

"""Subprocess runner for AgenticFL server-local training preflight jobs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from agenticfl.job_data import run_fed_job_recipe
from agenticfl.job_train import _build_fedavg_job_object


def _read_input(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("preflight runner input must be a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m agenticfl.flare.training_preflight_runner <input-json>", file=sys.stderr)
        return 2
    payload = _read_input(args[0])
    job = _build_fedavg_job_object(
        job_name=str(payload["job_name"]),
        training_plan=payload["training_plan"],
        training_code_spec=payload["training_code_spec"],
        client_configs=payload["client_configs"],
        min_clients=int(payload.get("min_clients") or 1),
    )
    recipe_run = run_fed_job_recipe(
        job,
        workspace_root=payload["workspace_root"],
        clients=[str(client) for client in payload.get("clients", [])],
        threads=int(payload.get("threads") or 1),
        log_config=str(payload.get("log_config") or "concise"),
    )
    result_path = Path(payload["result_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(recipe_run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
