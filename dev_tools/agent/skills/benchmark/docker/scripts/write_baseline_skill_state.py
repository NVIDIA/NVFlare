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

"""Write baseline no-skills state files for a benchmark Docker image."""

from __future__ import annotations

import json
import os
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(os.environ["BENCHMARK_AGENT_HOME"])
    agent = os.environ["BENCHMARK_DOCKER_AGENT"]
    reason = f"local baseline wheel does not package {agent} skills"
    write_json(
        root / os.environ["SKILLS_INSTALL_OUTPUT"],
        {"status": "skipped", "agent": agent, "reason": reason},
    )
    write_json(
        root / os.environ["SKILLS_LIST_OUTPUT"],
        {"status": "skipped", "agent": agent, "installed": [], "reason": reason},
    )


if __name__ == "__main__":
    main()
