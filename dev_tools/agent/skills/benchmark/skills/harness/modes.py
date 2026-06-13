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

"""Run-mode definitions shared by wrappers and reports."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class ModeSpec:
    label: str
    mode: str
    skills_enabled: bool


BENCHMARK_RUNS: tuple[ModeSpec, ...] = (
    ModeSpec("No skills baseline", "without_skills", False),
    ModeSpec("With skills", "with_skills", True),
)

# These aliases intentionally share the same tuple. They preserve wrapper/report
# vocabulary while the benchmark has one canonical two-mode run set.
PAIR_RUNS: tuple[ModeSpec, ...] = BENCHMARK_RUNS

KNOWN_RUNS: tuple[ModeSpec, ...] = BENCHMARK_RUNS
KNOWN_RUN_BY_MODE: dict[str, ModeSpec] = {item.mode: item for item in KNOWN_RUNS}


def mode_records(runs: Iterable[ModeSpec]) -> list[dict[str, object]]:
    return [asdict(item) for item in runs]


def mode_names(runs: Iterable[ModeSpec]) -> list[str]:
    return [item.mode for item in runs]


def mode_spec(mode: str) -> ModeSpec:
    try:
        return KNOWN_RUN_BY_MODE[mode]
    except KeyError as exc:
        valid_modes = ", ".join(KNOWN_RUN_BY_MODE)
        raise ValueError(f"Unknown mode {mode}; expected one of: {valid_modes}") from exc


def mode_shell_rows(runs: Iterable[ModeSpec]) -> list[str]:
    rows = []
    for item in runs:
        rows.append("|".join([item.mode, "true" if item.skills_enabled else "false"]))
    return rows


def select_mode(
    runs: Iterable[ModeSpec],
    *,
    skills_enabled: bool | None = None,
) -> str:
    for item in runs:
        if skills_enabled is not None and item.skills_enabled != skills_enabled:
            continue
        return item.mode
    raise RuntimeError("no benchmark mode matches the requested role")


BENCHMARK_MODE_NAMES = mode_names(BENCHMARK_RUNS)
PAIR_MODE_NAMES = mode_names(PAIR_RUNS)
NO_SKILLS_MODE = "without_skills"
WITH_SKILLS_MODE = "with_skills"
PAIR_WITHOUT_MODE = NO_SKILLS_MODE
PAIR_WITH_MODE = WITH_SKILLS_MODE


def benchmark_runs() -> list[dict[str, object]]:
    return mode_records(BENCHMARK_RUNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("benchmark-json", "benchmark-shell", "pair-json", "pair-shell"))
    args = parser.parse_args()

    if args.command == "benchmark-json":
        print(json.dumps(mode_records(BENCHMARK_RUNS), indent=2, sort_keys=True))
    elif args.command == "pair-json":
        print(json.dumps(mode_records(PAIR_RUNS), indent=2, sort_keys=True))
    else:
        runs = PAIR_RUNS if args.command == "pair-shell" else BENCHMARK_RUNS
        for row in mode_shell_rows(runs):
            print(row)


if __name__ == "__main__":
    main()
