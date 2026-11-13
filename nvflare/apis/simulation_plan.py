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

"""Simulation-only execution plans shared by task and Collab workflows.

Plans are run inputs, not part of the exported job. A server-side runtime owns
their consumption and traces so an unsupported section cannot be ignored.
"""

import hashlib
import json
import math
import pickle
import random
import time
from pathlib import Path

SIMULATION_PLAN_FILE = "simulation_plan.json"
SIMULATION_PLAN_PROP = "__simulation_plan_runtime__"
ROUND_PARTICIPATION = "round_participation"
ASYNC_EXECUTION = "async_execution"
FORMAT_VERSION = 1
PHASES = ("download", "train", "upload")


def _nonnegative_number(value, label):
    try:
        number = float(value) if type(value) in (int, float) else float("nan")
    except OverflowError:
        number = float("nan")
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return number


def _sample_delay(spec, rng):
    kind = spec["distribution"]
    if kind == "constant":
        return spec["value"]
    if kind == "exponential":
        return rng.expovariate(1 / spec["mean"]) if spec["mean"] else 0.0
    return rng.uniform(spec["min"], spec["max"])


def _validate_timing(value, names):
    if not isinstance(value, dict) or set(value) != {"starts", "seed", "profiles", "horizon"}:
        raise ValueError("timing requires starts, seed, profiles, and horizon")
    starts = value["starts"]
    if not isinstance(starts, list) or len(starts) != len(names):
        raise ValueError("timing.starts must contain one time per assignment")
    starts = [_nonnegative_number(v, "logical start") for v in starts]
    if starts != sorted(starts):
        raise ValueError("timing.starts must be nondecreasing")
    seed = value["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("timing.seed must be a nonnegative integer")
    horizon = _nonnegative_number(value["horizon"], "timing.horizon")
    if horizon <= 0:
        raise ValueError("timing.horizon must be positive")
    profiles = value["profiles"]
    if not isinstance(profiles, dict) or set(profiles) != set(names):
        raise ValueError("timing.profiles must cover exactly the logical clients")
    normalized_profiles = {}
    for name, profile in profiles.items():
        if not isinstance(profile, dict) or set(profile) != set(PHASES):
            raise ValueError(f"timing profile for {name} must define download, train, and upload")
        phases = {}
        for phase, spec in profile.items():
            if not isinstance(spec, dict):
                raise ValueError(f"{name}.{phase} delay must be an object")
            kind = spec.get("distribution")
            required = (
                {
                    "constant": {"distribution", "value"},
                    "exponential": {"distribution", "mean"},
                    "uniform": {"distribution", "min", "max"},
                }.get(kind)
                if isinstance(kind, str)
                else None
            )
            if required is None or set(spec) != required:
                raise ValueError(f"unsupported delay distribution for {name}.{phase}")
            values = {k: _nonnegative_number(v, f"{name}.{phase}.{k}") for k, v in spec.items() if k != "distribution"}
            if kind == "uniform" and values["min"] > values["max"]:
                raise ValueError(f"{name}.{phase}.min exceeds max")
            phases[phase] = {"distribution": kind, **values}
        normalized_profiles[name] = phases
    return {"starts": starts, "seed": seed, "profiles": normalized_profiles, "horizon": horizon}


class PlannedAsyncEvents:
    """Deterministic virtual events independent of worker completion timing."""

    def __init__(self, names: list[str], timing: dict):
        self.horizon = timing["horizon"]
        self.events = []
        for identity, (name, start) in enumerate(zip(names, timing["starts"])):
            self.events.append((start, 0, identity, "start"))
            instant = start
            for priority, phase in enumerate(PHASES, 1):
                spec = timing["profiles"][name][phase]
                digest = hashlib.sha256(f"{timing['seed']}:{identity}:{name}:{phase}".encode()).digest()
                rng = random.Random(int.from_bytes(digest[:8], "big"))
                instant += _sample_delay(spec, rng)
                if not math.isfinite(instant):
                    raise ValueError(f"non-finite virtual time for assignment {identity}")
                self.events.append((instant, priority, identity, phase))
        self.events.sort()

    def __iter__(self):
        for instant, _, identity, phase in self.events:
            if instant >= self.horizon:
                break
            yield instant, identity, phase


class PlannedRoundParticipation:
    """Selection and acceptance boundary for a round-based workflow."""

    def __init__(self, runtime, round_number: int, available: list[str], expected: int):
        self.runtime = runtime
        self.round_number = round_number
        self.clients = runtime.select_round(round_number, available, expected)
        self.accepted = []

    def record(self, client_name: str) -> None:
        self.accepted.append(client_name)

    def verify(self) -> None:
        self.runtime.validate_round_acceptance(self.round_number, self.accepted)


def validate_simulation_plan(value: dict) -> dict:
    """Return a normalized JSON plan, rejecting unknown or malformed controls."""
    if (
        not isinstance(value, dict)
        or type(value.get("format_version")) is not int
        or value["format_version"] != FORMAT_VERSION
    ):
        raise ValueError(f"simulation plan must be an object with format_version={FORMAT_VERSION}")
    unknown = set(value) - {"format_version", ROUND_PARTICIPATION, ASYNC_EXECUTION}
    if unknown:
        raise ValueError(f"unsupported simulation plan sections: {sorted(unknown)}")
    if ROUND_PARTICIPATION not in value and ASYNC_EXECUTION not in value:
        raise ValueError("simulation plan must contain a supported execution section")
    result = {"format_version": FORMAT_VERSION}
    if ROUND_PARTICIPATION in value:
        rounds = value[ROUND_PARTICIPATION]
        if not isinstance(rounds, dict) or not rounds:
            raise ValueError("round_participation must map round numbers to client lists")
        normalized = {}
        for raw_round, clients in rounds.items():
            try:
                round_number = int(raw_round)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid planned round: {raw_round!r}") from exc
            if str(round_number) != str(raw_round) or round_number < 0:
                raise ValueError(f"invalid planned round: {raw_round!r}")
            if not isinstance(clients, list) or not clients or any(not isinstance(c, str) or not c for c in clients):
                raise ValueError(f"round {round_number} must list nonempty client names")
            if len(clients) != len(set(clients)):
                raise ValueError(f"round {round_number} has duplicate client names")
            normalized[str(round_number)] = list(clients)
        result[ROUND_PARTICIPATION] = normalized
    if ASYNC_EXECUTION in value:
        async_plan = value[ASYNC_EXECUTION]
        if not isinstance(async_plan, dict) or set(async_plan) - {
            "logical_sequence",
            "arrival_sequence",
            "train_seeds",
            "timing",
            "physical_sites",
        }:
            raise ValueError("unsupported async_execution control")
        names = async_plan.get("logical_sequence")
        arrivals = async_plan.get("arrival_sequence")
        if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names):
            raise ValueError("async_execution.logical_sequence must list nonempty client names")
        timing = async_plan.get("timing")
        if (arrivals is None) == (timing is None):
            raise ValueError("async_execution requires exactly one of arrival_sequence or timing")
        if arrivals is not None and (
            not isinstance(arrivals, list)
            or any(type(i) is not int for i in arrivals)
            or sorted(arrivals) != list(range(len(names)))
        ):
            raise ValueError("async_execution.arrival_sequence must permute the assignment IDs")
        seeds = async_plan.get("train_seeds")
        if seeds is not None and (
            not isinstance(seeds, list) or len(seeds) != len(names) or any(type(s) is not int or s < 0 for s in seeds)
        ):
            raise ValueError("async_execution.train_seeds must contain one nonnegative integer per assignment")
        result[ASYNC_EXECUTION] = {"logical_sequence": list(names)}
        if arrivals is not None:
            result[ASYNC_EXECUTION]["arrival_sequence"] = list(arrivals)
        else:
            result[ASYNC_EXECUTION]["timing"] = _validate_timing(timing, names)
        if seeds is not None:
            result[ASYNC_EXECUTION]["train_seeds"] = list(seeds)
        sites = async_plan.get("physical_sites")
        if sites is not None:
            if (
                not isinstance(sites, list)
                or len(sites) != len(names)
                or any(not isinstance(s, str) or not s for s in sites)
            ):
                raise ValueError("async_execution.physical_sites must name one physical site per assignment")
            result[ASYNC_EXECUTION]["physical_sites"] = list(sites)
    return result


def load_simulation_plan(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return validate_simulation_plan(json.load(stream))


class SimulationPlanRuntime:
    """Server-local plan and append-only evidence for cooperating workflows."""

    def __init__(self, plan: dict, trace_dir: str | Path):
        self.plan = validate_simulation_plan(plan)
        self.trace_dir = Path(trace_dir)
        self._consumed = set()
        self._selected_rounds = {}
        self._accepted_rounds = set()
        self._started_at = time.monotonic()
        self.virtual_time = None
        self._finalized = False

    def select_round(self, round_number: int, available: list[str], expected: int | None) -> list[str]:
        rounds = self.plan[ROUND_PARTICIPATION]
        key = str(round_number)
        if key not in rounds:
            raise ValueError(f"no participation plan for round {round_number}")
        selected = rounds[key]
        if expected is not None and len(selected) != expected:
            raise ValueError(f"round {round_number} plans {len(selected)} clients; workflow expects {expected}")
        missing = set(selected) - set(available)
        if missing:
            raise RuntimeError(f"planned clients unavailable in round {round_number}: {sorted(missing)}")
        self._consumed.add(ROUND_PARTICIPATION)
        self._selected_rounds[round_number] = list(selected)
        self._trace("rounds.jsonl", {"round": round_number, "selected": selected, "available": sorted(available)})
        return list(selected)

    def validate_round_acceptance(self, round_number: int, accepted: list[str]) -> None:
        planned = self._selected_rounds.get(round_number)
        if planned is None:
            raise RuntimeError(f"round {round_number} was not selected through the simulation plan")
        self._trace("round_acceptance.jsonl", {"round": round_number, "accepted": accepted})
        if len(accepted) != len(planned) or set(accepted) != set(planned):
            raise RuntimeError(f"round {round_number} accepted {accepted}, expected exactly {planned}")
        self._accepted_rounds.add(round_number)

    def has_section(self, section: str) -> bool:
        return section in self.plan

    def claim(self, section: str) -> dict:
        if section not in self.plan:
            raise ValueError(f"simulation plan has no {section} section")
        self._consumed.add(section)
        return self.plan[section]

    def save_model(self, version: int, model) -> str:
        path = self.trace_dir / "models" / f"{version:08d}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("xb") as stream:
                pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
        return str(path)

    def save_result(self, assignment_id: int, result) -> None:
        path = self.trace_dir / "results" / f"{assignment_id:08d}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)

    def load_result(self, assignment_id: int):
        path = self.trace_dir / "results" / f"{assignment_id:08d}.pkl"
        with path.open("rb") as stream:
            return pickle.load(stream)

    def trace(self, filename: str, record: dict) -> None:
        self._trace(filename, record)

    def assert_consumed(self) -> None:
        unused = set(self.plan) - {"format_version"} - self._consumed
        if unused:
            raise RuntimeError(f"simulation plan was not consumed by this workflow: {sorted(unused)}")
        if ROUND_PARTICIPATION in self.plan:
            planned = {int(key) for key in self.plan[ROUND_PARTICIPATION]}
            if planned != self._accepted_rounds:
                raise RuntimeError(f"planned rounds were not fully accepted: {sorted(planned - self._accepted_rounds)}")

    def finalize(self) -> None:
        if not self._finalized:
            self._trace(
                "run.jsonl", {"wall_seconds": time.monotonic() - self._started_at, "virtual_time": self.virtual_time}
            )
            self._finalized = True

    def _trace(self, filename: str, record: dict) -> None:
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        with (self.trace_dir / filename).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
