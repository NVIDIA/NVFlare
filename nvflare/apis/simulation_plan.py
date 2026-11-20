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

"""Simulation-only execution plans for cooperating workflows.

Plans are run inputs, not part of the exported job. A server-side runtime owns
their consumption and traces so an unsupported section cannot be ignored.
"""

import hashlib
import json
import math
import pickle
import random
import time
from contextlib import contextmanager, nullcontext
from itertools import zip_longest
from pathlib import Path
from tempfile import TemporaryDirectory

SIMULATION_PLAN_FILE = "simulation_plan.json"
SIMULATION_PLAN_PROP = "__simulation_plan_runtime__"
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
    profile_keys = {"starts", "seed", "profiles", "horizon"}
    explicit_keys = {"starts", "delays", "horizon"}
    if isinstance(value, dict) and "events" in value:
        return _validate_ordered_timing(value, len(names))
    if not isinstance(value, dict) or set(value) not in (profile_keys, explicit_keys):
        raise ValueError("timing requires starts and horizon, with either delays or seed and profiles")
    starts = value["starts"]
    if not isinstance(starts, list) or len(starts) != len(names):
        raise ValueError("timing.starts must contain one time per assignment")
    starts = [_nonnegative_number(v, "logical start") for v in starts]
    if starts != sorted(starts):
        raise ValueError("timing.starts must be nondecreasing")
    horizon = _nonnegative_number(value["horizon"], "timing.horizon")
    if horizon <= 0:
        raise ValueError("timing.horizon must be positive")
    if "delays" in value:
        delays = value["delays"]
        if not isinstance(delays, list) or len(delays) != len(names):
            raise ValueError("timing.delays must contain one set of phase delays per assignment")
        normalized_delays = []
        for identity, durations in enumerate(delays):
            if not isinstance(durations, dict) or set(durations) != set(PHASES):
                raise ValueError(f"timing.delays[{identity}] must define download, train, and upload")
            normalized_delays.append(
                {
                    phase: _nonnegative_number(durations[phase], f"assignment {identity} {phase} delay")
                    for phase in PHASES
                }
            )
        return {"starts": starts, "delays": normalized_delays, "horizon": horizon}
    seed = value["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("timing.seed must be a nonnegative integer")
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


def _validate_ordered_timing(value, count):
    if set(value) - {"events", "horizon", "release_on_train", "speculative"} or "horizon" not in value:
        raise ValueError("ordered timing requires events and horizon")
    horizon = _nonnegative_number(value["horizon"], "timing.horizon")
    if horizon <= 0:
        raise ValueError("timing.horizon must be positive")
    options = {key: value[key] for key in ("release_on_train", "speculative") if key in value}
    for key, enabled in options.items():
        if type(enabled) is not bool:
            raise ValueError(f"timing.{key} must be a boolean")
    events = value["events"]
    if not isinstance(events, list) or not events:
        raise ValueError("timing.events must be a nonempty list")
    normalized = []
    next_phase = [0] * count
    phases = ("start", *PHASES)
    last_time = 0.0
    next_assignment = 0
    for index, event in enumerate(events):
        if not isinstance(event, dict) or set(event) != {"virtual_time", "assignment_id", "phase"}:
            raise ValueError(f"timing.events[{index}] requires virtual_time, assignment_id, and phase")
        instant = _nonnegative_number(event["virtual_time"], "event virtual time")
        identity = event["assignment_id"]
        phase = event["phase"]
        if type(identity) is not int or not 0 <= identity < count:
            raise ValueError(f"timing.events[{index}] names an invalid assignment")
        if next_phase[identity] == len(phases) or phase != phases[next_phase[identity]]:
            raise ValueError(f"unexpected phase for assignment {identity}: {phase}")
        if instant < last_time or (instant >= horizon and (index != len(events) - 1 or phase == "start")):
            raise ValueError(
                "ordered events must be nondecreasing, with at most one final phase at or beyond the horizon"
            )
        if phase == "start":
            if identity != next_assignment:
                raise ValueError("ordered starts must reserve sequential assignment IDs")
            next_assignment += 1
        next_phase[identity] += 1
        last_time = instant
        normalized.append({"virtual_time": instant, "assignment_id": identity, "phase": phase})
    if next_assignment != count:
        raise ValueError("timing.events must start every listed assignment")
    return {"horizon": horizon, "events": normalized, **options}


class PlannedAsyncEvents:
    """Deterministic virtual events independent of worker completion timing."""

    def __init__(self, names: list[str], timing: dict):
        self.horizon = timing["horizon"]
        self._ordered = "events" in timing
        if self._ordered:
            self.events = [
                (event["virtual_time"], 0, event["assignment_id"], event["phase"]) for event in timing["events"]
            ]
            return
        self.events = []
        delays = timing.get("delays")
        for identity, (name, start) in enumerate(zip(names, timing["starts"])):
            self.events.append((start, 0, identity, "start"))
            instant = start
            for priority, phase in enumerate(PHASES, 1):
                if delays is None:
                    spec = timing["profiles"][name][phase]
                    digest = hashlib.sha256(f"{timing['seed']}:{identity}:{name}:{phase}".encode()).digest()
                    rng = random.Random(int.from_bytes(digest[:8], "big"))
                    duration = _sample_delay(spec, rng)
                else:
                    duration = delays[identity][phase]
                instant += duration
                if not math.isfinite(instant):
                    raise ValueError(f"non-finite virtual time for assignment {identity}")
                self.events.append((instant, priority, identity, phase))
        self.events.sort()

    def __iter__(self):
        for instant, _, identity, phase in self.events:
            if not self._ordered and instant >= self.horizon:
                break
            yield instant, identity, phase


def validate_simulation_plan(value: dict) -> dict:
    """Return a normalized JSON plan, rejecting unknown or malformed controls."""
    if (
        not isinstance(value, dict)
        or type(value.get("format_version")) is not int
        or value["format_version"] != FORMAT_VERSION
    ):
        raise ValueError(f"simulation plan must be an object with format_version={FORMAT_VERSION}")
    unknown = set(value) - {"format_version", ASYNC_EXECUTION}
    if unknown:
        raise ValueError(f"unsupported simulation plan sections: {sorted(unknown)}")
    if ASYNC_EXECUTION not in value:
        raise ValueError("simulation plan must contain a supported execution section")
    result = {"format_version": FORMAT_VERSION}
    if ASYNC_EXECUTION in value:
        async_plan = value[ASYNC_EXECUTION]
        if not isinstance(async_plan, dict) or set(async_plan) - {
            "logical_sequence",
            "arrival_sequence",
            "train_seeds",
            "timing",
            "physical_sites",
            "base_versions",
            "retain_artifacts",
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
        versions = async_plan.get("base_versions")
        if versions is not None:
            if (
                not isinstance(versions, list)
                or len(versions) != len(names)
                or any(type(v) is not int or v < 0 for v in versions)
            ):
                raise ValueError("async_execution.base_versions must contain one nonnegative integer per assignment")
            result[ASYNC_EXECUTION]["base_versions"] = list(versions)
        if "retain_artifacts" in async_plan:
            retain = async_plan["retain_artifacts"]
            if type(retain) is not bool:
                raise ValueError("async_execution.retain_artifacts must be a boolean")
            result[ASYNC_EXECUTION]["retain_artifacts"] = retain
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


def _plan_digest(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _trace_line(record):
    return json.dumps(record, sort_keys=True) + "\n"


class SimulationPlanRuntime:
    """Server-local plan and append-only evidence for cooperating workflows."""

    def __init__(self, plan: dict, trace_dir: str | Path, *, plan_path: str | Path | None = None):
        self.plan = validate_simulation_plan(plan)
        self._initial_plan_sha256 = _plan_digest(self.plan)
        self._plan_path = Path(plan_path) if plan_path is not None else None
        self.trace_dir = Path(trace_dir)
        self._artifact_dir = self.trace_dir
        self._consumed = set()
        self._started_at = time.monotonic()
        self.virtual_time = None
        self._finalized = False
        self._logical_trace_verified = False

    def has_section(self, section: str) -> bool:
        return section in self.plan

    def claim(self, section: str) -> dict:
        if section not in self.plan:
            raise ValueError(f"simulation plan has no {section} section")
        if section in self._consumed:
            raise RuntimeError(f"simulation plan section {section} was already claimed")
        self._consumed.add(section)
        return self.plan[section]

    @contextmanager
    def artifacts(self, retain: bool):
        if retain:
            yield
            return
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="work-", dir=self.trace_dir) as directory:
            self._artifact_dir = Path(directory)
            try:
                yield
            finally:
                self._artifact_dir = self.trace_dir

    def _model_path(self, version):
        return self._artifact_dir / "models" / f"{version:08d}.pkl"

    def _result_path(self, identity):
        return self._artifact_dir / "results" / f"{identity:08d}.pkl"

    def save_model(self, version: int, model) -> str:
        path = self._model_path(version)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("xb") as stream:
                pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
        return str(path)

    def save_result(self, assignment_id: int, result) -> None:
        path = self._result_path(assignment_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)

    def load_result(self, assignment_id: int):
        path = self._result_path(assignment_id)
        with path.open("rb") as stream:
            return pickle.load(stream)

    def discard_assignment(self, assignment_id: int, unused_base_version: int | None) -> None:
        self._result_path(assignment_id).unlink()
        if unused_base_version is not None:
            self._model_path(unused_base_version).unlink()

    def trace(self, filename: str, record: dict) -> None:
        self._trace(filename, record)

    def verify_preflight(self, records: list[tuple[str, dict]]) -> None:
        """Check completed logical logs against preflight and the saved input plan."""
        self._logical_trace_verified = False
        try:
            if self._plan_path is not None and (
                _plan_digest(load_simulation_plan(self._plan_path)) != self._initial_plan_sha256
            ):
                raise RuntimeError("saved simulation plan differs from the plan used at launch")
            if _plan_digest(self.plan) != self._initial_plan_sha256:
                raise RuntimeError("simulation plan changed during execution")
            for filename in ("assignments.jsonl", "accepted.jsonl", "phases.jsonl"):
                path = self.trace_dir / filename
                expected = (record for trace, record in records if trace == filename)
                # A horizon can legitimately produce no records for a stream.
                with path.open(encoding="utf-8") if path.exists() else nullcontext(()) as stream:
                    for index, (record, line) in enumerate(zip_longest(expected, stream), 1):
                        if record is None or line != _trace_line(record):
                            raise RuntimeError(f"final simulation trace differs from preflight at {filename}:{index}")
        except (OSError, ValueError) as error:
            raise RuntimeError(f"cannot verify final simulation trace: {error}") from error
        self._logical_trace_verified = True

    def assert_consumed(self) -> None:
        unused = set(self.plan) - {"format_version"} - self._consumed
        if unused:
            raise RuntimeError(f"simulation plan was not consumed by this workflow: {sorted(unused)}")

    def finalize(self) -> None:
        if not self._finalized:
            self._trace(
                "run.jsonl",
                {
                    "wall_seconds": time.monotonic() - self._started_at,
                    "virtual_time": self.virtual_time,
                    "plan_sha256": self._initial_plan_sha256,
                    "logical_trace_verified": self._logical_trace_verified,
                },
            )
            self._finalized = True

    def _trace(self, filename: str, record: dict) -> None:
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        with (self.trace_dir / filename).open("a", encoding="utf-8") as stream:
            stream.write(_trace_line(record))
