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

"""Generate a FedRevive paper plan from prepared profiles before launching SimEnv."""

import argparse
import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nvflare.apis.simulation_plan import ASYNC_EXECUTION, FORMAT_VERSION, PHASES, validate_simulation_plan
from nvflare.collab import FedBuffScheduleState


@dataclass
class Phase:
    finish_time: float
    phase: str
    assignment_id: int

    def __lt__(self, other):
        # Match the reference heap: ties compare times only, not assignment IDs.
        return self.finish_time < other.finish_time


def sample_paper_delay(phase, profile, rng):
    """Sample one of the reference paper's three phase delays."""
    if phase == "download":
        return profile["download_mean"]
    if phase == "train":
        return float(rng.exponential(profile["train_mean"]))
    if phase == "upload":
        mean = profile["upload_mean"]
        width = profile.get("upload_half_width", 0.02)
        return float(rng.uniform(max(0.0, mean - width), mean + width))
    raise ValueError(f"unknown paper phase: {phase}")


def make_simulation_plan(
    profiles,
    run_seed,
    max_time,
    *,
    num_rounds,
    concurrency,
    min_open_slots,
    buffer_size,
    delay_sampler=None,
):
    """Generate an ordered SimEnv plan from logical clients and phase delays.

    Preserve the reference scheduling in Askin et al. (2025), "Reviving Stale
    Updates: Data-Free Knowledge Distillation for Asynchronous Federated
    Learning" (https://arxiv.org/abs/2511.00655). Logical clients become eligible
    after training, while their uploads continue to occupy slots.

    A custom ``delay_sampler(assignment_id, logical_name, phase, profile, rng)``
    returns a finite nonnegative duration. Its NumPy Generator is derived from
    the run seed, assignment ID and phase, independent of client selection and
    other samples. Omitting the sampler preserves the paper's shared RNG stream.
    """
    state = FedBuffScheduleState(num_rounds, concurrency, buffer_size, min_open_slots)
    if not max_time > 0:
        raise ValueError("max_time must be positive")
    if delay_sampler is not None and not callable(delay_sampler):
        raise TypeError("delay_sampler must be callable")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("runtime_profiles must map at least one logical client to its delay profile")
    rng = np.random.RandomState(run_seed)
    logical = sorted(profiles, key=_client_index)
    delays = {}
    for name, profile in profiles.items():
        try:
            durations = {key: float(profile[key]) for key in ("download_mean", "train_mean", "upload_mean")}
            durations["upload_half_width"] = float(profile.get("upload_half_width", 0.02))
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"invalid paper delay profile for {name}: {error}") from error
        if not all(math.isfinite(value) and value >= 0 for value in durations.values()):
            raise ValueError(f"paper schedule requires finite nonnegative delays for {name}")
        delays[name] = durations

    def sample(identity, name, phase):
        try:
            if delay_sampler is None:
                duration = sample_paper_delay(phase, delays[name], rng)
            else:
                phase_rng = np.random.default_rng([run_seed, identity, PHASES.index(phase)])
                duration = float(delay_sampler(identity, name, phase, delays[name].copy(), phase_rng))
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"cannot sample {phase} delay for assignment {identity} ({name}): {error}") from error
        if not math.isfinite(duration) or duration < 0:
            raise ValueError(f"{phase} delay for assignment {identity} ({name}) must be finite and nonnegative")
        return duration

    schedule = []
    sequence = []
    base_versions = []
    events = []
    released = set()
    instant = 0.0
    refill = True
    while not state.finished:
        if refill:
            refill = False
            count = min(state.get_available_slots(speculative=True), len(logical))
            if count:
                indices = rng.choice(len(logical), size=count, replace=False).tolist()
                selected = [logical[index] for index in indices]
                for index in sorted(indices, reverse=True):
                    del logical[index]
                for assignment in state.reserve(selected, speculative=True, released_assignments=released):
                    identity = assignment.assignment_id
                    name = assignment.logical_name
                    sequence.append(name)
                    base_versions.append(assignment.base_version)
                    schedule.append(_record(instant, identity, "start"))
                    heapq.heappush(events, Phase(instant + sample(identity, name, "download"), "download", identity))

        if not events:
            raise ValueError("paper schedule cannot make progress: no phase events remain")
        event = heapq.heappop(events)
        instant = event.finish_time
        if not math.isfinite(instant):
            raise ValueError(f"non-finite virtual time for assignment {event.assignment_id}")
        identity = event.assignment_id
        schedule.append(_record(instant, identity, event.phase))
        if instant >= max_time:
            break
        assignment = state.active[identity]
        name = assignment.logical_name
        if event.phase == "download":
            duration = sample(identity, name, "train")
            heapq.heappush(events, Phase(instant + duration, "train", identity))
        elif event.phase == "train":
            released.add(identity)
            logical.append(name)
            logical.sort(key=_client_index)
            duration = sample(identity, name, "upload")
            heapq.heappush(events, Phase(instant + duration, "upload", identity))
        else:
            state.accept(assignment, None)
            released.discard(identity)
            refill = True
    # Round-budget-only runs still serialize a finite horizon beyond all events.
    horizon = math.nextafter(instant, math.inf) if max_time == math.inf else max_time
    return {
        "format_version": FORMAT_VERSION,
        ASYNC_EXECUTION: {
            "logical_sequence": sequence,
            "base_versions": base_versions,
            "train_seeds": [run_seed * 1_000_000 + identity + 1 for identity in range(len(sequence))],
            "timing": {
                "events": schedule,
                "horizon": horizon,
                "release_on_train": True,
                "speculative": True,
            },
            "retain_artifacts": False,
        },
    }


def _record(instant, identity, phase):
    return {"virtual_time": instant, "assignment_id": identity, "phase": phase}


def _client_index(name):
    if isinstance(name, str):
        prefix, _, index = name.partition("-")
        if prefix == "client" and index.isdecimal():
            return int(index)
    raise ValueError(f"expected a logical client name of the form client-N, got {name!r}")


def plan_parser(description):
    """Build the command-line inputs shared by the offline planners."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("manifest", type=Path, help="Prepared FedRevive manifest.json")
    parser.add_argument("output", type=Path, help="Destination for the SimEnv plan JSON")
    parser.add_argument("--run-seed", type=int, default=10)
    parser.add_argument("--max-time", type=float, default=200.0)
    parser.add_argument("--num-rounds", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--buffer-size", type=int, default=1)
    parser.add_argument("--min-open-slots", type=int, default=1)
    return parser


def write_plan(args, *, delay_sampler=None):
    """Generate and validate a plan from parsed CLI arguments before writing it."""
    if args.output.resolve() == args.manifest.resolve():
        raise ValueError("plan output must differ from the input manifest")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    plan = make_simulation_plan(
        manifest["runtime_profiles"],
        args.run_seed,
        args.max_time,
        num_rounds=args.num_rounds,
        concurrency=args.concurrency,
        min_open_slots=args.min_open_slots,
        buffer_size=args.buffer_size,
        delay_sampler=delay_sampler,
    )
    args.output.write_text(json.dumps(validate_simulation_plan(plan), indent=2) + "\n", encoding="utf-8")


def main(*, delay_sampler=None):
    """Generate plan JSON from a manifest with an optional custom sampler."""
    parser = plan_parser(__doc__)
    args = parser.parse_args()
    try:
        write_plan(args, delay_sampler=delay_sampler)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
