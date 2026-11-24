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

"""Generate a FedRevive plan with log-normal or trace-sampled phase delays."""

import json
import math
from pathlib import Path

from schedule_standard import plan_parser, sample_paper_delay, write_plan

from nvflare.apis.simulation_plan import PHASES


def lognormal_training_sampler(sigma):
    """Replace exponential training delays with log-normal delays of the same mean."""
    variance = sigma * sigma
    if not math.isfinite(variance) or sigma < 0:
        raise ValueError("log-normal sigma must be nonnegative and have a finite square")

    def sample(assignment_id, logical_name, phase, profile, rng):
        if phase != "train":
            return sample_paper_delay(phase, profile, rng)
        mean = profile["train_mean"]
        return float(rng.lognormal(math.log(mean) - variance / 2, sigma)) if mean else 0.0

    return sample


def trace_delay_sampler(samples):
    """Sample independently with replacement from measured durations for each phase."""
    if not isinstance(samples, dict):
        raise ValueError("delay trace must be an object with download, train, and upload sample lists")
    phase_samples = {}
    for phase in PHASES:
        values = samples.get(phase)
        if not isinstance(values, list) or not values:
            raise ValueError(f"delay trace requires a nonempty {phase!r} sample list")
        error_message = f"delay trace {phase!r} samples must be finite nonnegative numbers"
        if any(type(value) not in (int, float) for value in values):
            raise ValueError(error_message)
        try:
            durations = tuple(float(value) for value in values)
        except OverflowError as error:
            raise ValueError(error_message) from error
        if any(not math.isfinite(duration) or duration < 0 for duration in durations):
            raise ValueError(error_message)
        phase_samples[phase] = durations

    def sample(assignment_id, logical_name, phase, profile, rng):
        durations = phase_samples[phase]
        return durations[rng.integers(len(durations))]

    return sample


def main():
    parser = plan_parser(__doc__)
    delays = parser.add_mutually_exclusive_group()
    delays.add_argument(
        "--train-lognormal-sigma", type=float, default=0.6, help="Log-normal training sigma (default: 0.6)"
    )
    delays.add_argument("--delay-trace", type=Path, help="JSON object with measured samples for each phase")
    args = parser.parse_args()
    try:
        if args.delay_trace is None:
            sampler = lognormal_training_sampler(args.train_lognormal_sigma)
        else:
            if args.output.resolve() == args.delay_trace.resolve():
                raise ValueError("plan output must differ from the input delay trace")
            sampler = trace_delay_sampler(json.loads(args.delay_trace.read_text(encoding="utf-8")))
        write_plan(args, delay_sampler=sampler)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
