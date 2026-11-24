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

"""Plot the assignment order or virtual-time phases of a SimEnv plan."""

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

from nvflare.apis.simulation_plan import (
    ASYNC_EXECUTION,
    PHASES,
    PlannedAsyncEvents,
    load_simulation_plan,
    validate_simulation_plan,
)

_PHASE_COLORS = dict(zip(PHASES, ("#4477AA", "#228833", "#CC6677")))


def plot_simulation_plan(
    plan: dict | str | Path,
    *,
    first_assignment: int = 0,
    limit: int = 30,
    base_mib: float | None = None,
    upload_mib: float | None = None,
):
    """Return a Matplotlib figure for consecutive reservation IDs in a plan.

    Timed plans use the simulator's virtual event calculator. Fixed plans
    show acceptance rank. The plot includes base versions when the plan
    supplies them. Timed plans with explicit base versions and temporary
    storage also show a peak file bound for the full plan. Supply both
    serialized payload sizes in MiB to estimate a size bound.
    """
    normalized = validate_simulation_plan(plan) if isinstance(plan, dict) else load_simulation_plan(plan)
    execution = normalized[ASYNC_EXECUTION]
    names = execution["logical_sequence"]
    versions = execution.get("base_versions")
    has_storage_bound = "timing" in execution and versions is not None and not execution.get("retain_artifacts", True)
    if base_mib is not None or upload_mib is not None:
        if not all(size is not None and math.isfinite(size) and size >= 0 for size in (base_mib, upload_mib)):
            raise ValueError("base_mib and upload_mib must both be finite nonnegative sizes")
        if not has_storage_bound:
            raise ValueError(
                "a storage size estimate requires a timed plan with base_versions and retain_artifacts=false"
            )
    if not 0 <= first_assignment < len(names):
        raise ValueError(f"first_assignment must be between 0 and {len(names) - 1}")
    if limit < 1:
        raise ValueError("limit must be positive")
    identities = range(first_assignment, min(first_assignment + limit, len(names)))

    from matplotlib.figure import Figure

    figure = Figure(figsize=(9, max(3, 1.8 + 0.32 * len(identities))), layout="constrained")
    axes = figure.subplots()
    if "timing" in execution:
        timing = execution["timing"]
        events = [event for event in PlannedAsyncEvents(names, timing) if event[0] < timing["horizon"]]
        _plot_timing(axes, timing, events, identities)
        if has_storage_bound:
            figure.suptitle(_storage_summary(events, versions, base_mib, upload_mib), x=0.02, ha="left", fontsize=10)
    else:
        positions = {identity: rank for rank, identity in enumerate(execution["arrival_sequence"], start=1)}
        axes.scatter([positions[identity] for identity in identities], list(identities), color="#4477AA", s=42)
        axes.set_title("Planned acceptance order", loc="left")
        axes.set_xlabel("Acceptance position (1-based)")
        axes.xaxis.get_major_locator().set_params(integer=True)

    names_by_row = [
        f"#{identity}  {names[identity]}" + (f"  (base v{versions[identity]})" if versions is not None else "")
        for identity in identities
    ]
    axes.set_yticks(list(identities), names_by_row, parse_math=False, usetex=False)
    axes.set_ylim(identities.stop - 0.5, identities.start - 0.5)
    axes.set_axisbelow(True)
    axes.grid(axis="x", color="#D8DEE4", linewidth=0.7)
    axes.spines[["top", "right"]].set_visible(False)
    return figure


def _storage_summary(events, versions, base_mib, upload_mib):
    accepted = {identity for _, identity, phase in events if phase == "upload"}
    base_users = Counter(versions[identity] for _, identity, phase in events if phase == "start")
    bases = set()
    uploads = peak_files = peak_bases = peak_uploads = 0
    peak_mib = 0.0
    for _, identity, phase in events:
        version = versions[identity]
        if phase == "start":
            bases.add(version)
            # A physical upload may arrive at any point before logical acceptance.
            uploads += identity in accepted
            peak_files = max(peak_files, len(bases) + uploads)
            peak_bases = max(peak_bases, len(bases))
            peak_uploads = max(peak_uploads, uploads)
            if base_mib is not None:
                peak_mib = max(peak_mib, len(bases) * base_mib + uploads * upload_mib)
        elif phase == "upload":
            uploads -= 1
            base_users[version] -= 1
            if base_users[version] == 0:
                bases.remove(version)
    summary = f"Peak payload bound  ≤ {peak_files} files  ·  Bases ≤ {peak_bases}  ·  Uploads ≤ {peak_uploads}"
    return summary if base_mib is None else f"{summary}  ·  ~{peak_mib:.2f} MiB"


def _plot_timing(axes, timing, events, identities):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    horizon = timing["horizon"]
    events_by_id = defaultdict(list)
    for instant, identity, phase in events:
        if identity in identities:
            events_by_id[identity].append((instant, phase))

    # An ordered plan can end at the publication budget before its horizon.
    end = min(horizon, timing["events"][-1]["virtual_time"]) if "events" in timing else horizon
    latest = 0.0
    has_open_phase = False
    for identity, events in events_by_id.items():
        start, _ = events[0]
        axes.scatter(start, identity, color="#252C36", s=14, zorder=3, clip_on=False)
        previous = start
        for instant, phase in events[1:]:
            if instant > previous:
                axes.barh(identity, instant - previous, left=previous, height=0.5, color=_PHASE_COLORS[phase])
            else:
                offset = (PHASES.index(phase) - 1) * 0.16
                axes.scatter(instant, identity + offset, marker="|", color=_PHASE_COLORS[phase], s=32, zorder=4)
            previous = instant
        latest = max(latest, previous)
        if events[-1][1] != "upload" and previous < end:
            next_phase = PHASES[len(events) - 1]
            axes.barh(identity, end - previous, left=previous, height=0.5, color=_PHASE_COLORS[next_phase], alpha=0.25)
            latest = max(latest, end)
            has_open_phase = True

    legend = [Patch(color=color, label=phase.title()) for phase, color in _PHASE_COLORS.items()]
    if has_open_phase:
        legend.append(Patch(color="#252C36", alpha=0.25, label="Open phase"))
    if latest >= horizon or not events_by_id:
        axes.axvline(horizon, color="#404852", linestyle="--", linewidth=1)
        legend.append(Line2D([], [], color="#404852", linestyle="--", label="Horizon"))
    if not events_by_id:
        axes.set_xlim(0, horizon)
        axes.text(0.5, 0.5, "No selected assignments start before the horizon", transform=axes.transAxes, ha="center")
    elif axes.get_xlim()[0] < 0:
        axes.set_xlim(left=0)
    axes.legend(handles=legend, loc="lower left", bbox_to_anchor=(0, 1), ncols=len(legend), frameon=False)
    axes.set_title("Planned assignment timeline", loc="left", pad=28)
    axes.set_xlabel("Virtual time (plan units)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="Input SimEnv plan JSON")
    parser.add_argument("output", type=Path, help="Chart destination, such as plan.png or plan.pdf")
    parser.add_argument("--first-assignment", type=int, default=0, help="First zero-based reservation ID to plot")
    parser.add_argument("--limit", type=int, default=30, help="Maximum assignments to plot")
    parser.add_argument("--base-mib", type=float, help="Estimated serialized size of one base model in MiB")
    parser.add_argument("--upload-mib", type=float, help="Estimated serialized size of one accepted upload in MiB")
    args = parser.parse_args()
    try:
        if args.plan.resolve() == args.output.resolve():
            raise ValueError("output must differ from the input plan")
        figure = plot_simulation_plan(
            args.plan,
            first_assignment=args.first_assignment,
            limit=args.limit,
            base_mib=args.base_mib,
            upload_mib=args.upload_mib,
        )
        figure.savefig(args.output, dpi=150)
    except ImportError as error:
        if error.name != "matplotlib":
            raise
        parser.error("install matplotlib to plot a plan")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
