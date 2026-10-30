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

"""Benchmark eval driver: per-suite loop, provenance-keyed caching, results.json.

Extracted from `cli/main.py` so the eval path has a single home before the
LIBERO benchmark lands. Backend/benchmark-agnostic and FL-framework-free; depends
only on `core/` (plus the runtime `builders` module).

Caching contract (spec §7/§10): each suite's result lands in
`eval_dir/suites/{suite}.json` as soon as it finishes, stamped with a
`provenance` key that identifies the *evaluated model*: the checkpoint's
path plus its `stat()` size and `mtime_ns`, the eval seed and
`n_episodes_per_task`, and `eval_config_hash` — a hash of the config that
decides how the checkpoint is turned into rollouts (`backend` + `eval`,
minus the suite list and bookkeeping fields). A re-run skips a suite ONLY
when its stored provenance matches the current invocation exactly.

The stat fields are what make the path safe to key on. Eval checkpoints are
named `round_{num_rounds:04d}.pt`, so retraining into the same `output_dir`
overwrites the same filename with different weights; keying on the path
alone silently reused the previous policy's numbers. Identity is stat-based
rather than a content hash so multi-GB checkpoints are not re-read on every
eval — an in-place rewrite always moves `mtime_ns`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from decent_vla.core.results import (
    SCHEMA_VERSION,
    DeviceInfo,
    RunResults,
    compute_aggregates,
    config_hash,
    git_commit,
    now_iso,
    tag_suites,
    write_results,
)
from decent_vla.runtime.builders import build_benchmark, resolve_device

if TYPE_CHECKING:
    from decent_vla.configs.schemas import RunConfig

logger = logging.getLogger(__name__)

# cfg.eval fields that steer bookkeeping, not the numbers — where the results
# land, whether training also ran, how often eval fires. Excluded so they never
# throw away finished rollouts.
_EVAL_CACHE_IGNORED = frozenset(
    {"output_dir", "only", "after_training", "every_n_rounds", "save_episodes", "personalized"}
)


def _eval_config_hash(cfg: RunConfig) -> str:
    """Hash of everything *except the weights* that decides a suite's numbers.

    Scope is deliberate. The full run `config_hash` would also cover `fl.*`,
    `data.*`, `name`, `wandb` — fields that either cannot move an eval number
    (names, output paths, tracking tags) or can only move it *through* the
    checkpoint, which the size/mtime_ns identity already catches. Keying on
    them would discard hours of LIBERO rollouts every time someone edits a
    wandb tag or a SLURM-facing field, so the key covers only the two things
    the checkpoint bytes don't: how the checkpoint becomes a policy
    (`backend`) and how that policy is rolled out (`eval`).

    Deny-list rather than allow-list inside those two blocks: a field added to
    EvalConfig/BackendConfig later invalidates by default. Over-invalidating
    costs GPU hours and says so in the log; under-invalidating prints a
    plausible number that belongs to a different model.
    """
    ev = {k: v for k, v in dataclasses.asdict(cfg.eval).items() if k not in _EVAL_CACHE_IGNORED}
    # The cache is per-suite, so the suite LIST is not part of any one suite's
    # identity — extending it must resume the finished suites, not re-run them.
    ev["benchmark"] = {k: v for k, v in ev["benchmark"].items() if k != "suites"}
    key: dict[str, Any] = {"backend": dataclasses.asdict(cfg.backend), "eval": ev}
    # The lerobot backend builds the policy and its preprocessing from the YAML at
    # backend.extra.train_config_path, read again at eval time. Key on that file's contents,
    # so an edit at the same path invalidates the cache. Added only when the path is set, so
    # the keys of configs without one are unchanged.
    tp_path = (cfg.backend.extra or {}).get("train_config_path")
    if tp_path:
        tp_file = Path(tp_path)
        key["train_config_sha256"] = hashlib.sha256(tp_file.read_bytes()).hexdigest() if tp_file.is_file() else None
    payload = json.dumps(key, sort_keys=True, separators=(",", ":"), default=str)
    return config_hash(payload)


def _checkpoint_identity(ckpt_path: str | Path | None) -> tuple[int | None, int | None]:
    """`(size, mtime_ns)` of the checkpoint, or `(None, None)` if there isn't one.

    One `stat()` — deliberately not a content hash, which would mean re-reading
    a multi-GB checkpoint on every eval to learn what a rewrite already told us.
    """
    if ckpt_path is None:
        return None, None
    p = Path(ckpt_path)
    if not p.exists():
        return None, None
    st = p.stat()
    return st.st_size, st.st_mtime_ns


def _horizon_config(policy):
    """The config object carrying an integer `n_action_steps`, or None.

    Benchmarks drive the ADAPTER's wrapper (LeRobotPolicy), not the lerobot
    policy itself, and the wrapper exposes no `.config` — so looking only at
    `policy.config` found nothing and raised on every real eval while every
    unit test, which passed a bare policy, kept passing. Try the wrapper's
    own config first, then the policy it wraps.
    """
    for candidate in (policy, getattr(policy, "_policy", None)):
        c = getattr(candidate, "config", None)
        if isinstance(getattr(c, "n_action_steps", None), int):
            return c
    return None


def effective_action_horizon(policy):
    """The action horizon `policy` will ACTUALLY roll out at, or None.

    Recorded into results.json provenance alongside the *requested*
    `cfg.eval.n_action_steps`. The two have silently diverged twice (the
    horizon broke at both config hops; eval dirs named `_nact10` ran
    open-loop at 50), so the artifact records what the policy reports AFTER
    `apply_action_horizon`, not what the config asked for.
    """
    cfg = _horizon_config(policy)
    return cfg.n_action_steps if cfg is not None else None


def apply_action_horizon(policy, n_action_steps):
    """Set the rollout action horizon on `policy`; return (old, new) or None.

    The horizon is the REPLANNING interval, not a cosmetic setting. A chunked
    VLA predicts `chunk_size` actions in one forward pass and pops them one per
    environment step, refilling only when the queue empties — so
    `config.n_action_steps` is how often the policy actually looks at the world.
    SmolVLA and pi0.5 both default it to 50, the entire chunk, i.e. fully open
    loop; `lerobot-eval` passes 10. Measured on one identical checkpoint over
    400 episodes: 70.0% at 10 versus 37.0% at 50, and the gap was not uniform
    across suites (-19 to -47 points). An eval that does not pin this is not
    comparable to one that does.

    `None` is a no-op, so existing configs keep whatever their policy declares.
    A policy with no `config.n_action_steps` raises rather than ignoring the
    request: silently running at some other horizon is the bug this exists to
    prevent.
    """
    if n_action_steps is None:
        return None
    cfg = _horizon_config(policy)
    if cfg is None:
        raise ValueError(
            f"eval.n_action_steps={n_action_steps} was requested but "
            f"{type(policy).__name__} exposes no integer config.n_action_steps; "
            "refusing to run at an unknown action horizon. n_action_steps is a "
            "lerobot knob — GR00T's equivalent is backend.extra.replan_steps "
            "(default: its full 16-step chunk horizon), which is what every "
            "existing GR00T result was measured at; leave eval.n_action_steps "
            "unset for GR00T."
        )
    old = cfg.n_action_steps
    if old == n_action_steps:
        return None
    cfg.n_action_steps = n_action_steps
    logger.info("eval action horizon: n_action_steps %d -> %d", old, n_action_steps)
    return (old, n_action_steps)


def run_eval(
    cfg: RunConfig,
    backend: Any,
    ckpt_path: str | Path | None,
    output_dir: Path,
    started_at: str,
    t0: float,
    cfg_dict: dict[str, Any],
    config_path: str | None = None,
) -> Path:
    """Evaluate `backend` on `cfg.eval`, after loading `ckpt_path` into it.

    `ckpt_path=None` evaluates the backend as built (the base model). A
    `ckpt_path` that does not exist raises FileNotFoundError before anything
    is evaluated or written: skipping the load would evaluate the base model
    under the checkpoint's name.

    Writes per-suite JSONs under `eval_dir/suites/` and the aggregated
    `results.json`; returns the results.json path. `config_path` (the YAML
    the run was launched from) lands in the results provenance block; falls
    back to `$DECENT_VLA_CONFIG`, which the CLI exports on every run.
    """
    device = resolve_device(cfg.device)
    if ckpt_path is not None:
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(
                f"eval checkpoint {ckpt_path} does not exist. Refusing to evaluate "
                "the base model under its name; fix the path, or wait until "
                "training has written it. To evaluate the base model, omit --ckpt "
                "and leave no checkpoint under the run's checkpoints/ directory."
            )
        backend.load_checkpoint(ckpt_path)
    policy = backend.build_policy()
    apply_action_horizon(policy, cfg.eval.n_action_steps)
    benchmark = build_benchmark(cfg.eval)

    # Idempotent per-suite eval (spec §7/§10): a crashed eval re-runs only the
    # missing suites. Caching is provenance-keyed, and the key has to identify
    # the evaluated MODEL, not the file it was read from: eval checkpoints are
    # named round_{num_rounds:04d}.pt, so a retrain into the same output_dir
    # lands different weights at the same path. Two halves cover that:
    #   * checkpoint path + size + mtime_ns — different weights, one stat() call
    #     (see _checkpoint_identity on why this is not a content hash);
    #   * eval_config_hash — same weights, different backend/eval config
    #     (dtype, benchmark, benchmark.extra, …), see _eval_config_hash for why
    #     it is a narrowed slice and not the run's whole config_hash.
    # Anything outside those two either cannot change the numbers or changes
    # them only via the checkpoint, which the stat identity already catches.
    eval_dir = Path(cfg.eval.output_dir or (output_dir / "eval"))
    suites_dir = eval_dir / "suites"
    suites_dir.mkdir(parents=True, exist_ok=True)
    ckpt_size, ckpt_mtime_ns = _checkpoint_identity(ckpt_path)
    provenance = {
        "checkpoint": str(ckpt_path) if ckpt_path is not None else None,
        "checkpoint_size": ckpt_size,
        "checkpoint_mtime_ns": ckpt_mtime_ns,
        "seed": cfg.eval.seed,
        "n_episodes_per_task": cfg.eval.n_episodes_per_task,
        "eval_config_hash": _eval_config_hash(cfg),
    }
    suites: dict[str, dict] = {}
    # Results, cache files and held-out tagging all use the canonical suite name, so a
    # config alias (LIBERO's `spatial`) matches the held-out spec's `libero_spatial`.
    canonical = getattr(benchmark, "canonical_suite", lambda s: s)
    canon_names = [canonical(s) for s in cfg.eval.benchmark.suites]
    dupes = sorted({n for n in canon_names if canon_names.count(n) > 1})
    if dupes:
        raise ValueError(
            f"eval.benchmark.suites names the same suite twice ({dupes}); list each suite once, "
            "by either its alias or its full name."
        )
    for config_suite in cfg.eval.benchmark.suites:
        suite = canonical(config_suite)
        suite_path = suites_dir / f"{suite}.json"
        # An alias selects tasks by its own name (LIBERO's task_filter is keyed by the config
        # name), so it is part of the key; keys for canonical names are unchanged.
        suite_prov = provenance if config_suite == suite else {**provenance, "config_suite": config_suite}
        if suite_path.exists():
            with suite_path.open() as f:
                cached = json.load(f)
            # The provenance key is cache metadata, not part of the suite
            # result — strip it before assembling results.json.
            if cached.pop("provenance", None) == suite_prov:
                logger.info("skipping %s: already evaluated", suite)
                suites[suite] = cached
                continue
            logger.info("re-evaluating %s: provenance changed", suite)
        bench_result = benchmark.evaluate(
            policy,
            suites=[config_suite],  # the benchmark resolves its own aliases
            n_episodes_per_task=cfg.eval.n_episodes_per_task,
            seed=cfg.eval.seed,
        )
        # One suite was requested; take its only result, whatever name the benchmark keys it by.
        (s,) = bench_result.suites.values()
        suite_dict = {
            "n_episodes": s.n_episodes,
            "n_success": s.n_success,
            "success_rate": s.success_rate,
            "per_task": [dataclasses.asdict(t) for t in s.per_task],
        }
        # Atomic write: a crash mid-write must not leave a torn JSON that a
        # later resume would skip as "already evaluated".
        tmp = suite_path.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump({**suite_dict, "provenance": suite_prov}, f, indent=2)
        tmp.rename(suite_path)
        suites[suite] = suite_dict

    # Tag every per-task entry seen/held_out (spec §8). Without a held-out
    # spec, everything is "seen".
    if cfg.data.held_out:
        from decent_vla.core.partitioner import load_heldout_json

        spec = load_heldout_json(cfg.data.held_out)
        heldout_eval_ids = {s: spec.eval_ids_for_suite(s) for s in {t.suite for t in spec.tasks}}
    else:
        heldout_eval_ids = {}
    suites = tag_suites(suites, heldout_eval_ids)
    aggregates = compute_aggregates(suites)

    # Partition provenance: hash of the manifest this run trained against.
    manifest_path = output_dir / "partition_manifest.json"
    partition_manifest_hash = None
    if manifest_path.exists():
        from decent_vla.core.partitioner import PartitionManifest

        partition_manifest_hash = PartitionManifest.read(manifest_path).content_hash()

    total_eps = sum(s["n_episodes"] for s in suites.values())
    total_succ = sum(s["n_success"] for s in suites.values())
    overall_rate = (total_succ / total_eps) if total_eps else 0.0
    wall = time.time() - t0
    # v1.2 provenance block (2026-08-14 audit): the artifact itself must
    # answer "what exactly was evaluated" — checkpoint identity, the horizon
    # the policy actually ran at, and the config — without the job log.
    # Distinct from the per-suite cache key above, which must NOT grow new
    # fields (that would invalidate every finished suite on the clusters).
    results_provenance = {
        "checkpoint": str(ckpt_path) if ckpt_path is not None else None,
        "checkpoint_size": ckpt_size,
        "checkpoint_mtime_ns": ckpt_mtime_ns,
        "n_action_steps_requested": cfg.eval.n_action_steps,
        "n_action_steps_effective": effective_action_horizon(policy),
        "eval_config_hash": provenance["eval_config_hash"],
        "config_path": config_path or os.environ.get("DECENT_VLA_CONFIG"),
    }
    results = RunResults(
        schema_version=SCHEMA_VERSION,
        run_id=cfg.name,
        backend=f"{backend.name}:{cfg.backend.name}",
        checkpoint=str(ckpt_path) if ckpt_path else None,
        benchmark=benchmark.name,
        config_hash=config_hash(cfg_dict),
        git_commit=git_commit(),
        started_at=started_at,
        finished_at=now_iso(),
        wall_time_s=wall,
        device=DeviceInfo(
            type=device.type,
            name=(torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"),
            count=1,
        ),
        n_episodes_per_task=cfg.eval.n_episodes_per_task,
        seed=cfg.eval.seed,
        suites=suites,
        overall={
            "n_episodes": total_eps,
            "n_success": total_succ,
            "success_rate": overall_rate,
        },
        aggregates=aggregates,
        partition_manifest_hash=partition_manifest_hash,
        provenance=results_provenance,
    )
    results_path = eval_dir / "results.json"
    write_results(results_path, results)
    print(f"Eval complete: {results_path}")
    print(f"Overall success rate: {overall_rate:.3f}")
    return results_path
