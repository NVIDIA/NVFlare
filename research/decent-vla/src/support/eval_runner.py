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

"""LIBERO evaluation with per-suite caching, checkpoint provenance and results.json."""

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

from src.libero import LIBEROBenchmark
from src.support.builders import resolve_device
from src.support.results import (
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

if TYPE_CHECKING:
    from src.config import RunConfig

logger = logging.getLogger(__name__)

# Output/bookkeeping fields do not change rollout success metrics.
_EVAL_CACHE_IGNORED = frozenset({"output_dir", "personalized"})


def _eval_config_hash(cfg: RunConfig) -> str:
    """Hash policy/evaluation settings and the companion training YAML for per-suite caching."""
    ev = {k: v for k, v in dataclasses.asdict(cfg.eval).items() if k not in _EVAL_CACHE_IGNORED}
    # The cache is per-suite, so the suite LIST is not part of any one suite's
    # identity — extending it must resume the finished suites, not re-run them.
    ev["benchmark"] = {k: v for k, v in ev["benchmark"].items() if k != "suites"}
    key: dict[str, Any] = {"backend": dataclasses.asdict(cfg.backend), "eval": ev}
    # The lerobot backend builds the policy and its preprocessing from the YAML at
    # backend.extra.train_config_path, read again at eval time. Key on that file's contents,
    # so an edit at the same path invalidates the cache.
    tp_path = (cfg.backend.extra or {}).get("train_config_path")
    if tp_path:
        tp_file = Path(tp_path)
        key["train_config_sha256"] = hashlib.sha256(tp_file.read_bytes()).hexdigest() if tp_file.is_file() else None
    payload = json.dumps(key, sort_keys=True, separators=(",", ":"), default=str)
    return config_hash(payload)


def _checkpoint_identity(ckpt_path: str | Path | None) -> tuple[int | None, int | None]:
    """Return checkpoint size and modification time without reading its tensors."""
    if ckpt_path is None:
        return None, None
    p = Path(ckpt_path)
    if not p.exists():
        return None, None
    st = p.stat()
    return st.st_size, st.st_mtime_ns


def _horizon_config(policy):
    """Find n_action_steps on a policy or its wrapped LeRobot policy."""
    for candidate in (policy, getattr(policy, "_policy", None)):
        c = getattr(candidate, "config", None)
        if isinstance(getattr(c, "n_action_steps", None), int):
            return c
    return None


def effective_action_horizon(policy):
    """Return the configured action replanning interval, when exposed by the policy."""
    cfg = _horizon_config(policy)
    return cfg.n_action_steps if cfg is not None else None


def apply_action_horizon(policy, n_action_steps):
    """Set the action replanning interval; return (old, new) when changed.

    None preserves the policy default. Policies without this setting reject
    an explicit request; GR00T uses backend.extra.replan_steps instead.
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
    """Load a checkpoint, evaluate each requested suite, and write results.json.

    Reuse cached suites only when checkpoint identity and evaluation settings
    match. A missing checkpoint raises; None evaluates the pretrained model.
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
    benchmark = LIBEROBenchmark.from_config(cfg.eval.benchmark)

    # Resume completed suites only when their checkpoint and settings match.
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
        # name), so it is part of the key.
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

    # Tag every per-task entry seen/held_out. Without a held-out
    # spec, everything is "seen".
    if cfg.data.held_out:
        from src.support.partitioning import load_heldout_json

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
        from src.support.partitioning import PartitionManifest

        partition_manifest_hash = PartitionManifest.read(manifest_path).content_hash()

    total_eps = sum(s["n_episodes"] for s in suites.values())
    total_succ = sum(s["n_success"] for s in suites.values())
    overall_rate = (total_succ / total_eps) if total_eps else 0.0
    wall = time.time() - t0
    # Record the requested and effective horizon with the evaluated checkpoint.
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
