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

"""Personalized eval for partial-federation (personalization) arms.

For a run trained with ``backend.shared_params`` (axis 2), the deployed
per-client model is the FINAL server shared subset overlaid on that client's
own persisted full state (its personal params, from
``client_states/<cid>_round_NNNN.pt``). Each client is evaluated on its OWN
tasks (from ``partition_manifest.json``); pooling the per-client results gives
the "personalized" success rate. The generic eval (server shared + pretrained
rest, all tasks) stays in ``eval_runner.run_eval``.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import logging
import re
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def assemble_personalized_state(
    backend: Any,
    client_weights: dict[str, torch.Tensor],
    server_shared: dict[str, torch.Tensor],
) -> None:
    """Load a client's deployed model: its full local state, then the final
    server aggregate over the shared subset.

    Fails loudly if the server blob carries keys outside the backend's shared
    scope — that means the eval config's ``shared_params`` don't match the
    training run, and proceeding would overwrite personal params.
    """
    shared_keys = set(backend.get_state("shared"))
    spill = set(server_shared) - shared_keys
    if spill:
        raise ValueError(
            f"server checkpoint has {len(spill)} key(s) outside the backend's "
            f"shared scope (e.g. {sorted(spill)[:3]}); eval config's "
            "backend.shared_params likely does not match the training run."
        )
    backend.set_state(client_weights, scope="all")
    backend.set_state(server_shared, scope="shared")


_WS = re.compile(r"[\s_]+")


def _norm(name: str) -> str:
    return _WS.sub(" ", name.strip().lower())


def match_task_names_to_ids(names: list[str], suite_tasks: dict[int, str]) -> list[int]:
    """Map partition-manifest task names onto benchmark task ids.

    Dataset task strings are often underscored ("Pick_up_the_black_bowl…")
    while benchmark task language is spaced — match on a normalized form.
    Unknown names raise: silently skipping a task would shrink the client's
    eval set and inflate nothing-visible.
    """
    by_norm = {_norm(lang): tid for tid, lang in suite_tasks.items()}
    out: list[int] = []
    for n in names:
        tid = by_norm.get(_norm(n))
        if tid is None:
            raise ValueError(f"task name {n!r} not found among benchmark tasks " f"{sorted(suite_tasks.values())[:3]}…")
        out.append(tid)
    return out


def pool_client_results(per_client: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pool per-client episode counts into the personalized aggregate."""
    n_eps = sum(c["n_episodes"] for c in per_client.values())
    n_succ = sum(c["n_success"] for c in per_client.values())
    return {
        "n_episodes": n_eps,
        "n_success": n_succ,
        "success_rate": (n_succ / n_eps) if n_eps else 0.0,
    }


def _client_task_filter(names: list[str], name_maps: dict[str, dict[int, str]]) -> dict[str, list[int]]:
    """Locate each manifest task name in exactly one suite → {suite: [ids]}."""
    out: dict[str, list[int]] = {}
    for n in names:
        hits = [
            (suite, tid) for suite, tasks in name_maps.items() for tid, lang in tasks.items() if _norm(lang) == _norm(n)
        ]
        if not hits:
            raise ValueError(
                f"manifest task {n!r} not found in any eval suite "
                f"({sorted(name_maps)}); dataset/benchmark task naming mismatch."
            )
        if len(hits) > 1:
            raise ValueError(f"manifest task {n!r} is ambiguous across suites: {hits}")
        suite, tid = hits[0]
        out.setdefault(suite, []).append(tid)
    return {s: sorted(ids) for s, ids in out.items()}


def run_personalized_eval(
    cfg: Any,
    backend_factory: Any,
    run_dir: str | Path,
    eval_dir: str | Path,
    ckpt_path: str | Path | None = None,
) -> Path:
    """Evaluate every client's personalized model on its OWN tasks.

    Reads the training run's artifacts from ``run_dir`` (partition manifest,
    server checkpoint, per-client states) and writes per-client JSONs
    plus the pooled ``personalized_results.json`` under ``eval_dir``.
    The server checkpoint is ``ckpt_path`` when given (a ``server_round_*.pt``
    blob), else ``server_round_final.pt``, else the latest periodic one; each
    client's state is its newest one saved at or before that round.
    Per-client results are provenance-cached like the generic eval: a crashed
    eval re-runs only the missing clients.
    """
    from decent_vla.core.partitioner import PartitionManifest
    from decent_vla.runtime.builders import build_benchmark
    from decent_vla.runtime.checkpoints import latest_server_checkpoint
    from decent_vla.runtime.client_checkpoint import client_checkpoints
    from decent_vla.runtime.eval_runner import _checkpoint_identity, _eval_config_hash, apply_action_horizon

    run_dir = Path(run_dir)
    eval_dir = Path(eval_dir)
    manifest = PartitionManifest.read(run_dir / "partition_manifest.json")
    ckpt_dir = run_dir / "checkpoints"
    server_blob = None
    if ckpt_path is not None:
        server_ckpt = Path(ckpt_path)
        if not server_ckpt.exists():
            raise FileNotFoundError(f"eval checkpoint {server_ckpt} does not exist")
    else:
        # The final blob, unless a later periodic checkpoint exists (a resumed run that
        # stopped before finishing leaves the previous run's final blob behind).
        server_ckpt = latest_server_checkpoint(ckpt_dir)
        final = ckpt_dir / "server_round_final.pt"
        if final.exists():
            final_blob = torch.load(final, map_location="cpu", weights_only=True)
            periodic_round = int(server_ckpt.stem.rsplit("_", 1)[1]) - 1 if server_ckpt else -1
            if int(final_blob["round"]) >= periodic_round:
                server_ckpt, server_blob = final, final_blob
            del final_blob
        if server_ckpt is None:
            raise FileNotFoundError(f"no server_round_*.pt under {ckpt_dir}")
    if server_blob is None:
        server_blob = torch.load(server_ckpt, map_location="cpu", weights_only=True)
    if not (isinstance(server_blob, dict) and "state" in server_blob and "round" in server_blob):
        raise ValueError(
            f"{server_ckpt} is not a server_round_*.pt checkpoint ('state' and 'round'); "
            "personalized eval pairs it with each client's newest state up to that round."
        )
    server_shared = server_blob["state"]
    server_round = int(server_blob["round"])
    server_size, server_mtime_ns = _checkpoint_identity(server_ckpt)
    eval_config_hash = _eval_config_hash(cfg)

    base_bench = build_benchmark(cfg.eval)
    if not hasattr(base_bench, "task_names"):
        raise TypeError(
            f"benchmark {base_bench.name!r} has no task_names() accessor; "
            "personalized eval needs it to map manifest tasks onto task ids."
        )
    suites = cfg.eval.benchmark.suites
    name_maps = {s: base_bench.task_names(s) for s in suites}

    clients_dir = eval_dir / "personalized_clients"
    clients_dir.mkdir(parents=True, exist_ok=True)
    per_client: dict[str, dict[str, Any]] = {}
    for cid in sorted(manifest.clients):
        # The client's newest state at or before the server round: a client that sat out the
        # last rounds keeps its older personal weights; a state from a later round is never used.
        earlier = [(r, p) for r, p in client_checkpoints(run_dir / "client_states", cid) if r <= server_round]
        if not earlier:
            raise FileNotFoundError(
                f"no client state for {cid} from round {server_round} or earlier under "
                f"{run_dir / 'client_states'}: personalized eval needs fl.persist_client_state on the "
                "training run."
            )
        client_round, client_ckpt = earlier[-1]
        if client_round < server_round:
            logger.warning(
                "%s: no client state from round %d; using its newest earlier one, from round %d",
                cid,
                server_round,
                client_round,
            )
        task_filter = _client_task_filter(list(manifest.clients[cid]["task_names"]), name_maps)
        client_size, client_mtime_ns = _checkpoint_identity(client_ckpt)
        # Keyed on the checkpoints' identities (not only their paths), the eval settings and the
        # client's tasks, so a retrain into the same directory or a changed split re-evaluates.
        provenance = {
            "server_checkpoint": str(server_ckpt),
            "server_checkpoint_size": server_size,
            "server_checkpoint_mtime_ns": server_mtime_ns,
            "client_checkpoint": str(client_ckpt),
            "client_checkpoint_size": client_size,
            "client_checkpoint_mtime_ns": client_mtime_ns,
            "seed": cfg.eval.seed,
            "n_episodes_per_task": cfg.eval.n_episodes_per_task,
            "n_action_steps": cfg.eval.n_action_steps,
            "eval_config_hash": eval_config_hash,
            "task_filter": task_filter,
        }
        client_path = clients_dir / f"{cid}.json"
        if client_path.exists():
            cached = json.loads(client_path.read_text())
            if cached.pop("provenance", None) == provenance:
                logger.info("skipping %s: already evaluated", cid)
                per_client[cid] = cached
                continue
            logger.info("re-evaluating %s: provenance changed", cid)

        client_blob = torch.load(client_ckpt, map_location="cpu", weights_only=False)
        backend = backend_factory()
        assemble_personalized_state(backend, client_blob["weights"], server_shared)
        policy = backend.build_policy()
        apply_action_horizon(policy, cfg.eval.n_action_steps)
        bench_cfg = dataclasses.replace(
            cfg.eval.benchmark,
            extra={**(cfg.eval.benchmark.extra or {}), "task_filter": task_filter},
        )
        bench = build_benchmark(dataclasses.replace(cfg.eval, benchmark=bench_cfg))
        result = bench.evaluate(
            policy,
            suites=sorted(task_filter),
            n_episodes_per_task=cfg.eval.n_episodes_per_task,
            seed=cfg.eval.seed,
        )
        suites_out = {
            s: {
                "n_episodes": r.n_episodes,
                "n_success": r.n_success,
                "success_rate": r.success_rate,
                "per_task": [dataclasses.asdict(t) for t in r.per_task],
            }
            for s, r in result.suites.items()
        }
        entry = {
            "client_id": cid,
            "client_round": client_round,
            "task_names": list(manifest.clients[cid]["task_names"]),
            "n_episodes": sum(s["n_episodes"] for s in suites_out.values()),
            "n_success": sum(s["n_success"] for s in suites_out.values()),
            "suites": suites_out,
        }
        entry["success_rate"] = entry["n_success"] / entry["n_episodes"] if entry["n_episodes"] else 0.0
        tmp = client_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({**entry, "provenance": provenance}, indent=2))
        tmp.rename(client_path)
        per_client[cid] = entry
        del backend, policy, client_blob
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = {
        "per_client": per_client,
        "personalized": pool_client_results(per_client),
        "server_checkpoint": str(server_ckpt),
        "seed": cfg.eval.seed,
        "n_episodes_per_task": cfg.eval.n_episodes_per_task,
    }
    results_path = eval_dir / "personalized_results.json"
    tmp = results_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=2))
    tmp.rename(results_path)
    logger.info(
        "personalized eval complete: %s (pooled %.3f over %d episodes)",
        results_path,
        results["personalized"]["success_rate"],
        results["personalized"]["n_episodes"],
    )
    return results_path
