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

"""`decent-vla` CLI — argparse-based subcommand dispatch.

Subcommands:
  list      — list installed backends / strategies / benchmarks
  info      — show details for one plugin
  doctor    — env diagnostics (Python, torch, plugins, errors)
  train     — centralized training (one of train/fl in the config)
  eval      — evaluate a checkpoint on a benchmark
  run       — train then eval, driven by a single YAML config
  partition — write partition_manifest.json + train_episodes.json without training
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

from decent_vla import __version__
from decent_vla.configs import (
    RunConfig,
    load_run_config,
)
from decent_vla.core.registry import discover
from decent_vla.core.results import now_iso
from decent_vla.runtime.builders import (
    build_backend_factory,
    build_strategy,
    partition_for,
    resolve_device,
)
from decent_vla.runtime.eval_runner import run_eval

logger = logging.getLogger("decent_vla.cli")


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# ----------------------------------------------------------------------------
# Subcommand implementations
# ----------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    reg = discover()
    bucket = {"backends": reg.backends, "strategies": reg.strategies, "benchmarks": reg.benchmarks}
    if args.what not in bucket:
        print(f"Unknown plugin group: {args.what}", file=sys.stderr)
        return 2
    items = bucket[args.what]
    if not items:
        print(f"(no {args.what} installed)")
        return 0
    for name in sorted(items):
        print(name)
    if reg.errors:
        print("\nErrors during discovery:", file=sys.stderr)
        for k, v in reg.errors.items():
            print(f"  {k}: {v}", file=sys.stderr)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    reg = discover()
    bucket = {"backend": reg.backends, "strategy": reg.strategies, "benchmark": reg.benchmarks}
    if args.what not in bucket:
        print(f"Unknown kind: {args.what} (expected one of {list(bucket)})", file=sys.stderr)
        return 2
    items = bucket[args.what]
    if args.name not in items:
        print(f"No {args.what} named {args.name!r} is installed.", file=sys.stderr)
        print(f"Available {args.what}s: {sorted(items)}", file=sys.stderr)
        return 1
    obj = items[args.name]
    print(f"{args.what}: {args.name}")
    print(f"  class:   {obj.__name__ if hasattr(obj, '__name__') else obj}")
    print(f"  module:  {getattr(obj, '__module__', '?')}")
    doc = (getattr(obj, "__doc__", None) or "").strip().splitlines()
    if doc:
        print(f"  doc:     {doc[0]}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    print(f"decent-vla {__version__}")
    print(f"Python:  {sys.version.split()[0]}  ({sys.executable})")
    print(f"Torch:   {torch.__version__}")
    print(f"CUDA:    available={torch.cuda.is_available()}  device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"         [{i}] {torch.cuda.get_device_name(i)}")
    # Informational dependency checks (spec §10) — never change the exit code.
    for pkg, extra in (
        ("nvflare", "[nvflare] extra"),
        ("peft", "[peft] extra"),
    ):
        try:
            ver = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            print(f"{pkg}: not installed ({extra})")
            continue
        print(f"{pkg}: {ver}")
    hf_home = os.environ.get("HF_HOME")
    if hf_home and not Path(hf_home).exists():
        print(f"WARN: HF_HOME={hf_home} does not exist")
    reg = discover()
    print(f"Backends:    {sorted(reg.backends) or '(none)'}")
    print(f"Strategies:  {sorted(reg.strategies) or '(none)'}")
    print(f"Benchmarks:  {sorted(reg.benchmarks) or '(none)'}")
    if reg.errors:
        print("Plugin import errors:")
        for k, v in reg.errors.items():
            print(f"  {k}: {v}")
        return 1
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """train (if cfg.train or cfg.fl set) then eval (if cfg.eval set)."""
    cfg = load_run_config(args.config)
    os.environ.setdefault("DECENT_VLA_CONFIG", args.config)
    device = resolve_device(cfg.device)
    started_at = now_iso()
    t0 = time.time()

    output_dir = Path(_run_output_dir(cfg))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Persist the normalized config for reproducibility.
    cfg_dict = dataclasses.asdict(cfg)
    cfg_path = output_dir / "config.json"
    with cfg_path.open("w") as f:
        json.dump(cfg_dict, f, indent=2, default=str)

    # ----- train (if not eval-only) -----
    eval_only = getattr(args, "eval_only", False) or (cfg.eval is not None and cfg.eval.only)
    do_train = cfg.fl is not None or cfg.train is not None
    if do_train and not eval_only:
        resume = getattr(args, "resume", None)
        if cfg.fl is not None:
            _run_fl_local(cfg, device, output_dir, resume=resume, config_path=args.config)
        else:
            _run_centralized(cfg, device, output_dir, resume=resume)

    # ----- eval (if not train-only) -----
    train_only = getattr(args, "train_only", False)
    if cfg.eval is not None and not train_only:
        backend_factory, _ = build_backend_factory(cfg, device)
        if cfg.eval.personalized:
            from decent_vla.runtime.personalized_eval import run_personalized_eval

            run_dir = Path(cfg.fl.output_dir) if cfg.fl is not None else output_dir
            eval_dir = Path(cfg.eval.output_dir or (output_dir / "eval"))
            run_personalized_eval(cfg, backend_factory, run_dir, eval_dir)
        else:
            backend = backend_factory()
            ckpt_path = args.ckpt or _latest_checkpoint(output_dir)
            run_eval(cfg, backend, ckpt_path, output_dir, started_at, t0, cfg_dict,
                     config_path=args.config)

    return 0


def cmd_train(args: argparse.Namespace) -> int:
    # cmd_run reloads the config from disk, so train-only must travel via args
    # (mutating cfg.eval here would be a dead write).
    return cmd_run(
        argparse.Namespace(
            config=args.config,
            ckpt=None,
            resume=getattr(args, "resume", None),
            train_only=True,
        )
    )


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = load_run_config(args.config)
    if cfg.eval is None:
        raise SystemExit("Config has no `eval` block; nothing to do.")
    # cmd_run reloads the config from disk, so eval-only must travel via args.
    return cmd_run(argparse.Namespace(config=args.config, ckpt=args.ckpt, eval_only=True))


def cmd_partition(args: argparse.Namespace) -> int:
    """Write partition_manifest.json + train_episodes.json without training.

    `train_episodes.json` is the flat sorted union of all client episode
    indices — the `--dataset.episodes` injection list for the centralised
    native-trainer path.
    """
    cfg = load_run_config(args.config)
    _, backend_cls = build_backend_factory(cfg, torch.device("cpu"))
    chunks, manifest = partition_for(cfg, backend_cls)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest.write(out / "partition_manifest.json")
    train_episodes = sorted(i for chunk in chunks for i in chunk)
    with (out / "train_episodes.json").open("w") as f:
        json.dump(train_episodes, f)
    print(
        f"Wrote {out / 'partition_manifest.json'} "
        f"({manifest.num_clients} clients, {manifest.total_episodes} episodes, "
        f"{manifest.content_hash()})"
    )
    print(f"Wrote {out / 'train_episodes.json'} ({len(train_episodes)} episodes)")
    return 0


# ----------------------------------------------------------------------------
# Training implementations
# ----------------------------------------------------------------------------


def _run_fl_local(
    cfg: RunConfig,
    device: torch.device,
    output_dir: Path,
    resume: str | None = None,
    config_path: str | None = None,
) -> None:
    from decent_vla.core.types import CheckpointSpec
    from decent_vla.runtime import observability as _obs
    from decent_vla.runtime.local_runner import LocalRunner
    from decent_vla.runtime.round_loop import save_server_checkpoint

    # Same `wandb:` config block as the NVFlare plane (see server_session).
    _obs.configure_wandb(cfg)

    if cfg.fl.mode == "nvflare":
        # Blocking in-process simulator run (or job export) — checkpoints land
        # in cfg.fl.output_dir via the shared server session, so cmd_run's
        # post-training eval works exactly as in local mode. Prefer the real
        # --config path over DECENT_VLA_CONFIG: cmd_run only setdefault()s the
        # env var, and a stale pre-set value would silently swap the config.
        from decent_vla.runtime.nvflare.entry import run_from_cli

        run_from_cli(cfg, config_path or os.environ["DECENT_VLA_CONFIG"], resume=resume)
        return
    if cfg.fl.mode != "local":
        raise SystemExit(f"Unknown fl.execution mode: {cfg.fl.mode!r}")

    backend_factory, backend_cls = build_backend_factory(cfg, device)
    strategy = build_strategy(cfg)

    # Generic partitioning for any backend (episode-level when the backend
    # exposes list_episodes, flat sample universe otherwise). The manifest is
    # the audit artifact — written BEFORE training step 1 (spec §5).
    chunks, manifest = partition_for(cfg, backend_cls)
    manifest.write(output_dir / "partition_manifest.json")
    client_ids = [f"c{i}" for i in range(cfg.data.num_clients)]

    # Opt-in client-per-GPU spread (fl.client_gpu_spread): give each client its
    # own dedicated CUDA device so a full-model client (pi0.5/smolvla full-FT)
    # never has to share/juggle one GPU. Engages only when there are at least as
    # many visible GPUs as clients; otherwise fall back to single-GPU juggling
    # so nothing regresses (e.g. LIBERO 8-client on one GPU).
    client_devices = None
    if getattr(cfg.fl, "client_gpu_spread", False) and device.type == "cuda":
        n_gpus = torch.cuda.device_count()
        if n_gpus >= len(client_ids):
            client_devices = [torch.device(f"cuda:{i}") for i in range(len(client_ids))]
            print(
                f"[client_gpu_spread] {len(client_ids)} clients on dedicated GPUs "
                f"{[str(d) for d in client_devices]}"
            )
        else:
            print(
                f"[client_gpu_spread] requested but only {n_gpus} visible GPU(s) for "
                f"{len(client_ids)} clients; falling back to single-GPU juggling"
            )

    start_round = 0
    initial_state: dict[str, torch.Tensor] | None = None
    if resume is not None:
        # Checkpoint blob is {"state": dict[str, Tensor], "round": int} —
        # weights_only-safe (tensors + primitives only).
        path = resume
        blob = torch.load(path, map_location="cpu", weights_only=True)
        # Resume restores the server state + round counter and fast-forwards
        # client LR schedules; client optimizer moments restart cold by design
        # (spec §9).
        if not (isinstance(blob, dict) and "state" in blob and "round" in blob):
            raise SystemExit(
                f"--resume expects a server_round_*.pt checkpoint with 'state'+'round'; "
                f"got keys {list(blob)[:5]} from {path}"
            )
        initial_state = blob["state"]
        start_round = int(blob["round"]) + 1
        if blob.get("strategy_state"):
            strategy.load_state_dict(blob["strategy_state"])
            print(f"Restored server strategy state ({strategy.name}) on resume")
        print(f"Resuming from {path}: starting at round {start_round}")

    ckpt_dir = output_dir / "checkpoints"

    # Scope provenance for freeze/personalization runs (both axes, resolved
    # names, payload bytes). Only when patterns are configured — writing it
    # costs one extra backend build on this path, and ordinary full-scope runs
    # have nothing non-obvious to record. Fresh starts only (resume keeps the
    # round-0 manifest).
    if resume is None and (cfg.backend.trainable_params or cfg.backend.shared_params):
        from decent_vla.runtime.scope_manifest import write_scope_manifest

        _boot = backend_factory()
        write_scope_manifest(output_dir, cfg.backend, cfg.fl, _boot)
        del _boot

    # cfg.fl.num_rounds is the TOTAL round count for the run, not "more rounds"
    # (as in runtime/server_session.run_fl_server). Without this, a
    # resume at round k would run the full num_rounds again from k and overshoot.
    remaining = max(0, cfg.fl.num_rounds - start_round)

    runner = LocalRunner(
        backend_factory=backend_factory,
        strategy=strategy,
        client_ids=client_ids,
        partition_indices=chunks,
        data_cfg=cfg.data,
        num_rounds=remaining,
        local_steps=cfg.fl.local_steps,
        optim_cfg=cfg.backend.optim,
        device=device,
        save_every_n_rounds=cfg.fl.save_every_n_rounds,
        checkpoint_cb=lambda r, state: save_server_checkpoint(ckpt_dir, r, state, strategy.state_dict()),
        min_clients=cfg.fl.min_clients,
        round_timeout_s=cfg.fl.round_timeout_s,
        start_round=start_round,
        initial_state=initial_state,
        persist_client_state=cfg.fl.persist_client_state,
        client_ckpt_dir=str(output_dir / "client_states"),
        client_state_keep=cfg.fl.client_state_keep,
        client_devices=client_devices,
        lean_host_memory=cfg.fl.lean_host_memory,
    )
    outcome = runner.run()

    # Materialize the final aggregated state into a backend and checkpoint it.
    final_backend = backend_factory()
    final_backend.set_state(outcome.final_state, scope="shared")
    final_round = start_round + outcome.rounds_completed
    ckpt = ckpt_dir / f"round_{final_round:04d}.pt"
    final_backend.save_checkpoint(str(ckpt), spec=CheckpointSpec())
    print(f"FL training complete: {outcome.rounds_completed} rounds, checkpoint at {ckpt}")


def _run_centralized(
    cfg: RunConfig, device: torch.device, output_dir: Path, resume: str | None = None
) -> None:
    """Centralized one-client local-mode training."""
    from decent_vla.configs.schemas import FLConfig, StrategyConfig

    cfg.fl = FLConfig(
        strategy=StrategyConfig(name="fedavg"),
        num_rounds=1,
        local_steps=cfg.train.num_steps,
        mode="local",
        output_dir=cfg.train.output_dir,
    )
    cfg.data.num_clients = 1
    _run_fl_local(cfg, device, output_dir, resume=resume)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _run_output_dir(cfg: RunConfig) -> str:
    if cfg.fl is not None:
        return cfg.fl.output_dir
    if cfg.train is not None:
        return cfg.train.output_dir
    if cfg.eval is not None and cfg.eval.output_dir:
        return cfg.eval.output_dir
    return f"outputs/{cfg.name}"


def _latest_checkpoint(output_dir: Path) -> Path | None:
    ckpt_dir = output_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    # server_round_*.pt are raw aggregated-state blobs for `--resume`, not
    # backend-format checkpoints; eval must prefer the backend ones.
    candidates = sorted(p for p in ckpt_dir.glob("*.pt") if not p.name.startswith("server_round_"))
    if not candidates:
        # Check whether there are any server-state blobs; if so, training did not
        # complete and we cannot produce a valid eval checkpoint automatically.
        server_blobs = sorted(ckpt_dir.glob("server_round_*.pt"))
        if server_blobs:
            raise SystemExit(
                f"Only server-state checkpoints found in {ckpt_dir}; "
                "run training to completion to produce a backend checkpoint, "
                "or pass --ckpt explicitly."
            )
        return None
    return candidates[-1]


# ----------------------------------------------------------------------------
# Top-level parser
# ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="decent-vla",
        description="Backend-agnostic federated learning for VLA policies.",
    )
    p.add_argument("--version", action="version", version=f"decent-vla {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="list installed backends, strategies, or benchmarks")
    pl.add_argument("what", choices=["backends", "strategies", "benchmarks"])
    pl.set_defaults(func=cmd_list)

    pi = sub.add_parser("info", help="show details for one plugin")
    pi.add_argument("what", choices=["backend", "strategy", "benchmark"])
    pi.add_argument("name")
    pi.set_defaults(func=cmd_info)

    pd = sub.add_parser("doctor", help="environment diagnostics")
    pd.set_defaults(func=cmd_doctor)

    pr = sub.add_parser("run", help="train (if configured) then eval, driven by one config")
    pr.add_argument("--config", required=True)
    pr.add_argument("--ckpt", default=None, help="checkpoint to eval (defaults to latest)")
    pr.add_argument(
        "--resume", default=None,
        help="server checkpoint (checkpoints/server_round_*.pt) to resume FL training from",
    )
    pr.set_defaults(func=cmd_run)

    pt = sub.add_parser("train", help="centralized or FL training (no eval)")
    pt.add_argument("--config", required=True)
    pt.add_argument(
        "--resume", default=None,
        help="server checkpoint (checkpoints/server_round_*.pt) to resume FL training from",
    )
    pt.set_defaults(func=cmd_train)

    pe = sub.add_parser("eval", help="evaluate a checkpoint")
    pe.add_argument("--config", required=True)
    pe.add_argument("--ckpt", default=None)
    pe.set_defaults(func=cmd_eval)

    pp = sub.add_parser(
        "partition",
        help="write partition_manifest.json + train_episodes.json without training",
    )
    pp.add_argument("--config", required=True)
    pp.add_argument("--output-dir", required=True)
    pp.set_defaults(func=cmd_partition)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
