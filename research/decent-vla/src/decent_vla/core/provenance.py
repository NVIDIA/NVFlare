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

"""Eval provenance sidecar: identify WHAT was evaluated, next to the artifacts.

The 2026-08-14 forensic audit of the pi0.5 r100 results could reconstruct
which checkpoint produced which numbers only because the Slurm job logs
still existed (the `[eval] ckpt=` echo lines). The artifacts themselves —
lerobot's `eval_info.json` in particular — carried no checkpoint identity,
no action horizon, no config identity. This module writes that identity
into the artifact directory at eval time, so the audit trail survives log
rotation, `eval_logs/` moves, and dir renames.

Two consumers:

  * `runtime/eval_runner.py` embeds the same identity fields directly in
    `results.json` (schema v1.2 `provenance` block).
  * Eval runs through lerobot's own `lerobot-eval` harness, where
    `eval_info.json` is written by upstream lerobot and cannot be extended,
    run
    `python -m decent_vla.core.provenance …` to drop a `provenance.json`
    sidecar into the harness output dir BEFORE the rollouts start — a
    crashed eval still leaves the identity behind.

core/ rules apply: stdlib-only, backend- and benchmark-agnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from decent_vla.core.results import git_commit, now_iso

SIDECAR_SCHEMA = "decent-vla/eval-provenance@1"
SIDECAR_NAME = "provenance.json"


def file_identity(path: str | Path, sha256: bool = False) -> dict[str, Any]:
    """`{path, size, mtime_ns[, sha256]}` for an existing file.

    stat-only by default: the size+mtime_ns pair is the repo's standard
    checkpoint identity (an in-place rewrite always moves mtime_ns) and costs
    one syscall. `sha256=True` adds a content hash — the strongest identity,
    at the price of reading the whole file; callers opt in per file.
    """
    p = Path(path)
    st = p.stat()
    ident: dict[str, Any] = {"path": str(p), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    if sha256:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        ident["sha256"] = h.hexdigest()
    return ident


def write_provenance(output_dir: str | Path, provenance: dict[str, Any]) -> Path:
    """Atomically write `provenance` as `{output_dir}/provenance.json`."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / SIDECAR_NAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(provenance, indent=2))
    tmp.replace(path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stamp a provenance.json sidecar into an eval output dir."
    )
    parser.add_argument("--output-dir", required=True,
                        help="eval output dir (created if missing)")
    parser.add_argument("--ckpt", required=True,
                        help="the checkpoint being evaluated")
    parser.add_argument("--config", default=None,
                        help="run config the eval was derived from")
    parser.add_argument("--n-action-steps", type=int, default=None,
                        help="the action horizon the harness is invoked with")
    parser.add_argument("--sha256", action="store_true",
                        help="content-hash the checkpoint (reads the whole "
                             "file; seconds per GB — worth it for a formal run)")
    parser.add_argument("--extra", action="append", default=[], metavar="KEY=VALUE",
                        help="free-form context (suites, n_episodes, slurm_job_id, …)")
    args = parser.parse_args(argv)

    # Resolve to absolute: launch scripts pass $CKPT as typed (usually relative
    # to the launch dir), and a relative path in the sidecar is ambiguous the
    # moment eval_logs/ moves.
    ckpt = Path(args.ckpt).resolve()
    # A stamp for a path that doesn't exist would be a fabricated audit
    # record — refuse rather than record.
    if not ckpt.is_file():
        raise SystemExit(f"provenance: checkpoint does not exist: {ckpt}")

    extra: dict[str, str] = {}
    for kv in args.extra:
        key, sep, value = kv.partition("=")
        if not sep:
            raise SystemExit(f"provenance: --extra expects KEY=VALUE, got {kv!r}")
        extra[key] = value

    prov: dict[str, Any] = {
        "schema": SIDECAR_SCHEMA,
        "written_at": now_iso(),
        "git_commit": git_commit(),
        "checkpoint": file_identity(ckpt, sha256=args.sha256),
        # Configs are tiny: always content-hashed.
        "config": (file_identity(Path(args.config).resolve(), sha256=True)
                   if args.config else None),
        "n_action_steps": args.n_action_steps,
        "extra": extra,
    }
    print(write_provenance(args.output_dir, prov))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
