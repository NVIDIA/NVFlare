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
#
# Authors: Anbang Liu, Junhan Zhao, and Ziyue Xu

"""Resume at audited global-round boundaries without changing local training."""

from __future__ import annotations

import fcntl
import os
import pickle
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from fabric_common import SITES, cpu_state, inside, read_json, sha256_file, state_digest, training_args, write_json


def atomic_save(payload: dict, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def validate_resume_run(root: Path, run_root: Path, setups: list, folds: list, manifest_root: Path) -> None:
    if not run_root.is_dir():
        raise FileNotFoundError(f"Resume run does not exist: {run_root}")
    # An intact checkpoint chain does not prove that training behavior is unchanged.
    # Require every recorded source file to match, including the FLARE adapters.
    for relative, digest in read_json(run_root / "code_sha256.json").items():
        if sha256_file(inside(root, relative)) != digest:
            raise ValueError(f"Training source changed since this run: {relative}")
    for setup in setups:
        output = run_root / setup["experiment"]
        path = output / "run_config.json"
        if not path.is_file():
            if output.exists() and any(output.glob("fold_*/runtime.json")):
                raise ValueError(f"Existing experiment has no run configuration: {output}")
            continue  # The previous invocation may have stopped before this encoder.
        saved = read_json(path)
        expected = {key: setup[key] for key in ("experiment", "model_name", "encoder", "input_dim")}
        expected.update(
            variant=setup.get("variant", "pooling"),
            model_options=setup.get("model_options", {}),
            settings=vars(training_args(root, setup, output, manifest_root)),
            source_fold_manifests=str(manifest_root),
        )
        for key, value in expected.items():
            default = "pooling" if key == "variant" else {} if key == "model_options" else None
            if saved.get(key, default) != value:
                raise ValueError(f"Resume configuration changed for {setup['experiment']}: {key}")
        if not set(folds).issubset(saved["folds"]):
            raise ValueError("Resume folds must belong to the original run configuration")


@contextmanager
def run_lock(run_root: Path):
    """Protect a run from two launchers, including pre-resume FLARE processes."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        runner = any(Path(arg).name == "run_fabric.py" for arg in argv if arg)
        same_name = f"--run-name={run_root.name}" in argv or any(
            a == "--run-name" and b == run_root.name for a, b in zip(argv, argv[1:])  # noqa: B905
        )
        flare = any("nvflare" in arg or Path(arg).name == "fabric_client.py" for arg in argv if arg)
        same_path = any(str(run_root) in arg for arg in argv)
        if (runner and (same_name or same_path)) or (flare and same_path):
            raise RuntimeError(f"Run is still active in process {entry.name}; refusing a second launcher")
    with (run_root / ".runner.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another launcher already holds this run") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _load_matching(path: Path, digest: str, specification: dict):
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state = cpu_state(payload.get("model_state_dict", payload.get("model", payload)))
    except (OSError, EOFError, RuntimeError, ValueError, TypeError, pickle.UnpicklingError):
        return None
    if state_digest(state) != digest:
        return None
    if list(state) != list(specification) or any(
        list(value.shape) != specification[key]["shape"] or str(value.dtype) != specification[key]["dtype"]
        for key, value in state.items()
    ):
        raise ValueError("Resume checkpoint parameter specification changed")
    return state


def inspect_fold(fold_dir: Path, setup: dict) -> dict:
    """Read-only recovery plan; a log alone is never treated as a checkpoint."""
    path = fold_dir / "runtime.json"
    if not path.is_file():
        if list(fold_dir.rglob("round_*.json")) or list(fold_dir.rglob("FL_global_model.pt")):
            raise ValueError(f"Training artifacts without their original runtime configuration: {fold_dir}")
        return {"action": "start", "completed_rounds": 0}
    runtime = read_json(path)
    if (
        Path(runtime["fold_dir"]).resolve() != fold_dir.resolve()
        or runtime["experiment"] != setup["experiment"]
        or runtime.get("variant", "pooling") != setup.get("variant", "pooling")
        or runtime.get("top_k") != setup.get("model_options", {}).get("top_k")
    ):
        raise ValueError("Resume runtime does not match the requested fold/model")
    for name, digest in runtime["manifest_sha256"].items():
        if sha256_file(inside(fold_dir, name)) != digest:
            raise ValueError(f"Saved fold manifest changed: {name}")
    paths = sorted((fold_dir / "server_audit").glob("round_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    if len(paths) > setup["settings"]["rounds"]:
        raise ValueError("Too many audited rounds")
    digests = [runtime["initial_state_sha256"]]
    for number, audit_path in enumerate(paths, 1):
        audit = read_json(audit_path)
        if (
            audit_path.name != f"round_{number}.json"
            or audit["round"] != number
            or audit["fold"] != runtime["fold"]
            or audit["sites_in_aggregation_order"] != list(SITES)
            or audit["patient_weights"] != [runtime["client_patients"][s] for s in SITES]
            or audit["input_state_sha256"] != digests[-1]
            or len(audit["local_logs"]) != len(SITES)
        ):
            raise ValueError(f"Incomplete/inconsistent aggregation chain: {audit_path}")
        digests.append(audit["output_state_sha256"])
    completed = len(paths)
    framework_files = sorted(fold_dir.rglob("FL_global_model.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    framework_checkpoint = None
    # A final round must also have been persisted by FLARE. If interrupted
    # between aggregation and framework persistence, safely replay that round.
    if completed == setup["settings"]["rounds"]:
        for candidate in framework_files:
            if _load_matching(candidate, digests[completed], runtime["state_spec"]) is not None:
                framework_checkpoint = candidate
                break
        if framework_checkpoint is None:
            completed -= 1
    candidates = [fold_dir / "server_checkpoints" / f"round_{completed}.pt", *framework_files]
    if completed == 0:
        candidates = [fold_dir / "initial_model.pt"]
    elif completed == setup["settings"]["rounds"]:
        candidates.insert(0, fold_dir / "global_model_round_final.pt")
    checkpoint = next(
        (
            p
            for p in candidates
            if p.is_file() and _load_matching(p, digests[completed], runtime["state_spec"]) is not None
        ),
        None,
    )
    if checkpoint is None:
        raise RuntimeError(f"No intact checkpoint matches audited round {completed} in {fold_dir}")
    required = (
        "nvflare_run.json",
        "global_model_round_final.pt",
        "global_test_predictions.tsv",
        "fold_result.tsv",
        "round_logs.tsv",
    )
    action = "resume"
    if completed == setup["settings"]["rounds"]:
        action = "complete" if all((fold_dir / n).is_file() for n in required) else "evaluate"
        if action == "complete":
            record = read_json(fold_dir / "nvflare_run.json")
            if record["final_state_sha256"] != digests[completed]:
                raise ValueError("Fold completion record differs from the audited final model")
            if (
                _load_matching(fold_dir / "global_model_round_final.pt", digests[completed], runtime["state_spec"])
                is None
            ):
                raise ValueError("Completed fold's final checkpoint is corrupt or changed")
    return {
        "action": action,
        "completed_rounds": completed,
        "checkpoint": str(checkpoint),
        "state_sha256": digests[completed],
        "framework_checkpoint": str(framework_checkpoint) if framework_checkpoint else None,
    }


def prepare_resume(fold_dir: Path, plan: dict) -> tuple[Path, Path]:
    """Retain original artifacts; archive only work after the recovery boundary."""
    runtime = read_json(fold_dir / "runtime.json")
    state = _load_matching(Path(plan["checkpoint"]), plan["state_sha256"], runtime["state_spec"])
    if state is None:
        raise ValueError("Resume checkpoint changed after validation")
    parent = fold_dir / "resume_attempts"
    parent.mkdir(exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="attempt_", dir=parent))
    checkpoint = attempt / "resume_model.pt"
    atomic_save({"model": state}, checkpoint)
    completed = plan["completed_rounds"]
    pending = []
    for directory in (fold_dir / "client_logs", fold_dir / "server_audit", fold_dir / "server_checkpoints"):
        for path in directory.rglob("round_*"):
            match = re.match(r"round_(\d+)(?:[_.])", path.name)
            if path.is_file() and match and int(match[1]) > completed:
                pending.append(path)
    final = fold_dir / "global_model_round_final.pt"
    if completed < 5 and final.exists():
        pending.append(final)
    for path in pending:
        destination = attempt / "previous_incomplete" / path.relative_to(fold_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination))
    runtime.update(resume_offset=completed, resume_checkpoint=str(checkpoint), resume_state_sha256=plan["state_sha256"])
    runtime_path = attempt / "runtime.json"
    write_json(runtime_path, runtime)
    write_json(attempt / "recovery.json", plan)
    return runtime_path, attempt
