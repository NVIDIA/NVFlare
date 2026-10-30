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

"""Compare original and native FLARE training with the same deterministic configuration."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def prepare_config(
    base,
    *,
    rounds=None,
    strategy=None,
    wire_dtype=None,
    num_clients=None,
    threads=None,
    gpu=None,
    num_workers=0,
    checkpoint_every=1,
    local_steps=None,
):
    """Apply the same explicit comparison controls to both implementations."""
    from nvflare.private.fed.utils.fed_utils import split_gpus

    cfg = copy.deepcopy(base)
    fl = cfg["fl"]
    fl["num_rounds"] = fl.pop("rounds", fl.get("num_rounds", 10)) if rounds is None else rounds
    fl.pop("rounds", None)
    fl.pop("execution", None)
    fl["mode"] = "nvflare"
    if strategy is not None:
        fl.setdefault("strategy", {})["name"] = strategy
    if wire_dtype is not None:
        fl["wire_dtype"] = wire_dtype
    if local_steps is not None:
        fl["local_steps"] = local_steps
    data = cfg.get("data") or {}
    cfg["data"] = data
    part = data.get("partitioning")
    default_count = data.get("num_clients", len(data.get("client_repo_ids") or [None]))
    count = part.get("num_clients", default_count) if isinstance(part, dict) else default_count
    count = int(count if num_clients is None else num_clients)
    if count > 100:
        raise ValueError(
            "The automatic comparison launcher supports at most 100 clients; use an exported job and CLI -m for larger experiments."
        )
    data["num_clients"] = count
    if isinstance(part, dict):
        part["num_clients"] = count
    data["num_workers"] = num_workers
    fl["min_clients"] = count
    fl["save_every_n_rounds"] = checkpoint_every
    nv = fl.setdefault("nvflare", {})
    nv["env"] = "simulator"
    nv["threads"] = threads if threads is not None else nv.get("threads") or count
    if gpu is not None:
        nv["gpu"] = gpu or None
    if count < 1 or fl["num_rounds"] < 1 or fl.get("local_steps", 50) < 1:
        raise ValueError("client, round and local-step counts must be positive")
    if num_workers < 0 or checkpoint_every < 0 or not 1 <= nv["threads"] <= count:
        raise ValueError("workers/checkpoint cadence must be nonnegative; threads must be in [1, num_clients]")
    groups = split_gpus(nv["gpu"]) if nv.get("gpu") else []
    # Strict parity needs the optimizer, RNG and data iterator to survive rounds.
    if (
        any(not g for g in groups)
        or (len(groups) <= 1 and nv["threads"] < count)
        or (len(groups) > 1 and len(groups) != count)
    ):
        raise ValueError(
            "Strict comparison requires resident clients: use threads=num_clients with zero/one GPU "
            "group, or one GPU-map entry per client. Sequential swapping does not preserve full "
            "VLA client state. See AUTHOR_COMPARISON.md."
        )
    cfg.setdefault("backend", {}).setdefault("extra", {})["deterministic"] = True
    cfg.setdefault("wandb", {})["enabled"] = False
    return cfg


def compare_outputs(
    reference, native, atol=1e-6, rtol=1e-5, *, checkpoint_every=1, expected_rounds=None, num_clients=None
):
    reference, native = Path(reference), Path(native)
    curves = [json.loads((p / "training_curve.json").read_text()) for p in (reference, native)]
    count = len(curves[0]) if expected_rounds is None else expected_rounds
    if count == 0 or any([r["round"] for r in curve] != list(range(count)) for curve in curves):
        raise AssertionError("missing, duplicate or unexpected completed rounds")
    manifest = [json.loads((p / "partition_manifest.json").read_text()) for p in (reference, native)]
    if manifest[0] != manifest[1]:
        raise AssertionError("different data partitions")
    report = []
    for ref, new in zip(*curves):
        clients = set(ref["client_losses"])
        if (
            not clients
            or clients != set(new["client_losses"])
            or (num_clients is not None and clients != {f"c{i}" for i in range(num_clients)})
        ):
            raise AssertionError("round/client participation differs from the required all-client set")
        for c in clients:
            if not np.isfinite([ref["client_losses"][c], new["client_losses"][c]]).all():
                raise AssertionError("non-finite client loss")
            np.testing.assert_allclose(new["client_losses"][c], ref["client_losses"][c], atol=atol, rtol=rtol)
        loss_error = max(abs(ref["client_losses"][c] - new["client_losses"][c]) for c in clients)
        r = ref["round"] + 1
        name = (
            f"server_round_{r:04d}.pt"
            if checkpoint_every and r % checkpoint_every == 0
            else "server_round_final.pt" if r == count else None
        )
        weight_error = None
        if name is not None:
            # mmap avoids eagerly loading unused Adam moments during comparison.
            states = [
                torch.load(str(p / "checkpoints" / name), map_location="cpu", weights_only=False, mmap=True)["state"]
                for p in (reference, native)
            ]
            if not states[0] or states[0].keys() != states[1].keys():
                raise AssertionError("missing or different federated parameter names")
            weight_error = 0.0
            for k in states[0]:
                if not all(torch.isfinite(s[k]).all() for s in states):
                    raise AssertionError(f"non-finite shared tensor: {k}")
                torch.testing.assert_close(states[1][k], states[0][k], atol=atol, rtol=rtol)
                if states[0][k].numel():
                    weight_error = max(weight_error, (states[1][k] - states[0][k]).abs().max().item())
            del states
        report.append(
            {"round": ref["round"], "max_loss_error": loss_error, "max_weight_error": weight_error, "checkpoint": name}
        )
    return report


def write_curves(out):
    """Write plotting-friendly per-client and unweighted mean loss curves."""
    curves = [json.loads((out / p / "training_curve.json").read_text()) for p in ("reference", "native")]
    with (out / "training_curves.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["round", "client", "reference_loss", "native_loss", "abs_error"])
        for ref, new in zip(*curves):
            for cid in sorted(ref["client_losses"]) + ["mean"]:
                a = ref["mean_client_loss"] if cid == "mean" else ref["client_losses"][cid]
                b = new["mean_client_loss"] if cid == "mean" else new["client_losses"][cid]
                writer.writerow([ref["round"] + 1, cid, a, b, abs(a - b)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--native-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--strategy", choices=["fedavg", "fedprox", "fedadam"])
    parser.add_argument("--wire-dtype", choices=["fp16", "fp32"])
    parser.add_argument("--num-clients", type=int)
    parser.add_argument("--local-steps", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--gpu", help="FLARE GPU groups; empty string clears the config GPU map")
    parser.add_argument("--num-workers", type=int, default=0, help="data workers per client; default 0 for comparison")
    parser.add_argument(
        "--checkpoint-every", type=int, default=1, help="weight-check cadence; 0 checks final weights only"
    )
    parser.add_argument("--compare-only", action="store_true", help="recheck existing outputs without retraining")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    metadata = out / "comparison_meta.json"
    if args.compare_only:
        info = json.loads(metadata.read_text())
    else:
        if args.reference_dir is None or args.config is None:
            parser.error("--reference-dir and --config are required unless --compare-only is used")
        if any((out / p).exists() for p in ("reference", "native", "reference.yaml", "native.yaml", metadata.name)):
            parser.error("choose a fresh comparison output directory")
        sources = {"reference": args.reference_dir.resolve(), "native": args.native_dir.resolve()}
        if not (sources["reference"] / "src/decent_vla").is_dir():
            parser.error(f"reference source is missing src/decent_vla: {sources['reference']}")
        if not (sources["native"] / "src/__init__.py").is_file():
            parser.error(f"native source is missing src/__init__.py: {sources['native']}")
        try:
            cfg = prepare_config(
                yaml.safe_load(args.config.read_text()),
                rounds=args.rounds,
                strategy=args.strategy,
                wire_dtype=args.wire_dtype,
                num_clients=args.num_clients,
                threads=args.threads,
                gpu=args.gpu,
                num_workers=args.num_workers,
                checkpoint_every=args.checkpoint_every,
                local_steps=args.local_steps,
            )
        except ValueError as exc:
            parser.error(str(exc))
        out.mkdir(parents=True, exist_ok=True)
        info = {
            "expected_rounds": cfg["fl"]["num_rounds"],
            "num_clients": cfg["data"]["num_clients"],
            "checkpoint_every": args.checkpoint_every,
            "sources": {k: str(v) for k, v in sources.items()},
            "simulator": {k: cfg["fl"]["nvflare"].get(k) for k in ("threads", "gpu")},
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nvflare": importlib.metadata.version("nvflare"),
        }
        metadata.write_text(json.dumps(info, indent=2))
        # Only the reference config carries its old CLI launcher settings.
        from src.configs.loader import load_run_config

        configs = {}
        for label in sources:
            raw = copy.deepcopy(cfg)
            raw["fl"]["output_dir"] = str(out / label)
            raw["fl"]["nvflare"]["workspace"] = str(out / label / "workspace")
            if label == "native":
                raw.pop("wandb", None)
                raw.get("backend", {}).pop("model", None)
                raw.get("data", {}).pop("revision", None)
                raw.get("data", {}).pop("task_key", None)
                raw.get("fl", {}).get("strategy", {}).pop("aggregate_lora_only", None)
                raw["fl"].pop("mode")
                for key in ("env", "threads", "gpu", "workspace", "job_dir"):
                    raw["fl"]["nvflare"].pop(key, None)
                for key in ("after_training", "every_n_rounds", "only", "eval_all_tasks", "save_episodes"):
                    (raw.get("eval") or {}).pop(key, None)
            config = out / f"{label}.yaml"
            config.write_text(yaml.safe_dump(raw))
            if label == "native":
                load_run_config(config)
            configs[label] = config
        for label, source in sources.items():
            env = {
                **os.environ,
                "PYTHONPATH": str(source / "src") if label == "reference" else str(source),
                "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
                "PYTHONHASHSEED": str(cfg.get("seed", 42)),
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "DECENT_VLA_NO_RESUME": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
            env.pop("DVLA_REFERENCE_PROBE", None)
            env.pop("DVLA_REFERENCE_PROBE_ACTIVE", None)
            if label == "reference":
                probe = Path(__file__).resolve().parent / "reference_probe"
                env["PYTHONPATH"] = str(probe) + os.pathsep + env["PYTHONPATH"]
                env["DVLA_REFERENCE_PROBE"] = "1"
                env["DVLA_REFERENCE_CURVE"] = str(out / label / "training_curve.json")
            print(f"Running {label}: {source}", flush=True)
            if label == "reference":
                commands = [
                    [
                        sys.executable,
                        "-c",
                        "import os; assert os.environ.get('DVLA_REFERENCE_PROBE_ACTIVE') == '1', 'reference determinism probe did not load'",
                    ],
                    [sys.executable, "-m", "decent_vla.cli.main", "train", "--config", str(configs[label])],
                ]
            else:
                env["DECENT_VLA_WORKDIR"] = str(source)
                simulator = [
                    str(Path(sys.executable).parent / "nvflare"),
                    "simulator",
                    str(out / "jobs" / cfg["name"]),
                    "-w",
                    str(out / "native/workspace"),
                    "-c",
                    ",".join(f"c{i}" for i in range(info["num_clients"])),
                    "-t",
                    str(info["simulator"]["threads"]),
                ]
                if info["simulator"]["gpu"]:
                    simulator += ["-gpu", info["simulator"]["gpu"]]
                commands = [
                    [
                        sys.executable,
                        str(source / "job.py"),
                        "--config",
                        str(configs[label]),
                        "--job-dir",
                        str(out / "jobs"),
                    ],
                    simulator,
                ]
            with (out / f"{label}.log").open("w") as log:
                for command in commands:
                    subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = compare_outputs(
        out / "reference",
        out / "native",
        args.atol,
        args.rtol,
        checkpoint_every=info["checkpoint_every"],
        expected_rounds=info["expected_rounds"],
        num_clients=info["num_clients"],
    )
    info["last_successful_check"] = {
        "atol": args.atol,
        "rtol": args.rtol,
        "weight_rounds": [r["round"] + 1 for r in report if r["checkpoint"] is not None],
    }
    metadata.write_text(json.dumps(info, indent=2))
    write_curves(out)
    (out / "comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
