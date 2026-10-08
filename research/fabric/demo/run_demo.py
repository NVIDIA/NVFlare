#!/usr/bin/env python3
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

"""Generate fictional feature bags and export or execute a two-round FLARE job."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import re
import shlex
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
DEMO_CODE = Path(__file__).resolve().parent
CODE = ROOT / "fabric"
sys.path.insert(0, str(CODE))

from fabric_common import SITES, cpu_state, inside, load_core, load_setup, read_json, state_digest, write_json
from fabric_runtime import create_runtime_tmp, launch_simulator


def create_inputs(directory: Path, setup: dict) -> dict:
    import numpy as np
    import pandas as pd
    import torch

    core = load_core()
    settings = dict(setup["settings"])
    settings.update(rounds=2, local_epochs=1, batch_size=2, num_workers=0, max_instances=32, device="cpu", amp=False)
    if setup.get("variant") == "topk":
        settings.update(model_variant="topk", dtfd_top_k=1, dtfd_pseudo_loss_weight=1.0, dtfd_eval_group_seed=42)
    counts = dict(zip(SITES, (4, 6, 8), strict=True))
    data = directory / "data"
    (data / "features").mkdir(parents=True)
    generator = np.random.default_rng(settings["seed"])
    all_ids = set()
    for site, count in [*counts.items(), ("test", 6)]:
        rows = []
        for index in range(count):
            case = f"synthetic-{site}-{index:03d}"
            if case in all_ids:
                raise ValueError("Synthetic train/test overlap")
            all_ids.add(case)
            label = index % 2
            features = generator.normal(size=(17 + 2 * (index % 4), setup["input_dim"])).astype(np.float32)
            features[:, :8] += 0.15 * (2 * label - 1)
            relative = f"features/{case}.npy"
            np.save(data / relative, features)
            rows.append(
                {
                    "patient_id": case,
                    "institution": "Synthetic",
                    "site": site,
                    "recurrence_label": label,
                    "tumor_grade": "synthetic",
                    "encoder": setup["encoder"],
                    "num_slides": 1,
                    "slide_ids": f"{case}-slide",
                    "feature_paths": relative,
                }
            )
        pd.DataFrame(rows).to_csv(data / f"{site}.csv", index=False)
    core.training.set_seed(settings["seed"])
    model = core.training.build_model(setup["model_name"], setup["input_dim"], SimpleNamespace(**settings))
    state = cpu_state(model.state_dict())
    torch.save({"model": state}, directory / "initial_model.pt")
    runtime = {
        "kind": "fabric_synthetic_demo",
        "root": str(ROOT),
        "demo_dir": str(directory),
        "encoder": setup["encoder"],
        "model_name": setup["model_name"],
        "variant": setup.get("variant", "pooling"),
        "input_dim": setup["input_dim"],
        "settings": settings,
        "client_counts": counts,
        "initial_state_sha256": state_digest(state),
        "state_spec": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in state.items()},
    }
    write_json(directory / "runtime.json", runtime)
    return runtime


def export_job(directory: Path, runtime: dict):
    from demo_components import DemoFedAvg

    from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
    from nvflare.client.config import ExchangeFormat, TransferType
    from nvflare.recipe import SimEnv

    settings = runtime["settings"]
    model_args = {
        "model_name": runtime["model_name"],
        "input_dim": runtime["input_dim"],
        "n_classes": 2,
        "embed_dim": settings["embed_dim"],
        "attn_dim": settings["attn_dim"],
        "dropout": settings["dropout"],
        "dtfd_pseudo_bags": settings["dtfd_pseudo_bags"],
    }
    if runtime["variant"] == "topk":
        model_args.update(dtfd_top_k=1, dtfd_eval_group_seed=42)
    recipe = FedAvgRecipe(
        name="fabric_demo",
        model={"class_path": "core.mil_models.MILModel", "args": model_args},
        initial_ckpt=str(directory / "initial_model.pt"),
        min_clients=3,
        num_rounds=2,
        train_script="demo_client.py",
        train_args=shlex.join(["--runtime", str(directory / "runtime.json")]),
        aggregator=DemoFedAvg(runtime_path=str(directory / "runtime.json")),
        launch_external_process=True,
        launch_once=False,
        command=shlex.join([sys.executable, "-B", "-u"]),
        server_expected_format=ExchangeFormat.PYTORCH,
        params_transfer_type=TransferType.FULL,
        per_site_config={site: {} for site in SITES},
        key_metric="",
        stop_cond=None,
        patience=None,
    )
    for source in sorted(CODE.rglob("*.py")):
        relative = source.relative_to(CODE)
        destination = None if relative.parent == Path(".") else str(relative.parent)
        for target in ("server", *SITES):
            recipe.job.add_file_to(str(source), target, dest_dir=destination)
    for name in ("demo_client.py", "demo_components.py"):
        for target in ("server", *SITES):
            recipe.job.add_file_to(str(DEMO_CODE / name), target)
    environment = SimEnv(clients=list(SITES), num_threads=1, workspace_root=str(directory / "nvflare_workspace"))
    recipe.export(str(directory / "exported_job"), env=environment)
    return directory / "exported_job" / recipe.job.name, Path(environment.workspace_root) / recipe.job.name


def verify_and_evaluate(directory: Path, runtime: dict, workspace: Path) -> dict:
    import pandas as pd
    import torch
    from demo_components import read_demo_manifest, training_args

    expected_digest = runtime["initial_state_sha256"]
    for number in range(1, 3):
        audit = read_json(directory / "server_audit" / f"round_{number}.json")
        if (
            audit["round"] != number
            or audit["clients"] != list(SITES)
            or audit["synthetic_case_counts"] != list(runtime["client_counts"].values())
            or audit["input_state_sha256"] != expected_digest
        ):
            raise RuntimeError("Incomplete or inconsistent demonstration aggregation")
        for site in SITES:
            log = read_json(directory / "client_logs" / site / f"round_{number}.json")
            if log["round"] != number or log["site"] != site or log["input_state_sha256"] != expected_digest:
                raise RuntimeError("Missing or inconsistent demonstration client task")
        expected_digest = audit["output_state_sha256"]
    files = list(workspace.rglob("FL_global_model.pt"))
    if len(files) != 1:
        raise RuntimeError("Expected one final FLARE checkpoint")
    persisted = torch.load(files[0], map_location="cpu", weights_only=True)
    state = cpu_state(persisted.get("model", persisted))
    aggregate = torch.load(directory / "aggregated_model.pt", map_location="cpu", weights_only=True)
    if state_digest(state) != expected_digest or state_digest(aggregate["model"]) != expected_digest:
        raise RuntimeError("Persisted FLARE model does not match the final aggregate")
    core = load_core()
    args = training_args(runtime)
    model = core.training.build_model(runtime["model_name"], runtime["input_dim"], args)
    model.load_state_dict(state, strict=True)
    test = read_demo_manifest(runtime, "test")
    train = pd.concat([read_demo_manifest(runtime, site) for site in SITES], ignore_index=True)
    if set(test["patient_id"]) & set(train["patient_id"]):
        raise RuntimeError("Synthetic train/test overlap")
    metrics, predictions = core.training.evaluate_model(model, test, train, args, args.seed, torch.device("cpu"))
    predictions.to_csv(directory / "synthetic_predictions.tsv", sep="\t", index=False)
    summary = {
        "status": "completed",
        "data": "entirely synthetic; not clinical performance",
        "encoder_shape": runtime["encoder"],
        "variant": runtime["variant"],
        "completed_rounds": 2,
        "completed_client_tasks": 6,
        "test_cases": len(test),
        "final_state_sha256": expected_digest,
        "flare_checkpoint_matches": True,
        "metrics_on_synthetic_data": metrics,
    }
    write_json(directory / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("uni", "virchow2"), default="uni")
    parser.add_argument("--variant", choices=("pooling", "topk"), default="pooling")
    parser.add_argument("--run-name", help="A new directory name inside .demo; existing runs are never overwritten")
    parser.add_argument("--execute", action="store_true", help="Run the exported FLARE job on CPU")
    args = parser.parse_args()
    name = args.run_name or f"{args.encoder}_{args.variant}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        parser.error("--run-name must contain only letters, numbers, underscores or hyphens")
    if importlib.metadata.version("nvflare") != "2.7.2":
        raise RuntimeError("This example was validated with nvflare==2.7.2; use the documented environment")
    setup = load_setup(ROOT, args.encoder, args.variant)
    directory = inside(inside(ROOT, ".demo"), name)
    directory.mkdir(parents=True, exist_ok=False)
    temporary = create_runtime_tmp(ROOT, directory)
    os.environ["TMPDIR"] = str(temporary)
    tempfile.tempdir = str(temporary)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    import torch

    torch.set_num_threads(1)
    runtime = create_inputs(directory, setup)
    previous = Path.cwd()
    try:
        os.chdir(DEMO_CODE)
        job, workspace = export_job(directory, runtime)
        print(f"[FABRIC demo] Exported CPU job using only fictional inputs: {job}", flush=True)
        if not args.execute:
            print("No training started. Re-run with --execute and a new run name to train.", flush=True)
            return
        launch_simulator(job, workspace, directory / "simulator.log")
        verify_and_evaluate(directory, runtime, workspace)
        print(
            f"[FABRIC demo] Completed: 2 rounds, 6 client tasks, matching FLARE checkpoint.\n"
            f"Summary: {directory / 'summary.json'}\nSynthetic metrics are not study results.",
            flush=True,
        )
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    main()
