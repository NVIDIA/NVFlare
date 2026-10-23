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

"""Plan by default; --execute runs the selected FABRIC model via FLARE."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from fabric_common import (
    EXPERIMENTS,
    SITES,
    VARIANTS,
    check_environment,
    check_inputs,
    cpu_state,
    inside,
    install_feature_reader,
    load_core,
    load_data_config,
    load_setup,
    make_spec,
    read_json,
    remap_feature_paths,
    sha256_file,
    state_digest,
    training_args,
    write_json,
)
from fabric_progress import Progress, loader_progress
from fabric_resume import atomic_save, inspect_fold, prepare_resume, run_lock, validate_resume_run
from fabric_runtime import create_runtime_tmp, launch_simulator

ROOT = Path(__file__).resolve().parents[1]
CODE = Path(__file__).resolve().parent


def build_recipe(runtime_path: Path, setup: dict):
    from fabric_aggregator import FabricFedAvg

    from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
    from nvflare.client.config import ExchangeFormat, TransferType

    runtime = read_json(runtime_path)
    settings = setup["settings"]
    model_args = {
        "model_name": setup["model_name"],
        "input_dim": setup["input_dim"],
        "n_classes": 2,
        "embed_dim": settings["embed_dim"],
        "attn_dim": settings["attn_dim"],
        "dropout": settings["dropout"],
        "dtfd_pseudo_bags": settings["dtfd_pseudo_bags"],
    }
    if setup.get("variant") == "topk":
        model_args.update(
            dtfd_top_k=setup["model_options"]["top_k"], dtfd_eval_group_seed=setup["model_options"]["eval_group_seed"]
        )
    recipe = FedAvgRecipe(
        name=f"fabric_{setup['experiment']}_fold_{runtime['fold']}",
        model={"class_path": "core.mil_models.MILModel", "args": model_args},
        initial_ckpt=runtime.get("resume_checkpoint", str(runtime_path.parent / "initial_model.pt")),
        min_clients=len(SITES),
        num_rounds=settings["rounds"] - runtime.get("resume_offset", 0),
        train_script="fabric_client.py",
        train_args=shlex.join(["--runtime", str(runtime_path)]),
        aggregator=FabricFedAvg(runtime_path=str(runtime_path)),
        launch_external_process=True,
        launch_once=False,
        command=shlex.join([sys.executable, "-B", "-u"]),
        server_expected_format=ExchangeFormat.PYTORCH,
        params_transfer_type=TransferType.FULL,
        per_site_config={site: {} for site in SITES},
        key_metric="",
        stop_cond=None,
        patience=None,  # Disable best-model selection as well.
    )
    # Explicitly bundle the local dependency closure for every app. No imports
    # or job resources point back to the original project source tree.
    for source in sorted(CODE.rglob("*.py")):
        relative = source.relative_to(CODE)
        destination = None if relative.parent == Path(".") else str(relative.parent)
        for target in ("server", *SITES):
            recipe.job.add_file_to(str(source), target, dest_dir=destination)
    return recipe


def run_fold(
    root: Path,
    setup: dict,
    fold: int,
    output_dir: Path,
    core,
    manifest_root: Path,
    prefix_map: dict[str, str],
    *,
    resume: bool = False,
) -> None:
    import pandas as pd
    import torch

    from nvflare.recipe import SimEnv

    fold_dir = inside(inside(root, "results"), output_dir / f"fold_{fold}")
    recovery = inspect_fold(fold_dir, setup) if resume else {"action": "start", "completed_rounds": 0}
    if recovery["action"] == "complete":
        print(f"[skip completed] {setup['encoder']} fold={fold}", flush=True)
        return
    fold_dir.mkdir(parents=True, exist_ok=resume)
    manifests = manifest_root / setup["experiment"] / f"fold_{fold}"
    if recovery["action"] == "start":
        for source in sorted(manifests.glob("*.csv")):
            destination = fold_dir / source.name
            if prefix_map:
                remap_feature_paths(pd.read_csv(source), prefix_map).to_csv(destination, index=False)
            else:
                shutil.copy2(source, destination)
    client_dfs = {site: pd.read_csv(fold_dir / f"client_{site}_train_manifest.csv") for site in SITES}
    test_df = pd.read_csv(fold_dir / "global_test_manifest.csv")
    train_args = training_args(root, setup, output_dir, manifest_root)
    if not torch.cuda.is_available():
        raise RuntimeError("Historical CUDA setup required; refusing silent CPU fallback")
    device = core.client_training.resolve_device(train_args)

    if recovery["action"] == "start":
        # NEW random initialization; unchanged for folds not started previously.
        core.training.set_seed(train_args.seed + fold * 1000)
        model = core.training.build_model(setup["model_name"], setup["input_dim"], train_args)
        initial_state = core.aggregation.copy_state_dict(model.state_dict())
        torch.save({"model": initial_state}, fold_dir / "initial_model.pt")
        model_config = {
            "model": setup["model_name"],
            "encoder": setup["encoder"],
            "input_dim": setup["input_dim"],
            "protocol": (
                "dtfd_topk_epoch5_fedavg_nvflare" if setup.get("variant") == "topk" else "dtfd_epoch5_fedavg_nvflare"
            ),
            "settings": vars(train_args),
        }
        runtime = {
            "root": str(root),
            "experiment": setup["experiment"],
            "fold": fold,
            "variant": setup.get("variant", "pooling"),
            "top_k": setup.get("model_options", {}).get("top_k"),
            "fold_dir": str(fold_dir),
            "model_config": model_config,
            "manifest_root": str(manifest_root),
            "client_patients": {site: len(df) for site, df in client_dfs.items()},
            "initial_state_sha256": state_digest(initial_state),
            "state_spec": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in initial_state.items()
            },
            "manifest_sha256": {p.name: sha256_file(p) for p in fold_dir.glob("*.csv")},
        }
        runtime_path = fold_dir / "runtime.json"
        write_json(runtime_path, runtime)
        work_dir = fold_dir
        del model, initial_state
    else:
        runtime_path, work_dir = prepare_resume(fold_dir, recovery)
        print(
            f"[resume] {setup['encoder']} fold={fold} completed_rounds={recovery['completed_rounds']} "
            f"action={recovery['action']}",
            flush=True,
        )

    start_stamp = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    # ScriptRunner's external command uses custom/<script>, so use a relative
    # script name and create/export/launch with CODE as the working directory.
    if recovery["completed_rounds"] < train_args.rounds:
        previous_cwd = Path.cwd()
        try:
            os.chdir(CODE)
            recipe = build_recipe(runtime_path, setup)
            env = SimEnv(clients=list(SITES), num_threads=1, workspace_root=str(work_dir / "nvflare_workspace"))
            recipe.export(str(work_dir / "exported_job"), env=env)
            workspace = inside(fold_dir, Path(env.workspace_root) / recipe.job.name)
            launch_simulator(work_dir / "exported_job" / recipe.job.name, workspace, work_dir / "simulator.log")
        finally:
            os.chdir(previous_cwd)
    else:
        # All rounds persisted, but evaluation was interrupted: evaluate only.
        framework_checkpoint = Path(recovery["framework_checkpoint"])
        workspace = framework_checkpoint.parents[3]
        if not (fold_dir / "global_model_round_final.pt").is_file():
            saved = torch.load(recovery["checkpoint"], map_location="cpu", weights_only=True)
            state = cpu_state(saved.get("model_state_dict", saved.get("model", saved)))
            runtime = read_json(runtime_path)
            atomic_save(
                {
                    "model_state_dict": state,
                    "config": runtime["model_config"],
                    "fold": fold,
                    "condition": setup["model_key"],
                    "completed_rounds": train_args.rounds,
                    "state_sha256": recovery["state_sha256"],
                },
                fold_dir / "global_model_round_final.pt",
            )
    flare_seconds = time.perf_counter() - start

    # SimEnv.get_status() is always None, not a success indicator. Require all
    # five audited aggregations AND the framework's persisted final weights.
    round_rows = []
    for round_number in range(1, train_args.rounds + 1):
        audit_path = fold_dir / "server_audit" / f"round_{round_number}.json"
        if not audit_path.is_file():
            raise RuntimeError(
                f"FLARE returned without completing aggregation round {round_number}. "
                f"Inspect {fold_dir / 'simulator.log'} and {workspace}; "
                "missing round records will not be created or skipped."
            )
        audit = read_json(audit_path)
        if audit["round"] != round_number or audit["sites_in_aggregation_order"] != list(SITES):
            raise RuntimeError("FLARE round audit is incomplete")
        round_rows.extend(audit["local_logs"])
    if len(round_rows) != len(SITES) * train_args.rounds:
        raise RuntimeError("Not all 15 local updates completed")
    checkpoint = torch.load(fold_dir / "global_model_round_final.pt", map_location="cpu", weights_only=True)
    if checkpoint["completed_rounds"] != train_args.rounds:
        raise RuntimeError("Final round missing")
    final_state = cpu_state(checkpoint["model_state_dict"])
    final_digest = state_digest(final_state)
    if final_digest != checkpoint["state_sha256"] or final_digest != audit["output_state_sha256"]:
        raise RuntimeError("Final checkpoint differs from the audited aggregate")
    framework_files = (
        [Path(recovery["framework_checkpoint"])]
        if recovery["action"] == "evaluate"
        else list(workspace.rglob("FL_global_model.pt"))
    )
    if len(framework_files) != 1:
        raise RuntimeError(f"Expected exactly one FLARE final checkpoint, found {len(framework_files)}")
    persisted = torch.load(framework_files[0], map_location="cpu", weights_only=True)
    persisted_state = cpu_state(persisted.get("model", persisted))
    if state_digest(persisted_state) != final_digest:
        raise RuntimeError("FLARE persisted model differs from the final FedAvg aggregate")

    evaluation_start = time.perf_counter()
    install_feature_reader(core, fold_dir / "evaluation_skipped_hdf5_rows.log")
    model = core.training.build_model(setup["model_name"], setup["input_dim"], train_args).to(device)
    model.load_state_dict(final_state, strict=True)
    with loader_progress(
        core.training,
        epochs=1,
        label=f"{setup['encoder']} fold_index={fold} test_evaluation",
        log_path=fold_dir / "evaluation_progress.log",
    ):
        metrics, predictions = core.training.evaluate_model(
            model,
            test_df,
            pd.concat(client_dfs.values(), ignore_index=True),
            train_args,
            seed=train_args.seed + fold,
            device=device,
        )
    predictions.insert(0, "fold", fold)
    predictions.insert(1, "test_set", "global_cv_test")
    predictions.to_csv(fold_dir / "global_test_predictions.tsv", sep="\t", index=False)
    pd.DataFrame(round_rows).to_csv(fold_dir / "round_logs.tsv", sep="\t", index=False)
    spec = make_spec(core, root, setup, output_dir, manifest_root)
    seconds = time.perf_counter() - start
    row = core.client_training.result_row(
        core.experiment_utils.condition_for(spec),
        fold,
        output_dir,
        metrics,
        predictions,
        seconds,
    )
    row = core.experiment_utils.normalized_result_row(spec, row, fold, predictions, train_args.threshold, output_dir)
    pd.DataFrame([row]).to_csv(fold_dir / "fold_result.tsv", sep="\t", index=False)
    core.experiment_utils.write_fold_runtime(
        fold_dir, row, start_stamp, datetime.now(timezone.utc).isoformat(), seconds
    )
    write_json(
        fold_dir / "nvflare_run.json",
        {
            "completion_evidence": "5 server audits + 15 client updates + matching FLARE final checkpoint",
            "workspace": str(workspace),
            "framework_final_checkpoint": str(framework_files[0]),
            "final_state_sha256": final_digest,
            "nvflare_job_wall_seconds": flare_seconds,
            "evaluation_seconds": time.perf_counter() - evaluation_start,
            "client_training_seconds_sum": sum(float(item["seconds"]) for item in round_rows),
            "fold_wall_seconds": seconds,
            **(
                {
                    "resumed_from_round": recovery["completed_rounds"],
                    "timing_scope": "This invocation only; client_training_seconds_sum covers all completed rounds",
                }
                if recovery["action"] != "start"
                else {}
            ),
        },
    )
    core.experiment_utils.refresh_summaries(spec, train_args.threshold)
    print(
        f"[completed] {setup['encoder']} fold={fold} auc={metrics['auc']:.6f} "
        f"bacc={metrics['bacc']:.6f} acc={metrics['acc']:.6f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=[*EXPERIMENTS, "both"], default="both")
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        default="pooling",
        help="pooling: attention-weighted aggregation; topk: supervised high/low patch selection",
    )
    parser.add_argument(
        "--top-k", type=int, help="Top-k variant only: patches per high/low set per pseudo-bag (default: 1)"
    )
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    parser.add_argument("--run-name", help="Output directory name; existing runs require --resume")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run from audited round checkpoints; skip completed folds",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data_paths.local.json"),
        help="Private manifest root and optional feature path-prefix mapping",
    )
    parser.add_argument("--execute", action="store_true", help="Actually start FLARE and training")
    args = parser.parse_args()
    if len(set(args.folds)) != len(args.folds):
        parser.error("Duplicate fold numbers are not allowed")
    if args.top_k is not None and (args.variant != "topk" or args.top_k < 1):
        parser.error("--top-k must be positive and requires --variant topk")
    if args.resume and not args.run_name:
        parser.error("--resume requires the original --run-name")
    name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        parser.error("--run-name must contain only letters, numbers, underscores or hyphens")
    run_root = inside(inside(ROOT, "results"), name)
    experiments = EXPERIMENTS if args.experiment == "both" else (args.experiment,)
    setups = [load_setup(ROOT, name, args.variant, args.top_k) for name in experiments]
    data = load_data_config(ROOT, args.data_config)
    manifest_root = Path(data["manifest_root"])
    prefix_map = data["feature_path_prefix_map"]
    checks = {setup["experiment"]: check_inputs(setup, manifest_root, prefix_map) for setup in setups}
    if args.resume:
        validate_resume_run(ROOT, run_root, setups, args.folds, manifest_root)
        for setup in setups:
            for fold in args.folds:
                plan = inspect_fold(run_root / setup["experiment"] / f"fold_{fold}", setup)
                print(
                    f"[resume plan] {setup['encoder']} fold={fold}: {plan['action']}; "
                    f"completed_rounds={plan['completed_rounds']}/5",
                    flush=True,
                )
    print(
        json.dumps(
            {
                "mode": "execute" if args.execute else "plan_only",
                "experiments": list(experiments),
                "folds": args.folds,
                "variant": args.variant,
                "model_options": setups[0].get("model_options", {}),
                "rounds": 5,
                "local_epochs": 5,
                "sites": list(SITES),
                "inputs": checks,
                "manifests": str(manifest_root),
                "features": "external, read-only",
            },
            indent=2,
        )
    )
    if not args.execute:
        print("No training or FLARE job started. Use --execute only when ready.")
        return

    check_environment(ROOT)
    from check_static import perform_checks

    # Runtime is independent of the original project code/results directories.
    perform_checks(ROOT)
    run_root.mkdir(parents=True, exist_ok=args.resume)
    with run_lock(run_root):
        execute_run(args, run_root, setups, manifest_root, prefix_map)


def execute_run(args, run_root: Path, setups: list, manifest_root: Path, prefix_map: dict) -> None:
    # Keep provenance for the first invocation. Each resume records the recovery
    # wrapper versions separately, without rewriting any completed experiment.
    metadata_root = run_root
    if args.resume:
        history = run_root / "resume_history"
        history.mkdir(exist_ok=True)
        metadata_root = Path(tempfile.mkdtemp(prefix="attempt_", dir=history))
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["MPLCONFIGDIR"] = str(run_root / "runtime_cache/matplotlib")
    temporary_root = create_runtime_tmp(ROOT, metadata_root)
    os.environ["TMPDIR"] = str(temporary_root)
    tempfile.tempdir = str(temporary_root)
    write_json(
        metadata_root / "code_sha256.json",
        {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(CODE.rglob("*.py"))},
    )
    core = load_core()
    for setup in setups:
        output_dir = run_root / setup["experiment"]
        if args.resume:
            remaining = [
                fold for fold in args.folds if inspect_fold(output_dir / f"fold_{fold}", setup)["action"] != "complete"
            ]
            if not remaining:
                print(f"[skip completed] {setup['encoder']} selected folds already finished", flush=True)
                # Do not touch completed outputs unless summary generation itself
                # was interrupted after writing the last fold completion marker.
                summaries = ("fold_results.tsv", "mean_std_summary.tsv", "all_fold_predictions.tsv")
                newest_fold = max(
                    (output_dir / f"fold_{fold}" / "nvflare_run.json").stat().st_mtime for fold in args.folds
                )
                if any(
                    not (output_dir / n).is_file() or (output_dir / n).stat().st_mtime < newest_fold for n in summaries
                ):
                    core.experiment_utils.refresh_summaries(
                        make_spec(core, ROOT, setup, output_dir, manifest_root), setup["settings"]["threshold"]
                    )
                continue
        else:
            remaining = args.folds
        output_dir.mkdir(exist_ok=args.resume)
        config = {
            "schema_version": 1,
            "experiment": setup["experiment"],
            "model_name": setup["model_name"],
            "encoder": setup["encoder"],
            "variant": setup.get("variant", "pooling"),
            "model_options": setup.get("model_options", {}),
            "input_dim": setup["input_dim"],
            "settings": vars(training_args(ROOT, setup, output_dir, manifest_root)),
            "folds": args.folds,
            "source_fold_manifests": str(manifest_root),
            "execution_framework": "NVIDIA FLARE 2.7.2",
        }
        if not (args.resume and (output_dir / "run_config.json").is_file()):
            write_json(output_dir / "run_config.json", config)
        progress = Progress(
            len(remaining), f"{setup['encoder']} selected_folds", output_dir / "logs/progress.log", unit="fold"
        )
        progress.show("starting", force=True)
        for fold in remaining:
            log_path = output_dir / "logs" / f"{setup['model_key']}_fold{fold}.log"
            with core.experiment_utils.tee_to_log(log_path):
                progress.show(f"starting_fold_index={fold}", force=True)
                run_fold(ROOT, setup, fold, output_dir, core, manifest_root, prefix_map, resume=args.resume)
                progress.advance(f"completed_fold_index={fold}", force=True)
    write_json(
        run_root / "completed.json",
        {"experiments": [s["experiment"] for s in setups], "folds": args.folds, "variant": args.variant},
    )


if __name__ == "__main__":
    main()
