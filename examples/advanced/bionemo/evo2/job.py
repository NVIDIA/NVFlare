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
"""Run federated Evo2 LoRA fine-tuning for splice-site classification."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
import time
from pathlib import Path

import evo2_adapter_checkpoint as adapter_checkpoint
import evo2_runtime
import provenance
from evo2_aggregator import ExactSchemaFedAvgAggregator
from evo2_persistor import CPUTrainablePTFileModelPersistor

from nvflare.app_common.app_constant import DefaultCheckpointFileName
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.recipe import SimEnv, set_per_site_config

EXTERNAL_PROCESS_TIMEOUT = 3600
TENSOR_TRANSFER_TIMEOUT = 1800
TRAINING_LOCK_FILENAME = ".evo2_training.lock"


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--manifest", default=None, help="Defaults to DATA_DIR/manifest.json")
    parser.add_argument("--initial-checkpoint", default="./models/evo2_lora_init.pt")
    parser.add_argument("--base-checkpoint", default="./models/evo2_1b_bf16_mbridge")
    parser.add_argument("--classifier-file", default=None)
    parser.add_argument("--workspace", default="/tmp/nvflare/evo2_splice_fedavg")
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--gpu", default="[0]", help='Simulator GPU config, for example "[0]"')
    parser.add_argument("--backend", choices=("bionemo", "mock"), default="bionemo")
    parser.add_argument("--local-steps", type=int, default=20)
    parser.add_argument("--seq-length", type=int, default=600)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--min-learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lora-dim", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora-target-modules",
        default="linear_qkv,linear_proj,linear_fc1,linear_fc2,dense_projection,dense",
    )
    parser.add_argument("--mock-delta", type=float, default=0.01)
    return parser


def _parse_gpu_string(gpu_string: str) -> list[str]:
    if not gpu_string or not gpu_string.strip():
        return []
    normalized = gpu_string.replace(" ", "")
    return [normalized] if re.fullmatch(r"\[[0-9]+\]", normalized) else []


def _load_manifest(path: str | os.PathLike[str]) -> dict:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Dataset manifest not found: {manifest_path}. Run prepare_data.py before launching the job."
        )
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("format_version") != 2:
        raise ValueError(f"Unsupported manifest format_version in {manifest_path}: {manifest.get('format_version')!r}")
    if manifest.get("audit", {}).get("status") != "passed":
        raise ValueError(f"Dataset manifest {manifest_path} does not contain a passed leakage audit.")
    return manifest


def _resolve_file(data_dir: Path, relative_path: str, label: str) -> Path:
    path = (data_dir / relative_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return path


def _validate_manifest_file_identities(data_dir: Path, manifest: dict) -> dict:
    """Verify that every audited JSONL output still matches the manifest."""

    try:
        files = manifest["files"]
        counts = manifest["counts"]
        identities = manifest["file_identities"]
        site_files = files["sites"]
        site_counts = counts["sites"]
        site_identities = identities["sites"]
        scalar_entries = {
            "validation": (files["validation"], int(counts["validation"])),
            "test": (files["test"], int(counts["test"])),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Dataset manifest is missing valid file paths, counts, or content identities.") from exc

    if not isinstance(site_files, dict) or not isinstance(site_counts, dict) or not isinstance(site_identities, dict):
        raise ValueError("Dataset manifest site files, counts, and content identities must be dictionaries.")
    if set(site_files) != set(site_counts) or set(site_files) != set(site_identities):
        raise ValueError("Dataset manifest site file, count, and content-identity keys do not match.")

    observed = {"sites": {}}

    def verify(name: str, relative_path: str, expected_rows: int, expected_identity: dict) -> dict:
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"Dataset manifest path for {name} must be a non-empty string.")
        if not isinstance(expected_identity, dict) or set(expected_identity) != {"sha256", "bytes", "rows"}:
            raise ValueError(f"Dataset manifest content identity for {name} is malformed.")
        if expected_identity.get("rows") != expected_rows:
            raise ValueError(
                f"Dataset manifest row count and content identity disagree for {name}: "
                f"{expected_rows} != {expected_identity.get('rows')!r}."
            )
        path = _resolve_file(data_dir, relative_path, name)
        actual = provenance.jsonl_identity(path, expected_rows=expected_rows, label=name)
        actual_payload = {field: actual[field] for field in ("sha256", "bytes", "rows")}
        if actual_payload != expected_identity:
            raise ValueError(
                f"Dataset file {name} no longer matches its audited manifest identity: "
                f"expected {expected_identity}, observed {actual_payload}. Run prepare_data.py again."
            )
        return actual

    for name, (relative_path, expected_rows) in scalar_entries.items():
        if name not in identities:
            raise ValueError(f"Dataset manifest is missing the {name} content identity.")
        observed[name] = verify(name, relative_path, expected_rows, identities[name])
    for site_name in sorted(site_files):
        try:
            expected_rows = int(site_counts[site_name]["count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Dataset manifest has an invalid row count for {site_name}.") from exc
        observed["sites"][site_name] = verify(
            f"{site_name} training",
            site_files[site_name],
            expected_rows,
            site_identities[site_name],
        )
    return observed


def _site_plan(args: argparse.Namespace, manifest: dict) -> list[dict]:
    if args.num_clients < 2:
        raise ValueError("Federated Evo2 training requires at least two clients.")
    data_dir = Path(args.data_dir).resolve()
    available_sites = manifest.get("files", {}).get("sites", {})
    counts = manifest.get("counts", {}).get("sites", {})
    selected_names = [f"site-{index}" for index in range(1, args.num_clients + 1)]
    missing = [name for name in selected_names if name not in available_sites]
    if missing:
        raise ValueError(f"Manifest does not contain requested sites: {missing}")

    plan = []
    for index, name in enumerate(selected_names, start=1):
        sample_count = int(counts[name]["count"])
        train_file = _resolve_file(data_dir, available_sites[name], f"{name} training")
        plan.append(
            {
                "client_name": name,
                "train_file": train_file,
                "train_identity": provenance.jsonl_identity(
                    train_file,
                    expected_rows=sample_count,
                    label=f"{name} training",
                ),
                "sample_count": sample_count,
                "mock_delta": args.mock_delta * index,
            }
        )
    return plan


def _build_train_args(
    args: argparse.Namespace,
    *,
    train_file: Path,
    validation_file: Path,
    site_name: str,
    sample_count: int,
    mock_delta: float,
) -> str:
    values = [
        "--backend",
        args.backend,
        "--train-file",
        str(train_file),
        "--validation-file",
        str(validation_file),
        "--base-checkpoint",
        str(Path(args.base_checkpoint).resolve()),
        "--work-dir",
        str(Path(args.workspace).resolve() / "client_work" / site_name),
        "--local-steps",
        str(args.local_steps),
        "--sample-count",
        str(sample_count),
        "--seq-length",
        str(args.seq_length),
        "--micro-batch-size",
        str(args.micro_batch_size),
        "--global-batch-size",
        str(args.global_batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--min-learning-rate",
        str(args.min_learning_rate),
        "--warmup-iters",
        str(args.warmup_iters),
        "--eval-iters",
        str(args.eval_iters),
        "--seed",
        str(args.seed),
        "--lora-dim",
        str(args.lora_dim),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--lora-target-modules",
        args.lora_target_modules,
        "--mock-delta",
        str(mock_delta),
    ]
    if args.classifier_file:
        values.extend(["--classifier-file", str(Path(args.classifier_file).resolve())])
    return shlex.join(values)


def _configure_timeouts(recipe: FedAvgRecipe, clients: list[str]) -> None:
    recipe.add_client_config(
        {
            "EXTERNAL_PRE_INIT_TIMEOUT": EXTERNAL_PROCESS_TIMEOUT,
            "get_task_timeout": EXTERNAL_PROCESS_TIMEOUT,
            "submit_task_result_timeout": EXTERNAL_PROCESS_TIMEOUT,
            "tensor_min_download_timeout": TENSOR_TRANSFER_TIMEOUT,
            "max_resends": 3,
        },
        clients=clients,
    )
    recipe.add_server_config(
        {
            "streaming_per_request_timeout": TENSOR_TRANSFER_TIMEOUT,
            "tensor_min_download_timeout": TENSOR_TRANSFER_TIMEOUT,
        }
    )


def _build_external_command(args: argparse.Namespace) -> str:
    command = [
        sys.executable,
        "-u",
        "custom/sequential_launcher.py",
        "--lock-file",
        str(Path(args.workspace).resolve() / TRAINING_LOCK_FILENAME),
        "--",
    ]
    if args.backend == "bionemo":
        command.extend(["torchrun", "--standalone", "--nproc_per_node=1"])
    else:
        command.extend([sys.executable, "-u"])
    return shlex.join(command)


def collect_training_input_provenance(
    args: argparse.Namespace,
    manifest: dict,
    plan: list[dict],
    validation_file: Path,
) -> dict:
    """Capture the exact data, backbone, and classifier used by a run."""

    inputs = {
        "train_files": {entry["client_name"]: entry["train_identity"] for entry in plan},
        "validation_file": provenance.jsonl_identity(
            validation_file,
            expected_rows=int(manifest["counts"]["validation"]),
            label="validation",
        ),
        "base_checkpoint": None,
        "classifier_file": None,
    }
    if args.backend == "bionemo":
        inputs["base_checkpoint"] = provenance.directory_identity(args.base_checkpoint)
        classifier_file = evo2_runtime.resolve_classifier_path(args.classifier_file)
        inputs["classifier_file"] = provenance.file_identity(classifier_file)
    return inputs


def validate_inputs(args: argparse.Namespace) -> tuple[dict, list[dict], Path, dict]:
    if args.num_rounds <= 0 or args.local_steps <= 0:
        raise ValueError("--num-rounds and --local-steps must be positive.")
    if not _parse_gpu_string(args.gpu):
        raise ValueError('--gpu must name exactly one GPU in simulator bracket syntax, for example "[0]".')

    initial_checkpoint = Path(args.initial_checkpoint).resolve()
    if not initial_checkpoint.is_file():
        raise FileNotFoundError(
            f"Initial trainable checkpoint not found: {initial_checkpoint}. Run prepare_initial_model.py first."
        )
    if args.backend == "bionemo" and not Path(args.base_checkpoint).is_dir():
        raise FileNotFoundError(f"Megatron Bridge base checkpoint not found: {Path(args.base_checkpoint).resolve()}")

    data_dir = Path(args.data_dir).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else data_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)
    manifest_identities = _validate_manifest_file_identities(data_dir, manifest)
    validation_file = _resolve_file(data_dir, manifest["files"]["validation"], "validation")
    plan = _site_plan(args, manifest)
    training_inputs = collect_training_input_provenance(args, manifest, plan, validation_file)

    checkpoint_metadata = adapter_checkpoint.load_nvflare_checkpoint_metadata(initial_checkpoint)
    provenance.validate_initialization_metadata(
        checkpoint_metadata,
        backend=args.backend,
        seed=args.seed,
        seq_length=args.seq_length,
        lora_dim=args.lora_dim,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=evo2_runtime.parse_lora_targets(args.lora_target_modules),
        base_checkpoint_identity=training_inputs["base_checkpoint"],
        classifier_file_identity=training_inputs["classifier_file"],
        exchange_dtype=adapter_checkpoint.EXCHANGE_DTYPE_NAME,
        initialization_data_identity=(manifest_identities["validation"] if args.backend == "bionemo" else None),
    )
    return manifest, plan, validation_file, training_inputs


def create_recipe(args: argparse.Namespace, plan: list[dict], validation_file: Path) -> FedAvgRecipe:
    clients = [entry["client_name"] for entry in plan]
    per_site_config = {
        entry["client_name"]: {
            "train_args": _build_train_args(
                args,
                train_file=entry["train_file"],
                validation_file=validation_file,
                site_name=entry["client_name"],
                sample_count=entry["sample_count"],
                mock_delta=entry["mock_delta"],
            )
        }
        for entry in plan
    }
    aggregation_weights = {entry["client_name"]: float(entry["sample_count"]) for entry in plan}
    model_persistor = CPUTrainablePTFileModelPersistor(
        source_ckpt_file_full_name=str(Path(args.initial_checkpoint).resolve()),
        allow_numpy_conversion=False,
    )
    aggregator = ExactSchemaFedAvgAggregator(
        schema_checkpoint=str(Path(args.initial_checkpoint).resolve()),
        aggregation_weights=aggregation_weights,
    )
    recipe = FedAvgRecipe(
        name="evo2-splice-fedavg-lora",
        min_clients=len(clients),
        num_rounds=args.num_rounds,
        model_persistor=model_persistor,
        aggregator=aggregator,
        train_script="client.py",
        launch_external_process=True,
        command=_build_external_command(args),
        server_expected_format=ExchangeFormat.PYTORCH,
        params_transfer_type=TransferType.DIFF,
        aggregation_weights=aggregation_weights,
        key_metric="",
        launch_once=False,
        shutdown_timeout=60,
        server_memory_gc_rounds=1,
        client_memory_gc_rounds=1,
        cuda_empty_cache=True,
    )
    set_per_site_config(recipe, per_site_config)
    recipe.add_client_file("evo2_adapter_checkpoint.py", clients=clients)
    recipe.add_client_file("evo2_runtime.py", clients=clients)
    recipe.add_client_file("sequential_launcher.py", clients=clients)
    recipe.add_server_file("evo2_adapter_checkpoint.py")
    recipe.add_server_file("evo2_aggregator.py")
    recipe.add_server_file("evo2_persistor.py")
    recipe.add_server_file("provenance.py")
    _configure_timeouts(recipe, clients)
    return recipe


def create_sim_env(args: argparse.Namespace, plan: list[dict]) -> SimEnv:
    return SimEnv(
        clients=[entry["client_name"] for entry in plan],
        num_threads=1,
        gpu_config=args.gpu,
        workspace_root=str(Path(args.workspace).resolve()),
    )


def _load_global_checkpoint_metadata(path: Path) -> dict:
    try:
        return adapter_checkpoint.load_nvflare_checkpoint_metadata(path)
    except Exception as exc:
        raise RuntimeError(f"Could not load global checkpoint metadata from {path}: {exc}") from exc


def collect_run_summary(
    args: argparse.Namespace,
    plan: list[dict],
    result_dir: str | os.PathLike[str],
    started_at: float,
    manifest: dict,
    training_inputs: dict,
) -> dict:
    """Validate the final global checkpoint and record the core run configuration."""

    result_dir = Path(result_dir).resolve()
    initial_checkpoint = Path(args.initial_checkpoint).resolve()
    initial_state = adapter_checkpoint.load_nvflare_checkpoint(initial_checkpoint)
    schema = adapter_checkpoint.ValidatedTrainableStateSchema(initial_state)
    initial_metadata = provenance.resolve_initialization_metadata(
        adapter_checkpoint.load_nvflare_checkpoint_metadata(initial_checkpoint)
    )
    global_checkpoint = result_dir / "server" / "simulate_job" / "app_server" / DefaultCheckpointFileName.GLOBAL_MODEL
    if not global_checkpoint.is_file() or global_checkpoint.stat().st_mtime < started_at:
        raise RuntimeError(
            f"Simulation did not create a fresh global checkpoint at {global_checkpoint}. Check the server/client logs."
        )

    final_state = adapter_checkpoint.load_nvflare_checkpoint(global_checkpoint, schema=schema)
    final_metadata = _load_global_checkpoint_metadata(global_checkpoint)
    if final_metadata.get("current_round") != args.num_rounds - 1:
        raise RuntimeError(
            f"Global checkpoint {global_checkpoint} is not from the final round: expected "
            f"meta_props.current_round={args.num_rounds - 1}, received {final_metadata.get('current_round')!r}."
        )
    if final_metadata.get("nr_aggregated") != len(plan):
        raise RuntimeError(
            f"Global checkpoint {global_checkpoint} has incomplete aggregation metadata: expected "
            f"meta_props.nr_aggregated={len(plan)}, received {final_metadata.get('nr_aggregated')!r}."
        )
    if final_metadata.get("initialization") != initial_metadata:
        raise RuntimeError(f"Global checkpoint {global_checkpoint} does not preserve initialization metadata.")

    changed_tensors = sum(not initial_state[name].equal(final_state[name]) for name in initial_state)
    manifest_path = Path(args.manifest).resolve() if args.manifest else Path(args.data_dir).resolve() / "manifest.json"
    summary = {
        "backend": args.backend,
        "exchange_dtype": adapter_checkpoint.EXCHANGE_DTYPE_NAME,
        "num_clients": len(plan),
        "num_rounds": args.num_rounds,
        "local_steps": args.local_steps,
        "global_checkpoint": str(global_checkpoint),
        "global_checkpoint_sha256": provenance.sha256_file(global_checkpoint),
        "global_checkpoint_round": final_metadata["current_round"],
        "global_checkpoint_contributors": final_metadata["nr_aggregated"],
        "trainable_tensors_changed": changed_tensors,
        "initial_checkpoint": str(initial_checkpoint),
        "initial_checkpoint_sha256": provenance.sha256_file(initial_checkpoint),
        "initialization_metadata": initial_metadata,
        "aggregation_weights": {entry["client_name"]: entry["sample_count"] for entry in plan},
        "dataset": {
            "manifest": str(manifest_path),
            "manifest_sha256": provenance.sha256_file(manifest_path),
            "source": manifest.get("source"),
            "settings": manifest.get("settings"),
            "audit": manifest.get("audit"),
        },
        "training_inputs": training_inputs,
        "configuration": {
            "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
            "classifier_file": str(Path(args.classifier_file).resolve()) if args.classifier_file else None,
            "gpu": args.gpu,
            "seq_length": args.seq_length,
            "micro_batch_size": args.micro_batch_size,
            "global_batch_size": args.global_batch_size,
            "learning_rate": args.learning_rate,
            "min_learning_rate": args.min_learning_rate,
            "warmup_iters": args.warmup_iters,
            "eval_iters": args.eval_iters,
            "seed": args.seed,
            "lora_dim": args.lora_dim,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": list(evo2_runtime.parse_lora_targets(args.lora_target_modules)),
        },
    }
    summary_path = Path(args.workspace).resolve() / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary["summary_path"] = str(summary_path)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    args = define_parser().parse_args(argv)
    manifest, plan, validation_file, training_inputs = validate_inputs(args)
    recipe = create_recipe(args, plan, validation_file)
    started_at = time.time()
    run = recipe.execute(create_sim_env(args, plan))
    result_dir = run.get_result()
    summary = collect_run_summary(args, plan, result_dir, started_at, manifest, training_inputs)
    print(f"Result can be found in: {result_dir}")
    print(f"Global checkpoint: {summary['global_checkpoint']}")
    print(f"Run summary: {summary['summary_path']}")


if __name__ == "__main__":
    main()
