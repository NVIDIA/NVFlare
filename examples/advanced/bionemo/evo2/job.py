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
"""Run federated or baseline Evo2 parameter-efficient fine-tuning."""

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
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.recipe import SimEnv, set_per_site_config

EXTERNAL_PROCESS_TIMEOUT = 3600
TENSOR_TRANSFER_TIMEOUT = 1800
TRAINING_LOCK_FILENAME = ".evo2_training.lock"


class Evo2FedAvgRecipe(FedAvgRecipe):
    """FedAvg recipe that supports numbering a validated continuation run."""

    def __init__(self, *, start_round: int = 0, **kwargs):
        self.start_round = start_round
        super().__init__(**kwargs)

    def _create_controller(self, persistor_id: str, model_params, model_aggregator) -> FedAvg:
        controller = super()._create_controller(persistor_id, model_params, model_aggregator)
        controller.start_round = self.start_round
        return controller


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--manifest", default=None, help="Defaults to DATA_DIR/manifest.json")
    parser.add_argument("--initial-checkpoint", default="./models/evo2_lora_init.pt")
    parser.add_argument("--base-checkpoint", default="./models/evo2_1b_bf16_mbridge")
    parser.add_argument("--classifier-file", default=None)
    parser.add_argument("--workspace", default="/tmp/nvflare/evo2_splice_fedavg")
    parser.add_argument(
        "--require-fresh-workspace",
        action="store_true",
        help="Fail before launch when the requested workspace path already exists",
    )
    parser.add_argument("--mode", choices=("fedavg", "local", "pooled"), default="fedavg")
    parser.add_argument("--site-index", type=int, default=1, help="Dataset site used by --mode local")
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument(
        "--start-round",
        type=int,
        default=0,
        help="Logical first round; a prior global checkpoint with a matching continuation signature is required",
    )
    parser.add_argument("--num-threads", type=int, default=1)
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
    parser.add_argument("--peft-mode", choices=("lora", "head-only"), default="lora")
    parser.add_argument("--lora-dim", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora-target-modules",
        default="linear_qkv,linear_proj,linear_fc1,linear_fc2,dense_projection,dense",
    )
    parser.add_argument(
        "--server-tensor-device",
        choices=("cpu",),
        default="cpu",
        help="Device for returned trainable tensors; the server-side state is CPU-only",
    )
    parser.add_argument("--mock-delta", type=float, default=0.01)
    parser.add_argument(
        "--persist-client-training-state",
        action="store_true",
        help="Resume each site's native BioNeMo optimizer, scheduler, RNG, and sampler state across rounds",
    )
    return parser


def _parse_gpu_string(gpu_string: str) -> list[str]:
    if not gpu_string or not gpu_string.strip():
        return []
    normalized = gpu_string.replace(" ", "")
    return [normalized] if re.fullmatch(r"\[[0-9]+\]", normalized) else []


def _validate_workspace_precondition(args: argparse.Namespace) -> None:
    if not getattr(args, "require_fresh_workspace", False):
        return
    requested_workspace = Path(args.workspace).expanduser()
    workspace = requested_workspace.resolve()
    if requested_workspace.is_symlink() or workspace.exists():
        raise FileExistsError(
            f"Fresh Evo2 workspace already exists: {workspace}. Choose a new path or omit "
            "--require-fresh-workspace for an intentional continuation workflow."
        )


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
            "pooled_train": (files["pooled_train"], int(counts["train"])),
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
    data_dir = Path(args.data_dir).resolve()
    files = manifest.get("files", {})
    counts = manifest.get("counts", {})
    if args.mode == "fedavg":
        available_sites = files.get("sites", {})
        if args.num_clients < 2:
            raise ValueError("--mode fedavg requires at least two clients.")
        selected_names = [f"site-{index}" for index in range(1, args.num_clients + 1)]
        missing = [name for name in selected_names if name not in available_sites]
        if missing:
            raise ValueError(f"Manifest does not contain requested sites: {missing}")
        plan = []
        for index, name in enumerate(selected_names, start=1):
            sample_count = int(counts["sites"][name]["count"])
            train_file = _resolve_file(data_dir, available_sites[name], f"{name} training")
            plan.append(
                {
                    "client_name": name,
                    "train_file": train_file,
                    "train_identity": provenance.jsonl_identity(
                        train_file, expected_rows=sample_count, label=f"{name} training"
                    ),
                    "sample_count": sample_count,
                    "mock_delta": args.mock_delta * index,
                }
            )
        return plan

    if args.mode == "local":
        source_name = f"site-{args.site_index}"
        relative_path = files.get("sites", {}).get(source_name)
        if relative_path is None:
            raise ValueError(f"Manifest does not contain requested local dataset {source_name!r}.")
        sample_count = int(counts["sites"][source_name]["count"])
        train_file = _resolve_file(data_dir, relative_path, f"{source_name} training")
        return [
            {
                "client_name": source_name,
                "train_file": train_file,
                "train_identity": provenance.jsonl_identity(
                    train_file, expected_rows=sample_count, label=f"{source_name} training"
                ),
                "sample_count": sample_count,
                "mock_delta": args.mock_delta * args.site_index,
            }
        ]

    relative_path = files.get("pooled_train")
    if not relative_path:
        raise ValueError("Manifest does not define files.pooled_train.")
    sample_count = int(counts["train"])
    train_file = _resolve_file(data_dir, relative_path, "pooled training")
    return [
        {
            "client_name": "site-1",
            "train_file": train_file,
            "train_identity": provenance.jsonl_identity(
                train_file, expected_rows=sample_count, label="pooled training"
            ),
            "sample_count": sample_count,
            "mock_delta": args.mock_delta,
        }
    ]


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
        "--peft-mode",
        args.peft_mode,
        "--lora-dim",
        str(args.lora_dim),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--lora-target-modules",
        args.lora_target_modules,
        "--server-tensor-device",
        args.server_tensor_device,
        "--mock-delta",
        str(mock_delta),
    ]
    if args.classifier_file:
        values.extend(["--classifier-file", str(Path(args.classifier_file).resolve())])
    if getattr(args, "persist_client_training_state", False):
        values.extend(
            [
                "--training-state-dir",
                str(Path(args.workspace).resolve() / "client_training_state" / site_name),
            ]
        )
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
    lock_file = Path(args.workspace).resolve() / TRAINING_LOCK_FILENAME
    command = [
        sys.executable,
        "-u",
        "custom/sequential_launcher.py",
        "--lock-file",
        str(lock_file),
        "--",
    ]
    if args.backend == "bionemo":
        command.extend(["torchrun", "--standalone", "--nproc_per_node=1"])
    else:
        command.extend([sys.executable, "-u"])
    return shlex.join(command)


def _portable_jsonl_identity(identity: dict, label: str) -> dict:
    if not isinstance(identity, dict):
        raise ValueError(f"{label} content identity must be a dictionary.")
    required = ("sha256", "bytes", "rows")
    missing = [name for name in required if name not in identity]
    if missing:
        raise ValueError(f"{label} content identity is missing fields: {missing}.")
    return {name: identity[name] for name in required}


def build_continuation_signature(
    args: argparse.Namespace, manifest: dict, plan: list[dict], training_inputs: dict
) -> dict:
    """Bind a stateless continuation to the exact federation and training protocol."""

    for name in ("source", "settings", "audit"):
        if not isinstance(manifest.get(name), dict):
            raise ValueError(f"Dataset manifest field {name!r} must be a dictionary.")
    train_files = training_inputs.get("train_files")
    if not isinstance(train_files, dict):
        raise ValueError("Training input provenance is missing train-file identities.")
    clients = []
    for entry in sorted(plan, key=lambda item: item["client_name"]):
        client_name = entry["client_name"]
        if client_name not in train_files:
            raise ValueError(f"Training input provenance is missing the identity for {client_name}.")
        clients.append(
            {
                "client_name": client_name,
                "sample_weight": float(entry["sample_count"]),
                "train_file": _portable_jsonl_identity(train_files[client_name], f"{client_name} training file"),
            }
        )
    payload = {
        "backend": args.backend,
        "backend_settings": {"mock_delta": args.mock_delta} if args.backend == "mock" else {},
        "mode": args.mode,
        "clients": clients,
        "dataset_manifest": {
            "format_version": manifest.get("format_version"),
            "source": manifest["source"],
            "settings": manifest["settings"],
            "audit": manifest["audit"],
            "validation_file": _portable_jsonl_identity(training_inputs.get("validation_file"), "validation file"),
        },
        "sampler_budget": {
            "seed": args.seed,
            "local_steps": args.local_steps,
            "micro_batch_size": args.micro_batch_size,
            "global_batch_size": args.global_batch_size,
        },
        "optimizer_schedule": {
            "learning_rate": args.learning_rate,
            "min_learning_rate": args.min_learning_rate,
            "warmup_iters": args.warmup_iters,
        },
    }
    return provenance.make_continuation_signature(payload)


def validate_inputs(args: argparse.Namespace) -> tuple[dict, list[dict], Path, dict, dict]:
    if args.num_rounds <= 0 or args.local_steps <= 0:
        raise ValueError("--num-rounds and --local-steps must be positive.")
    if args.start_round < 0:
        raise ValueError("--start-round must be non-negative.")
    if args.num_threads != 1:
        raise ValueError("This one-H100 example requires --num-threads 1 so sites run sequentially.")
    if not _parse_gpu_string(args.gpu):
        raise ValueError('--gpu must name exactly one GPU in simulator bracket syntax, for example "[0]".')
    if getattr(args, "persist_client_training_state", False) and args.backend != "bionemo":
        raise ValueError("--persist-client-training-state is supported only by the BioNeMo backend.")
    if getattr(args, "persist_client_training_state", False) and args.start_round > 0:
        raise ValueError(
            "--start-round greater than zero cannot be combined with --persist-client-training-state; "
            "start from round zero to build the complete site-local state chain."
        )
    initial_checkpoint = Path(args.initial_checkpoint).resolve()
    if not initial_checkpoint.is_file():
        raise FileNotFoundError(
            f"Initial trainable checkpoint not found: {initial_checkpoint}. Run prepare_initial_model.py first."
        )
    if args.backend == "bionemo" and not Path(args.base_checkpoint).is_dir():
        raise FileNotFoundError(f"Megatron Bridge base checkpoint not found: {Path(args.base_checkpoint).resolve()}")

    manifest_path = Path(args.manifest).resolve() if args.manifest else Path(args.data_dir).resolve() / "manifest.json"
    manifest = _load_manifest(manifest_path)
    manifest_identities = _validate_manifest_file_identities(Path(args.data_dir).resolve(), manifest)
    validation_file = _resolve_file(Path(args.data_dir).resolve(), manifest["files"]["validation"], "validation")
    plan = _site_plan(args, manifest)
    training_inputs = collect_training_input_provenance(args, manifest, plan, validation_file)
    continuation_signature = build_continuation_signature(args, manifest, plan, training_inputs)
    checkpoint_metadata = adapter_checkpoint.load_nvflare_checkpoint_metadata(initial_checkpoint)
    provenance.validate_initialization_metadata(
        checkpoint_metadata,
        backend=args.backend,
        peft_mode=args.peft_mode,
        seed=args.seed,
        seq_length=args.seq_length,
        lora_dim=args.lora_dim,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=evo2_runtime.parse_lora_targets(args.lora_target_modules),
        base_checkpoint_identity=training_inputs["base_checkpoint"],
        classifier_file_identity=training_inputs["classifier_file"],
        exchange_dtype=adapter_checkpoint.EXCHANGE_DTYPE_NAME,
        initialization_data_identity=manifest_identities["pooled_train"] if args.backend == "bionemo" else None,
    )
    if args.start_round > 0:
        source_round = checkpoint_metadata.get("current_round")
        source_contributors = checkpoint_metadata.get("nr_aggregated")
        if "initialization" not in checkpoint_metadata:
            raise ValueError("--start-round requires a prior global checkpoint with one initialization metadata layer.")
        if type(source_round) is not int or source_round != args.start_round - 1:
            raise ValueError(
                f"--start-round {args.start_round} requires an initial checkpoint from round "
                f"{args.start_round - 1}, received current_round={source_round!r}."
            )
        if type(source_contributors) is not int or source_contributors != len(plan):
            raise ValueError(
                f"Continuation checkpoint contributor metadata must equal the {len(plan)} planned clients, "
                f"received nr_aggregated={source_contributors!r}."
            )
        try:
            source_signature = provenance.validate_continuation_signature(
                checkpoint_metadata.get("continuation_signature"),
                context="Continuation checkpoint signature",
            )
        except ValueError as exc:
            raise ValueError(f"--start-round requires a valid continuation signature: {exc}") from exc
        if source_signature != continuation_signature:
            raise ValueError(
                "Continuation checkpoint signature does not match this federation or training protocol: "
                f"expected sha256={continuation_signature['sha256']}, observed sha256={source_signature['sha256']}."
            )
    return manifest, plan, validation_file, training_inputs, continuation_signature


def create_recipe(
    args: argparse.Namespace, plan: list[dict], validation_file: Path, continuation_signature: dict
) -> Evo2FedAvgRecipe:
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
        continuation_signature=continuation_signature,
    )
    command = _build_external_command(args)
    recipe = Evo2FedAvgRecipe(
        start_round=args.start_round,
        name=f"evo2-splice-{args.mode}-{args.peft_mode}",
        min_clients=len(clients),
        num_rounds=args.num_rounds,
        model_persistor=model_persistor,
        aggregator=aggregator,
        train_script="client.py",
        launch_external_process=True,
        command=command,
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
    recipe.add_client_file("provenance.py", clients=clients)
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


def collect_training_input_provenance(
    args: argparse.Namespace, manifest: dict, plan: list[dict], validation_file: Path
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


def _load_global_checkpoint_metadata(path: Path) -> dict:
    """Load and validate the metadata stored by the PyTorch model persistor."""

    try:
        return adapter_checkpoint.load_nvflare_checkpoint_metadata(path)
    except Exception as exc:
        raise RuntimeError(f"Could not load global checkpoint metadata from {path}: {exc}") from exc


def collect_run_summary(
    args: argparse.Namespace,
    plan: list[dict],
    result_dir: str | os.PathLike[str],
    started_at: float,
    manifest: dict | None = None,
    training_inputs: dict | None = None,
    continuation_signature: dict | None = None,
) -> dict:
    """Validate expected artifacts and write a compact resource/exchange report."""

    result_dir = Path(result_dir).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else Path(args.data_dir).resolve() / "manifest.json"
    if manifest is None:
        manifest = _load_manifest(manifest_path)
    if training_inputs is None:
        validation_file = _resolve_file(Path(args.data_dir).resolve(), manifest["files"]["validation"], "validation")
        training_inputs = collect_training_input_provenance(args, manifest, plan, validation_file)
    expected_continuation_signature = build_continuation_signature(args, manifest, plan, training_inputs)
    if continuation_signature is None:
        continuation_signature = expected_continuation_signature
    else:
        continuation_signature = provenance.validate_continuation_signature(
            continuation_signature,
            context="Run continuation signature",
        )
        if continuation_signature != expected_continuation_signature:
            raise RuntimeError("Run continuation signature does not match the supplied run configuration.")
    initial_checkpoint = Path(args.initial_checkpoint).resolve()
    initial_state = adapter_checkpoint.load_nvflare_checkpoint(initial_checkpoint)
    trainable_state_schema = adapter_checkpoint.ValidatedTrainableStateSchema(initial_state)
    persist_client_training_state = bool(getattr(args, "persist_client_training_state", False))
    initial_trainable_schema = (
        evo2_runtime._trainable_schema_from_validated_state(initial_state) if persist_client_training_state else None
    )
    initial_state_sha256 = (
        evo2_runtime._trainable_state_sha256_from_validated_state(initial_state)
        if persist_client_training_state
        else None
    )
    source_checkpoint_metadata = adapter_checkpoint.load_nvflare_checkpoint_metadata(initial_checkpoint)
    initial_metadata = provenance.resolve_initialization_metadata(source_checkpoint_metadata)
    global_checkpoint = result_dir / "server" / "simulate_job" / "app_server" / DefaultCheckpointFileName.GLOBAL_MODEL
    if not global_checkpoint.is_file() or global_checkpoint.stat().st_mtime < started_at:
        raise RuntimeError(
            f"Simulation did not create a fresh global checkpoint at {global_checkpoint}. Check the server/client logs."
        )
    try:
        adapter_checkpoint.load_nvflare_checkpoint(global_checkpoint, schema=trainable_state_schema)
    except Exception as exc:
        raise RuntimeError(f"Final global trainable checkpoint {global_checkpoint} is invalid: {exc}") from exc
    global_checkpoint_metadata = _load_global_checkpoint_metadata(global_checkpoint)
    expected_final_round = args.start_round + args.num_rounds - 1
    checkpoint_round = global_checkpoint_metadata.get("current_round")
    if checkpoint_round != expected_final_round:
        raise RuntimeError(
            f"Global checkpoint {global_checkpoint} is not from the final round: expected "
            f"meta_props.current_round={expected_final_round}, received {checkpoint_round!r}."
        )
    checkpoint_contributors = global_checkpoint_metadata.get("nr_aggregated")
    if checkpoint_contributors != len(plan):
        raise RuntimeError(
            f"Global checkpoint {global_checkpoint} has incomplete aggregation metadata: expected "
            f"meta_props.nr_aggregated={len(plan)}, received {checkpoint_contributors!r}."
        )
    if global_checkpoint_metadata.get("initialization") != initial_metadata:
        raise RuntimeError(
            f"Global checkpoint {global_checkpoint} does not preserve the exact initialization metadata."
        )
    try:
        global_continuation_signature = provenance.validate_continuation_signature(
            global_checkpoint_metadata.get("continuation_signature"),
            context=f"Global checkpoint {global_checkpoint} continuation signature",
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if global_continuation_signature != continuation_signature:
        raise RuntimeError(f"Global checkpoint {global_checkpoint} does not preserve the continuation signature.")

    global_round_checkpoints = []
    global_round_state_sha256 = {}
    logical_rounds = range(args.start_round, args.start_round + args.num_rounds)
    for round_index in logical_rounds:
        round_checkpoint = global_checkpoint.with_name(
            f"{global_checkpoint.stem}_round_{round_index:03d}{global_checkpoint.suffix}"
        )
        if not round_checkpoint.is_file() or round_checkpoint.stat().st_mtime < started_at:
            raise RuntimeError(
                f"Simulation did not create a fresh round {round_index} checkpoint at {round_checkpoint}."
            )
        try:
            round_state = adapter_checkpoint.load_nvflare_checkpoint(round_checkpoint, schema=trainable_state_schema)
            round_metadata = _load_global_checkpoint_metadata(round_checkpoint)
        except Exception as exc:
            raise RuntimeError(f"Global round checkpoint {round_checkpoint} is invalid: {exc}") from exc
        if round_metadata.get("current_round") != round_index:
            raise RuntimeError(
                f"Global round checkpoint {round_checkpoint} has meta_props.current_round="
                f"{round_metadata.get('current_round')!r}; expected {round_index}."
            )
        if round_metadata.get("nr_aggregated") != len(plan):
            raise RuntimeError(
                f"Global round checkpoint {round_checkpoint} has meta_props.nr_aggregated="
                f"{round_metadata.get('nr_aggregated')!r}; expected {len(plan)}."
            )
        if round_metadata.get("initialization") != initial_metadata:
            raise RuntimeError(
                f"Global round checkpoint {round_checkpoint} does not preserve the exact initialization metadata."
            )
        try:
            round_continuation_signature = provenance.validate_continuation_signature(
                round_metadata.get("continuation_signature"),
                context=f"Global round checkpoint {round_checkpoint} continuation signature",
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if round_continuation_signature != continuation_signature:
            raise RuntimeError(
                f"Global round checkpoint {round_checkpoint} does not preserve the continuation signature."
            )
        global_round_checkpoints.append(
            {
                "round": round_index,
                "path": str(round_checkpoint),
                "sha256": provenance.sha256_file(round_checkpoint),
                "mebibytes": adapter_checkpoint.state_dict_size_mb(round_state),
            }
        )
        if persist_client_training_state:
            global_round_state_sha256[round_index] = evo2_runtime._trainable_state_sha256_from_validated_state(
                round_state
            )

    metric_paths = [
        path
        for path in (Path(args.workspace).resolve() / "client_work").glob("*/*/round_metrics.json")
        if path.stat().st_mtime >= started_at
    ]
    expected_metrics = len(plan) * args.num_rounds
    if len(metric_paths) != expected_metrics:
        raise RuntimeError(
            f"Expected {expected_metrics} fresh client metric files, found {len(metric_paths)}. "
            "A client task may have failed; inspect the simulator logs."
        )
    round_metrics = []
    metric_pair_counts = {}
    for path in sorted(metric_paths):
        with path.open(encoding="utf-8") as file:
            metric = json.load(file)
        site_name = metric.get("site_name")
        round_index = metric.get("round")
        if not isinstance(site_name, str) or type(round_index) is not int:
            raise RuntimeError(
                f"Round metrics at {path} must contain a string site_name and integer round; "
                f"received site_name={site_name!r}, round={round_index!r}."
            )
        metric_pair = (site_name, round_index)
        metric_pair_counts[metric_pair] = metric_pair_counts.get(metric_pair, 0) + 1
        local_checkpoint_value = metric.get("local_checkpoint")
        if not isinstance(local_checkpoint_value, str) or not local_checkpoint_value:
            raise RuntimeError(f"Round metrics at {path} do not contain a local_checkpoint path.")
        local_checkpoint = Path(local_checkpoint_value)
        if not local_checkpoint.is_file() or local_checkpoint.stat().st_mtime < started_at:
            raise RuntimeError(f"Round metrics at {path} do not reference a fresh local trainable checkpoint.")
        try:
            local_state = adapter_checkpoint.load_nvflare_checkpoint(local_checkpoint, schema=trainable_state_schema)
            local_metadata = adapter_checkpoint.load_nvflare_checkpoint_metadata(local_checkpoint)
        except Exception as exc:
            raise RuntimeError(f"Local trainable checkpoint referenced by {path} is invalid: {exc}") from exc
        if local_metadata.get("site_name") != site_name or local_metadata.get("round") != round_index:
            raise RuntimeError(
                f"Local trainable checkpoint metadata does not match {path}: expected site_name={site_name!r}, "
                f"round={round_index}, received site_name={local_metadata.get('site_name')!r}, "
                f"round={local_metadata.get('round')!r}."
            )
        metric["local_checkpoint_sha256"] = provenance.sha256_file(local_checkpoint)
        if persist_client_training_state:
            training_state_manifest_value = metric.get("training_state_manifest")
            expected_training_state_manifest = (
                Path(args.workspace).resolve()
                / "client_training_state"
                / site_name
                / f"round_{round_index:03d}"
                / evo2_runtime.TRAINING_STATE_MANIFEST_FILENAME
            )
            if not isinstance(training_state_manifest_value, str) or (
                Path(training_state_manifest_value).resolve() != expected_training_state_manifest
            ):
                raise RuntimeError(
                    f"Round metrics at {path} do not reference the expected site-private training-state manifest "
                    f"{expected_training_state_manifest}."
                )
            if (
                not expected_training_state_manifest.is_file()
                or expected_training_state_manifest.stat().st_mtime < started_at
            ):
                raise RuntimeError(
                    f"Round metrics at {path} do not reference a fresh persistent client training-state manifest."
                )
            state_manifest = evo2_runtime._read_state_manifest(expected_training_state_manifest)
            expected_counters = evo2_runtime._expected_counters(round_index, args.local_steps, args.global_batch_size)
            if (
                state_manifest["site_name"] != site_name
                or state_manifest["round_index"] != round_index
                or state_manifest["counters"] != expected_counters
            ):
                raise RuntimeError(
                    f"Persistent client training-state manifest {expected_training_state_manifest} does not match "
                    f"site={site_name!r}, round={round_index}, and the configured counters."
                )
            if state_manifest["configuration_sha256"] != evo2_runtime._json_sha256(
                state_manifest["configuration"]
            ) or state_manifest["trainable_schema_sha256"] != evo2_runtime._json_sha256(
                state_manifest["trainable_schema"]
            ):
                raise RuntimeError(
                    f"Persistent client training-state manifest {expected_training_state_manifest} has invalid "
                    "configuration or schema digests."
                )
            if state_manifest["trainable_schema"] != initial_trainable_schema:
                raise RuntimeError(
                    f"Persistent client training-state manifest {expected_training_state_manifest} does not match "
                    "the federated trainable schema."
                )
            expected_updated_sha256 = evo2_runtime._trainable_state_sha256_from_validated_state(local_state)
            if state_manifest["updated_local_state_sha256"] != expected_updated_sha256:
                raise RuntimeError(
                    f"Persistent client training-state manifest {expected_training_state_manifest} does not match "
                    "the referenced local trainable checkpoint."
                )
            expected_incoming_sha256 = (
                initial_state_sha256 if round_index == args.start_round else global_round_state_sha256[round_index - 1]
            )
            if state_manifest["incoming_global_state_sha256"] != expected_incoming_sha256:
                raise RuntimeError(
                    f"Persistent client training-state manifest {expected_training_state_manifest} does not match "
                    "the global trainable checkpoint supplied to that round."
                )
            checkpoint_dir = expected_training_state_manifest.parent
            if evo2_runtime._checkpoint_payload_identity(checkpoint_dir) != state_manifest["payload"]:
                raise RuntimeError(
                    f"Persistent client training-state payload for {site_name} round {round_index} was modified."
                )
            evo2_runtime._validate_native_checkpoint(
                checkpoint_dir,
                expected_counters["end_step"],
                expected_counters["end_train_samples"],
            )
            metric["training_state_manifest_sha256"] = provenance.sha256_file(expected_training_state_manifest)
        metric["path"] = str(path)
        round_metrics.append(metric)

    expected_metric_pairs = {(entry["client_name"], round_index) for entry in plan for round_index in logical_rounds}
    actual_metric_pairs = set(metric_pair_counts)
    duplicate_metric_pairs = sorted(pair for pair, count in metric_pair_counts.items() if count > 1)
    missing_metric_pairs = sorted(expected_metric_pairs - actual_metric_pairs)
    unexpected_metric_pairs = sorted(actual_metric_pairs - expected_metric_pairs)
    if duplicate_metric_pairs or missing_metric_pairs or unexpected_metric_pairs:
        raise RuntimeError(
            "Fresh client metric files do not cover the planned client/round pairs exactly: "
            f"duplicates={duplicate_metric_pairs}, missing={missing_metric_pairs}, "
            f"unexpected={unexpected_metric_pairs}."
        )

    if persist_client_training_state:
        metrics_by_pair = {(metric["site_name"], metric["round"]): metric for metric in round_metrics}
        for entry in plan:
            previous_manifest_sha256 = None
            configuration_sha256 = None
            for round_index in logical_rounds:
                metric = metrics_by_pair[(entry["client_name"], round_index)]
                state_manifest = evo2_runtime._read_state_manifest(Path(metric["training_state_manifest"]))
                if state_manifest["previous_manifest_sha256"] != previous_manifest_sha256:
                    raise RuntimeError(
                        f"Persistent client training-state manifest chain is broken for {entry['client_name']} "
                        f"at round {round_index}."
                    )
                if configuration_sha256 is None:
                    configuration_sha256 = state_manifest["configuration_sha256"]
                elif state_manifest["configuration_sha256"] != configuration_sha256:
                    raise RuntimeError(
                        f"Persistent client training-state configuration changed for {entry['client_name']} "
                        f"at round {round_index}."
                    )
                previous_manifest_sha256 = metric["training_state_manifest_sha256"]

    summary = {
        "backend": args.backend,
        "exchange_dtype": adapter_checkpoint.EXCHANGE_DTYPE_NAME,
        "mode": args.mode,
        "peft_mode": args.peft_mode,
        "num_clients": len(plan),
        "num_rounds": args.num_rounds,
        "start_round": args.start_round,
        "local_steps": args.local_steps,
        "global_checkpoint": str(global_checkpoint),
        "global_checkpoint_sha256": provenance.sha256_file(global_checkpoint),
        "global_checkpoint_round": checkpoint_round,
        "global_checkpoint_contributors": checkpoint_contributors,
        "global_round_checkpoints": global_round_checkpoints,
        "initial_checkpoint": str(initial_checkpoint),
        "initial_checkpoint_sha256": provenance.sha256_file(initial_checkpoint),
        "initialization_metadata": initial_metadata,
        "continuation_signature": continuation_signature,
        "aggregation_weights": {entry["client_name"]: entry["sample_count"] for entry in plan},
        "dataset": {
            "manifest": str(manifest_path),
            "source": manifest.get("source"),
            "settings": manifest.get("settings"),
            "audit": manifest.get("audit"),
        },
        "training_inputs": training_inputs,
        "configuration": {
            "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
            "classifier_file": str(Path(args.classifier_file).resolve()) if args.classifier_file else None,
            "num_threads": args.num_threads,
            "gpu": args.gpu,
            "seq_length": args.seq_length,
            "micro_batch_size": args.micro_batch_size,
            "global_batch_size": args.global_batch_size,
            "learning_rate": args.learning_rate,
            "min_learning_rate": args.min_learning_rate,
            "warmup_iters": args.warmup_iters,
            "eval_iters": args.eval_iters,
            "seed": args.seed,
            "lora_dim": args.lora_dim if args.peft_mode == "lora" else None,
            "lora_alpha": args.lora_alpha if args.peft_mode == "lora" else None,
            "lora_dropout": args.lora_dropout if args.peft_mode == "lora" else None,
            "lora_target_modules": (
                list(evo2_runtime.parse_lora_targets(args.lora_target_modules)) if args.peft_mode == "lora" else []
            ),
            "server_tensor_device": args.server_tensor_device,
        },
        "total_client_runtime_seconds": sum(metric["runtime_seconds"] for metric in round_metrics),
        "peak_client_gpu_memory_mebibytes": max(
            (metric["peak_gpu_memory_mebibytes"] for metric in round_metrics), default=0.0
        ),
        "total_received_mebibytes": sum(metric["received_mebibytes"] for metric in round_metrics),
        "total_sent_mebibytes": sum(metric["sent_mebibytes"] for metric in round_metrics),
        "round_metrics": round_metrics,
    }
    if persist_client_training_state:
        summary["configuration"]["persist_client_training_state"] = True
        summary["total_persistent_training_state_mebibytes"] = sum(
            metric.get("persistent_training_state_mebibytes", 0.0) for metric in round_metrics
        )
    summary_path = Path(args.workspace).resolve() / "run_summary.json"
    summary["summary_path"] = str(summary_path)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    args = define_parser().parse_args(argv)
    _validate_workspace_precondition(args)
    manifest, plan, validation_file, training_inputs, continuation_signature = validate_inputs(args)
    recipe = create_recipe(args, plan, validation_file, continuation_signature)
    started_at = time.time()
    run = recipe.execute(create_sim_env(args, plan))
    result_dir = run.get_result()
    summary = collect_run_summary(
        args,
        plan,
        result_dir,
        started_at,
        manifest,
        training_inputs,
        continuation_signature,
    )
    print(f"Result can be found in: {result_dir}")
    print(f"Global checkpoint: {summary['global_checkpoint']}")
    print(f"Run summary: {summary['summary_path']}")


if __name__ == "__main__":
    main()
