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
"""Evaluate a reloaded global Evo2 trainable checkpoint on the held-out test set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from collections.abc import Sequence
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import TypedDict

import evo2_adapter_checkpoint as adapter_checkpoint
import evo2_runtime
import provenance
import torch

_ROW_INDEX_KEY = "_nvflare_row_index"


class ClassificationMetrics(TypedDict):
    num_examples: int
    class_ids: list[int]
    accuracy: float
    macro_f1: float
    confusion_matrix: list[list[int]]
    classification_report: dict[str, object]


class EvaluationCoverage(TypedDict):
    identity: str
    expected_rows: int
    observed_rows: int
    unique_rows: int
    sampler_order_sha256: str


class _RowIndexedDataset(torch.utils.data.Dataset):
    """Add the source JSONL row index without changing the model inputs."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        if not isinstance(row, dict):
            raise TypeError(f"The pinned Evo2 dataset must return a dictionary, received {type(row).__name__}.")
        if _ROW_INDEX_KEY in row:
            raise ValueError(f"The pinned Evo2 dataset unexpectedly contains {_ROW_INDEX_KEY!r}.")
        indexed_row = dict(row)
        indexed_row[_ROW_INDEX_KEY] = torch.tensor(index, dtype=torch.long)
        return indexed_row


@contextmanager
def _row_indexed_dataset_provider(dataset_provider):
    """Wrap datasets built by the pinned provider for one evaluation run."""

    provider_class = type(dataset_provider)
    original_build_datasets = getattr(provider_class, "build_datasets", None)
    if not callable(original_build_datasets):
        raise TypeError("The pinned Evo2 dataset provider does not expose build_datasets().")

    def build_datasets_with_row_indices(self, context):
        datasets = original_build_datasets(self, context)
        if not isinstance(datasets, (list, tuple)) or len(datasets) != 3:
            raise TypeError("The pinned Evo2 dataset provider must return the train, validation, and test datasets.")
        return tuple(_RowIndexedDataset(dataset) if dataset is not None else None for dataset in datasets)

    provider_class.build_datasets = build_datasets_with_row_indices
    try:
        yield
    finally:
        provider_class.build_datasets = original_build_datasets


def define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("bionemo", "mock"), default="bionemo")
    parser.add_argument("--checkpoint", required=True, help="NVFlare global_model.pt or initialization checkpoint")
    parser.add_argument("--base-checkpoint", default="./models/evo2_1b_bf16_mbridge")
    parser.add_argument("--test-file", default="./data/test.jsonl")
    parser.add_argument("--manifest", default="./data/manifest.json")
    parser.add_argument("--output", default="./evaluation.json")
    parser.add_argument("--confusion-matrix", default=None)
    parser.add_argument("--work-dir", default="/tmp/nvflare/evo2_evaluate")
    parser.add_argument("--classifier-file", default=None)
    parser.add_argument("--seq-length", type=int, default=600)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=4,
        help="Evaluation microbatch size; the test row count must be divisible by this value",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=32,
        help="Evaluation global batch size; must be divisible by --micro-batch-size",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lora-dim", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora-target-modules",
        default="linear_qkv,linear_proj,linear_fc1,linear_fc2,dense_projection,dense",
    )
    return parser


def _read_labels(path: str | os.PathLike[str]) -> list[int]:
    labels = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                labels.append(int(json.loads(line)["label"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid label in {path} at line {line_number}.") from exc
    if not labels:
        raise ValueError(f"Test file is empty: {path}")
    return labels


def _validate_manifest_binding(args: argparse.Namespace) -> dict:
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict) or manifest.get("format_version") != 2:
        version = manifest.get("format_version") if isinstance(manifest, dict) else type(manifest).__name__
        raise ValueError(f"Unsupported dataset manifest format_version: {version!r}.")
    if manifest.get("audit", {}).get("status") != "passed":
        raise ValueError(f"Dataset manifest {manifest_path} does not contain a passed leakage audit.")

    try:
        relative_path = manifest["files"]["test"]
        expected_rows = manifest["counts"]["test"]
        expected_identity = manifest["file_identities"]["test"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Dataset manifest {manifest_path} is missing the test split identity.") from exc
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("Dataset manifest test path must be a non-empty string.")
    if type(expected_rows) is not int or expected_rows <= 0:
        raise ValueError("Dataset manifest test row count must be a positive integer.")
    if not isinstance(expected_identity, dict) or set(expected_identity) != {"sha256", "bytes", "rows"}:
        raise ValueError("Dataset manifest test content identity is malformed.")
    if expected_identity["rows"] != expected_rows:
        raise ValueError(
            "Dataset manifest test row count and content identity disagree: "
            f"{expected_rows} != {expected_identity['rows']!r}."
        )

    expected_path = (manifest_path.parent / relative_path).resolve()
    test_path = Path(args.test_file).resolve()
    if test_path != expected_path:
        raise ValueError(
            f"--test-file does not match manifest.files.test: expected {expected_path}, observed {test_path}."
        )
    observed_identity = provenance.jsonl_identity(
        test_path,
        expected_rows=expected_rows,
        label="Manifest test split",
    )
    observed_payload = {field: observed_identity[field] for field in ("sha256", "bytes", "rows")}
    if observed_payload != expected_identity:
        raise ValueError(
            "Manifest test split no longer matches its audited content identity: "
            f"expected {expected_identity}, observed {observed_payload}. Run prepare_data.py again."
        )
    return {
        "path": str(manifest_path),
        "sha256": provenance.sha256_file(manifest_path),
        "format_version": 2,
        "split_role": "test",
        "file_identity": observed_payload,
    }


_sha256_file = provenance.sha256_file


def calculate_classification_metrics(labels: Sequence[int], predictions: Sequence[int]) -> ClassificationMetrics:
    """Calculate classification metrics using an explicit, stable class order.

    Args:
        labels: Ground-truth class identifiers.
        predictions: Predicted class identifiers in the same example order.

    Returns:
        Accuracy, macro-F1, confusion matrix, and per-class classification metrics.

    Raises:
        ValueError: If the inputs have different lengths or contain no examples.
        RuntimeError: If scikit-learn is unavailable.
    """

    if len(labels) != len(predictions):
        raise ValueError(f"labels and predictions differ in length: {len(labels)} != {len(predictions)}")
    if not labels:
        raise ValueError("At least one prediction is required.")
    try:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    except ImportError as exc:
        raise RuntimeError("Evaluation requires scikit-learn") from exc

    class_ids = sorted(set(labels) | set(predictions))
    return {
        "num_examples": len(labels),
        "class_ids": class_ids,
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=class_ids, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=class_ids).tolist(),
        "classification_report": classification_report(
            labels, predictions, labels=class_ids, output_dict=True, zero_division=0
        ),
    }


def _recording_forward_step(upstream, predictions: list[int], labels_seen: list[int], row_indices_seen: list[int]):
    def forward_step(state, data_iterator, model, return_schedule_plan=False):
        if return_schedule_plan:
            raise NotImplementedError("Schedule plans are not used by the Evo2 classifier.")
        batch = next(data_iterator)
        row_indices = batch.get(_ROW_INDEX_KEY)
        if not isinstance(row_indices, torch.Tensor):
            raise RuntimeError("Evaluation batch is missing the JSONL row-index tensor.")
        input_ids = batch["input_ids"].cuda(non_blocking=True)
        pool_mask = batch["pool_mask"].cuda(non_blocking=True)
        labels = batch["labels"].cuda(non_blocking=True)
        if row_indices.numel() != labels.numel():
            raise RuntimeError(
                "Evaluation batch row-index and label counts differ: " f"{row_indices.numel()} != {labels.numel()}."
            )
        logits = model(input_ids=input_ids, pool_mask=pool_mask)
        predictions.extend(logits.detach().argmax(dim=-1).cpu().tolist())
        labels_seen.extend(labels.detach().cpu().tolist())
        row_indices_seen.extend(row_indices.detach().cpu().reshape(-1).tolist())
        return logits, partial(upstream._classification_loss_fn, labels)

    return forward_step


def _validate_exact_bionemo_coverage(num_examples: int, micro_batch_size: int) -> None:
    """Reject batch sizes for which Megatron's cyclic sampler drops test rows."""

    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive.")
    if num_examples % micro_batch_size:
        raise ValueError(
            "Exact BioNeMo evaluation requires the test row count to be divisible by --micro-batch-size; "
            f"received {num_examples} rows and microbatch {micro_batch_size}. The pinned cyclic sampler drops "
            "an incomplete microbatch before cycling."
        )


def _validate_exact_row_coverage(
    row_indices: Sequence[int],
    labels: Sequence[int],
    predictions: Sequence[int],
    source_labels: Sequence[int],
) -> EvaluationCoverage:
    """Require one prediction for every JSONL row and the label at that row."""

    expected_rows = len(source_labels)
    observed_rows = len(row_indices)
    if len(labels) != observed_rows or len(predictions) != observed_rows:
        raise RuntimeError(
            "Evaluation row-index, label, and prediction counts differ: "
            f"rows={observed_rows}, labels={len(labels)}, predictions={len(predictions)}."
        )
    invalid_indices = [index for index in row_indices if type(index) is not int]
    if invalid_indices:
        raise RuntimeError(f"Evaluation returned non-integer JSONL row indices: {invalid_indices[:10]}.")

    counts = Counter(row_indices)
    expected = set(range(expected_rows))
    observed = set(row_indices)
    duplicates = sorted(index for index, count in counts.items() if count > 1)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if observed_rows != expected_rows or duplicates or missing or unexpected:
        raise RuntimeError(
            "Evaluation did not cover every held-out JSONL row exactly once: "
            f"expected={expected_rows}, observed={observed_rows}, duplicates={duplicates[:10]}, "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}."
        )

    label_mismatches = [
        (row_index, source_labels[row_index], observed_label)
        for row_index, observed_label in zip(row_indices, labels)
        if observed_label != source_labels[row_index]
    ]
    if label_mismatches:
        raise RuntimeError(
            "Evaluation labels do not match their held-out JSONL rows: "
            f"(row_index, expected, observed)={label_mismatches[:10]}."
        )

    sampler_order = json.dumps(list(row_indices), separators=(",", ":")).encode("ascii")
    return {
        "identity": "zero_based_jsonl_line_index",
        "expected_rows": expected_rows,
        "observed_rows": observed_rows,
        "unique_rows": len(observed),
        "sampler_order_sha256": hashlib.sha256(sampler_order).hexdigest(),
    }


def _run_bionemo_evaluation(
    args: argparse.Namespace, state, num_examples: int
) -> tuple[list[int], list[int], list[int]]:
    _validate_exact_bionemo_coverage(num_examples, args.micro_batch_size)
    upstream = evo2_runtime.load_classifier_module(args.classifier_file)
    eval_iters = math.ceil(num_examples / args.global_batch_size)
    config = evo2_runtime.build_classifier_config(
        upstream,
        base_checkpoint=str(Path(args.base_checkpoint).resolve()),
        train_file=str(Path(args.test_file).resolve()),
        validation_file=None,
        test_file=str(Path(args.test_file).resolve()),
        result_dir=str(Path(args.work_dir).resolve()),
        experiment_name="global_test",
        train_iters=0,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        learning_rate=1e-6,
        min_learning_rate=1e-6,
        warmup_iters=0,
        eval_interval=1,
        eval_iters=eval_iters,
        seed=args.seed,
        lora_dim=args.lora_dim,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=evo2_runtime.parse_lora_targets(args.lora_target_modules),
    )
    config.validation.skip_train = True
    predictions = []
    labels_seen = []
    row_indices_seen = []
    callback = evo2_runtime.make_exchange_callback(
        state,
        extract_after_training=False,
        load_on_data_init=True,
    )
    with _row_indexed_dataset_provider(config.dataset):
        upstream.pretrain(
            config,
            _recording_forward_step(upstream, predictions, labels_seen, row_indices_seen),
            callbacks=[callback],
        )
    return row_indices_seen[:num_examples], labels_seen[:num_examples], predictions[:num_examples]


def _mock_predictions(labels: Sequence[int], state) -> list[int]:
    head_biases = [tensor for name, tensor in state.items() if name.endswith("classification_head.bias")]
    predicted_class = int(head_biases[-1].float().argmax().item()) if head_biases else 0
    return [predicted_class] * len(labels)


def _write_confusion_matrix(path: Path, metrics: ClassificationMetrics) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Confusion-matrix rendering requires matplotlib") from exc

    matrix = metrics["confusion_matrix"]
    class_ids = metrics["class_ids"]
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(matrix, cmap="Greens")
    figure.colorbar(image, ax=axis)
    axis.set(
        xlabel="Predicted class",
        ylabel="True class",
        xticks=range(len(class_ids)),
        yticks=range(len(class_ids)),
        xticklabels=class_ids,
        yticklabels=class_ids,
        title="Evo2 splice-site classification",
    )
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, str(value), ha="center", va="center")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def evaluate(args: argparse.Namespace) -> dict:
    manifest_binding = _validate_manifest_binding(args)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_state = adapter_checkpoint.load_nvflare_checkpoint(checkpoint_path)
    checkpoint_metadata = adapter_checkpoint.load_nvflare_checkpoint_metadata(checkpoint_path)
    source_labels = _read_labels(args.test_file)
    lora_target_modules = evo2_runtime.parse_lora_targets(args.lora_target_modules)
    base_checkpoint = None
    classifier_file = None
    base_checkpoint_identity = None
    classifier_file_identity = None
    if args.backend == "bionemo":
        base_checkpoint = Path(args.base_checkpoint).resolve()
        if not base_checkpoint.is_dir():
            raise FileNotFoundError(f"Base checkpoint not found: {base_checkpoint}")
        classifier_file = evo2_runtime.resolve_classifier_path(args.classifier_file)
        base_checkpoint_identity = provenance.directory_identity(base_checkpoint)
        classifier_file_identity = provenance.file_identity(classifier_file)
    initialization_metadata = provenance.validate_initialization_metadata(
        checkpoint_metadata,
        backend=args.backend,
        seed=args.seed,
        seq_length=args.seq_length,
        lora_dim=args.lora_dim,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_target_modules,
        base_checkpoint_identity=base_checkpoint_identity,
        classifier_file_identity=classifier_file_identity,
        exchange_dtype=adapter_checkpoint.EXCHANGE_DTYPE_NAME,
    )
    if args.backend == "bionemo":
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        row_indices, labels, predictions = _run_bionemo_evaluation(args, checkpoint_state, len(source_labels))
    else:
        started = time.monotonic()
        row_indices = list(range(len(source_labels)))
        labels = source_labels
        predictions = _mock_predictions(labels, checkpoint_state)

    evaluation_coverage = _validate_exact_row_coverage(row_indices, labels, predictions, source_labels)
    metrics = calculate_classification_metrics(labels, predictions)
    evaluation_signature = {
        "backend": args.backend,
        "test_file_sha256": _sha256_file(args.test_file),
        "dataset_manifest_sha256": manifest_binding["sha256"],
        "split_role": "test",
        "base_checkpoint_sha256": base_checkpoint_identity["sha256"] if base_checkpoint_identity else None,
        "classifier_file_sha256": classifier_file_identity["sha256"] if classifier_file_identity else None,
        "seq_length": args.seq_length,
        "micro_batch_size": args.micro_batch_size,
        "global_batch_size": args.global_batch_size,
        "seed": args.seed,
        "row_identity": evaluation_coverage["identity"],
        "peft_mode": "lora",
        "lora_dim": args.lora_dim,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": list(lora_target_modules),
    }
    report: dict[str, object] = dict(metrics)
    report.update(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "checkpoint_tensors": len(checkpoint_state),
            "checkpoint_mebibytes": adapter_checkpoint.state_dict_size_mb(checkpoint_state),
            "runtime_seconds": time.monotonic() - started,
            "test_file": str(Path(args.test_file).resolve()),
            "dataset_manifest": manifest_binding,
            "base_checkpoint": str(base_checkpoint) if base_checkpoint else None,
            "classifier_file": str(classifier_file) if classifier_file else None,
            "initialization_metadata": initialization_metadata,
            "evaluation_signature": evaluation_signature,
            "evaluation_coverage": evaluation_coverage,
            "peak_gpu_memory_mebibytes": (
                torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
            ),
        }
    )
    output_path = Path(args.output).resolve()
    matrix_path = (
        Path(args.confusion_matrix).resolve()
        if args.confusion_matrix
        else output_path.with_name(f"{output_path.stem}_confusion_matrix.png")
    )
    report["confusion_matrix_plot"] = str(matrix_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
        file.write("\n")
    _write_confusion_matrix(matrix_path, metrics)
    print(
        f"accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, "
        f"examples={metrics['num_examples']}, report={output_path}"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    args = define_parser().parse_args(argv)
    evaluate(args)


if __name__ == "__main__":
    main()
