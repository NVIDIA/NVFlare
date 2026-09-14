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

import importlib.util
import json
import sys
from collections import OrderedDict
from pathlib import Path

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
pytestmark = pytest.mark.skipif(
    not (HAS_TORCH and HAS_SKLEARN),
    reason="PyTorch and scikit-learn are required for Evo2 evaluation tests",
)


def _example_dir() -> Path:
    return Path(__file__).parents[5] / "examples" / "advanced" / "bionemo" / "evo2"


def _load_evaluate_module():
    example_dir = _example_dir()
    module_names = ("adapter_checkpoint", "evo2_runtime")
    previous_modules = {name: sys.modules.pop(name, None) for name in module_names}
    sys.path.insert(0, str(example_dir))
    try:
        spec = importlib.util.spec_from_file_location("evo2_evaluate_under_test", example_dir / "evaluate.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(example_dir))
        for name in module_names:
            sys.modules.pop(name, None)
        for name, previous in previous_modules.items():
            if previous is not None:
                sys.modules[name] = previous


def test_classification_metrics_use_macro_f1_and_explicit_confusion_order():
    evaluate = _load_evaluate_module()

    metrics = evaluate.calculate_classification_metrics(
        labels=[0, 0, 1, 1, 2, 2],
        predictions=[0, 1, 1, 1, 0, 2],
    )

    assert metrics["num_examples"] == 6
    assert metrics["class_ids"] == [0, 1, 2]
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]

    with pytest.raises(ValueError, match="differ in length"):
        evaluate.calculate_classification_metrics([0], [0, 1])
    with pytest.raises(ValueError, match="At least one"):
        evaluate.calculate_classification_metrics([], [])


def test_reference_comparison_reports_metric_deltas(tmp_path):
    evaluate = _load_evaluate_module()
    reference = tmp_path / "initialization.json"
    signature = {"test_file_sha256": "same-data-and-settings"}
    reference.write_text(
        json.dumps({"accuracy": 0.4, "macro_f1": 0.3, "evaluation_signature": signature}), encoding="utf-8"
    )

    comparison = evaluate.compare_with_reference(
        {"accuracy": 0.5, "macro_f1": 0.35, "evaluation_signature": signature},
        reference,
        improvement_metric="macro_f1",
    )

    assert comparison["reference_report"] == str(reference.resolve())
    assert comparison["metric_deltas"] == pytest.approx({"accuracy": 0.1, "macro_f1": 0.05})
    assert comparison["required_metric"] == "macro_f1"
    assert comparison["improved"] is True

    with pytest.raises(ValueError, match="different held-out dataset"):
        evaluate.compare_with_reference(
            {"accuracy": 0.5, "macro_f1": 0.35, "evaluation_signature": {"test_file_sha256": "other"}},
            reference,
            improvement_metric="macro_f1",
        )


def test_require_improvement_requires_a_reference_report():
    evaluate = _load_evaluate_module()
    args = evaluate.define_parser().parse_args(["--checkpoint", "unused.pt", "--require-improvement"])

    with pytest.raises(ValueError, match="requires --reference-report"):
        evaluate.evaluate(args)


def test_performance_gate_is_opt_in_and_checks_all_configured_metrics():
    evaluate = _load_evaluate_module()
    metrics = evaluate.calculate_classification_metrics(
        labels=[0, 0, 1, 1, 2, 2],
        predictions=[0, 0, 1, 1, 2, 2],
    )

    disabled = evaluate.assess_performance_gate(metrics)
    passing = evaluate.assess_performance_gate(
        metrics,
        min_accuracy=0.9563,
        min_macro_f1=0.956,
        min_class_recall=0.94,
        min_class_f1=0.94,
    )

    assert disabled["enabled"] is False
    assert disabled["passed"] is True
    assert passing["enabled"] is True
    assert passing["passed"] is True
    assert passing["criteria"]["minimum_accuracy"] == 0.9563
    assert passing["observed"]["per_class"] == {
        "0": {"recall": 1.0, "f1": 1.0},
        "1": {"recall": 1.0, "f1": 1.0},
        "2": {"recall": 1.0, "f1": 1.0},
    }


def test_performance_gate_records_aggregate_class_and_improvement_failures():
    evaluate = _load_evaluate_module()
    metrics = evaluate.calculate_classification_metrics(
        labels=[0, 0, 1, 1, 2, 2],
        predictions=[0, 1, 1, 1, 0, 2],
    )
    metrics["comparison"] = {
        "metric_deltas": {"accuracy": 0.01, "macro_f1": -0.02},
        "required_metric": "macro_f1",
        "improved": False,
    }

    result = evaluate.assess_performance_gate(
        metrics,
        min_accuracy=0.70,
        min_macro_f1=0.70,
        min_class_recall=0.60,
        min_class_f1=0.60,
        require_improvement=True,
        improvement_metric="macro_f1",
    )

    assert result["passed"] is False
    assert result["failures"] == [
        {"metric": "accuracy", "minimum": 0.70, "observed": pytest.approx(4 / 6)},
        {"metric": "macro_f1", "minimum": 0.70, "observed": pytest.approx((0.5 + 0.8 + 2 / 3) / 3)},
        {"metric": "recall", "minimum": 0.60, "observed": 0.5, "class_id": 0},
        {"metric": "f1", "minimum": 0.60, "observed": 0.5, "class_id": 0},
        {"metric": "recall", "minimum": 0.60, "observed": 0.5, "class_id": 2},
        {
            "metric": "improvement:macro_f1",
            "minimum": 0.0,
            "exclusive_minimum": True,
            "observed": -0.02,
        },
    ]


def test_performance_threshold_cli_values_are_probabilities():
    evaluate = _load_evaluate_module()

    args = evaluate.define_parser().parse_args(
        [
            "--checkpoint",
            "unused.pt",
            "--min-accuracy",
            "0.9563",
            "--min-macro-f1",
            "0.956",
            "--min-class-recall",
            "0.94",
            "--min-class-f1",
            "0.94",
        ]
    )
    assert (args.min_accuracy, args.min_macro_f1, args.min_class_recall, args.min_class_f1) == (
        0.9563,
        0.956,
        0.94,
        0.94,
    )

    with pytest.raises(SystemExit):
        evaluate.define_parser().parse_args(["--checkpoint", "unused.pt", "--min-accuracy", "1.01"])


def test_exact_bionemo_coverage_requires_complete_microbatches():
    evaluate = _load_evaluate_module()

    evaluate._validate_exact_bionemo_coverage(num_examples=3000, micro_batch_size=4)
    with pytest.raises(ValueError, match="test row count to be divisible"):
        evaluate._validate_exact_bionemo_coverage(num_examples=3000, micro_batch_size=16)


def test_row_indexed_dataset_provider_preserves_model_inputs():
    import torch

    evaluate = _load_evaluate_module()
    source_rows = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "pool_mask": torch.tensor([1.0, 1.0, 1.0]),
            "labels": torch.tensor(2),
        }
    ]

    class Provider:
        def build_datasets(self, _context):
            return source_rows, None, source_rows

    original_build_datasets = Provider.build_datasets
    provider = Provider()
    with evaluate._row_indexed_dataset_provider(provider):
        train, validation, test = provider.build_datasets(None)
        indexed_row = train[0]

        assert validation is None
        assert test[0][evaluate._ROW_INDEX_KEY].item() == 0
        assert indexed_row[evaluate._ROW_INDEX_KEY].item() == 0
        assert set(indexed_row) == {*source_rows[0], evaluate._ROW_INDEX_KEY}
        for name, tensor in source_rows[0].items():
            assert indexed_row[name] is tensor
            assert torch.equal(indexed_row[name], tensor)
    assert Provider.build_datasets is original_build_datasets


def test_exact_row_coverage_rejects_duplicate_or_mislabeled_same_class_rows():
    evaluate = _load_evaluate_module()
    source_labels = [0, 0, 1]

    coverage = evaluate._validate_exact_row_coverage(
        row_indices=[2, 0, 1],
        labels=[1, 0, 0],
        predictions=[1, 0, 1],
        source_labels=source_labels,
    )
    assert coverage["identity"] == "zero_based_jsonl_line_index"
    assert coverage["expected_rows"] == coverage["observed_rows"] == coverage["unique_rows"] == 3
    assert len(coverage["sampler_order_sha256"]) == 64

    with pytest.raises(RuntimeError, match="every held-out JSONL row exactly once"):
        evaluate._validate_exact_row_coverage(
            row_indices=[0, 0, 2],
            labels=[0, 0, 1],
            predictions=[0, 0, 1],
            source_labels=source_labels,
        )
    with pytest.raises(RuntimeError, match="labels do not match"):
        evaluate._validate_exact_row_coverage(
            row_indices=[2, 0, 1],
            labels=[0, 0, 1],
            predictions=[0, 0, 1],
            source_labels=source_labels,
        )


def test_recording_forward_step_keeps_row_index_out_of_model_inputs(monkeypatch):
    import torch

    evaluate = _load_evaluate_module()
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, non_blocking=True: self)
    batch = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "pool_mask": torch.ones(2, 2),
        "labels": torch.tensor([0, 2]),
        evaluate._ROW_INDEX_KEY: torch.tensor([7, 3]),
    }
    model_inputs = {}

    def model(**kwargs):
        model_inputs.update(kwargs)
        return torch.tensor([[3.0, 1.0, 0.0], [0.0, 1.0, 3.0]])

    class Upstream:
        @staticmethod
        def _classification_loss_fn(labels, logits):
            return labels, logits

    predictions = []
    labels_seen = []
    row_indices_seen = []
    forward_step = evaluate._recording_forward_step(Upstream, predictions, labels_seen, row_indices_seen)
    forward_step(None, iter([batch]), model)

    assert set(model_inputs) == {"input_ids", "pool_mask"}
    assert model_inputs["input_ids"] is batch["input_ids"]
    assert model_inputs["pool_mask"] is batch["pool_mask"]
    assert predictions == [0, 2]
    assert labels_seen == [0, 2]
    assert row_indices_seen == [7, 3]


def test_checkpoint_directory_identity_is_content_and_layout_sensitive(tmp_path):
    evaluate = _load_evaluate_module()
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "iter_0000001").mkdir(parents=True)
    first = checkpoint / "latest_checkpointed_iteration.txt"
    second = checkpoint / "iter_0000001" / "weights.distcp"
    first.write_text("1\n", encoding="utf-8")
    second.write_bytes(b"weights")

    original = evaluate._sha256_directory(checkpoint)
    second.write_bytes(b"changed")
    assert evaluate._sha256_directory(checkpoint) != original

    second.write_bytes(b"weights")
    second.rename(checkpoint / "iter_0000001" / "renamed.distcp")
    assert evaluate._sha256_directory(checkpoint) != original


def test_checkpoint_directory_identity_frames_file_content_lengths(tmp_path):
    evaluate = _load_evaluate_module()
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()

    def encoded_path(name):
        value = name.encode()
        return len(value).to_bytes(8, byteorder="big") + value

    (left / "a").write_bytes(b"")
    (left / "b").write_bytes(b"PAYLOAD" + encoded_path("c"))
    (right / "a").write_bytes(encoded_path("b") + b"PAYLOAD")
    (right / "c").write_bytes(b"")

    assert sum(path.stat().st_size for path in left.iterdir()) == sum(path.stat().st_size for path in right.iterdir())
    assert evaluate._sha256_directory(left) != evaluate._sha256_directory(right)


def test_mock_evaluation_reloads_nvflare_checkpoint_and_writes_report(tmp_path, monkeypatch):
    import torch

    evaluate = _load_evaluate_module()
    checkpoint = tmp_path / "global_model.pt"
    state = OrderedDict(
        [
            ("decoder.layers.0.adapter.linear_in.weight", torch.ones(2, 2)),
            ("decoder.classification_head.weight", torch.zeros(3, 2)),
            ("decoder.classification_head.bias", torch.tensor([0.0, 2.0, -1.0])),
        ]
    )
    initialization_metadata = {
        "backend": "mock",
        "base_checkpoint": str((tmp_path / "unused-base").resolve()),
        "exchange_dtype": "float32",
        "peft_mode": "lora",
        "seed": 1234,
        "seq_length": 600,
        "lora_dim": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "lora_target_modules": [
            "linear_qkv",
            "linear_proj",
            "linear_fc1",
            "linear_fc2",
            "dense_projection",
            "dense",
        ],
        "training_inputs": {"data_file": None, "base_checkpoint": None, "classifier_file": None},
    }
    evaluate.adapter_checkpoint.save_nvflare_checkpoint(state, checkpoint, metadata=initialization_metadata)
    test_file = tmp_path / "test.jsonl"
    test_file.write_text(
        "".join(
            json.dumps({"sequence": f"ACGT{index}", "label": label}) + "\n" for index, label in enumerate([1, 1, 0, 2])
        ),
        encoding="utf-8",
    )
    test_identity = evaluate.provenance.jsonl_identity(test_file)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format_version": 2,
                "audit": {"status": "passed"},
                "counts": {"test": 4},
                "files": {"test": "test.jsonl"},
                "file_identities": {"test": {field: test_identity[field] for field in ("sha256", "bytes", "rows")}},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "evaluation.json"

    def fake_matrix_writer(path, metrics):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics["confusion_matrix"]), encoding="utf-8")

    monkeypatch.setattr(evaluate, "_write_confusion_matrix", fake_matrix_writer)
    monkeypatch.setattr(
        evaluate.evo2_runtime,
        "load_classifier_module",
        lambda *_args, **_kwargs: pytest.fail("mock evaluation must not import BioNeMo"),
    )
    args = evaluate.define_parser().parse_args(
        [
            "--backend",
            "mock",
            "--checkpoint",
            str(checkpoint),
            "--test-file",
            str(test_file),
            "--manifest",
            str(manifest),
            "--split-role",
            "test",
            "--output",
            str(output),
        ]
    )

    metrics = evaluate.evaluate(args)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert metrics["checkpoint"] == str(checkpoint.resolve())
    assert metrics["checkpoint_tensors"] == 3
    assert len(metrics["checkpoint_sha256"]) == 64
    assert metrics["num_examples"] == 4
    assert metrics["test_file"] == str(test_file.resolve())
    assert metrics["dataset_manifest"]["path"] == str(manifest.resolve())
    assert metrics["dataset_manifest"]["split_role"] == "test"
    assert metrics["dataset_manifest"]["file_identity"] == {
        field: test_identity[field] for field in ("sha256", "bytes", "rows")
    }
    assert len(metrics["evaluation_signature"]["test_file_sha256"]) == 64
    assert metrics["evaluation_signature"]["dataset_manifest_sha256"] == metrics["dataset_manifest"]["sha256"]
    assert metrics["evaluation_signature"]["split_role"] == "test"
    assert metrics["evaluation_signature"]["base_checkpoint_sha256"] is None
    assert metrics["evaluation_signature"]["classifier_file_sha256"] is None
    assert metrics["evaluation_signature"]["row_identity"] == "zero_based_jsonl_line_index"
    assert metrics["evaluation_signature"]["lora_target_modules"] == initialization_metadata["lora_target_modules"]
    assert metrics["initialization_metadata"] == initialization_metadata
    assert metrics["evaluation_coverage"]["identity"] == "zero_based_jsonl_line_index"
    assert metrics["evaluation_coverage"]["unique_rows"] == 4
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["macro_f1"] == pytest.approx(2 / 9)
    assert metrics["confusion_matrix"] == [[0, 1, 0], [0, 2, 0], [0, 1, 0]]
    assert metrics["performance_gate"] == saved["performance_gate"]
    assert metrics["performance_gate"]["enabled"] is False
    assert metrics["performance_gate"]["passed"] is True
    assert saved["accuracy"] == pytest.approx(metrics["accuracy"])
    assert Path(metrics["confusion_matrix_plot"]).read_text(encoding="utf-8") == "[[0, 1, 0], [0, 2, 0], [0, 1, 0]]"


def test_manifest_binding_rejects_unpaired_arguments_wrong_path_and_modified_content(tmp_path):
    evaluate = _load_evaluate_module()
    test_file = tmp_path / "test.jsonl"
    test_file.write_text(json.dumps({"sequence": "ACGT", "label": 0}) + "\n", encoding="utf-8")
    identity = evaluate.provenance.jsonl_identity(test_file)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format_version": 2,
                "audit": {"status": "passed"},
                "counts": {"test": 1},
                "files": {"test": "test.jsonl"},
                "file_identities": {"test": {field: identity[field] for field in ("sha256", "bytes", "rows")}},
            }
        ),
        encoding="utf-8",
    )

    unpaired = evaluate.define_parser().parse_args(["--checkpoint", "unused.pt", "--manifest", str(manifest)])
    with pytest.raises(ValueError, match="must be provided together"):
        evaluate._validate_manifest_binding(unpaired)

    other_file = tmp_path / "other.jsonl"
    other_file.write_text(test_file.read_text(encoding="utf-8"), encoding="utf-8")
    wrong_path = evaluate.define_parser().parse_args(
        [
            "--checkpoint",
            "unused.pt",
            "--test-file",
            str(other_file),
            "--manifest",
            str(manifest),
            "--split-role",
            "test",
        ]
    )
    with pytest.raises(ValueError, match="does not match manifest.files.test"):
        evaluate._validate_manifest_binding(wrong_path)

    test_file.write_text(test_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    changed_content = evaluate.define_parser().parse_args(
        [
            "--checkpoint",
            "unused.pt",
            "--test-file",
            str(test_file),
            "--manifest",
            str(manifest),
            "--split-role",
            "test",
        ]
    )
    with pytest.raises(ValueError, match="blank row"):
        evaluate._validate_manifest_binding(changed_content)


def test_manifest_binding_is_required_unless_unbound_evaluation_is_explicit():
    evaluate = _load_evaluate_module()

    default_args = evaluate.define_parser().parse_args(["--checkpoint", "unused.pt"])
    with pytest.raises(ValueError, match="requires --manifest and --split-role"):
        evaluate._validate_manifest_binding(default_args)

    unbound_args = evaluate.define_parser().parse_args(["--checkpoint", "unused.pt", "--allow-unbound-evaluation"])
    assert evaluate._validate_manifest_binding(unbound_args) is None

    conflicting_args = evaluate.define_parser().parse_args(
        [
            "--checkpoint",
            "unused.pt",
            "--manifest",
            "manifest.json",
            "--split-role",
            "test",
            "--allow-unbound-evaluation",
        ]
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        evaluate._validate_manifest_binding(conflicting_args)


def test_manifest_binding_accepts_validation_and_rejects_invalid_manifest_metadata(tmp_path):
    evaluate = _load_evaluate_module()
    validation_file = tmp_path / "validation.jsonl"
    validation_file.write_text(json.dumps({"sequence": "ACGT", "label": 0}) + "\n", encoding="utf-8")
    identity = evaluate.provenance.jsonl_identity(validation_file)
    identity_payload = {field: identity[field] for field in ("sha256", "bytes", "rows")}
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "format_version": 2,
        "audit": {"status": "passed"},
        "counts": {"validation": 1},
        "files": {"validation": "validation.jsonl"},
        "file_identities": {"validation": identity_payload},
    }
    args = evaluate.define_parser().parse_args(
        [
            "--checkpoint",
            "unused.pt",
            "--test-file",
            str(validation_file),
            "--manifest",
            str(manifest_path),
            "--split-role",
            "validation",
        ]
    )

    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    binding = evaluate._validate_manifest_binding(args)
    assert binding["split_role"] == "validation"
    assert binding["file_identity"] == identity_payload

    invalid_cases = [
        ({**manifest, "format_version": 1}, "Unsupported dataset manifest format_version"),
        ({**manifest, "audit": {"status": "failed"}}, "does not contain a passed leakage audit"),
        (
            {**manifest, "file_identities": {"validation": {"sha256": identity_payload["sha256"]}}},
            "content identity is malformed",
        ),
    ]
    for invalid_manifest, expected_error in invalid_cases:
        manifest_path.write_text(json.dumps(invalid_manifest), encoding="utf-8")
        with pytest.raises(ValueError, match=expected_error):
            evaluate._validate_manifest_binding(args)
