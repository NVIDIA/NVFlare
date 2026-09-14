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
from pathlib import Path

import pytest


def _load_module():
    module_path = Path(__file__).parents[5] / "examples" / "advanced" / "bionemo" / "evo2" / "summarize_baselines.py"
    spec = importlib.util.spec_from_file_location("evo2_summarize_baselines", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _signature(split_role: str) -> dict:
    return {
        "backend": "bionemo",
        "base_checkpoint_sha256": "b" * 64,
        "classifier_file_sha256": "c" * 64,
        "dataset_manifest_sha256": "d" * 64,
        "global_batch_size": 32,
        "lora_alpha": 32,
        "lora_dim": 16,
        "lora_dropout": 0.1,
        "lora_target_modules": [
            "linear_qkv",
            "linear_proj",
            "linear_fc1",
            "linear_fc2",
            "dense_projection",
            "dense",
        ],
        "micro_batch_size": 8,
        "peft_mode": "lora",
        "row_identity": "zero_based_jsonl_line_index",
        "seed": 1234,
        "seq_length": 600,
        "split_role": split_role,
        "test_file_sha256": ("e" if split_role == "validation" else "f") * 64,
    }


def _evaluation(checkpoint_sha256: str, split_role: str, accuracy: float, macro_f1: float) -> dict:
    signature = _signature(split_role)
    return {
        "accuracy": accuracy,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_manifest": {
            "file_identity": {
                "bytes": 1234,
                "rows": 3000,
                "sha256": signature["test_file_sha256"],
            },
            "format_version": 2,
            "path": "/validation/data/manifest.json",
            "sha256": signature["dataset_manifest_sha256"],
            "split_role": split_role,
        },
        "evaluation_coverage": {
            "expected_rows": 3000,
            "identity": "zero_based_jsonl_line_index",
            "observed_rows": 3000,
            "sampler_order_sha256": "a" * 64,
            "unique_rows": 3000,
        },
        "evaluation_signature": signature,
        "macro_f1": macro_f1,
        "num_examples": 3000,
    }


def _build_campaign(tmp_path: Path, module):
    checkpoint_sha256 = {
        "initialization": "1" * 64,
        "local_sites": {"1": "2" * 64, "2": "3" * 64, "3": "4" * 64},
        "primary_fl_final": "5" * 64,
        "secondary_selected_fl": "6" * 64,
    }
    manifest_path = tmp_path / "campaign.json"
    _write_json(
        manifest_path,
        {
            "checkpoint_sha256": checkpoint_sha256,
            "format_version": 1,
            "protocol": {
                "global_batch_size": 96,
                "local_steps": 888,
                "num_rounds": 1,
                "seed": 1234,
            },
            "resources": {"fl_total_optimizer_steps": 2664, "local_optimizer_steps_per_site": 888},
        },
    )

    metrics = {
        "initialization": {"validation": (0.33, 0.17), "test": (0.34, 0.18)},
        "primary_fl_final": {"validation": (0.95, 0.945), "test": (0.95, 0.945)},
        "secondary_selected_fl": {"validation": (0.96, 0.958), "test": (0.956, 0.955)},
        "site_1": {"validation": (0.95, 0.90), "test": (0.90, 0.89)},
        "site_2": {"validation": (0.91, 0.92), "test": (0.92, 0.91)},
        "site_3": {"validation": (0.93, 0.92), "test": (0.94, 0.93)},
    }
    report_paths = {}
    for role, split_metrics in metrics.items():
        if role.startswith("site_"):
            checkpoint = checkpoint_sha256["local_sites"][role.removeprefix("site_")]
        else:
            checkpoint = checkpoint_sha256[role]
        report_paths[role] = {}
        for split_role, (accuracy, macro_f1) in split_metrics.items():
            path = tmp_path / f"{role}_{split_role}.json"
            _write_json(path, _evaluation(checkpoint, split_role, accuracy, macro_f1))
            report_paths[role][split_role] = path

    output_path = tmp_path / "summary.json"
    argv = [
        "--campaign-manifest",
        str(manifest_path),
        "--primary-fl-validation",
        str(report_paths["primary_fl_final"]["validation"]),
        "--primary-fl-test",
        str(report_paths["primary_fl_final"]["test"]),
        "--secondary-selected-fl-validation",
        str(report_paths["secondary_selected_fl"]["validation"]),
        "--secondary-selected-fl-test",
        str(report_paths["secondary_selected_fl"]["test"]),
        "--initialization-validation",
        str(report_paths["initialization"]["validation"]),
        "--initialization-test",
        str(report_paths["initialization"]["test"]),
        "--output",
        str(output_path),
    ]
    for site_id in (1, 2, 3):
        argv.extend(
            [
                "--site-validation",
                f"{site_id}={report_paths[f'site_{site_id}']['validation']}",
                "--site-test",
                f"{site_id}={report_paths[f'site_{site_id}']['test']}",
            ]
        )
    return module.define_parser().parse_args(argv), report_paths, manifest_path, output_path


def _mutate_report(path: Path, mutation) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    mutation(report)
    _write_json(path, report)


def test_summarize_writes_deterministic_metrics_deltas_and_metadata(tmp_path):
    module = _load_module()
    args, _report_paths, _manifest_path, output_path = _build_campaign(tmp_path, module)

    summary = module.summarize(args)
    first_bytes = output_path.read_bytes()
    assert module.summarize(args) == summary
    assert output_path.read_bytes() == first_bytes

    assert summary["protocol"] == {
        "global_batch_size": 96,
        "local_steps": 888,
        "num_rounds": 1,
        "seed": 1234,
    }
    assert summary["resources"] == {"fl_total_optimizer_steps": 2664, "local_optimizer_steps_per_site": 888}
    assert summary["local"]["best_site"]["site_id"] == 3
    assert list(summary["local"]["sites"]) == ["1", "2", "3"]
    assert summary["local"]["equal_site_summary"]["test"]["accuracy"] == pytest.approx(
        {"mean": 0.92, "population_std": 0.016329931618554495, "min": 0.90, "max": 0.94}
    )
    assert summary["local"]["equal_site_summary"]["test"]["macro_f1"] == pytest.approx(
        {"mean": 0.91, "population_std": 0.016329931618554495, "min": 0.89, "max": 0.93}
    )
    primary_test_deltas = summary["deltas"]["primary_fl_final"]["test"]
    expected_site_deltas = {
        "1": {"accuracy": 0.05, "macro_f1": 0.055},
        "2": {"accuracy": 0.03, "macro_f1": 0.035},
        "3": {"accuracy": 0.01, "macro_f1": 0.015},
    }
    for site_id, expected in expected_site_deltas.items():
        assert primary_test_deltas["minus_each_site"][site_id] == pytest.approx(expected)
    assert primary_test_deltas["minus_equal_site_mean"] == pytest.approx({"accuracy": 0.03, "macro_f1": 0.035})
    assert primary_test_deltas["minus_validation_selected_best"] == pytest.approx({"accuracy": 0.01, "macro_f1": 0.015})
    assert summary["bionemo_accuracy"]["reference_accuracy"] == 0.966
    accuracy_gaps = summary["bionemo_accuracy"]["observed_minus_reference"]
    assert accuracy_gaps["local_sites"] == pytest.approx({"1": -0.066, "2": -0.046, "3": -0.026})
    assert {name: value for name, value in accuracy_gaps.items() if name != "local_sites"} == pytest.approx(
        {
            "initialization": -0.626,
            "local_equal_site_mean": -0.046,
            "local_validation_selected_best": -0.026,
            "primary_fl_final": -0.016,
            "secondary_selected_fl": -0.01,
        }
    )


def test_best_local_tie_breaks_on_lowest_site_id_after_validation_metrics(tmp_path):
    module = _load_module()
    args, report_paths, _manifest_path, _output_path = _build_campaign(tmp_path, module)
    _mutate_report(report_paths["site_2"]["validation"], lambda report: report.update(accuracy=0.93))

    summary = module.summarize(args)

    assert summary["local"]["best_site"]["site_id"] == 2
    assert summary["deltas"]["primary_fl_final"]["test"]["minus_validation_selected_best"] == pytest.approx(
        {"accuracy": 0.03, "macro_f1": 0.035}
    )


@pytest.mark.parametrize(
    "values, message",
    (
        (["1=a", "2=b"], "exactly three times"),
        (["1=a", "2=b", "2=c"], "duplicate site ID 2"),
        (["1=a", "2=b", "4=c"], "unsupported site ID 4"),
    ),
)
def test_site_inputs_require_exact_distinct_ids(values, message):
    module = _load_module()

    with pytest.raises(ValueError, match=message):
        module._parse_site_paths(values, "--site-validation")


def test_summarize_rejects_checkpoint_hash_mismatch(tmp_path):
    module = _load_module()
    args, report_paths, _manifest_path, _output_path = _build_campaign(tmp_path, module)
    _mutate_report(report_paths["site_1"]["test"], lambda report: report.update(checkpoint_sha256="9" * 64))

    with pytest.raises(ValueError, match="does not match the campaign manifest"):
        module.summarize(args)


def test_summarize_rejects_mismatched_current_signatures(tmp_path):
    module = _load_module()
    args, report_paths, _manifest_path, _output_path = _build_campaign(tmp_path, module)
    _mutate_report(
        report_paths["site_1"]["validation"],
        lambda report: report["evaluation_signature"].update(seed=999),
    )

    with pytest.raises(ValueError, match="Validation evaluation_signature values do not match"):
        module.summarize(args)


def test_summarize_rejects_cross_split_setting_mismatch(tmp_path):
    module = _load_module()
    args, report_paths, _manifest_path, _output_path = _build_campaign(tmp_path, module)
    for role in report_paths:
        _mutate_report(
            report_paths[role]["test"],
            lambda report: report["evaluation_signature"].update(global_batch_size=64),
        )

    with pytest.raises(ValueError, match="do not match apart from split identity"):
        module.summarize(args)


def test_summarize_rejects_missing_current_signature_field(tmp_path):
    module = _load_module()
    args, report_paths, _manifest_path, _output_path = _build_campaign(tmp_path, module)
    _mutate_report(
        report_paths["initialization"]["validation"],
        lambda report: report["evaluation_signature"].pop("dataset_manifest_sha256"),
    )

    with pytest.raises(ValueError, match="does not use the current evaluation_signature"):
        module.summarize(args)


@pytest.mark.parametrize("location", ("signature", "manifest"))
def test_summarize_rejects_split_role_mismatch(tmp_path, location):
    module = _load_module()
    args, report_paths, _manifest_path, _output_path = _build_campaign(tmp_path, module)

    def mutate(report):
        if location == "signature":
            report["evaluation_signature"]["split_role"] = "test"
        else:
            report["dataset_manifest"]["split_role"] = "test"

    _mutate_report(report_paths["site_2"]["validation"], mutate)

    with pytest.raises(ValueError, match="split_role must be 'validation'"):
        module.summarize(args)


@pytest.mark.parametrize("field", ("num_examples", "expected_rows", "observed_rows", "unique_rows"))
def test_summarize_rejects_incomplete_coverage(tmp_path, field):
    module = _load_module()
    args, report_paths, _manifest_path, _output_path = _build_campaign(tmp_path, module)

    def mutate(report):
        if field == "num_examples":
            report[field] = 2999
        else:
            report["evaluation_coverage"][field] = 2999

    _mutate_report(report_paths["site_3"]["test"], mutate)

    with pytest.raises(ValueError, match="num_examples=3000|complete 3000-row coverage"):
        module.summarize(args)


def test_summarize_rejects_malformed_dataset_file_identity(tmp_path):
    module = _load_module()
    args, report_paths, _manifest_path, _output_path = _build_campaign(tmp_path, module)
    _mutate_report(
        report_paths["site_3"]["test"],
        lambda report: report["dataset_manifest"]["file_identity"].pop("bytes"),
    )

    with pytest.raises(ValueError, match="split file identity is malformed"):
        module.summarize(args)


def test_campaign_manifest_requires_all_local_site_hashes(tmp_path):
    module = _load_module()
    args, _report_paths, manifest_path, _output_path = _build_campaign(tmp_path, module)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint_sha256"]["local_sites"].pop("3")
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="must contain exactly site IDs 1, 2, and 3"):
        module.summarize(args)


def test_campaign_manifest_rejects_boolean_format_version(tmp_path):
    module = _load_module()
    args, _report_paths, manifest_path, _output_path = _build_campaign(tmp_path, module)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = True
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="Unsupported campaign manifest format_version"):
        module.summarize(args)
