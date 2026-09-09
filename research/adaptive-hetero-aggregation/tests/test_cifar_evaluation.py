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

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_DIR / "cifar10_evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import eval_split  # noqa: E402
from evaluate_result import checkpoint_meta, checkpoint_state_dict, find_server_checkpoint  # noqa: E402
from protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    canonical_config_hash,
    common_run_config,
    condition_config,
    method_run_config,
)
from run_campaign import _completed_keys  # noqa: E402
from summarize_results import (  # noqa: E402
    _load_rows,
    render_markdown,
    summarize,
    validate_complete_matrix,
    validate_config_provenance,
)


def _campaign_args(methods=None):
    return SimpleNamespace(
        methods=methods or ["fedavg", "adaptive"],
        n_clients=8,
        num_rounds=50,
        aggregation_epochs=4,
        batch_size=64,
        lr=5e-2,
        validation_fraction=0.10,
        num_workers=2,
        num_threads=None,
        gpu_config=None,
        eval_batch_size=256,
        eval_num_workers=0,
        eval_device=None,
        fedprox_mu=0.01,
        fedce_mode="plus",
        sample_exponent=0.65,
        representation_exponent=0.70,
        metric_prior_strength=100.0,
        max_blend_factor=0.20,
        activation_warmup_rounds=3,
        activation_patience=2,
        min_weight=0.0,
        max_weight=1.0,
        allow_changing_cohort_evidence=False,
    )


def _result_row(method: str, participation: float, seed: int, accuracy: float, args=None, alpha: float = 0.1) -> dict:
    args = args or _campaign_args()
    common = common_run_config(
        n_clients=args.n_clients,
        num_rounds=args.num_rounds,
        aggregation_epochs=args.aggregation_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        validation_fraction=args.validation_fraction,
        num_workers=args.num_workers,
        num_threads=args.num_threads,
        gpu_config=args.gpu_config,
        eval_batch_size=args.eval_batch_size,
        eval_num_workers=args.eval_num_workers,
        eval_device=args.eval_device,
    )
    method_config = method_run_config(
        method,
        fedprox_mu=args.fedprox_mu,
        fedce_mode=args.fedce_mode,
        sample_exponent=args.sample_exponent,
        representation_exponent=args.representation_exponent,
        metric_prior_strength=args.metric_prior_strength,
        max_blend_factor=args.max_blend_factor,
        activation_warmup_rounds=args.activation_warmup_rounds,
        activation_patience=args.activation_patience,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        allow_changing_cohort_evidence=args.allow_changing_cohort_evidence,
    )
    condition = condition_config(alpha, participation, seed)
    experiment = {
        "protocol_version": PROTOCOL_VERSION,
        "common": common,
        "method": method_config,
        "condition": condition,
    }
    row = {
        "protocol_version": PROTOCOL_VERSION,
        "method": method,
        "alpha": alpha,
        "participation_rate": participation,
        "seed": seed,
        "global_accuracy": accuracy,
        "worst_client_accuracy": accuracy - 0.10,
        "validation_fraction": args.validation_fraction,
        "common_config": common,
        "common_config_hash": canonical_config_hash(common),
        "method_config": method_config,
        "method_config_hash": canonical_config_hash(method_config),
        "condition_config": condition,
        "experiment_config": experiment,
        "experiment_config_hash": canonical_config_hash(experiment),
    }
    if method == "adaptive":
        row["adaptive_telemetry"] = {
            "adaptive_aggregation_rounds": 50,
            "adaptive_active_rounds": 20,
            "adaptive_activation_rate": 0.4,
            "adaptive_mean_active_blend_factor": 0.12,
            "adaptive_max_observed_blend_factor": 0.18,
            "adaptive_cohort_change_count": 3 if participation < 1.0 else 0,
        }
    else:
        row["adaptive_telemetry"] = {}
    return row


def test_find_server_checkpoint_prefers_final_round_model(tmp_path):
    server_dir = tmp_path / "server" / "simulate_job" / "app_server"
    server_dir.mkdir(parents=True)
    best = server_dir / "best_FL_global_model.pt"
    final = server_dir / "FL_global_model.pt"
    best.write_bytes(b"best")
    final.write_bytes(b"final")

    assert find_server_checkpoint(str(tmp_path)) == final


def test_checkpoint_state_dict_and_meta_support_nvflare_pt_format(tmp_path):
    path = tmp_path / "FL_global_model.pt"
    expected = {"weight": torch.tensor([1.0, 2.0])}
    telemetry = {"adaptive_active_rounds": 4, "adaptive_activation_rate": 0.4}
    torch.save({"model": expected, "meta_props": telemetry}, path)

    loaded = checkpoint_state_dict(path)

    assert set(loaded) == {"weight"}
    assert torch.equal(loaded["weight"], expected["weight"])
    assert checkpoint_meta(path) == telemetry


def test_integer_allocation_preserves_total_and_nonnegativity():
    allocated = eval_split._integer_allocation(17, np.asarray([0.1, 0.2, 0.7]))

    assert allocated.sum() == 17
    assert np.all(allocated >= 0)


def test_train_validation_split_is_disjoint_and_exhaustive(tmp_path, monkeypatch):
    assignment_root = tmp_path / "assignment"
    train_root = tmp_path / "train"
    validation_root = tmp_path / "validation"
    assignment_root.mkdir()
    site1 = np.arange(0, 20, dtype=np.int64)
    site2 = np.arange(20, 40, dtype=np.int64)
    np.save(assignment_root / "site-1.npy", site1)
    np.save(assignment_root / "site-2.npy", site2)
    labels = np.asarray([index % 10 for index in range(40)], dtype=np.int64)
    monkeypatch.setattr(eval_split, "load_cifar10_data", lambda: labels)

    eval_split.create_train_validation_splits(
        assignment_root=str(assignment_root),
        train_output_root=str(train_root),
        validation_output_root=str(validation_root),
        n_clients=2,
        seed=7,
        validation_fraction=0.20,
    )

    for site_name, assigned in (("site-1", site1), ("site-2", site2)):
        train = np.load(train_root / f"{site_name}.npy")
        validation = np.load(validation_root / f"{site_name}.npy")
        assert set(train).isdisjoint(set(validation))
        assert set(train) | set(validation) == set(assigned)
        assert len(validation) == 4
        assert len(train) == 16


def test_training_clients_use_held_out_training_validation_not_cifar_test():
    client_files = (
        "baseline_sgd_client.py",
        "fedprox_client.py",
        "scaffold_client.py",
        "fedce_client.py",
        "adaptive_client.py",
    )
    for filename in client_files:
        source = (EVAL_DIR / filename).read_text()
        assert "create_local_datasets" in source
        assert "train=False" not in source
        assert "create_datasets(" not in source


def test_summary_reports_ci_paired_deltas_and_activation_rate():
    rows = []
    for seed, adaptive, fedavg in ((7, 0.80, 0.75), (19, 0.82, 0.78), (31, 0.81, 0.77)):
        rows.extend(
            [
                _result_row("adaptive", 1.0, seed, adaptive),
                _result_row("fedavg", 1.0, seed, fedavg),
            ]
        )

    result = summarize(rows)

    adaptive_summary = next(item for item in result["summaries"] if item["method"] == "adaptive")
    assert adaptive_summary["metrics"]["global_accuracy"]["n"] == 3
    assert adaptive_summary["metrics"]["global_accuracy"]["ci95_low"] is not None
    assert adaptive_summary["adaptive_telemetry"]["activation_rate"]["mean"] == pytest.approx(0.4)
    paired = result["paired_comparisons"][0]
    assert paired["seeds"] == [7, 19, 31]
    assert paired["delta_reference_minus_baseline"]["global_accuracy"]["mean"] > 0.0


def test_complete_matrix_rejects_missing_partial_participation_row():
    methods = ["fedavg", "adaptive"]
    seeds = [7, 19]
    rows = []
    for method in methods:
        for participation in (1.0, 0.75):
            for seed in seeds:
                if method == "fedavg" and participation == 0.75 and seed == 19:
                    continue
                rows.append(_result_row(method, participation, seed, 0.75))

    with pytest.raises(ValueError, match="incomplete CIFAR-10 evidence matrix"):
        validate_complete_matrix(rows, methods, [0.1], [1.0, 0.75], seeds)


def test_complete_matrix_and_markdown_include_partial_participation_and_activation():
    methods = ["fedavg", "adaptive"]
    seeds = [7, 19]
    rows = []
    for method in methods:
        for participation in (1.0, 0.75):
            for seed in seeds:
                accuracy = 0.80 if method == "adaptive" else 0.76
                rows.append(_result_row(method, participation, seed, accuracy))

    validate_complete_matrix(rows, methods, [0.1], [1.0, 0.75], seeds)
    markdown = render_markdown(summarize(rows))

    assert "participation=100%" in markdown
    assert "participation=75%" in markdown
    assert "fedavg" in markdown
    assert "adaptive" in markdown
    assert "Adaptive activation" in markdown
    assert "40.0%" in markdown
    assert "Paired adaptive-minus-baseline deltas" in markdown
    assert "+4.00" in markdown


def test_provenance_rejects_mixed_method_configuration():
    rows = [_result_row("fedavg", 1.0, 7, 0.75), _result_row("adaptive", 1.0, 7, 0.80)]
    changed = _campaign_args()
    changed.max_blend_factor = 0.35
    rows.append(_result_row("adaptive", 1.0, 19, 0.81, args=changed))

    with pytest.raises(ValueError, match="multiple method configurations"):
        validate_config_provenance(rows)


def test_provenance_rejects_mixed_execution_configuration():
    rows = [_result_row("fedavg", 1.0, 7, 0.75), _result_row("adaptive", 1.0, 7, 0.80)]
    changed = _campaign_args()
    changed.num_threads = 2
    rows.append(_result_row("adaptive", 1.0, 19, 0.81, args=changed))

    with pytest.raises(ValueError, match="multiple common experiment configurations"):
        validate_config_provenance(rows)


def test_provenance_rejects_missing_or_inconsistent_adaptive_telemetry():
    row = _result_row("adaptive", 0.75, 7, 0.80)
    row["adaptive_telemetry"]["adaptive_activation_rate"] = 0.9

    with pytest.raises(ValueError, match="inconsistent activation rate"):
        validate_config_provenance([row])


def test_campaign_resume_ignores_stale_protocol_and_mismatched_configuration(tmp_path):
    args = _campaign_args()
    path = tmp_path / "runs.jsonl"
    current = _result_row("adaptive", 0.75, 19, 0.8, args=args)
    stale = dict(current, protocol_version="older_protocol", seed=7)
    changed = _campaign_args()
    changed.max_blend_factor = 0.35
    mismatched = _result_row("adaptive", 0.75, 31, 0.8, args=changed)
    path.write_text(json.dumps(current) + "\n" + json.dumps(stale) + "\n" + json.dumps(mismatched) + "\n")

    assert _completed_keys(path, args) == {("adaptive", 0.1, 0.75, 19)}
    assert _load_rows(str(path), protocol_version=PROTOCOL_VERSION) == [current, mismatched]
