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

import numpy as np
import pytest
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_DIR / "cifar10_evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import eval_split  # noqa: E402
from evaluate_result import checkpoint_state_dict, find_server_checkpoint  # noqa: E402
from protocol import PROTOCOL_VERSION  # noqa: E402
from run_campaign import _completed_keys  # noqa: E402
from summarize_results import _load_rows, render_markdown, summarize, validate_complete_matrix  # noqa: E402


def test_find_server_checkpoint_prefers_final_round_model(tmp_path):
    server_dir = tmp_path / "server" / "simulate_job" / "app_server"
    server_dir.mkdir(parents=True)
    best = server_dir / "best_FL_global_model.pt"
    final = server_dir / "FL_global_model.pt"
    best.write_bytes(b"best")
    final.write_bytes(b"final")

    assert find_server_checkpoint(str(tmp_path)) == final


def test_checkpoint_state_dict_supports_nvflare_pt_format(tmp_path):
    path = tmp_path / "FL_global_model.pt"
    expected = {"weight": torch.tensor([1.0, 2.0])}
    torch.save({"model": expected, "meta": {}}, path)

    loaded = checkpoint_state_dict(path)

    assert set(loaded) == {"weight"}
    assert torch.equal(loaded["weight"], expected["weight"])


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


def _result_row(method: str, participation: float, seed: int, accuracy: float) -> dict:
    return {
        "method": method,
        "alpha": 0.1,
        "participation_rate": participation,
        "seed": seed,
        "global_accuracy": accuracy,
        "worst_client_accuracy": accuracy - 0.10,
    }


def test_summary_reports_ci_and_paired_split_seed_deltas():
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


def test_complete_matrix_and_markdown_include_partial_participation_main_results():
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
    assert "Paired adaptive-minus-baseline deltas" in markdown
    assert "+4.00" in markdown


def test_campaign_resume_uses_protocol_condition_seed_and_validation_fraction(tmp_path):
    path = tmp_path / "runs.jsonl"
    current = {
        "protocol_version": PROTOCOL_VERSION,
        "method": "adaptive",
        "alpha": 0.1,
        "participation_rate": 0.75,
        "seed": 19,
        "validation_fraction": 0.10,
        "global_accuracy": 0.8,
        "worst_client_accuracy": 0.7,
    }
    stale = dict(current, protocol_version="older_protocol", seed=7)
    path.write_text(json.dumps(current) + "\n" + json.dumps(stale) + "\n")

    assert _completed_keys(path) == {("adaptive", 0.1, 0.75, 19, 0.10)}
    assert _load_rows(str(path), protocol_version=PROTOCOL_VERSION) == [current]
