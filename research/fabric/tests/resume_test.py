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

"""Verify round-boundary recovery using generated states and audit records."""

import shutil
from pathlib import Path

import pytest
import torch
from fabric_aggregator import FabricFedAvg
from fabric_common import SITES, read_json, sha256_file, state_digest, training_args, write_json
from fabric_resume import atomic_save, inspect_fold, prepare_resume, validate_resume_run


def run_round(case, aggregator, state, client_update, relative_round, offset=0):
    for site in reversed(SITES):
        aggregator.accept_model(client_update(case, site, state, relative_round, offset))
    rng = torch.get_rng_state().clone()
    output = aggregator.aggregate_model()
    assert torch.equal(rng, torch.get_rng_state())
    assert output.current_round == relative_round
    return output.params


@pytest.fixture(params=["pooling", "topk"])
def resumable_source_run(make_fold, client_update, request):
    case = make_fold(variant=request.param)
    code = case.root / "fabric"
    shutil.copytree(Path(__file__).resolve().parents[1] / "fabric", code, ignore=shutil.ignore_patterns("__pycache__"))
    run_root = case.folder.parents[1]
    write_json(
        run_root / "code_sha256.json",
        {str(path.relative_to(case.root)): sha256_file(path) for path in sorted(code.rglob("*.py"))},
    )
    config = {key: case.setup[key] for key in ("experiment", "model_name", "encoder", "input_dim")}
    config.update(
        variant=case.setup.get("variant", "pooling"),
        model_options=case.setup.get("model_options", {}),
        settings=vars(training_args(case.root, case.setup, case.folder.parent, Path(case.runtime["manifest_root"]))),
        source_fold_manifests=case.runtime["manifest_root"],
        folds=[0],
    )
    write_json(case.folder.parent / "run_config.json", config)
    run_round(case, FabricFedAvg(str(case.path)), case.initial, client_update, 0)
    return case


def test_resume_accepts_unchanged_source(resumable_source_run):
    case = resumable_source_run
    validate_resume_run(case.root, case.folder.parents[1], [case.setup], [0], Path(case.runtime["manifest_root"]))
    plan = inspect_fold(case.folder, case.setup)
    assert (plan["action"], plan["completed_rounds"]) == ("resume", 1)


@pytest.mark.parametrize(
    "relative",
    [
        "fabric/fabric_client.py",
        "fabric/fabric_aggregator.py",
        "fabric/run_fabric.py",
        "fabric/fabric_resume.py",
        "fabric/check_static.py",
        "fabric/core/client_training.py",
    ],
)
def test_resume_rejects_changed_source_with_valid_checkpoint(resumable_source_run, relative):
    case = resumable_source_run
    run_root = case.folder.parents[1]
    original_records = {path: path.read_bytes() for path in run_root.rglob("*") if path.is_file()}
    source = case.root / relative
    source.write_text(source.read_text() + "\n# Synthetic source edit after interruption.\n")
    # An intact checkpoint chain cannot prove that the training protocol is unchanged.
    assert inspect_fold(case.folder, case.setup)["completed_rounds"] == 1
    with pytest.raises(ValueError, match="Training source changed since this run") as error:
        validate_resume_run(case.root, run_root, [case.setup], [0], Path(case.runtime["manifest_root"]))
    assert relative in str(error.value)
    assert {path: path.read_bytes() for path in run_root.rglob("*") if path.is_file()} == original_records


@pytest.mark.parametrize("encoder", ["uni", "virchow2"])
@pytest.mark.parametrize("variant", ["pooling", "topk"])
@pytest.mark.parametrize("completed", [0, 3, 4])
def test_resumed_rounds_match_uninterrupted_aggregation(make_fold, client_update, encoder, variant, completed):
    baseline = make_fold("baseline", encoder, variant)
    aggregator = FabricFedAvg(str(baseline.path))
    state, expected = baseline.initial, []
    for number in range(5):
        state = run_round(baseline, aggregator, state, client_update, number)
        expected.append(state_digest(state))
    case = make_fold("resumed", encoder, variant)
    aggregator = FabricFedAvg(str(case.path))
    state = case.initial
    for number in range(completed):
        state = run_round(case, aggregator, state, client_update, number)
    pending = case.folder / "client_logs" / SITES[0] / f"round_{completed + 1}.json"
    write_json(pending, {"status": "synthetic unfinished round"})
    committed = {path.name: path.read_bytes() for path in (case.folder / "server_audit").glob("*.json")}
    plan = inspect_fold(case.folder, case.setup)
    assert (plan["action"], plan["completed_rounds"]) == ("resume", completed)
    runtime_path, attempt = prepare_resume(case.folder, plan)
    assert not pending.exists()
    assert (attempt / "previous_incomplete" / pending.relative_to(case.folder)).is_file()
    assert committed == {path.name: path.read_bytes() for path in (case.folder / "server_audit").glob("*.json")}
    runtime = read_json(runtime_path)
    assert runtime["resume_offset"] == completed
    state = torch.load(attempt / "resume_model.pt", map_location="cpu", weights_only=True)["model"]
    aggregator = FabricFedAvg(str(runtime_path))
    for relative in range(5 - completed):
        state = run_round(case, aggregator, state, client_update, relative, completed)
        assert state_digest(state) == expected[relative + completed]
    # A final aggregate alone is insufficient: FLARE must also have persisted it.
    assert inspect_fold(case.folder, case.setup)["completed_rounds"] == 4
    atomic_save({"model": state}, case.folder / "nvflare_workspace/FL_global_model.pt")
    assert inspect_fold(case.folder, case.setup)["action"] == "evaluate"
    for name in ("global_test_predictions.tsv", "fold_result.tsv", "round_logs.tsv"):
        (case.folder / name).write_text("synthetic-complete\n")
    write_json(case.folder / "nvflare_run.json", {"final_state_sha256": expected[-1]})
    assert inspect_fold(case.folder, case.setup)["action"] == "complete"


def test_final_persistence_gap_replays_only_the_uncommitted_round(make_fold, client_update):
    case = make_fold()
    aggregator, state = FabricFedAvg(str(case.path)), case.initial
    for number in range(5):
        state = run_round(case, aggregator, state, client_update, number)
    plan = inspect_fold(case.folder, case.setup)
    assert plan["completed_rounds"] == 4
    _, attempt = prepare_resume(case.folder, plan)
    assert not (case.folder / "server_audit/round_5.json").exists()
    assert (attempt / "previous_incomplete/server_audit/round_5.json").is_file()
    assert not (case.folder / "global_model_round_final.pt").exists()
    assert (attempt / "previous_incomplete/global_model_round_final.pt").is_file()


@pytest.mark.parametrize("artifact", ["manifest", "audit", "checkpoint"])
def test_changed_recovery_inputs_are_rejected(make_fold, client_update, artifact):
    case = make_fold()
    aggregator = FabricFedAvg(str(case.path))
    run_round(case, aggregator, case.initial, client_update, 0)
    if artifact == "manifest":
        (case.folder / f"client_{SITES[0]}_train_manifest.csv").write_text("changed\n")
        error, message = ValueError, "manifest changed"
    elif artifact == "audit":
        path = case.folder / "server_audit/round_1.json"
        audit = read_json(path)
        audit["input_state_sha256"] = "changed"
        write_json(path, audit)
        error, message = ValueError, "aggregation chain"
    else:
        atomic_save({"model": case.initial}, case.folder / "server_checkpoints/round_1.pt")
        error, message = RuntimeError, "No intact checkpoint"
    with pytest.raises(error, match=message):
        inspect_fold(case.folder, case.setup)


def test_interrupted_save_preserves_the_last_committed_checkpoint(tmp_path, monkeypatch):
    destination = tmp_path / "checkpoint.pt"
    atomic_save({"weight": torch.tensor([1.0])}, destination)
    original = destination.read_bytes()

    def interrupted_save(payload, path):
        path.write_bytes(b"partial checkpoint")
        raise RuntimeError("interrupted save")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(RuntimeError, match="interrupted save"):
        atomic_save({"weight": torch.tensor([2.0])}, destination)
    assert destination.read_bytes() == original
