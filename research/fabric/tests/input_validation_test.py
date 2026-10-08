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

"""Reject invalid inputs using fictional manifests and small CPU feature arrays."""

import csv
from types import SimpleNamespace
from unittest.mock import Mock

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from core import training
from fabric_common import SITES, check_inputs


@pytest.mark.parametrize("variant", ["pooling", "topk"])
@pytest.mark.parametrize("stage", ["training", "evaluation"])
@pytest.mark.parametrize("loader_kind", ["dataloader", "iterator"])
def test_empty_loaders_raise_without_updating_weights(variant, stage, loader_kind, monkeypatch):
    args = SimpleNamespace(
        amp=False, model_variant=variant, threshold=0.5, max_instances=4, batch_size=2, num_workers=0
    )
    frame = pd.DataFrame(columns=[*training.PREDICTION_METADATA, "recurrence_label"])
    loader = training.make_loader(frame, args, seed=42, training=False, pin_memory=False)
    if loader_kind == "iterator":
        loader = iter(loader)
    device = torch.device("cpu")
    model = torch.nn.Linear(4, 2)
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    with pytest.raises(ValueError, match=f"{stage.capitalize()} loader yielded no patients"):
        if stage == "training":
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            training.train_one_epoch(
                model,
                loader,
                torch.nn.CrossEntropyLoss(),
                optimizer,
                device,
                args,
                torch.amp.GradScaler("cuda", enabled=False),
            )
        else:
            monkeypatch.setattr(training, "make_loader", lambda *args, **kwargs: loader)
            criterion_frame = pd.DataFrame({"recurrence_label": [0, 1]})
            training.evaluate_model(model, frame, criterion_frame, args, seed=42, device=device)
    assert all(torch.equal(value, initial[key]) for key, value in model.state_dict().items())


@pytest.fixture(params=["uni", "virchow2"])
def feature_inputs(make_fold, request):
    case = make_fold(encoder=request.param)
    manifest_root = case.root / "manifests"
    feature_root = case.root / "features"
    feature_root.mkdir()

    def create(suffix, wrong_width=False, single_patch=False):
        paths = [feature_root / f"{name}{suffix}" for name in ("first", "second")]
        for index, path in enumerate(paths):
            width = case.setup["input_dim"] + int(wrong_width and index == 1)
            values = np.zeros((width,) if single_patch else (2, width), dtype=np.float32)
            if suffix in {".h5", ".hdf5"}:
                with h5py.File(path, "w") as handle:
                    handle.create_dataset("features", data=values)
            elif suffix == ".npy":
                np.save(path, values)
            else:
                torch.save({"features": torch.from_numpy(values)}, path)
        features = ";".join(f"synthetic-features/{path.name}" for path in paths)
        # The study runner requires 804 cases; all identifiers here are fabricated.
        for fold in case.setup["folds"]:
            folder = manifest_root / case.setup["experiment"] / f"fold_{fold}"
            folder.mkdir(parents=True)
            for site in (*SITES, "test"):
                filename = "global_test_manifest.csv" if site == "test" else f"client_{site}_train_manifest.csv"
                with (folder / filename).open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(("patient_id", "site", "recurrence_label", "feature_paths"))
                    for index in range(804):
                        assigned_site = "test" if index % 5 == fold else SITES[index % len(SITES)]
                        if assigned_site == site:
                            writer.writerow((f"synthetic-{index}", site, index % 2, features))
        return case.setup, manifest_root, {"synthetic-features": str(feature_root)}, paths

    return create


@pytest.mark.parametrize("suffix", [".h5", ".hdf5", ".npy", ".pt", ".pth"])
def test_preflight_accepts_expected_widths_and_checks_unique_remapped_files(feature_inputs, suffix, monkeypatch):
    setup, manifests, mapping, paths = feature_inputs(suffix)
    reader = Mock(wraps=training.read_feature_shape)
    monkeypatch.setattr(training, "read_feature_shape", reader)
    result = check_inputs(setup, manifests, mapping)
    assert result["patients"] == 804
    assert result["feature_files"] == 2
    assert result["feature_bytes"] == sum(path.stat().st_size for path in paths)
    assert reader.call_count == 2
    assert {call.args[0] for call in reader.call_args_list} == set(paths)


@pytest.mark.parametrize("suffix", [".h5", ".hdf5", ".npy", ".pt", ".pth"])
def test_preflight_rejects_wrong_width_before_training(feature_inputs, suffix):
    setup, manifests, mapping, paths = feature_inputs(suffix, wrong_width=True)
    with pytest.raises(ValueError, match="Feature dimension mismatch") as error:
        check_inputs(setup, manifests, mapping)
    assert str(paths[1]) in str(error.value)
    assert f"expected {setup['input_dim']}" in str(error.value)
    assert f"got {setup['input_dim'] + 1}" in str(error.value)


def test_preflight_keeps_single_patch_feature_support(feature_inputs):
    setup, manifests, mapping, _ = feature_inputs(".npy", single_patch=True)
    assert check_inputs(setup, manifests, mapping)["feature_files"] == 2


def test_manifest_only_validation_does_not_open_features(feature_inputs):
    setup, manifests, mapping, paths = feature_inputs(".npy")
    for path in paths:
        path.unlink()
    result = check_inputs(setup, manifests, mapping, check_features=False)
    assert result["patients"] == 804
    assert result["feature_bytes"] == 0
