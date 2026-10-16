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
from pathlib import Path

import numpy as np
import pytest


def load_module(name):
    path = Path(__file__).parents[1] / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"gen3_example_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_matches_independent_closed_form_control(tmp_path):
    training = load_module("client")
    verification = load_module("verify_result")
    manifest = load_module("prepare_data").prepare_data(tmp_path)
    arrays = [
        training.load_arrays([tmp_path / record["file_name"] for record in records])
        for records in manifest["sites"].values()
    ]
    weights = np.zeros(3)
    for _ in range(3):
        updates = [training.train_round(weights, features, labels)[0] for features, labels in arrays]
        weights = np.average(updates, axis=0, weights=[len(labels) for _, labels in arrays])
    expected = verification.expected_model(tmp_path / "manifest.json", tmp_path)
    np.testing.assert_allclose(weights, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("authoring_path", ["example", "data"])
def test_export_packages_manifest_beside_trainer_without_credentials(monkeypatch, tmp_path, authoring_path):
    load_module("prepare_data").prepare_data(tmp_path / "data")
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] if authoring_path == "example" else tmp_path))
    monkeypatch.setenv("GEN3_CLIENT_SECRET", "must-not-enter-job-package")
    load_module("job").make_recipe(tmp_path / "data/manifest.json").export(str(tmp_path / "jobs"))
    files = list((tmp_path / "jobs").rglob("*"))
    trainers = [path for path in files if path.name == "client.py"]
    assert trainers
    for path in trainers:
        assert (path.parent / "manifest.json").is_file()
    for path in files:
        if path.is_file():
            assert b"must-not-enter-job-package" not in path.read_bytes()
