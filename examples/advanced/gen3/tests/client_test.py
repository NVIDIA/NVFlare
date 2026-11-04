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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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


@pytest.fixture
def client_main(monkeypatch, tmp_path):
    training = load_module("client")
    sites = {
        site: [{"object_id": f"{site}-{i}", "file_name": f"{site}-{i}.csv"} for i in range(2)]
        for site in ("site-1", "site-2")
    }
    records = {record["object_id"]: record for assigned in sites.values() for record in assigned}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sites": sites}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["client.py", "--manifest", str(manifest)])
    for name, value in {
        "GEN3_ENDPOINT": "https://commons.example",
        "GEN3_CLIENT_ID": "study-client",
        "GEN3_CLIENT_SECRET": "test-secret",
    }.items():
        monkeypatch.setenv(name, value)

    def download(object_id, directory):
        (Path(directory) / records[object_id]["file_name"]).write_text("x0,x1,y\n1,0,2.5\n0,1,-2.5\n")
        return True

    files = SimpleNamespace(download_single=Mock(side_effect=download))
    monkeypatch.setattr(training, "Gen3Auth", Mock(return_value=object()))
    monkeypatch.setattr(training, "Gen3File", Mock(return_value=files))
    flare = SimpleNamespace(
        init=Mock(),
        system_info=Mock(return_value={"site_name": "site-1"}),
        is_running=Mock(side_effect=[True, True, False]),
        receive=Mock(
            side_effect=[
                SimpleNamespace(params={training.NPConstants.NUMPY_KEY: np.zeros(3)}, current_round=round_number)
                for round_number in range(2)
            ]
        ),
        send=Mock(),
        FLModel=SimpleNamespace,
    )
    monkeypatch.setattr(training, "flare", flare)
    return training, sites, files.download_single


@pytest.mark.parametrize("site", ["site-1", "site-2"])
def test_main_downloads_only_assigned_objects_once_across_rounds(client_main, site):
    training, sites, download = client_main
    training.flare.system_info.return_value = {"site_name": site}
    training.main()

    assert [call.args[0] for call in download.call_args_list] == [record["object_id"] for record in sites[site]]
    training.Gen3Auth.assert_called_once_with(
        endpoint="https://commons.example", client_credentials=("study-client", "test-secret")
    )
    training.Gen3File.assert_called_once_with(auth_provider=training.Gen3Auth.return_value)
    sent = [call.args[0] for call in training.flare.send.call_args_list]
    assert [model.current_round for model in sent] == [0, 1]
    assert all(model.meta[training.FLMetaKey.NUM_STEPS_CURRENT_ROUND] == 4 for model in sent)


@pytest.mark.parametrize("failed_object", [0, 1])
def test_main_sends_no_model_when_a_download_fails(client_main, failed_object):
    training, sites, download = client_main
    failed_id = sites["site-1"][failed_object]["object_id"]
    successful_download = download.side_effect
    download.side_effect = lambda object_id, directory: (
        False if object_id == failed_id else successful_download(object_id, directory)
    )

    with pytest.raises(RuntimeError, match=f"Download failed for {failed_id}"):
        training.main()

    assert download.call_count == failed_object + 1
    training.flare.receive.assert_not_called()
    training.flare.send.assert_not_called()


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
