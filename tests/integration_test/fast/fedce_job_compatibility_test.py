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

"""Manual CPU qualification of the shipped FedCE client; requires MONAI."""

import json
import math
import os
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from nvflare.app_opt.pt.fedce import FedCEConstants
from nvflare.app_opt.pt.recipes.fedce import FedCERecipe
from nvflare.recipe import SimEnv

ROOT = Path(__file__).resolve().parents[3]
RESEARCH = ROOT / "research/fed-ce/jobs/fedce_prostate"
FIXTURE = ROOT / "tests/integration_test/data/fedce_task/fedce_observer.py"
CLIENTS = ["client_I2CVB", "client_MSD"]


def _site_record(root, site, name):
    paths = [p for p in root.rglob(name) if site in p.parts and p.parent.name == "fedce_evidence"]
    assert len(paths) == 1, (site, name, paths)
    return torch.load(paths[0], map_location="cpu", weights_only=True)


def _synthetic_segmentation_data(root):
    images = root / "dataset_2D"
    datalists = root / "datalist_2D"
    images.mkdir(parents=True)
    datalists.mkdir()
    rows, cols = np.mgrid[:32, :32]
    rng = np.random.default_rng(173)
    for site_index, site in enumerate(CLIENTS):
        records = []
        for index in range(4):
            label = ((rows - 16) ** 2 + (cols - 15 - site_index) ** 2 < (6 + index) ** 2).astype(np.float32)
            image = np.clip(30 + 170 * label + rng.normal(0, 10, label.shape), 0, 255).astype(np.float32)
            image_name, label_name = f"{site}_{index}_image.npy", f"{site}_{index}_label.npy"
            np.save(images / image_name, image)
            np.save(images / label_name, 255 * label)
            records.append({"image": image_name, "label": label_name})
        (datalists / f"{site}.json").write_text(json.dumps({"training": records[:2], "validation": records[2:]}))


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("fedce_mode", ["plus", "times"])
def test_shipped_fedce_job_preserves_previous_model_and_score_history(tmp_path, monkeypatch, fedce_mode):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(ROOT), os.environ.get("PYTHONPATH", ""))))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.delenv("NVFLARE_HOME", raising=False)
    data = tmp_path / "data"
    _synthetic_segmentation_data(data)
    model_class = runpy.run_path(str(RESEARCH / "model.py"))["UNet"]
    torch.manual_seed(173)
    model = model_class(in_channels=1, out_channels=1)
    initial = tmp_path / "initial.pt"
    torch.save({"model": model.state_dict()}, initial)
    source = RESEARCH / "client.py"
    script = tmp_path / "shipped_client.py"
    shutil.copy2(source, script)
    observer = tmp_path / "fedce_observer.py"
    shutil.copy2(FIXTURE, observer)
    model_script = tmp_path / "model.py"
    shutil.copy2(RESEARCH / "model.py", model_script)
    recipe = FedCERecipe(
        name=f"fedce_{fedce_mode}_job",
        model={"path": "model.UNet", "args": {"in_channels": 1, "out_channels": 1}},
        initial_ckpt=str(initial),
        trainable_param_names=[name for name, _ in model.named_parameters()],
        min_clients=2,
        num_rounds=3,
        train_script=str(observer),
        train_args=[
            "--data-root",
            str(data),
            "--batch-size",
            "2",
            "--cache-rate",
            "0",
            "--local-epochs",
            "1",
            "--num-workers",
            "0",
            "--seed",
            "173",
        ],
        fedce_mode=fedce_mode,
        key_metric="dice",
    )
    recipe.add_server_file(str(model_script))
    recipe.add_client_file(str(model_script))
    recipe.add_client_file(str(script))
    env = SimEnv(clients=CLIENTS, num_threads=2, workspace_root=str(tmp_path / "simulation"))
    run = recipe.execute(env)
    result = run.get_result(timeout=900, clean_up=False)
    assert result is not None and not env.last_run_failed
    root = Path(result)
    deployed = list(root.rglob("shipped_client.py"))
    assert len(deployed) >= 2 and all(p.read_bytes() == source.read_bytes() for p in deployed)

    for site in CLIENTS:
        minus_scores = [0.0]
        pids = set()
        for current_round in range(3):
            incoming = _site_record(root, site, f"{current_round}_input.pt")
            output = _site_record(root, site, f"{current_round}_output.pt")
            pids.add(output["pid"])
            assert len(output["validations"]) == (1 if current_round == 0 else 2)
            assert all(math.isfinite(score) and 0 <= score <= 1 for score in output["validations"])
            assert output["metrics"]["dice"] == output["validations"][0]
            assert any(torch.count_nonzero(value).item() for value in output["params"].values())
            assert all(torch.isfinite(value).all() for value in output["params"].values())
            if current_round > 0:
                weights = incoming["meta"][FedCEConstants.CONTRIBUTION_WEIGHTS]
                assert set(weights) == set(CLIENTS)
                assert sum(weights.values()) == pytest.approx(1.0)
                assert all(0 < weight < 1 for weight in weights.values())
                minus_scores.append(output["validations"][1])
            assert output["meta"][FedCEConstants.MINUS_MODEL_SCORE] == pytest.approx(1 - np.mean(minus_scores))
        assert len(pids) == 1

    model_files = list(root.rglob("FL_global_model.pt"))
    assert len(model_files) == 1
    final = torch.load(model_files[0], map_location="cpu", weights_only=True)["model"]
    assert all(torch.isfinite(value).all() for value in final.values())
    assert any(not torch.equal(final[name], value) for name, value in model.state_dict().items())
    assert not list(root.rglob("completion.json"))
