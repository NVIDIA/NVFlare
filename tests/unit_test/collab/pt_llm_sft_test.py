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

import importlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_ADVANCED_EXAMPLES_ROOT = Path(__file__).resolve().parents[3] / "examples" / "advanced"


@pytest.fixture
def llm_module(monkeypatch):
    monkeypatch.syspath_prepend(str(_ADVANCED_EXAMPLES_ROOT))
    return importlib.import_module("collab.pt_llm_sft.pt_llm_sft")


def test_recipe_finalizes_and_env_preserves_gpu_config(llm_module, tmp_path):
    args = llm_module.define_parser().parse_args([])
    args.data_root = tmp_path / "data"
    args.output_root = tmp_path / "results"
    args.workspace_root = str(tmp_path / "workspace")
    args.num_clients = 2
    args.gpu_config = "0,1"

    recipe = llm_module.make_recipe(args)
    env = llm_module.make_env(args)

    job = recipe.finalize()
    assert recipe.finalize() is job
    assert env.num_clients == 2
    assert env.gpu_config == "0,1"


def test_average_model_states_uses_example_counts(llm_module):
    updates = {
        "site-1": {
            "weights": {
                "weight": torch.tensor([1.0, 3.0], dtype=torch.float16),
                "step": torch.tensor(1, dtype=torch.int64),
            },
            "num_examples": 1,
            "train_loss": 2.0,
        },
        "site-2": {
            "weights": {
                "weight": torch.tensor([5.0, 7.0], dtype=torch.float16),
                "step": torch.tensor(2, dtype=torch.int64),
            },
            "num_examples": 3,
            "train_loss": 6.0,
        },
    }

    averaged, average_loss = llm_module.average_model_states(updates, min_clients=2)

    torch.testing.assert_close(averaged["weight"], torch.tensor([4.0, 6.0], dtype=torch.float16))
    assert averaged["weight"].dtype == torch.float16
    assert averaged["step"].item() == 1
    assert average_loss == pytest.approx(5.0)


def test_average_model_states_requires_quorum(llm_module):
    update = {
        "site-1": {
            "weights": {"weight": torch.tensor([1.0])},
            "num_examples": 1,
            "train_loss": 1.0,
        }
    }

    with pytest.raises(RuntimeError, match="need at least 2"):
        llm_module.average_model_states(update, min_clients=2)


def test_prepare_synthetic_data_writes_each_site(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(_ADVANCED_EXAMPLES_ROOT))
    prepare_module = importlib.import_module("collab.pt_llm_sft.prepare_data")

    manifest = prepare_module.prepare_data(tmp_path, num_clients=2)

    assert manifest["sites"] == ["site-1", "site-2"]
    assert manifest["site_counts"] == {
        "site-1": {"train": 6, "valid": 1},
        "site-2": {"train": 6, "valid": 1},
    }
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == manifest
    for site_name in manifest["sites"]:
        assert len((tmp_path / site_name / "train.jsonl").read_text(encoding="utf-8").splitlines()) == 6
        assert len((tmp_path / site_name / "valid.jsonl").read_text(encoding="utf-8").splitlines()) == 1
