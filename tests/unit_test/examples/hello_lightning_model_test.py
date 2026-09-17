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

import pytest

pl = pytest.importorskip("pytorch_lightning")
torch = pytest.importorskip("torch")
pytest.importorskip("torchmetrics")
DataLoader = torch.utils.data.DataLoader
TensorDataset = torch.utils.data.TensorDataset


def _load_model_module():
    repo_root = Path(__file__).parents[3]
    module_path = repo_root / "examples" / "hello-world" / "hello-lightning" / "model.py"
    spec = importlib.util.spec_from_file_location("hello_lightning_model", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_validation_uses_recipe_metric_without_renaming_test_metric():
    module = _load_model_module()
    model = module.LitNet()
    loader = DataLoader(
        TensorDataset(torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 3])),
        batch_size=2,
    )
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    validation_metrics = trainer.validate(model, dataloaders=loader, verbose=False)[0]
    test_metrics = trainer.test(model, dataloaders=loader, verbose=False)[0]

    assert "accuracy" in validation_metrics
    assert "test_acc_epoch" in test_metrics
    assert "accuracy" not in test_metrics
