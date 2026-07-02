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

"""Deterministic tests for the packaged conversion templates.

These run the shipped PyTorch evaluation template, the custom
``ModelAggregator`` template, and the Lightning evaluation template against
toy models so template rot is caught here rather than only in expensive,
nondeterministic LLM evals.
"""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SKILLS_ROOT = REPO_ROOT / "skills"
PT_TEMPLATES = SKILLS_ROOT / "nvflare-convert-pytorch" / "references" / "templates"
LIGHTNING_TEMPLATES = SKILLS_ROOT / "nvflare-convert-lightning" / "references" / "templates"
SHARED_TEMPLATES = SKILLS_ROOT / "_shared" / "templates"


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"template_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FloatOverflow:
    def __float__(self):
        raise OverflowError("step count too large")


def test_pytorch_eval_template_computes_metric_against_toy_model():
    torch = pytest.importorskip("torch")
    module = _load_module(PT_TEMPLATES / "client_with_eval.py")

    model = torch.nn.Linear(4, 2)
    features = torch.randn(6, 4)
    labels = torch.randint(0, 2, (6,))
    val_loader = [(features, labels)]

    metric = module.evaluate(model, val_loader, device="cpu")

    assert isinstance(metric, float)
    assert 0.0 <= metric <= 1.0


def test_pytorch_eval_template_restores_training_mode():
    # evaluate() must not leave the model in eval mode, or a later training round
    # would run with dropout/batchnorm disabled.
    torch = pytest.importorskip("torch")
    module = _load_module(PT_TEMPLATES / "client_with_eval.py")

    model = torch.nn.Linear(4, 2)
    model.train()
    features = torch.randn(4, 4)
    labels = torch.randint(0, 2, (4,))

    module.evaluate(model, [(features, labels)], device="cpu")

    assert model.training is True


def test_pytorch_eval_template_fails_closed_on_empty_data():
    pytest.importorskip("torch")
    module = _load_module(PT_TEMPLATES / "client_with_eval.py")

    with pytest.raises(RuntimeError):
        module.evaluate(_DummyModel(), [], device="cpu")


def test_custom_aggregator_template_step_weighted_average():
    import numpy as np

    from nvflare.apis.dxo import MetaKey
    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator()

    aggregator.accept_model(FLModel(params={"w": np.array([2.0])}, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 1}))
    aggregator.accept_model(FLModel(params={"w": np.array([4.0])}, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 3}))
    result = aggregator.aggregate_model()

    # (2*1 + 4*3) / (1 + 3) = 14 / 4 = 3.5
    assert result.params["w"][0] == pytest.approx(3.5)


def test_custom_aggregator_template_materializes_lazy_disk_offload_refs():
    # With enable_tensor_disk_offload=True, params can arrive as lazy references
    # exposing materialize() instead of in-memory arrays. The template must
    # materialize before the weighted-sum math rather than doing value * weight
    # on the ref (which would raise TypeError).
    import numpy as np

    from nvflare.apis.dxo import MetaKey
    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator()

    class _LazyRef:
        def __init__(self, array):
            self._array = array

        def materialize(self):
            return self._array

        def __mul__(self, other):  # pragma: no cover - must never be reached
            raise TypeError("lazy ref must be materialized before weighted math")

    aggregator.accept_model(FLModel(params={"w": _LazyRef(np.array([2.0]))}, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 1}))
    aggregator.accept_model(FLModel(params={"w": _LazyRef(np.array([4.0]))}, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 3}))
    result = aggregator.aggregate_model()

    # (2*1 + 4*3) / (1 + 3) = 3.5, computed on the materialized arrays.
    assert result.params["w"][0] == pytest.approx(3.5)


def test_custom_aggregator_template_averages_per_key_with_mismatched_keys():
    # A parameter present in only one client is averaged over just that client's
    # weight (not diluted), and a key missing from the first client does not
    # raise KeyError.
    import numpy as np

    from nvflare.apis.dxo import MetaKey
    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator()

    aggregator.accept_model(FLModel(params={"shared": np.array([2.0])}, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 1}))
    aggregator.accept_model(
        FLModel(
            params={"shared": np.array([4.0]), "only_b": np.array([9.0])},
            meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 3},
        )
    )
    result = aggregator.aggregate_model()

    # shared: (2*1 + 4*3)/(1+3) = 3.5 ; only_b: 9 present only in client B -> 9.0
    assert result.params["shared"][0] == pytest.approx(3.5)
    assert result.params["only_b"][0] == pytest.approx(9.0)


@pytest.mark.parametrize(
    "bad_steps",
    [
        -5,
        0,
        float("nan"),
        float("inf"),
        "abc",
        True,
        None,
        pytest.param(10**10000, id="oversized-int"),
        pytest.param(_FloatOverflow(), id="overflow"),
    ],
)
def test_custom_aggregator_template_falls_back_to_unit_weight_for_bad_step_counts(bad_steps):
    # Negative / non-finite / non-numeric / bool / missing / overflowing step
    # metadata must fall back to weight 1.0 (never corrupt or crash the average).
    import numpy as np

    from nvflare.apis.dxo import MetaKey
    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator()

    meta = {} if bad_steps is None else {MetaKey.NUM_STEPS_CURRENT_ROUND: bad_steps}
    aggregator.accept_model(FLModel(params={"w": np.array([2.0])}, meta=meta))
    aggregator.accept_model(FLModel(params={"w": np.array([4.0])}, meta=meta))
    result = aggregator.aggregate_model()

    # Both weights coerced to 1.0 -> plain mean (2 + 4) / 2 = 3.0.
    assert result.params["w"][0] == pytest.approx(3.0)


def test_custom_aggregator_template_resets_between_rounds():
    import numpy as np

    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator()

    aggregator.accept_model(FLModel(params={"w": np.array([1.0])}))
    aggregator.aggregate_model()

    # aggregate_model resets stats; a second aggregate with no accepts must fail.
    with pytest.raises(RuntimeError):
        aggregator.aggregate_model()


def test_lightning_eval_template_reports_validation_metric():
    torch = pytest.importorskip("torch")
    pl = pytest.importorskip("pytorch_lightning")
    from torch.utils.data import DataLoader, TensorDataset

    module = _load_module(LIGHTNING_TEMPLATES / "lightning_client.py")

    class ToyLightning(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 2)

        def forward(self, x):
            return self.layer(x)

        def validation_step(self, batch, batch_idx):
            features, labels = batch
            loss = torch.nn.functional.cross_entropy(self(features), labels)
            self.log("val_loss", loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    loader = DataLoader(TensorDataset(torch.randn(6, 4), torch.randint(0, 2, (6,))), batch_size=3)
    trainer = pl.Trainer(logger=False, enable_checkpointing=False, enable_progress_bar=False, devices=1)

    metrics = module.validate_global_model(trainer, ToyLightning(), dataloaders=loader)

    assert "val_loss" in metrics


def test_lightning_template_eval_only_mode_skips_training():
    # FedEval / evaluation-only: main(evaluate_only=True) must validate but never
    # call trainer.fit, so a converted eval-only job does not train.
    module = _load_module(LIGHTNING_TEMPLATES / "lightning_client.py")

    calls = []

    class _FakeTrainer:
        callback_metrics = {"val_loss": 0.1}

        def validate(self, *a, **k):
            calls.append("validate")

        def fit(self, *a, **k):
            calls.append("fit")

    fake = _FakeTrainer()
    import types

    fake_flare = types.SimpleNamespace(
        patch=lambda trainer: None,
        receive=lambda: None,
        _running=[True, False],
        is_running=lambda: fake_flare._running.pop(0) if fake_flare._running else False,
    )
    module.flare = fake_flare  # patch the module-level flare handle

    try:
        module.main(model=object(), datamodule=object(), trainer_factory=lambda: fake, evaluate_only=True)
    finally:
        pass

    assert "validate" in calls
    assert "fit" not in calls


class _DummyModel:
    training = True

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self

    def __call__(self, *_args, **_kwargs):  # pragma: no cover - never reached on empty loader
        raise AssertionError("model should not be called when the loader is empty")
