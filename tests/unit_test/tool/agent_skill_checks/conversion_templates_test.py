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

import ast
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SKILLS_ROOT = REPO_ROOT / "skills"
PT_TEMPLATES = SKILLS_ROOT / "nvflare-convert-pytorch" / "assets"
LIGHTNING_TEMPLATES = SKILLS_ROOT / "nvflare-convert-lightning" / "assets"
SHARED_TEMPLATES = SKILLS_ROOT / "nvflare-shared" / "assets"


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


def test_pytorch_eval_template_initializes_flare_before_training_setup():
    source = (PT_TEMPLATES / "client_with_eval.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    main_func = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")

    init_line = None
    setup_line = None
    loop_line = None
    for node in ast.walk(main_func):
        if isinstance(node, ast.While):
            loop_line = node.lineno
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "flare"
            and node.func.attr == "init"
        ):
            init_line = node.lineno
        if isinstance(node.func, ast.Name) and node.func.id == "train_setup_factory":
            setup_line = node.lineno

    assert init_line is not None
    assert setup_line is not None
    assert loop_line is not None
    assert init_line < setup_line < loop_line


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

    class _LazyRef:
        def __init__(self, array):
            self._array = array

        def materialize(self):
            return self._array

        def __mul__(self, other):  # pragma: no cover - must never be reached
            raise TypeError("lazy ref must be materialized before weighted math")

    # Production accepts NVFlare's concrete disk-offload ref type. Inject this
    # test double explicitly so arbitrary objects with a materialize() method
    # are never trusted by duck typing.
    aggregator = module.WeightedAggregator(trusted_lazy_types=(_LazyRef,))

    aggregator.accept_model(FLModel(params={"w": _LazyRef(np.array([2.0]))}, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 1}))
    aggregator.accept_model(FLModel(params={"w": _LazyRef(np.array([4.0]))}, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 3}))
    result = aggregator.aggregate_model()

    # (2*1 + 4*3) / (1 + 3) = 3.5, computed on the materialized arrays.
    assert result.params["w"][0] == pytest.approx(3.5)


def test_custom_aggregator_template_rejects_mismatched_parameter_schema():
    # Missing or additional keys usually mean incompatible model definitions.
    # Fail the round instead of silently producing a hybrid global model.
    import numpy as np

    from nvflare.apis.dxo import MetaKey
    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator()

    aggregator.accept_model(FLModel(params={"shared": np.array([2.0])}, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 1}))
    with pytest.raises(ValueError, match="schema"):
        aggregator.accept_model(
            FLModel(
                params={"shared": np.array([4.0]), "only_b": np.array([9.0])},
                meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 3},
            )
        )

    # A rejected client does not partially mutate the accumulator.
    assert aggregator.aggregate_model().params["shared"][0] == pytest.approx(2.0)


def test_custom_aggregator_template_preserves_weighted_metrics():
    import numpy as np

    from nvflare.apis.dxo import MetaKey
    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator()

    aggregator.accept_model(
        FLModel(
            params={"w": np.array([2.0])},
            metrics={"accuracy": 0.5},
            meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 1},
        )
    )
    aggregator.accept_model(
        FLModel(
            params={"w": np.array([4.0])},
            metrics={"accuracy": 1.0},
            meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 3},
        )
    )

    result = aggregator.aggregate_model()

    assert result.metrics == {"accuracy": pytest.approx(0.875)}


def test_custom_aggregator_accepts_ordered_state_dict_parameters():
    from collections import OrderedDict

    import numpy as np

    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator()
    aggregator.accept_model(FLModel(params=OrderedDict([("w", np.array([2.0]))])))

    assert aggregator.aggregate_model().params["w"][0] == pytest.approx(2.0)


def test_custom_aggregator_template_disables_metrics_if_any_client_omits_them():
    import numpy as np

    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator()

    aggregator.accept_model(FLModel(params={"w": np.array([2.0])}, metrics={"accuracy": 0.5}))
    aggregator.accept_model(FLModel(params={"w": np.array([4.0])}, metrics=None))

    assert aggregator.aggregate_model().metrics is None


def test_custom_aggregator_template_rejects_untrusted_materializer_without_calling_it():
    import numpy as np

    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator(trusted_lazy_types=())
    called = False

    class _UntrustedRef:
        def materialize(self):
            nonlocal called
            called = True
            return np.array([2.0])

    with pytest.raises(TypeError, match="unsupported numeric type"):
        aggregator.accept_model(FLModel(params={"w": _UntrustedRef()}))

    assert called is False


def test_custom_aggregator_template_does_not_probe_untrusted_materialize_attribute():
    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator(trusted_lazy_types=())
    probed = False

    class _AttributeTrap:
        def __getattribute__(self, name):
            nonlocal probed
            if name == "materialize":
                probed = True
                raise AssertionError("untrusted attribute must not be probed")
            return super().__getattribute__(name)

    with pytest.raises(TypeError, match="unsupported numeric type"):
        aggregator.accept_model(FLModel(params={"w": _AttributeTrap()}))

    assert probed is False


def test_custom_aggregator_template_rejects_spoofed_numeric_module_without_arithmetic():
    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator()
    multiplied = False

    class _SpoofedArray:
        __module__ = "numpy"

        def __mul__(self, other):
            nonlocal multiplied
            multiplied = True
            return self

    with pytest.raises(TypeError, match="unsupported numeric type"):
        aggregator.accept_model(FLModel(params={"w": _SpoofedArray()}))

    assert multiplied is False


def test_custom_aggregator_template_rejects_numeric_subclasses_without_dispatch():
    import numpy as np

    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator()
    dispatched = False

    class _ArraySubclass(np.ndarray):
        def __array_ufunc__(self, *args, **kwargs):
            nonlocal dispatched
            dispatched = True
            return super().__array_ufunc__(*args, **kwargs)

    value = np.array([1.0]).view(_ArraySubclass)
    with pytest.raises(TypeError, match="unsupported numeric type"):
        aggregator.accept_model(FLModel(params={"w": value}))

    assert dispatched is False


def test_custom_aggregator_template_bounds_parameter_bytes_and_metric_keys():
    import numpy as np

    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    byte_bounded = module.WeightedAggregator(max_param_bytes=8)
    with pytest.raises(ValueError, match="parameter byte size"):
        byte_bounded.accept_model(FLModel(params={"w": np.array([1.0, 2.0], dtype=np.float64)}))

    metric_bounded = module.WeightedAggregator(max_metric_keys=1)
    with pytest.raises(ValueError, match="metric key count"):
        metric_bounded.accept_model(FLModel(params={"w": np.array([1.0])}, metrics={"accuracy": 0.5, "loss": 1.0}))

    union_bounded = module.WeightedAggregator(max_metric_keys=1)
    union_bounded.accept_model(FLModel(params={"w": np.array([1.0])}, metrics={"accuracy": 0.5}))
    with pytest.raises(ValueError, match="metric key union"):
        union_bounded.accept_model(FLModel(params={"w": np.array([2.0])}, metrics={"loss": 1.0}))


def test_custom_aggregator_preflights_production_lazy_ref_before_materializing(monkeypatch, tmp_path):
    from nvflare.app_common.abstract.fl_model import FLModel
    from nvflare.app_opt.pt import lazy_tensor_dict
    from nvflare.app_opt.pt.lazy_tensor_dict import LazyTensorRef, _TempDirRef

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    loaded = False

    class FakeSlice:
        @staticmethod
        def get_shape():
            return [4]

        @staticmethod
        def get_dtype():
            return "F32"

    class FakeSafeOpen:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return None

        @staticmethod
        def get_slice(key):
            assert key == "w"
            return FakeSlice()

        @staticmethod
        def get_tensor(key):
            nonlocal loaded
            loaded = True
            raise AssertionError("oversized lazy tensor must not be materialized")

    monkeypatch.setattr(lazy_tensor_dict, "safe_open", lambda *args, **kwargs: FakeSafeOpen())
    temp_dir = tmp_path / "offload"
    temp_dir.mkdir()
    ref = LazyTensorRef(
        file_path=str(temp_dir / "unused.safetensors"),
        key="w",
        temp_ref=_TempDirRef(str(temp_dir)),
    )
    aggregator = module.WeightedAggregator(max_param_bytes=15)

    with pytest.raises(ValueError, match="lazy tensor byte size"):
        aggregator.accept_model(FLModel(params={"w": ref}))

    assert loaded is False
    assert aggregator._accepted_count == 0


def test_custom_aggregator_bounds_promoted_parameter_contribution():
    import numpy as np

    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator(max_param_bytes=4)

    # The input is four bytes, but multiplying int8 values by a float weight
    # promotes the contribution to a 32-byte float64 array.
    with pytest.raises(ValueError, match="weighted parameter contribution byte size"):
        aggregator.accept_model(FLModel(params={"w": np.array([1, 2, 3, 4], dtype=np.int8)}))

    assert aggregator._accepted_count == 0
    assert aggregator._weighted_sum == {}


def test_custom_aggregator_rejects_oversized_input_before_finiteness_scan(monkeypatch):
    import numpy as np

    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator(max_param_bytes=1)
    called = False
    original_isfinite = np.isfinite

    def recording_isfinite(value):
        nonlocal called
        called = True
        return original_isfinite(value)

    monkeypatch.setattr(np, "isfinite", recording_isfinite)
    with pytest.raises(ValueError, match="parameter byte size"):
        aggregator.accept_model(FLModel(params={"w": np.array([1.0], dtype=np.float64)}))

    assert called is False
    assert aggregator._accepted_count == 0


def test_custom_aggregator_detaches_torch_updates_and_bounds_schema_device_layout():
    import torch

    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator()
    aggregator.accept_model(FLModel(params={"w": torch.tensor([1.0], requires_grad=True)}))
    aggregator.accept_model(FLModel(params={"w": torch.tensor([3.0], requires_grad=True)}))

    assert aggregator._weighted_sum["w"].grad_fn is None
    result = aggregator.aggregate_model()
    assert result.params["w"].grad_fn is None
    assert result.params["w"].item() == pytest.approx(2.0)

    device_aggregator = module.WeightedAggregator()
    device_aggregator.accept_model(FLModel(params={"w": torch.ones(1)}))
    with pytest.raises(ValueError, match="schema"):
        device_aggregator.accept_model(FLModel(params={"w": torch.ones(1, device="meta")}))

    layout_aggregator = module.WeightedAggregator()
    layout_aggregator.accept_model(FLModel(params={"w": torch.ones(1)}))
    sparse = torch.sparse_coo_tensor(indices=torch.tensor([[0]]), values=torch.tensor([1.0]), size=(1,))
    with pytest.raises(ValueError, match="schema"):
        layout_aggregator.accept_model(FLModel(params={"w": sparse}))


@pytest.mark.parametrize("dtype", ["datetime64[D]", "timedelta64[D]"])
def test_custom_aggregator_rejects_non_numeric_numpy_dtypes(dtype):
    import numpy as np

    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator()
    value = np.array(["2026-01-01" if dtype.startswith("datetime") else 1], dtype=dtype)

    with pytest.raises(TypeError, match="unsupported NumPy dtype"):
        aggregator.accept_model(FLModel(params={"w": value}))

    assert aggregator._accepted_count == 0


def test_custom_aggregator_checks_accumulator_and_result_sizes(monkeypatch):
    import numpy as np

    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator(max_param_bytes=16)
    original_value_nbytes = module._value_nbytes
    calls = 0

    def oversized_third_value(value):
        nonlocal calls
        calls += 1
        # Raw value, contribution, then accumulator are checked in order.
        return 17 if calls == 3 else original_value_nbytes(value)

    monkeypatch.setattr(module, "_value_nbytes", oversized_third_value)
    with pytest.raises(ValueError, match="parameter accumulator byte size"):
        aggregator.accept_model(FLModel(params={"w": np.array([1.0], dtype=np.float64)}))
    assert aggregator._accepted_count == 0

    monkeypatch.setattr(module, "_value_nbytes", original_value_nbytes)
    aggregator.accept_model(FLModel(params={"w": np.array([1.0], dtype=np.float64)}))
    monkeypatch.setattr(module, "_value_nbytes", lambda value: 17)
    with pytest.raises(ValueError, match="aggregated parameter result byte size"):
        aggregator.aggregate_model()


def test_custom_aggregator_rejects_numpy_object_dtype_without_element_dispatch():
    import numpy as np

    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator()
    dispatched = []

    class ElementTrap:
        def __float__(self):
            dispatched.append("float")
            raise AssertionError("object element must not be converted")

        def __mul__(self, other):
            dispatched.append("mul")
            raise AssertionError("object element arithmetic must not run")

        def __rmul__(self, other):
            dispatched.append("rmul")
            raise AssertionError("object element arithmetic must not run")

    value = np.empty(1, dtype=object)
    value[0] = ElementTrap()

    with pytest.raises(TypeError, match="NumPy object dtype"):
        aggregator.accept_model(FLModel(params={"w": value}))

    assert dispatched == []
    assert aggregator._accepted_count == 0


def test_custom_aggregator_template_rejects_shape_or_dtype_drift():
    import numpy as np

    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")

    shape_aggregator = module.WeightedAggregator()
    shape_aggregator.accept_model(FLModel(params={"w": np.array([1.0], dtype=np.float32)}))
    with pytest.raises(ValueError, match="schema"):
        shape_aggregator.accept_model(FLModel(params={"w": np.array([1.0, 2.0], dtype=np.float32)}))

    dtype_aggregator = module.WeightedAggregator()
    dtype_aggregator.accept_model(FLModel(params={"w": np.array([1.0], dtype=np.float32)}))
    with pytest.raises(ValueError, match="schema"):
        dtype_aggregator.accept_model(FLModel(params={"w": np.array([1.0], dtype=np.float64)}))


def test_custom_aggregator_template_rejects_excessive_finite_step_weight():
    import numpy as np

    from nvflare.apis.dxo import MetaKey
    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator(max_step_weight=100.0)

    with pytest.raises(ValueError, match="exceeds configured maximum"):
        aggregator.accept_model(FLModel(params={"w": np.array([2.0])}, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 101.0}))


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
