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
import itertools
import sys
from collections import OrderedDict
from pathlib import Path

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_NVFLARE_RUNTIME_DEPS = importlib.util.find_spec("msgpack") is not None
pytestmark = pytest.mark.skipif(
    not (HAS_TORCH and HAS_NVFLARE_RUNTIME_DEPS),
    reason="PyTorch and NVFlare runtime dependencies are required for Evo2 aggregator tests",
)


def _example_dir() -> Path:
    return Path(__file__).parents[1]


def _load_example_modules():
    example_dir = _example_dir()
    module_names = ("evo2_adapter_checkpoint", "evo2_aggregator")
    previous_modules = {name: sys.modules.pop(name, None) for name in module_names}
    sys.path.insert(0, str(example_dir))
    try:
        modules = []
        for module_name in module_names:
            module_path = example_dir / f"{module_name}.py"
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            modules.append(module)
        return modules
    finally:
        sys.path.remove(str(example_dir))
        for name in module_names:
            sys.modules.pop(name, None)
        for name, previous in previous_modules.items():
            if previous is not None:
                sys.modules[name] = previous


def _reference_state():
    import torch

    return OrderedDict(
        {
            "decoder.layers.0.linear_qkv.adapter.linear_in.weight": torch.zeros(2, 3),
            "decoder.classification_head.weight": torch.zeros(3, 3),
            "decoder.classification_head.bias": torch.zeros(3),
        }
    )


def _make_aggregator(tmp_path: Path, aggregation_weights=None):
    adapter_checkpoint, evo2_aggregator = _load_example_modules()
    checkpoint = tmp_path / "initial.pt"
    adapter_checkpoint.save_nvflare_checkpoint(
        _reference_state(), checkpoint, metadata={"backend": "mock", "exchange_dtype": "float32"}
    )
    aggregator = evo2_aggregator.ExactSchemaFedAvgAggregator(
        schema_checkpoint=str(checkpoint),
        aggregation_weights=aggregation_weights or {"site-1": 1.0, "site-2": 3.0},
    )
    return aggregator


def _model(params, *, client_name: str, params_type=None, metrics=None, exchange_dtype="float32"):
    from nvflare.app_common.abstract.fl_model import FLModel, ParamsType

    metadata = {"client_name": client_name, "current_round": 0}
    if exchange_dtype is not None:
        metadata["exchange_dtype"] = exchange_dtype
    return FLModel(
        params=params,
        params_type=params_type or ParamsType.DIFF,
        metrics=metrics,
        meta=metadata,
    )


def test_exact_schema_aggregator_applies_static_sample_weighted_fedavg(tmp_path):
    import torch

    aggregator = _make_aggregator(tmp_path)
    first = OrderedDict((name, torch.ones_like(value)) for name, value in _reference_state().items())
    second = OrderedDict((name, torch.full_like(value, 3.0)) for name, value in _reference_state().items())

    aggregator.accept_model(_model(first, client_name="site-1", metrics={"accuracy": 0.5}))
    aggregator.accept_model(_model(second, client_name="site-2", metrics={"accuracy": 1.0}))
    result = aggregator.aggregate_model()

    assert result.params_type.value == "DIFF"
    assert all(torch.equal(value, torch.full_like(value, 2.5)) for value in result.params.values())
    assert all(value.dtype == torch.float32 for value in result.params.values())
    assert result.metrics == {"accuracy": pytest.approx(0.875)}
    assert result.meta == {
        "exchange_dtype": "float32",
        "initialization": {"backend": "mock", "exchange_dtype": "float32"},
    }


@pytest.mark.parametrize("exchange_dtype", (None, "bfloat16"))
def test_aggregator_rejects_missing_or_incompatible_exchange_dtype_metadata(tmp_path, exchange_dtype):
    adapter_checkpoint, evo2_aggregator = _load_example_modules()
    checkpoint = tmp_path / "initial.pt"
    metadata = {"backend": "mock"}
    if exchange_dtype is not None:
        metadata["exchange_dtype"] = exchange_dtype
    adapter_checkpoint.save_nvflare_checkpoint(_reference_state(), checkpoint, metadata=metadata)

    with pytest.raises(ValueError, match="exchange_dtype='float32'"):
        evo2_aggregator.ExactSchemaFedAvgAggregator(
            schema_checkpoint=str(checkpoint),
            aggregation_weights={"site-1": 1.0},
        )


@pytest.mark.parametrize("exchange_dtype", (None, "bfloat16"))
def test_aggregator_rejects_client_updates_without_canonical_exchange_dtype(tmp_path, exchange_dtype):
    import torch

    aggregator = _make_aggregator(tmp_path)
    params = OrderedDict((name, torch.ones_like(value)) for name, value in _reference_state().items())

    with pytest.raises(ValueError, match="client update metadata.*exchange_dtype='float32'"):
        aggregator.accept_model(_model(params, client_name="site-1", exchange_dtype=exchange_dtype))

    assert aggregator.contributors == set()


@pytest.mark.parametrize("failure", ("params_type", "missing", "extra", "shape", "dtype", "bfloat16"))
def test_exact_schema_aggregator_rejects_incompatible_client_updates(tmp_path, failure):
    import torch

    from nvflare.app_common.abstract.fl_model import ParamsType

    aggregator = _make_aggregator(tmp_path)
    params = OrderedDict((name, value.clone()) for name, value in _reference_state().items())
    params_type = ParamsType.DIFF
    expected_error = (ValueError, KeyError)

    if failure == "params_type":
        params_type = ParamsType.FULL
        expected_error = ValueError
    elif failure == "missing":
        params.pop("decoder.classification_head.bias")
        expected_error = KeyError
    elif failure == "extra":
        params["decoder.classification_head.extra"] = torch.zeros(1)
        expected_error = KeyError
    elif failure == "shape":
        params["decoder.classification_head.bias"] = torch.zeros(4)
        expected_error = ValueError
    elif failure == "dtype":
        params["decoder.classification_head.bias"] = torch.zeros(3, dtype=torch.float64)
        expected_error = ValueError
    else:
        params = OrderedDict((name, value.to(torch.bfloat16)) for name, value in params.items())
        expected_error = ValueError

    with pytest.raises(expected_error):
        aggregator.accept_model(_model(params, client_name="site-1", params_type=params_type))


def test_exact_schema_aggregator_refuses_partial_round_after_rejection(tmp_path):
    import torch

    aggregator = _make_aggregator(tmp_path)
    valid = OrderedDict((name, torch.ones_like(value)) for name, value in _reference_state().items())
    aggregator.accept_model(_model(valid, client_name="site-1"))

    malformed = OrderedDict(valid)
    malformed.pop("decoder.classification_head.bias")
    with pytest.raises(KeyError):
        aggregator.accept_model(_model(malformed, client_name="site-2"))
    with pytest.raises(RuntimeError, match=r"missing=\['site-2'\]"):
        aggregator.aggregate_model()


@pytest.mark.parametrize(
    "non_finite_value",
    (float("nan"), float("inf"), float("-inf")),
    ids=("nan", "positive_inf", "negative_inf"),
)
def test_exact_schema_aggregator_rejects_non_finite_update_without_accepting_contributor(tmp_path, non_finite_value):
    import torch

    aggregator = _make_aggregator(tmp_path)
    invalid = OrderedDict((name, torch.ones_like(value)) for name, value in _reference_state().items())
    invalid["decoder.classification_head.bias"][0] = non_finite_value

    with pytest.raises(ValueError, match="Evo2 DIFF from site-1 contains non-finite values"):
        aggregator.accept_model(_model(invalid, client_name="site-1"))

    assert aggregator.contributors == set()
    assert aggregator.params_helper.get_len() == 0
    assert aggregator.params_helper.total == {}
    assert aggregator.params_helper.counts == {}

    valid = OrderedDict((name, torch.ones_like(value)) for name, value in _reference_state().items())
    aggregator.accept_model(_model(valid, client_name="site-1"))
    assert aggregator.contributors == {"site-1"}
    assert set(aggregator.param_contributions) == {"site-1"}
    assert aggregator.params_helper.get_len() == 0


def test_fp32_aggregation_is_arrival_order_independent_across_all_permutations_and_preserves_one_ulp(tmp_path):
    import torch

    weights = {"site-1": 1.0, "site-2": 1.0, "site-3": 1.0}
    values = {
        "site-1": 1.0e8,
        "site-2": -1.0e8,
        "site-3": 3.0 * torch.finfo(torch.float32).eps,
    }

    def aggregate(path, order):
        aggregator = _make_aggregator(path, weights)
        for client_name in order:
            params = OrderedDict(
                (name, torch.full_like(reference, values[client_name]))
                for name, reference in _reference_state().items()
            )
            aggregator.accept_model(_model(params, client_name=client_name))
        return aggregator.aggregate_model().params

    client_names = tuple(weights)
    results = [
        aggregate(tmp_path / f"order-{index}", order)
        for index, order in enumerate(itertools.permutations(client_names))
    ]
    forward = results[0]
    expected_ulp = torch.finfo(torch.float32).eps

    assert all(torch.equal(forward[name], result[name]) for result in results[1:] for name in forward)
    assert all(torch.equal(value, torch.full_like(value, expected_ulp)) for value in forward.values())
    for value in forward.values():
        base = torch.ones_like(value)
        assert torch.equal(base + value, torch.nextafter(base, torch.full_like(base, float("inf"))))
