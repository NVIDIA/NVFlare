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

import pytest

torch = pytest.importorskip("torch")

from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_opt.pt.fedce import FedCEConstants, FedCEModelAggregator, PTFedCEHelper, _normalize


def _result(client, update, minus_score, round_number=0):
    return FLModel(
        params={"weight": torch.tensor(update, dtype=torch.float32)},
        params_type=ParamsType.DIFF,
        current_round=round_number,
        meta={
            "client_name": client,
            FedCEConstants.MINUS_MODEL_SCORE: minus_score,
        },
    )


def test_fedce_weights_favor_higher_minus_model_score():
    aggregator = FedCEModelAggregator(mode="plus")
    aggregator.accept_model(_result("site-1", [1.0, 0.0], 0.9))
    aggregator.accept_model(_result("site-2", [0.0, 1.0], 0.1))

    result = aggregator.aggregate_model()
    weights = result.meta[FedCEConstants.CONTRIBUTION_WEIGHTS]

    assert result.params_type == ParamsType.DIFF
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["site-1"] > weights["site-2"]
    assert result.params["weight"].tolist() == pytest.approx([weights["site-1"], weights["site-2"]])


def test_fedce_reset_keeps_constant_space_cosine_mean_for_next_round():
    aggregator = FedCEModelAggregator()
    aggregator.accept_model(_result("site-1", [1.0], 0.7))
    aggregator.accept_model(_result("site-2", [2.0], 0.3))
    first = aggregator.aggregate_model().meta[FedCEConstants.CONTRIBUTION_WEIGHTS]

    aggregator.reset_stats()
    aggregator.accept_model(_result("site-1", [2.0], 0.7, round_number=1))
    aggregator.accept_model(_result("site-2", [1.0], 0.3, round_number=1))
    aggregator.aggregate_model()

    assert aggregator._contribution_weights
    assert first.keys() == aggregator._contribution_weights.keys()
    assert aggregator._cosine_counts["site-1"] == 2
    assert set(aggregator._cosine_means) == {"site-1", "site-2"}


def test_fedce_recovers_prior_weights_from_dispatched_model_metadata():
    aggregator = FedCEModelAggregator()
    prior_weights = {"site-1": 0.8, "site-2": 0.2}
    site_1 = _result("site-1", [1.0], 0.7)
    site_1.meta["props"] = {FedCEConstants.CONTRIBUTION_WEIGHTS: prior_weights}

    aggregator.accept_model(site_1)
    aggregator.accept_model(_result("site-2", [2.0], 0.3))

    assert aggregator._get_prior_weights(["site-1", "site-2"]) == prior_weights


def test_fedce_carries_weights_across_partial_participation(caplog):
    aggregator = FedCEModelAggregator()
    aggregator.accept_model(_result("site-1", [1.0, 0.0], 0.5))
    aggregator.accept_model(_result("site-3", [0.0, 1.0], 0.5))
    first_weights = aggregator.aggregate_model().meta[FedCEConstants.CONTRIBUTION_WEIGHTS]

    aggregator.reset_stats()
    aggregator.accept_model(_result("site-2", [1.0, 1.0], 0.5, round_number=1))
    aggregator.accept_model(_result("site-3", [0.0, 1.0], 0.5, round_number=1))

    assert aggregator._get_prior_weights(["site-2", "site-3"]) == {"site-2": 0.5, "site-3": 0.5}
    with caplog.at_level("WARNING"):
        second = aggregator.aggregate_model()

    second_weights = second.meta[FedCEConstants.CONTRIBUTION_WEIGHTS]
    assert "partial participation" in caplog.text
    assert set(second_weights) == {"site-1", "site-2", "site-3"}
    assert second_weights["site-1"] == pytest.approx(first_weights["site-1"])
    assert second_weights["site-2"] + second_weights["site-3"] == pytest.approx(1.0)
    assert PTFedCEHelper.get_contribution_weight(second, "site-1") == pytest.approx(first_weights["site-1"])


def test_fedce_requires_minus_model_score():
    aggregator = FedCEModelAggregator()
    result = FLModel(
        params={"weight": torch.tensor([1.0])},
        params_type=ParamsType.DIFF,
        meta={"client_name": "site-1"},
    )

    with pytest.raises(ValueError, match=FedCEConstants.MINUS_MODEL_SCORE):
        aggregator.accept_model(result)


def test_fedce_requires_diff_results():
    aggregator = FedCEModelAggregator()
    result = FLModel(
        params={"weight": torch.tensor([1.0])},
        params_type=ParamsType.FULL,
        meta={"client_name": "site-1", FedCEConstants.MINUS_MODEL_SCORE: 1.0},
    )

    with pytest.raises(ValueError, match="ParamsType.DIFF"):
        aggregator.accept_model(result)


def test_fedce_helper_builds_minus_model_and_attaches_score():
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(10.0)
    previous = {"weight": torch.tensor([[4.0]])}

    minus_model = PTFedCEHelper.make_minus_model(model, previous, contribution_weight=0.25)
    result = FLModel(params={"weight": torch.tensor([[1.0]])}, params_type=ParamsType.DIFF)
    PTFedCEHelper.set_minus_model_score(result, 0.8)

    assert minus_model.weight.item() == pytest.approx(12.0)
    assert result.meta[FedCEConstants.MINUS_MODEL_SCORE] == pytest.approx(0.8)


def test_fedce_helper_reads_weight_and_handles_non_trainable_buffers():
    class BufferedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([10.0]))
            self.register_buffer("counter", torch.tensor([3], dtype=torch.int64))
            self.register_buffer("untouched", torch.tensor([4], dtype=torch.int64))

    global_model = FLModel(meta={FedCEConstants.CONTRIBUTION_WEIGHTS: {"site-1": 0.4, "site-3": 0.6}})
    assert PTFedCEHelper.get_contribution_weight(global_model, "site-1") == pytest.approx(0.4)
    assert PTFedCEHelper.get_contribution_weight(global_model, "site-2") == pytest.approx(0.5)
    assert PTFedCEHelper.get_contribution_weight(global_model, "site-2", default=0.2) == pytest.approx(0.2)
    assert PTFedCEHelper.get_contribution_weight(FLModel(meta={}), "site-1") == pytest.approx(0.0)

    model = BufferedModel()
    minus_model = PTFedCEHelper.make_minus_model(
        model,
        {"weight": torch.tensor([4.0]), "counter": torch.tensor([99], dtype=torch.int64)},
        contribution_weight=0.25,
    )

    assert minus_model.counter.item() == 3
    assert minus_model.untouched.item() == 4
    assert minus_model.weight.item() == pytest.approx(12.0)


@pytest.mark.parametrize("weight", [-0.1, 1.0])
def test_fedce_helper_rejects_invalid_contribution_weight(weight):
    with pytest.raises(ValueError, match="contribution_weight must be"):
        PTFedCEHelper.make_minus_model(torch.nn.Linear(1, 1), {}, weight)


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")]])
def test_fedce_normalize_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        _normalize(values, epsilon=1e-6)


def test_fedce_aggregator_validates_configuration_and_results():
    with pytest.raises(ValueError, match="mode must be"):
        FedCEModelAggregator(mode="invalid")
    with pytest.raises(ValueError, match="epsilon must be"):
        FedCEModelAggregator(epsilon=0.0)

    aggregator = FedCEModelAggregator()
    with pytest.raises(ValueError, match="missing FLModel.meta"):
        aggregator.accept_model(
            FLModel(
                params={"weight": torch.tensor([1.0])},
                params_type=ParamsType.DIFF,
                meta={FedCEConstants.MINUS_MODEL_SCORE: 1.0},
            )
        )
    with pytest.raises(ValueError, match="empty parameters"):
        aggregator.accept_model(
            FLModel(
                params={},
                params_type=ParamsType.DIFF,
                meta={"client_name": "site-1", FedCEConstants.MINUS_MODEL_SCORE: 1.0},
            )
        )
    with pytest.raises(ValueError, match="empty result set"):
        aggregator.aggregate_model()

    result = _result("site-1", [1.0], 1.0)
    aggregator.accept_model(result)
    with pytest.raises(ValueError, match="more than one result"):
        aggregator.accept_model(result)


def test_fedce_times_mode_handles_zero_updates_and_aggregates_metrics():
    aggregator = FedCEModelAggregator(mode="times")
    site_1 = _result("site-1", [0.0, 0.0], 0.8)
    site_1.metrics = {"accuracy": 0.5, "label": "ignored"}
    site_2 = _result("site-2", [1.0, 0.0], 0.2)
    site_2.metrics = {"accuracy": 1.0, "label": "ignored"}
    aggregator.accept_model(site_1)
    aggregator.accept_model(site_2)

    result = aggregator.aggregate_model()

    weights = result.meta[FedCEConstants.CONTRIBUTION_WEIGHTS]
    expected_accuracy = 0.5 * weights["site-1"] + weights["site-2"]
    assert result.metrics["accuracy"] == pytest.approx(expected_accuracy)
    assert "label" not in result.metrics
    assert result.meta["nr_aggregated"] == 2


def test_fedce_rejects_updates_without_common_trainable_parameters():
    aggregator = FedCEModelAggregator(trainable_param_names=["missing"])
    aggregator.accept_model(_result("site-1", [1.0], 0.5))
    aggregator.accept_model(_result("site-2", [2.0], 0.5))

    with pytest.raises(ValueError, match="no common trainable parameters"):
        aggregator.aggregate_model()


def test_fedce_rejects_updates_with_inconsistent_parameter_sets():
    aggregator = FedCEModelAggregator()
    aggregator.accept_model(_result("site-1", [1.0], 0.5))
    site_2 = _result("site-2", [2.0], 0.5)
    site_2.params["bias"] = torch.tensor([1.0])
    aggregator.accept_model(site_2)

    with pytest.raises(
        ValueError,
        match=r"client 'site-2' parameters do not match client 'site-1'.*unexpected=\['bias'\]",
    ):
        aggregator.aggregate_model()


@pytest.mark.parametrize(
    ("site_1_update", "site_2_update", "expected_shape", "received_shape"),
    [
        ([1.0, 2.0], [[1.0, 2.0, 3.0]], "(2,)", "(1, 3)"),
        ([[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0, 3.0, 4.0], "(2, 2)", "(4,)"),
    ],
)
def test_fedce_rejects_updates_with_inconsistent_parameter_shapes(
    site_1_update, site_2_update, expected_shape, received_shape
):
    aggregator = FedCEModelAggregator()
    aggregator.accept_model(_result("site-1", site_1_update, 0.5))
    aggregator.accept_model(_result("site-2", site_2_update, 0.5))

    with pytest.raises(ValueError, match=r"client 'site-2' parameter 'weight' shape does not match") as exc_info:
        aggregator.aggregate_model()

    assert f"expected={expected_shape}" in str(exc_info.value)
    assert f"got={received_shape}" in str(exc_info.value)


def test_fedce_rejects_buffer_shape_mismatch_outside_scoring_set():
    aggregator = FedCEModelAggregator(trainable_param_names=["weight"])
    site_1 = _result("site-1", [1.0], 0.5)
    site_1.params["running_mean"] = torch.tensor([1.0, 2.0])
    site_2 = _result("site-2", [2.0], 0.5)
    site_2.params["running_mean"] = torch.tensor([[1.0, 2.0]])
    aggregator.accept_model(site_1)
    aggregator.accept_model(site_2)

    with pytest.raises(ValueError, match=r"parameter 'running_mean' shape does not match"):
        aggregator.aggregate_model()


def test_fedce_rejects_lazy_parameter_shape_mismatch():
    class LazyValue:
        def __init__(self, value):
            self.value = torch.tensor(value)

        def materialize(self):
            return self.value

    aggregator = FedCEModelAggregator()
    site_1 = _result("site-1", [1.0], 0.5)
    site_1.params["weight"] = LazyValue([1.0, 2.0])
    site_2 = _result("site-2", [2.0], 0.5)
    site_2.params["weight"] = LazyValue([[1.0, 2.0]])
    aggregator.accept_model(site_1)
    aggregator.accept_model(site_2)

    with pytest.raises(ValueError, match=r"parameter 'weight' shape does not match"):
        aggregator.aggregate_model()


def test_fedce_materializes_lazy_parameters():
    class LazyValue:
        @staticmethod
        def materialize():
            return torch.tensor([1.0, 2.0])

    assert FedCEModelAggregator._get_param_shape(LazyValue()) == (2,)
    flattened = FedCEModelAggregator._flatten_params({"weight": LazyValue()}, ["weight"])

    assert flattened.tolist() == [1.0, 2.0]
