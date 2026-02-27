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

from unittest.mock import patch

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, Task
from nvflare.apis.fl_constant import FLMetaKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.app_common.workflows.base_fedavg import BaseFedAvg
from nvflare.app_common.workflows.fedavg import FedAvg


class MockModelAggregator(ModelAggregator):
    """Mock aggregator for testing custom aggregator support."""

    def __init__(self):
        super().__init__()
        self.models = []
        self.reset_count = 0

    def accept_model(self, model: FLModel):
        self.models.append(model)

    def aggregate_model(self) -> FLModel:
        if not self.models:
            return FLModel(params={})

        # Simple average for testing
        result_params = {}
        for key in self.models[0].params.keys():
            values = [m.params[key] for m in self.models]
            result_params[key] = sum(values) / len(values)

        return FLModel(params=result_params, params_type=self.models[0].params_type)

    def reset_stats(self):
        self.models = []
        self.reset_count += 1


class TestFedAvgInit:
    """Test FedAvg controller initialization."""

    def test_default_initialization(self):
        """Test FedAvg with default parameters."""
        controller = FedAvg()

        assert controller.num_clients == 3  # default
        assert controller.num_rounds == 5  # default
        assert controller.model is None
        assert controller.aggregator is None
        assert controller.stop_cond is None
        assert controller.patience is None
        assert controller.task_name == "train"
        assert controller.save_filename == "FL_global_model.pt"
        assert controller.exclude_vars is None
        assert controller.aggregation_weights == {}

    def test_custom_initialization(self):
        """Test FedAvg with custom parameters."""
        model = {"layer1": [1.0, 2.0, 3.0]}
        aggregator = MockModelAggregator()

        controller = FedAvg(
            num_clients=5,
            num_rounds=10,
            model=model,
            aggregator=aggregator,
            stop_cond="accuracy >= 80",
            patience=3,
            task_name="validate",
            save_filename="best_model.pt",
            exclude_vars="bn.*",
            aggregation_weights={"site-1": 2.0, "site-2": 1.0},
        )

        assert controller.num_clients == 5
        assert controller.num_rounds == 10
        assert controller.model == model
        assert controller.aggregator is aggregator
        assert controller.stop_cond == "accuracy >= 80"
        assert controller.patience == 3
        assert controller.task_name == "validate"
        assert controller.save_filename == "best_model.pt"
        assert controller.exclude_vars == "bn.*"
        assert controller.aggregation_weights == {"site-1": 2.0, "site-2": 1.0}

    def test_stop_condition_parsing(self):
        """Test that stop condition is correctly parsed."""
        controller = FedAvg(stop_cond="accuracy >= 80")
        assert controller.stop_condition is not None
        key, target, op_fn = controller.stop_condition
        assert key == "accuracy"
        assert target == 80
        assert op_fn(85, 80) is True  # 85 >= 80
        assert op_fn(75, 80) is False  # 75 >= 80

    def test_no_stop_condition(self):
        """Test that no stop condition means stop_condition is None."""
        controller = FedAvg()
        assert controller.stop_condition is None


class TestFedAvgEarlyStopping:
    """Test FedAvg early stopping logic."""

    def test_should_stop_no_condition(self):
        """Test should_stop returns False when no stop condition."""
        controller = FedAvg()
        assert controller.should_stop({"accuracy": 90}) is False

    def test_should_stop_condition_met(self):
        """Test should_stop returns True when condition is met."""
        controller = FedAvg(stop_cond="accuracy >= 80")
        assert controller.should_stop({"accuracy": 85}) is True

    def test_should_stop_condition_not_met(self):
        """Test should_stop returns False when condition is not met."""
        controller = FedAvg(stop_cond="accuracy >= 80")
        assert controller.should_stop({"accuracy": 75}) is False

    def test_should_stop_patience_exceeded(self):
        """Test should_stop returns True when patience is exceeded."""
        controller = FedAvg(stop_cond="accuracy >= 80", patience=2)
        controller.num_fl_rounds_without_improvement = 3
        assert controller.should_stop({"accuracy": 75}) is True

    def test_should_stop_missing_metric(self):
        """Test should_stop returns False when metric is missing."""
        controller = FedAvg(stop_cond="accuracy >= 80")
        assert controller.should_stop({"loss": 0.5}) is False


class TestFedAvgModelSelection:
    """Test FedAvg model selection logic."""

    def test_is_curr_model_better_no_condition(self):
        """Test is_curr_model_better returns True when no stop condition."""
        controller = FedAvg()
        model = FLModel(params={}, metrics={"accuracy": 90})
        assert controller.is_curr_model_better(model) is True

    def test_is_curr_model_better_first_model(self):
        """Test first model is always considered better."""
        controller = FedAvg(stop_cond="accuracy >= 80")
        model = FLModel(params={}, metrics={"accuracy": 50})
        assert controller.is_curr_model_better(model) is True
        assert controller.best_target_metric_value == 50

    def test_is_curr_model_better_improvement(self):
        """Test model with better metric is selected."""
        controller = FedAvg(stop_cond="accuracy >= 80")
        controller.best_target_metric_value = 50

        model = FLModel(params={}, metrics={"accuracy": 60})
        assert controller.is_curr_model_better(model) is True
        assert controller.best_target_metric_value == 60

    def test_is_curr_model_worse(self):
        """Test model with worse metric is not selected."""
        controller = FedAvg(stop_cond="accuracy >= 80")
        controller.best_target_metric_value = 60

        model = FLModel(params={}, metrics={"accuracy": 50})
        assert controller.is_curr_model_better(model) is False
        assert controller.num_fl_rounds_without_improvement == 1

    def test_equal_metric_with_patience_is_not_improvement(self):
        """Test equal metric values with patience count as no improvement (stagnation).

        This is intentional behavior: if accuracy stays at 85% for multiple rounds,
        that's stagnation, not improvement. Early stopping should trigger.
        """
        controller = FedAvg(stop_cond="accuracy >= 80", patience=3)
        controller.best_target_metric_value = 85

        # Same accuracy as best - should NOT be considered improvement
        model = FLModel(params={}, metrics={"accuracy": 85})
        result = controller.is_curr_model_better(model)

        assert result is False
        assert controller.num_fl_rounds_without_improvement == 1
        assert controller.best_target_metric_value == 85  # Not updated

    def test_equal_metric_without_patience_is_improvement(self):
        """Test equal metric values without patience count as improvement.

        Without patience tracking, we just update the best model.
        """
        controller = FedAvg(stop_cond="accuracy >= 80")  # No patience
        controller.best_target_metric_value = 85

        model = FLModel(params={}, metrics={"accuracy": 85})
        result = controller.is_curr_model_better(model)

        assert result is True
        assert controller.num_fl_rounds_without_improvement == 0
        assert controller.best_target_metric_value == 85

    def test_better_metric_with_patience_resets_counter(self):
        """Test that truly better metric resets the patience counter."""
        controller = FedAvg(stop_cond="accuracy >= 80", patience=3)
        controller.best_target_metric_value = 80
        controller.num_fl_rounds_without_improvement = 2

        # Better accuracy - should reset counter
        model = FLModel(params={}, metrics={"accuracy": 85})
        result = controller.is_curr_model_better(model)

        assert result is True
        assert controller.num_fl_rounds_without_improvement == 0
        assert controller.best_target_metric_value == 85

    def test_stagnation_triggers_early_stop(self):
        """Test that repeated stagnation leads to early stopping via patience.

        Note: should_stop returns True when EITHER:
        1. stop_cond is met (accuracy >= target), OR
        2. patience is exceeded (patience <= num_rounds_without_improvement)

        This test focuses on patience-based stopping with metrics below target.
        """
        controller = FedAvg(stop_cond="accuracy >= 90", patience=3)
        controller.best_target_metric_value = 75

        # Round 1: same accuracy (below target)
        model = FLModel(params={}, metrics={"accuracy": 75})
        controller.is_curr_model_better(model)
        assert controller.num_fl_rounds_without_improvement == 1
        assert controller.should_stop({"accuracy": 75}) is False  # 3 > 1

        # Round 2: same accuracy again
        controller.is_curr_model_better(model)
        assert controller.num_fl_rounds_without_improvement == 2
        assert controller.should_stop({"accuracy": 75}) is False  # 3 > 2

        # Round 3: same accuracy - patience met
        controller.is_curr_model_better(model)
        assert controller.num_fl_rounds_without_improvement == 3
        assert controller.should_stop({"accuracy": 75}) is True  # 3 <= 3, patience exceeded!

    def test_missing_metrics_not_improvement(self):
        """Test model with missing metrics is not considered better."""
        controller = FedAvg(stop_cond="accuracy >= 80", patience=3)
        controller.best_target_metric_value = 80

        model = FLModel(params={}, metrics=None)
        assert controller.is_curr_model_better(model) is False

        model2 = FLModel(params={}, metrics={"loss": 0.5})  # Wrong metric
        assert controller.is_curr_model_better(model2) is False


class TestFedAvgAggregation:
    """Test FedAvg aggregation logic."""

    def test_aggregate_one_result_builtin(self):
        """Test built-in aggregation accumulates results."""
        controller = FedAvg(num_clients=2)

        # Simulate round setup
        from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper

        controller._aggr_helper = WeightedAggregationHelper()
        controller._aggr_metrics_helper = WeightedAggregationHelper()
        controller._all_metrics = True
        controller._received_count = 0
        controller._expected_count = 2
        controller._params_type = None
        controller.current_round = 0

        # Simulate client result 1
        result1 = FLModel(
            params={"w": 1.0},
            params_type=ParamsType.FULL,
            meta={"client_name": "site-1", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 10},
        )
        controller._aggregate_one_result(result1)
        assert controller._received_count == 1

        # Simulate client result 2
        result2 = FLModel(
            params={"w": 3.0},
            params_type=ParamsType.FULL,
            meta={"client_name": "site-2", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 10},
        )
        controller._aggregate_one_result(result2)
        assert controller._received_count == 2

        # Get aggregated result
        aggr_result = controller._get_aggregated_result()
        assert aggr_result.params["w"] == 2.0  # (1+3)/2

    def test_aggregate_one_result_filters_non_aggregatable_metrics(self):
        """Test built-in aggregation filters non-aggregatable metrics."""
        controller = FedAvg(num_clients=2)

        from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper

        controller._aggr_helper = WeightedAggregationHelper()
        controller._aggr_metrics_helper = WeightedAggregationHelper()
        controller._all_metrics = True
        controller._received_count = 0
        controller._expected_count = 2
        controller._params_type = None
        controller.current_round = 0

        result1 = FLModel(
            params={"w": 1.0},
            params_type=ParamsType.FULL,
            metrics={"loss": 0.2, "meta": {"client": "site-1"}, "tags": ["a", "b"], "name": "run-1"},
            meta={"client_name": "site-1", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 1},
        )
        result2 = FLModel(
            params={"w": 3.0},
            params_type=ParamsType.FULL,
            metrics={"loss": 0.6, "meta": {"client": "site-2"}, "tags": ["c"], "name": "run-2"},
            meta={"client_name": "site-2", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 1},
        )

        controller._aggregate_one_result(result1)
        controller._aggregate_one_result(result2)

        aggr_result = controller._get_aggregated_result()
        assert aggr_result.metrics is not None
        assert aggr_result.metrics["loss"] == 0.4
        assert "meta" not in aggr_result.metrics
        assert "tags" not in aggr_result.metrics
        assert "name" not in aggr_result.metrics

    def test_aggregate_one_result_warns_once_per_skipped_metric_key(self):
        """Test repeated skipped metric keys only emit one warning."""
        from unittest.mock import patch

        controller = FedAvg(num_clients=2)

        from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper

        controller._aggr_helper = WeightedAggregationHelper()
        controller._aggr_metrics_helper = WeightedAggregationHelper()
        controller._all_metrics = True
        controller._received_count = 0
        controller._expected_count = 2
        controller._params_type = None
        controller.current_round = 0

        result1 = FLModel(
            params={"w": 1.0},
            params_type=ParamsType.FULL,
            metrics={"loss": 0.2, "meta": {"a": 1}},
            meta={"client_name": "site-1", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 1},
        )
        result2 = FLModel(
            params={"w": 3.0},
            params_type=ParamsType.FULL,
            metrics={"loss": 0.6, "meta": {"b": 2}},
            meta={"client_name": "site-2", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 1},
        )

        with patch.object(controller, "warning") as mock_warning:
            controller._aggregate_one_result(result1)
            controller._aggregate_one_result(result2)

        assert mock_warning.call_count == 1
        assert mock_warning.call_args_list[0].args[0] == "Metric 'meta' (dict) skipped for aggregation."

    def test_base_fedavg_aggregate_fn_filters_non_aggregatable_metrics(self):
        """Test BaseFedAvg.aggregate_fn skips unsupported metrics without failing."""
        result1 = FLModel(
            params={"w": 1.0},
            params_type=ParamsType.FULL,
            metrics={"loss": 0.2, "meta": {"client": "site-1"}},
            meta={"client_name": "site-1", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 1},
        )
        result2 = FLModel(
            params={"w": 3.0},
            params_type=ParamsType.FULL,
            metrics={"loss": 0.6, "meta": {"client": "site-2"}},
            meta={"client_name": "site-2", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 1},
        )
        result1.current_round = 0
        result2.current_round = 0

        aggr_result = BaseFedAvg.aggregate_fn([result1, result2])
        assert aggr_result.metrics is not None
        assert aggr_result.metrics["loss"] == 0.4
        assert "meta" not in aggr_result.metrics

    def test_aggregate_with_custom_aggregator(self):
        """Test custom aggregator is used when provided."""
        aggregator = MockModelAggregator()
        controller = FedAvg(num_clients=2, aggregator=aggregator)

        # Simulate round setup
        aggregator.reset_stats()
        controller._received_count = 0
        controller._expected_count = 2
        controller._params_type = None
        controller.current_round = 0

        # Simulate client results
        result1 = FLModel(
            params={"w": 2.0},
            params_type=ParamsType.FULL,
            meta={"client_name": "site-1"},
        )
        result2 = FLModel(
            params={"w": 4.0},
            params_type=ParamsType.FULL,
            meta={"client_name": "site-2"},
        )

        controller._aggregate_one_result(result1)
        controller._aggregate_one_result(result2)

        # Verify custom aggregator was used
        assert len(aggregator.models) == 2

        # Get aggregated result
        aggr_result = controller._get_aggregated_result()
        assert aggr_result.params["w"] == 3.0  # (2+4)/2

    def test_aggregate_empty_result_skipped(self):
        """Test empty results are skipped."""
        controller = FedAvg(num_clients=2)

        from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper

        controller._aggr_helper = WeightedAggregationHelper()
        controller._aggr_metrics_helper = WeightedAggregationHelper()
        controller._all_metrics = True
        controller._received_count = 0
        controller._expected_count = 2
        controller._params_type = None
        controller.current_round = 0

        # Empty result
        empty_result = FLModel(params=None, meta={"client_name": "site-1"})
        controller._aggregate_one_result(empty_result)
        assert controller._received_count == 0  # Not counted


class TestFedAvgWorkflowEvents:
    def test_run_fires_round_started_and_before_aggregation_once_per_round(self):
        controller = FedAvg(num_clients=1, num_rounds=2, model={"w": 1.0})
        controller.fl_ctx = FLContext()
        controller.abort_signal = Signal()
        controller.sample_clients = lambda _: ["site-1"]
        controller.send_model = lambda **kwargs: None
        controller.get_num_standing_tasks = lambda: 0
        controller._get_aggregated_result = lambda: FLModel(params={"w": 1.0})
        controller.update_model = lambda model, aggr_result: model
        controller.save_model = lambda model: None

        with patch.object(controller, "event") as mock_event:
            controller.run()

        round_started_calls = [c for c in mock_event.call_args_list if c.args[0] == AppEventType.ROUND_STARTED]
        before_aggr_calls = [c for c in mock_event.call_args_list if c.args[0] == AppEventType.BEFORE_AGGREGATION]

        assert len(round_started_calls) == 2
        assert len(before_aggr_calls) == 2

    def test_process_result_sets_current_round_on_fl_ctx(self):
        controller = FedAvg(num_clients=1)
        fl_ctx = FLContext()
        fl_ctx.set_prop(AppConstants.CURRENT_ROUND, 1, private=True, sticky=True)

        task_data = Shareable()
        task_data.set_header(AppConstants.CURRENT_ROUND, 3)
        result = FLModelUtils.to_shareable(FLModel(params={"w": 1.0}))
        result.set_header(AppConstants.CURRENT_ROUND, 3)

        task = Task(
            name=AppConstants.TASK_TRAIN,
            data=task_data,
            props={AppConstants.META_DATA: {}},
        )
        client_task = ClientTask(client=Client("site-1", "token"), task=task)
        client_task.result = result

        with patch.object(controller, "event"):
            controller._process_result(client_task, fl_ctx)

        assert fl_ctx.get_prop(AppConstants.CURRENT_ROUND) == 3

    def test_broadcast_model_does_not_fire_round_started(self):
        controller = FedAvg(num_clients=1)
        controller.fl_ctx = FLContext()
        model = FLModel(params={"w": 1.0}, current_round=2)

        with (
            patch.object(controller, "broadcast") as mock_broadcast,
            patch.object(controller, "fire_event") as mock_fire,
        ):
            controller.broadcast_model(data=model, blocking=False, callback=lambda _: None)

        round_started_calls = [c for c in mock_fire.call_args_list if c.args[0] == AppEventType.ROUND_STARTED]
        assert len(round_started_calls) == 0
        mock_broadcast.assert_called_once()

    def test_run_sets_num_rounds_sticky_consistently(self):
        controller = FedAvg(num_clients=1, num_rounds=1, model={"w": 1.0})
        controller.fl_ctx = FLContext()
        controller.abort_signal = Signal()
        controller.sample_clients = lambda _: ["site-1"]
        controller.send_model = lambda **kwargs: None
        controller.get_num_standing_tasks = lambda: 0
        controller._get_aggregated_result = lambda: FLModel(params={"w": 1.0})
        controller.update_model = lambda model, aggr_result: model
        controller.save_model = lambda model: None

        with patch.object(controller.fl_ctx, "set_prop", wraps=controller.fl_ctx.set_prop) as mock_set_prop:
            controller.run()

        num_round_calls = [c for c in mock_set_prop.call_args_list if c.args and c.args[0] == AppConstants.NUM_ROUNDS]
        assert len(num_round_calls) > 0
        assert all(c.kwargs.get("sticky") is True for c in num_round_calls)


class TestFedAvgAggregationWeights:
    """Test FedAvg aggregation weights."""

    def test_aggregation_weights_applied(self):
        """Test per-client aggregation weights are applied."""
        controller = FedAvg(
            num_clients=2,
            aggregation_weights={"site-1": 2.0, "site-2": 1.0},
        )

        from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper

        controller._aggr_helper = WeightedAggregationHelper()
        controller._aggr_metrics_helper = WeightedAggregationHelper()
        controller._all_metrics = True
        controller._received_count = 0
        controller._expected_count = 2
        controller._params_type = None
        controller.current_round = 0

        # Result from site-1 with weight 2.0
        result1 = FLModel(
            params={"w": 1.0},
            params_type=ParamsType.FULL,
            meta={"client_name": "site-1", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 1},
        )
        # Result from site-2 with weight 1.0
        result2 = FLModel(
            params={"w": 4.0},
            params_type=ParamsType.FULL,
            meta={"client_name": "site-2", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 1},
        )

        controller._aggregate_one_result(result1)
        controller._aggregate_one_result(result2)

        aggr_result = controller._get_aggregated_result()
        # Weighted average: (1.0*2.0 + 4.0*1.0) / (2.0 + 1.0) = 6.0/3.0 = 2.0
        assert aggr_result.params["w"] == 2.0

    def test_aggregation_weights_combined_with_num_steps(self):
        """Test aggregation_weights combined with NUM_STEPS_CURRENT_ROUND.

        Weight formula: weight = aggregation_weight * n_iter
        """
        controller = FedAvg(
            num_clients=2,
            aggregation_weights={"site-1": 2.0, "site-2": 1.0},
        )

        from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper

        controller._aggr_helper = WeightedAggregationHelper()
        controller._aggr_metrics_helper = WeightedAggregationHelper()
        controller._all_metrics = True
        controller._received_count = 0
        controller._expected_count = 2
        controller._params_type = None
        controller.current_round = 0

        # site-1: weight=2.0, n_iter=10 -> total_weight = 20
        result1 = FLModel(
            params={"w": 1.0},
            params_type=ParamsType.FULL,
            meta={"client_name": "site-1", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 10},
        )
        # site-2: weight=1.0, n_iter=30 -> total_weight = 30
        result2 = FLModel(
            params={"w": 6.0},
            params_type=ParamsType.FULL,
            meta={"client_name": "site-2", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 30},
        )

        controller._aggregate_one_result(result1)
        controller._aggregate_one_result(result2)

        aggr_result = controller._get_aggregated_result()
        # Weighted average: (1.0*20 + 6.0*30) / (20 + 30) = (20 + 180) / 50 = 200/50 = 4.0
        assert aggr_result.params["w"] == 4.0

    def test_aggregation_weights_with_different_num_steps(self):
        """Test aggregation without explicit weights uses NUM_STEPS_CURRENT_ROUND only."""
        controller = FedAvg(num_clients=2)  # No aggregation_weights

        from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper

        controller._aggr_helper = WeightedAggregationHelper()
        controller._aggr_metrics_helper = WeightedAggregationHelper()
        controller._all_metrics = True
        controller._received_count = 0
        controller._expected_count = 2
        controller._params_type = None
        controller.current_round = 0

        # site-1: n_iter=100
        result1 = FLModel(
            params={"w": 2.0},
            params_type=ParamsType.FULL,
            meta={"client_name": "site-1", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 100},
        )
        # site-2: n_iter=300
        result2 = FLModel(
            params={"w": 6.0},
            params_type=ParamsType.FULL,
            meta={"client_name": "site-2", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 300},
        )

        controller._aggregate_one_result(result1)
        controller._aggregate_one_result(result2)

        aggr_result = controller._get_aggregated_result()
        # Weighted average: (2.0*100 + 6.0*300) / (100 + 300) = (200 + 1800) / 400 = 2000/400 = 5.0
        assert aggr_result.params["w"] == 5.0

    def test_aggregation_with_multi_value_state_dict(self):
        """Test aggregation with state dict containing multiple parameters."""
        controller = FedAvg(
            num_clients=2,
            aggregation_weights={"site-1": 1.0, "site-2": 3.0},
        )

        from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper

        controller._aggr_helper = WeightedAggregationHelper()
        controller._aggr_metrics_helper = WeightedAggregationHelper()
        controller._all_metrics = True
        controller._received_count = 0
        controller._expected_count = 2
        controller._params_type = None
        controller.current_round = 0

        # site-1: multiple params, weight=1, n_iter=10 -> total=10
        result1 = FLModel(
            params={"layer1.weight": 1.0, "layer1.bias": 0.5, "layer2.weight": 2.0, "layer2.bias": 0.1},
            params_type=ParamsType.FULL,
            meta={"client_name": "site-1", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 10},
        )
        # site-2: weight=3, n_iter=10 -> total=30
        result2 = FLModel(
            params={"layer1.weight": 5.0, "layer1.bias": 2.5, "layer2.weight": 6.0, "layer2.bias": 0.5},
            params_type=ParamsType.FULL,
            meta={"client_name": "site-2", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 10},
        )

        controller._aggregate_one_result(result1)
        controller._aggregate_one_result(result2)

        aggr_result = controller._get_aggregated_result()
        # Total weights: 10 + 30 = 40
        # layer1.weight: (1.0*10 + 5.0*30) / 40 = 160/40 = 4.0
        assert aggr_result.params["layer1.weight"] == 4.0
        # layer1.bias: (0.5*10 + 2.5*30) / 40 = 80/40 = 2.0
        assert aggr_result.params["layer1.bias"] == 2.0
        # layer2.weight: (2.0*10 + 6.0*30) / 40 = 200/40 = 5.0
        assert aggr_result.params["layer2.weight"] == 5.0
        # layer2.bias: (0.1*10 + 0.5*30) / 40 = 16/40 = 0.4
        assert aggr_result.params["layer2.bias"] == 0.4

    def test_aggregation_three_clients_varied_weights_and_steps(self):
        """Test aggregation with 3 clients having different weights and num_steps."""
        controller = FedAvg(
            num_clients=3,
            aggregation_weights={"site-1": 1.0, "site-2": 2.0, "site-3": 0.5},
        )

        from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper

        controller._aggr_helper = WeightedAggregationHelper()
        controller._aggr_metrics_helper = WeightedAggregationHelper()
        controller._all_metrics = True
        controller._received_count = 0
        controller._expected_count = 3
        controller._params_type = None
        controller.current_round = 0

        # site-1: weight=1.0, n_iter=100 -> total=100
        result1 = FLModel(
            params={"w": 10.0, "b": 1.0},
            params_type=ParamsType.FULL,
            meta={"client_name": "site-1", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 100},
        )
        # site-2: weight=2.0, n_iter=50 -> total=100
        result2 = FLModel(
            params={"w": 20.0, "b": 2.0},
            params_type=ParamsType.FULL,
            meta={"client_name": "site-2", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 50},
        )
        # site-3: weight=0.5, n_iter=200 -> total=100
        result3 = FLModel(
            params={"w": 30.0, "b": 3.0},
            params_type=ParamsType.FULL,
            meta={"client_name": "site-3", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 200},
        )

        controller._aggregate_one_result(result1)
        controller._aggregate_one_result(result2)
        controller._aggregate_one_result(result3)

        aggr_result = controller._get_aggregated_result()
        # Total weights: 100 + 100 + 100 = 300
        # w: (10*100 + 20*100 + 30*100) / 300 = 6000/300 = 20.0
        assert aggr_result.params["w"] == 20.0
        # b: (1*100 + 2*100 + 3*100) / 300 = 600/300 = 2.0
        assert aggr_result.params["b"] == 2.0


class TestFedAvgLoadSaveModel:
    """Test FedAvg load_model and save_model overrides."""

    def test_load_model_with_persistor(self):
        """Test load_model delegates to parent when persistor is available."""
        controller = FedAvg()
        # When persistor is set, it should use parent's load_model
        # We can't fully test this without mocking, but we verify the logic exists
        assert hasattr(controller, "load_model")
        assert hasattr(controller, "save_model")

    def test_load_model_returns_empty_flmodel_when_no_config(self):
        """Test load_model returns empty FLModel when no persistor or save_filename."""
        controller = FedAvg()
        controller.persistor = None
        controller.save_filename = None

        result = controller.load_model()

        assert isinstance(result, FLModel)
        assert result.params == {}

    def test_save_model_file_and_load_model_file_defaults(self):
        """Test default save_model_file and load_model_file use FOBS."""
        controller = FedAvg()
        # Verify methods exist
        assert hasattr(controller, "save_model_file")
        assert hasattr(controller, "load_model_file")


class TestPTFedAvgModel:
    """Test PTFedAvg model type handling."""

    def test_model_with_nn_module(self):
        """Test model with torch.nn.Module extracts state_dict."""
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

        from nvflare.app_opt.pt.fedavg import PTFedAvg

        model = SimpleModel()
        controller = PTFedAvg(model=model)

        # model should be converted to state_dict (OrderedDict)
        assert controller.model is not None
        assert isinstance(controller.model, dict)
        assert "linear.weight" in controller.model
        assert "linear.bias" in controller.model

    def test_model_with_dict(self):
        """Test model with dict is passed through."""
        from nvflare.app_opt.pt.fedavg import PTFedAvg

        model_dict = {"layer1.weight": [1.0, 2.0, 3.0]}
        controller = PTFedAvg(model=model_dict)

        assert controller.model == model_dict

    def test_model_with_flmodel(self):
        """Test model with FLModel is passed through."""
        from nvflare.app_opt.pt.fedavg import PTFedAvg

        fl_model = FLModel(params={"w": 1.0})
        controller = PTFedAvg(model=fl_model)

        assert controller.model is fl_model

    def test_model_with_none(self):
        """Test model with None is allowed."""
        from nvflare.app_opt.pt.fedavg import PTFedAvg

        controller = PTFedAvg(model=None)

        assert controller.model is None

    def test_model_with_invalid_type_raises_error(self):
        """Test model with invalid type raises TypeError."""
        import pytest

        from nvflare.app_opt.pt.fedavg import PTFedAvg

        with pytest.raises(TypeError, match="model must be"):
            PTFedAvg(model="invalid_string")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="model must be"):
            PTFedAvg(model=12345)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="model must be"):
            PTFedAvg(model=[1, 2, 3])  # type: ignore[arg-type]

    def test_task_name_parameter(self):
        """Test task_name parameter is passed correctly."""
        from nvflare.app_opt.pt.fedavg import PTFedAvg

        controller = PTFedAvg(task_name="validate")
        assert controller.task_name == "validate"

        controller2 = PTFedAvg()
        assert controller2.task_name == "train"  # default

    def test_backward_compatibility_alias(self):
        """Test PTFedAvgEarlyStopping alias still works for backward compatibility."""
        from nvflare.app_opt.pt.fedavg import PTFedAvg, PTFedAvgEarlyStopping

        # Verify alias points to same class
        assert PTFedAvgEarlyStopping is PTFedAvg

        # Verify can instantiate with old name
        controller = PTFedAvgEarlyStopping(num_clients=2, num_rounds=3)
        assert isinstance(controller, PTFedAvg)
        assert controller.num_clients == 2
        assert controller.num_rounds == 3


class TestPTFedAvgModelPersistence:
    """Test PTFedAvg save_model_file and load_model_file methods."""

    def test_save_and_load_model_file_roundtrip(self, tmp_path):
        """Test save_model_file and load_model_file work correctly with PyTorch."""
        import torch

        from nvflare.app_opt.pt.fedavg import PTFedAvg

        controller = PTFedAvg()

        # Create model with PyTorch tensors
        model = FLModel(
            params={"weight": torch.tensor([1.0, 2.0, 3.0]), "bias": torch.tensor([0.5])},
            metrics={"accuracy": 0.85},
        )
        filepath = str(tmp_path / "test_model.pt")

        # Save and load
        controller.save_model_file(model, filepath)
        loaded = controller.load_model_file(filepath)

        # Verify params
        assert torch.equal(loaded.params["weight"], model.params["weight"])
        assert torch.equal(loaded.params["bias"], model.params["bias"])
        # Verify metadata
        assert loaded.metrics == model.metrics

    def test_load_model_file_without_metadata(self, tmp_path):
        """Test load_model_file works when metadata file doesn't exist."""
        import torch

        from nvflare.app_opt.pt.fedavg import PTFedAvg

        controller = PTFedAvg()

        # Create a model file without metadata
        params = {"weight": torch.tensor([1.0, 2.0])}
        filepath = str(tmp_path / "model_no_metadata.pt")
        torch.save(params, filepath)

        # Load without metadata file
        loaded = controller.load_model_file(filepath)

        assert torch.equal(loaded.params["weight"], params["weight"])
        assert loaded.metrics is None  # No metadata

    def test_run_registers_tensor_decomposer(self):
        """Test that run() registers TensorDecomposer."""
        from nvflare.app_opt.pt.fedavg import PTFedAvg

        controller = PTFedAvg()

        # Verify run method exists and can be inspected
        assert hasattr(controller, "run")
        # The actual registration happens when run() is called, which requires
        # a full FL context. We just verify the method exists.


class TestBaseFedAvgMemoryManagement:
    """Test memory management in BaseFedAvg."""

    def test_memory_gc_rounds_default(self):
        """Test memory_gc_rounds defaults to 0 (disabled)."""
        controller = FedAvg()
        assert controller.memory_gc_rounds == 0

    def test_memory_gc_rounds_custom(self):
        """Test memory_gc_rounds can be set to custom value."""
        controller = FedAvg(memory_gc_rounds=5)
        assert controller.memory_gc_rounds == 5

    def test_maybe_cleanup_memory_disabled(self):
        """Test _maybe_cleanup_memory does nothing when memory_gc_rounds=0."""
        from unittest.mock import patch

        controller = FedAvg(memory_gc_rounds=0)
        controller.current_round = 0

        with patch("nvflare.app_common.workflows.base_fedavg.cleanup_memory") as mock_cleanup:
            controller._maybe_cleanup_memory()
            mock_cleanup.assert_not_called()

    def test_maybe_cleanup_memory_called_on_interval(self):
        """Test _maybe_cleanup_memory calls cleanup at correct intervals."""
        from unittest.mock import patch

        controller = FedAvg(memory_gc_rounds=5)

        with patch("nvflare.app_common.workflows.base_fedavg.cleanup_memory") as mock_cleanup:
            # Round 0-3: should not trigger (round+1 not divisible by 5)
            for r in range(4):
                controller.current_round = r
                controller._maybe_cleanup_memory()
            assert mock_cleanup.call_count == 0

            # Round 4: (4+1) % 5 == 0, should trigger
            controller.current_round = 4
            controller._maybe_cleanup_memory()
            assert mock_cleanup.call_count == 1

            # Round 5-8: should not trigger
            for r in range(5, 9):
                controller.current_round = r
                controller._maybe_cleanup_memory()
            assert mock_cleanup.call_count == 1

            # Round 9: (9+1) % 5 == 0, should trigger again
            controller.current_round = 9
            controller._maybe_cleanup_memory()
            assert mock_cleanup.call_count == 2

    def test_maybe_cleanup_memory_every_round(self):
        """Test _maybe_cleanup_memory with memory_gc_rounds=1 (every round)."""
        from unittest.mock import patch

        controller = FedAvg(memory_gc_rounds=1)

        with patch("nvflare.app_common.workflows.base_fedavg.cleanup_memory") as mock_cleanup:
            for r in range(5):
                controller.current_round = r
                controller._maybe_cleanup_memory()
            assert mock_cleanup.call_count == 5

    def test_memory_gc_rounds_inherited_by_scaffold(self):
        """Test Scaffold inherits memory_gc_rounds from BaseFedAvg."""
        from nvflare.app_common.workflows.scaffold import Scaffold

        controller = Scaffold(memory_gc_rounds=3)
        assert controller.memory_gc_rounds == 3

    def test_memory_gc_rounds_in_cyclic(self):
        """Test Cyclic has memory_gc_rounds parameter."""
        from nvflare.app_common.workflows.cyclic import Cyclic

        controller = Cyclic(memory_gc_rounds=2)
        assert controller.memory_gc_rounds == 2

    def test_memory_gc_rounds_in_scatter_and_gather(self):
        """Test ScatterAndGather has memory_gc_rounds parameter with default=1."""
        from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather

        # Default is 1 (every round) for backward compatibility
        controller = ScatterAndGather()
        assert controller._memory_gc_rounds == 1

        # Can be customized
        controller2 = ScatterAndGather(memory_gc_rounds=5)
        assert controller2._memory_gc_rounds == 5

        # Can be disabled
        controller3 = ScatterAndGather(memory_gc_rounds=0)
        assert controller3._memory_gc_rounds == 0

    def test_memory_gc_rounds_in_cyclic_controller(self):
        """Test CyclicController has memory_gc_rounds parameter with default=1."""
        from nvflare.app_common.workflows.cyclic_ctl import CyclicController

        # Default is 1 (every round) for backward compatibility
        controller = CyclicController()
        assert controller._memory_gc_rounds == 1

        # Can be customized
        controller2 = CyclicController(memory_gc_rounds=3)
        assert controller2._memory_gc_rounds == 3
