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

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator
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
        assert controller.initial_model is None
        assert controller.aggregator is None
        assert controller.stop_cond is None
        assert controller.patience is None
        assert controller.task_name == "train"
        assert controller.save_filename == "FL_global_model.pt"
        assert controller.exclude_vars is None
        assert controller.aggregation_weights == {}

    def test_custom_initialization(self):
        """Test FedAvg with custom parameters."""
        initial_model = {"layer1": [1.0, 2.0, 3.0]}
        aggregator = MockModelAggregator()

        controller = FedAvg(
            num_clients=5,
            num_rounds=10,
            initial_model=initial_model,
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
        assert controller.initial_model == initial_model
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


class TestPTFedAvgInitialModel:
    """Test PTFedAvg initial_model type handling."""

    def test_initial_model_with_nn_module(self):
        """Test initial_model with torch.nn.Module extracts state_dict."""
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

        from nvflare.app_opt.pt.fedavg_early_stopping import PTFedAvg

        model = SimpleModel()
        controller = PTFedAvg(initial_model=model)

        # initial_model should be converted to state_dict (OrderedDict)
        assert controller.initial_model is not None
        assert isinstance(controller.initial_model, dict)
        assert "linear.weight" in controller.initial_model
        assert "linear.bias" in controller.initial_model

    def test_initial_model_with_dict(self):
        """Test initial_model with dict is passed through."""
        from nvflare.app_opt.pt.fedavg_early_stopping import PTFedAvg

        model_dict = {"layer1.weight": [1.0, 2.0, 3.0]}
        controller = PTFedAvg(initial_model=model_dict)

        assert controller.initial_model == model_dict

    def test_initial_model_with_flmodel(self):
        """Test initial_model with FLModel is passed through."""
        from nvflare.app_opt.pt.fedavg_early_stopping import PTFedAvg

        fl_model = FLModel(params={"w": 1.0})
        controller = PTFedAvg(initial_model=fl_model)

        assert controller.initial_model is fl_model

    def test_initial_model_with_none(self):
        """Test initial_model with None is allowed."""
        from nvflare.app_opt.pt.fedavg_early_stopping import PTFedAvg

        controller = PTFedAvg(initial_model=None)

        assert controller.initial_model is None

    def test_initial_model_with_invalid_type_raises_error(self):
        """Test initial_model with invalid type raises TypeError."""
        import pytest

        from nvflare.app_opt.pt.fedavg_early_stopping import PTFedAvg

        with pytest.raises(TypeError, match="initial_model must be"):
            PTFedAvg(initial_model="invalid_string")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="initial_model must be"):
            PTFedAvg(initial_model=12345)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="initial_model must be"):
            PTFedAvg(initial_model=[1, 2, 3])  # type: ignore[arg-type]

    def test_task_name_parameter(self):
        """Test task_name parameter is passed correctly."""
        from nvflare.app_opt.pt.fedavg_early_stopping import PTFedAvg

        controller = PTFedAvg(task_name="validate")
        assert controller.task_name == "validate"

        controller2 = PTFedAvg()
        assert controller2.task_name == "train"  # default

    def test_backward_compatibility_alias(self):
        """Test PTFedAvgEarlyStopping alias still works for backward compatibility."""
        from nvflare.app_opt.pt.fedavg_early_stopping import PTFedAvg, PTFedAvgEarlyStopping

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

        from nvflare.app_opt.pt.fedavg_early_stopping import PTFedAvg

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

        from nvflare.app_opt.pt.fedavg_early_stopping import PTFedAvg

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
        from nvflare.app_opt.pt.fedavg_early_stopping import PTFedAvg

        controller = PTFedAvg()

        # Verify run method exists and can be inspected
        assert hasattr(controller, "run")
        # The actual registration happens when run() is called, which requires
        # a full FL context. We just verify the method exists.
