from typing import Optional, Dict

from experimental.fl_api.common.aggregator.aggregators import DefaultAggregator, FuncAggregatorWrapper
from experimental.fl_api.common.interfaces.message_type import FedModel
from experimental.fl_api.common.interfaces.strategy import StrategyConfig


def get_aggregator(strategy_config: Optional[StrategyConfig] = None):
    aggregator = None
    if strategy_config:
        if hasattr(strategy_config, "aggregator") and strategy_config.aggregator is not None:
            aggregator = strategy_config.aggregator
        elif hasattr(strategy_config, "aggregation_fn") and strategy_config.aggregation_fn:
            aggregator = FuncAggregatorWrapper(strategy_config.aggregation_fn)
        else:
            aggregator = DefaultAggregator()
    return aggregator


def aggregate(model: FedModel, updates: Dict[str, FedModel], strategy_config: StrategyConfig) -> FedModel:
    aggregator = get_aggregator(strategy_config)
    result: FedModel = aggregator.aggregate(list(updates.values()))
    return update_model(model, result)


def update_model(model: FedModel, model_update: FedModel, replace_meta: bool = True) -> FedModel:
    if model.params_type != "FULL":
        raise RuntimeError(f"params_type {model.params_type} of `model` not supported! Expected `ParamsType.FULL`.")

    if replace_meta:
        model.meta = model_update.meta
    else:
        model.meta.update(model_update.meta)

    model.metrics = model_update.metrics

    if model_update.params_type == "FULL":
        model.params = model_update.params
    elif model_update.params_type == "DIFF":
        for v_name, v_value in model_update.params.items():
            model.params[v_name] = model.params[v_name] + v_value
    else:
        raise RuntimeError(f"params_type {model_update.params_type} of `model_update` not supported!")
    return model
