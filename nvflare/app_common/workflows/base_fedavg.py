# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import math
from threading import Lock
from typing import Any, Dict, List, Optional

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.abstract.model import make_model_learnable
from nvflare.app_common.aggregators.weighted_aggregation_helper import (
    AggregationStatsKey,
    WeightedAggregationHelper,
    compute_key_match_stats,
    filter_aggregatable_metrics,
)
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.fuel.utils.memory_utils import cleanup_memory
from nvflare.fuel.utils.validation_utils import check_non_negative_int
from nvflare.security.logging import secure_format_exception

from .model_controller import ModelController


def make_fedavg_metrics_aggregation_info(
    key_metric: Optional[str] = None,
    key_metric_mode: Optional[str] = None,
    key_metric_mode_source: Optional[str] = None,
    weight_key: str = FLMetaKey.NUM_STEPS_CURRENT_ROUND,
    weight_formula: Optional[str] = None,
    site_weights: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    aggregation = {
        "method": "weighted_average",
        "weight_key": weight_key,
        "metric_policy": "finite_numeric_metrics_only_per_key_denominator",
    }
    if weight_formula:
        aggregation["weight_formula"] = weight_formula

    info = {
        "metric_source": "client_reported_flmodel_metrics",
        "aggregation": aggregation,
    }
    if key_metric and key_metric_mode in ("max", "min"):
        key_metric_info = {"name": key_metric, "mode": key_metric_mode}
        if key_metric_mode_source:
            key_metric_info["mode_source"] = key_metric_mode_source
        info["key_metric"] = key_metric_info
    if site_weights:
        info["site_weights"] = site_weights
    return info


def make_key_metric_info_from_stop_condition(stop_cond, stop_condition) -> Optional[Dict[str, Any]]:
    if not stop_cond or not stop_condition:
        return None
    tokens = stop_cond.split(" ")
    if len(tokens) != 3:
        return None
    op = tokens[1]
    if op in (">", ">="):
        mode = "max"
    elif op in ("<", "<="):
        mode = "min"
    else:
        return None
    return {
        "name": stop_condition[0],
        "mode": mode,
        "mode_source": "derived_from_stop_condition",
    }


def _get_client_name(result: FLModel) -> str:
    meta = result.meta or {}
    value = meta.get("client_name", AppConstants.CLIENT_UNKNOWN)
    return value if isinstance(value, str) and value else AppConstants.CLIENT_UNKNOWN


def _get_num_steps_weight(result: FLModel) -> float:
    meta = result.meta or {}
    value = meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND)
    if value is None or isinstance(value, bool):
        return 1.0
    try:
        weight = float(value)
    except (TypeError, ValueError, OverflowError):
        return 1.0
    if weight == 0:
        return 0.0
    if not math.isfinite(weight) or weight <= 0:
        return 1.0
    return weight


def _is_zero_step_result(result: FLModel) -> bool:
    return _get_num_steps_weight(result) == 0.0


def _release_params(result: FLModel):
    if not isinstance(result.params, dict):
        return
    for value in result.params.values():
        release = getattr(value, "release", None)
        if callable(release):
            release()


def _without_zero_step_results(results: List[FLModel], on_skip=None) -> List[FLModel]:
    kept = []
    for result in results:
        if not _is_zero_step_result(result):
            kept.append(result)
            continue
        if on_skip:
            on_skip(result)
        _release_params(result)
    return kept


class _UnchangedModelResult(FLModel):
    pass


def _keeps_model(aggr_result) -> bool:
    return type(aggr_result) is _UnchangedModelResult


_TASK_REPLY_COUNT = "_fedavg_task_reply_count"


class _ReplyCountedResults(list):
    def __init__(self, results, reply_count: int):
        super().__init__(results)
        self.reply_count = reply_count


def _make_unchanged_model_result(current_round) -> FLModel:
    return _UnchangedModelResult(
        params={},
        params_type=ParamsType.DIFF,
        current_round=current_round,
        meta={
            "nr_aggregated": 0,
            "current_round": current_round,
            AppConstants.METRICS_AGGREGATION_INFO: make_fedavg_metrics_aggregation_info(),
        },
    )


def _aggregate_fl_model_metrics(results: List[FLModel]) -> Optional[Dict[str, Any]]:
    """Aggregate FLModel metrics across results with FedAvg-compatible semantics.

    Metrics are weighted averages of client metric values. For non-linear metrics
    such as AUROC, this is not the same as computing the metric from pooled
    predictions across all clients.
    """
    aggr_metrics_helper = WeightedAggregationHelper()
    for _result in results:
        if _is_zero_step_result(_result):
            continue
        if _result.metrics is None:
            return None
        aggregatable = filter_aggregatable_metrics(_result.metrics)
        if aggregatable:
            aggr_metrics_helper.add(
                data=aggregatable,
                weight=_get_num_steps_weight(_result),
                contributor_name=_get_client_name(_result),
                contribution_round=_result.current_round,
            )

    aggr_metrics = aggr_metrics_helper.get_result()
    return aggr_metrics or None


class BaseFedAvg(ModelController):
    def __init__(
        self,
        *args,
        num_clients: int = 3,
        num_rounds: int = 5,
        start_round: int = 0,
        memory_gc_rounds: int = 0,
        **kwargs,
    ):
        """The base controller for FedAvg Workflow. *Note*: This class is based on the `ModelController`.

        Implements [FederatedAveraging](https://arxiv.org/abs/1602.05629).

        A model persistor can be configured via the `persistor_id` argument of the `ModelController`.
        The model persistor is used to load the initial global model which is sent to a list of clients.
        Each client sends it's updated weights after local training which is aggregated.
        Next, the global model is updated.
        The model_persistor will also save the model after training.

        Provides the default implementations for the follow routines:
            - def aggregate(self, results: List[FLModel], aggregate_fn=None) -> FLModel
            - def update_model(self, aggr_result)

        The `run` routine needs to be implemented by the derived class:

            - def run(self)

        Args:
            num_clients (int, optional): The number of clients. Defaults to 3. NOTE: this argument should not be here
            we will remove this argument in next release.
            num_rounds (int, optional): The total number of training rounds. Defaults to 5.
            start_round (int, optional): The starting round number.
            memory_gc_rounds (int, optional): Run memory cleanup (gc.collect + malloc_trim) every N rounds.
                Set to 0 to disable. Defaults to 0 (disabled).
        """
        super().__init__(*args, **kwargs)

        check_non_negative_int("memory_gc_rounds", memory_gc_rounds)

        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.start_round = start_round
        self.memory_gc_rounds = memory_gc_rounds

        self.current_round = None
        self._reply_count = 0
        self._reply_lock = Lock()
        self._task_reply_count = None

    def _prepare_task(self, data, task_name, timeout, callback):
        task = super()._prepare_task(data=data, task_name=task_name, timeout=timeout, callback=callback)
        self._task_reply_count = [0]
        task.set_prop(_TASK_REPLY_COUNT, self._task_reply_count)
        return task

    def _process_result(self, client_task, fl_ctx) -> None:
        with self._reply_lock:
            self._reply_count += 1
            task_reply_count = client_task.task.get_prop(_TASK_REPLY_COUNT)
            if task_reply_count is not None:
                task_reply_count[0] += 1
        super()._process_result(client_task, fl_ctx)

    def send_model_and_wait(self, *args, **kwargs) -> List[FLModel]:
        self._task_reply_count = None
        results = super().send_model_and_wait(*args, **kwargs)
        with self._reply_lock:
            reply_count = self._task_reply_count[0] if self._task_reply_count else 0
            self._task_reply_count = None
        return _ReplyCountedResults(results, reply_count)

    def _maybe_cleanup_memory(self):
        """Perform memory cleanup if configured (every N rounds based on memory_gc_rounds)."""
        if self.current_round is None:
            return
        if self.memory_gc_rounds > 0 and (self.current_round + 1) % self.memory_gc_rounds == 0:
            self.info(f"Memory cleanup at round {self.current_round + 1}")
            cleanup_memory()

    @staticmethod
    def _check_results(results: List[FLModel]):
        empty_clients = []
        for _result in results:
            if not _result.params:
                empty_clients.append(_get_client_name(_result))

        if len(empty_clients) > 0:
            raise ValueError(f"Result from client(s) {empty_clients} is empty!")

    @staticmethod
    def aggregate_fn(results: List[FLModel]) -> FLModel:
        """Aggregate model params and metrics across results with weighted averaging.

        Note:
            Metric values that do not support weighted arithmetic are skipped during
            aggregation. If no aggregatable metrics remain after filtering, the
            aggregated metrics are returned as ``None``.
        """
        if not results:
            raise ValueError("received empty results for aggregation.")
        kept_results = _without_zero_step_results(results)
        if not kept_results:
            return _make_unchanged_model_result(results[0].current_round)
        results = kept_results

        aggr_helper = WeightedAggregationHelper()
        for _result in results:
            aggr_helper.add(
                data=_result.params,
                weight=_get_num_steps_weight(_result),
                contributor_name=_get_client_name(_result),
                contribution_round=_result.current_round,
            )

        aggr_params = aggr_helper.get_result()
        aggr_metrics = _aggregate_fl_model_metrics(results)

        aggr_result = FLModel(
            params=aggr_params,
            params_type=results[0].params_type,
            metrics=aggr_metrics,
            meta={
                "nr_aggregated": len(results),
                "current_round": results[0].current_round,
                AppConstants.METRICS_AGGREGATION_INFO: make_fedavg_metrics_aggregation_info(),
            },
        )
        return aggr_result

    def aggregate(self, results: List[FLModel], aggregate_fn=None) -> FLModel:
        """Called by the `run` routine to aggregate the training results of clients.

        Args:
            results: a list of FLModel containing training results of the clients.
            aggregate_fn: a function that turns the list of FLModel into one resulting (aggregated) FLModel.

        Returns: aggregated FLModel.

        """
        self.debug("Start aggregation.")
        # FLContext is reused across rounds. Clear the prior round's value before any
        # aggregation work so an empty/unsupported result set or stats-computation failure
        # cannot expose stale statistics to AFTER_AGGREGATION handlers.
        if self.fl_ctx:
            self._set_ctx_prop_preserving_attrs(fl_ctx=self.fl_ctx, key=AppConstants.AGGREGATION_STATS, value=None)
        self.event(AppEventType.BEFORE_AGGREGATION)
        reply_count = results.reply_count if isinstance(results, _ReplyCountedResults) else 0
        kept_results = self._drop_zero_step_results(results)
        if results and not kept_results:
            if reply_count > len(results):
                raise RuntimeError("no accepted contributions to aggregate")
            self.warning(f"All results reported 0 steps at round {self.current_round}, keeping the global model.")
            aggr_result = _make_unchanged_model_result(self.current_round)
            self._publish_zero_contribution_stats()
            self._results = []
            self.fire_event_with_data(
                AppEventType.AFTER_AGGREGATION, self.fl_ctx, AppConstants.AGGREGATION_RESULT, aggr_result
            )
            return aggr_result
        results = kept_results
        self._check_results(results)

        if not aggregate_fn:
            aggregate_fn = self.aggregate_fn

        self.info(f"aggregating {len(results)} update(s) at round {self.current_round}")
        try:
            aggr_result = aggregate_fn(results)
        except Exception as e:
            error_msg = f"Exception in aggregate call: {secure_format_exception(e)}"
            self.exception(error_msg)
            self.panic(error_msg)
            return FLModel()
        self._results = []
        self._set_metrics_aggregation_info(aggr_result)

        # Reporting stats must never break a working aggregation: params is typed Any (custom
        # aggregate_fn callers may use ndarrays etc.), so only dict params contribute key stats
        # and any unexpected error is swallowed.
        try:
            contributions = {_get_client_name(r): list(r.params.keys()) for r in results if isinstance(r.params, dict)}
            if contributions:
                aggr_stats = compute_key_match_stats(contributions)
                aggr_stats[AggregationStatsKey.ROUND] = self.current_round
                if self.fl_ctx:
                    self.fl_ctx.set_prop(AppConstants.AGGREGATION_STATS, aggr_stats, private=True, sticky=False)
        except Exception as e:
            self.debug(f"unable to compute aggregation key stats: {secure_format_exception(e)}")

        self.fire_event_with_data(
            AppEventType.AFTER_AGGREGATION, self.fl_ctx, AppConstants.AGGREGATION_RESULT, aggr_result
        )

        self.debug("End aggregation.")

        return aggr_result

    def _publish_zero_contribution_stats(self):
        if self.fl_ctx:
            aggr_stats = compute_key_match_stats({})
            aggr_stats[AggregationStatsKey.ROUND] = self.current_round
            self.fl_ctx.set_prop(AppConstants.AGGREGATION_STATS, aggr_stats, private=True, sticky=False)

    def _drop_zero_step_results(self, results: List[FLModel]) -> List[FLModel]:
        return _without_zero_step_results(
            results,
            on_skip=lambda result: self.warning(
                f"Skipping result from client {_get_client_name(result)} at round {self.current_round}: 0 steps."
            ),
        )

    def _set_metrics_aggregation_info(self, aggr_result: FLModel):
        if not isinstance(aggr_result, FLModel):
            return
        aggr_result.meta = aggr_result.meta or {}
        info = aggr_result.meta.get(AppConstants.METRICS_AGGREGATION_INFO)
        if isinstance(info, dict):
            info = dict(info)
        else:
            info = make_fedavg_metrics_aggregation_info()
        key_metric_info = make_key_metric_info_from_stop_condition(
            getattr(self, "stop_cond", None), getattr(self, "stop_condition", None)
        )
        if key_metric_info and "key_metric" not in info:
            info["key_metric"] = key_metric_info
        aggr_result.meta[AppConstants.METRICS_AGGREGATION_INFO] = info

    def update_model(self, model, aggr_result):
        """Called by the `run` routine to update the current global model (self.model) given the aggregated result.

        Args:
            model: FLModel to be updated.
            aggr_result: aggregated FLModel.

        Returns: None.

        """
        if _keeps_model(aggr_result):
            return model

        self.event(AppEventType.BEFORE_SHAREABLE_TO_LEARNABLE)

        model = FLModelUtils.update_model(model, aggr_result)

        # persistor uses Learnable format to save model
        ml = make_model_learnable(weights=model.params, meta_props=model.meta)
        self.fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, ml, private=True, sticky=True)

        self.event(AppEventType.AFTER_SHAREABLE_TO_LEARNABLE)

        return model
