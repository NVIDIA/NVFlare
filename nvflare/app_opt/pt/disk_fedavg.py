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

"""Recipe-composed disk aggregation and persistence for sequential PyTorch FedAvg."""

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import torch

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, WorkspaceConstants
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.abstract.model import ModelLearnableKey, make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator
from nvflare.app_common.aggregators.weighted_aggregation_helper import (
    AggregationStatsKey,
    WeightedAggregationHelper,
    filter_aggregatable_metrics,
)
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.workflows.base_fedavg import (
    _get_client_name,
    _get_num_steps_weight,
    make_fedavg_metrics_aggregation_info,
)
from nvflare.fuel.utils import fobs

from .decomposers import TensorDecomposer
from .lazy_tensor_dict import TensorMetadata, _LazyRef, safetensors_dtype, safetensors_refs, write_safetensors

CURRENT_MODEL = ".FL_global_model.current.safetensors"
SAVED_MODEL = "FL_global_model.safetensors"


def _model_path(fl_ctx, name):
    root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
    if not root:
        raise RuntimeError("disk-backed FedAvg requires APP_ROOT")
    return os.path.join(root, name)


def _weighted_metadata(item):
    if item.dtype.startswith(("F", "BF", "C")):
        return item
    dtype = torch.get_default_dtype()
    return TensorMetadata(item.shape, safetensors_dtype(dtype), math.prod(item.shape) * dtype.itemsize)


class DiskFedAvgAggregator(ModelAggregator):
    """Weighted FULL/DIFF FedAvg with only one key's tensors live at a time.

    FedAvg serializes result callbacks and completes all tasks before aggregation.
    Contributions must be disk-backed tensor refs. The current global model comes
    from GLOBAL_MODEL in the existing workflow context, not from a persistor API.
    """

    def __init__(self, aggregation_weights: Optional[dict] = None, exclude_vars: Optional[str] = None):
        super().__init__()
        self.aggregation_weights = aggregation_weights or {}
        self.exclude_vars = exclude_vars
        self._exclude_vars = re.compile(exclude_vars) if exclude_vars else None
        self._metrics = WeightedAggregationHelper()
        self._warned_metrics = set()
        self.reset_stats()

    def handle_event(self, event_type, fl_ctx):
        super().handle_event(event_type, fl_ctx)
        if event_type == AppEventType.BEFORE_AGGREGATION:
            self.fl_ctx = fl_ctx
        if event_type == EventType.END_RUN:
            self.reset_stats()

    def reset_stats(self):
        for params, _, _ in getattr(self, "_contributions", []):
            for ref in params.values():
                ref.release()
        self._contributions = []
        self._metadata = {}
        self._skipped = set()
        self._params_type = None
        self._failed = False
        self._all_metrics = True
        self._metrics.reset_stats()

    def accept_model(self, model):
        try:
            if self._failed:
                raise RuntimeError("disk-backed FedAvg round already failed")
            self._accept_model(model)
        except Exception:
            # The controller catches callback errors; prevent it from publishing a partial round.
            self._failed = True
            if isinstance(model.params, dict):
                for ref in model.params.values():
                    if isinstance(ref, _LazyRef):
                        ref.release()
            raise

    def _accept_model(self, model):
        if model.params_type not in (ParamsType.FULL, ParamsType.DIFF):
            raise ValueError("disk-backed FedAvg requires FULL or DIFF parameters")
        if self._params_type is not None and model.params_type != self._params_type:
            raise ValueError("cannot mix FULL and DIFF contributions in one round")
        if not isinstance(model.params, dict) or not model.params:
            raise TypeError("disk-backed FedAvg requires a non-empty parameter dict")

        metadata = {}
        skipped = set()
        for key, ref in model.params.items():
            if not isinstance(key, str) or not isinstance(ref, _LazyRef):
                raise TypeError("disk-backed FedAvg requires string keys and lazy tensor refs")
            if self._exclude_vars and self._exclude_vars.search(key):
                skipped.add(key)
                continue
            item = ref.get_metadata()
            if key in self._metadata and item != self._metadata[key]:
                raise ValueError(f"tensor '{key}' has different shape or dtype across contributions")
            metadata[key] = item

        contributor = _get_client_name(model)
        weight = self.aggregation_weights.get(contributor, 1.0) * _get_num_steps_weight(model)
        # Validate the complete contribution before changing accepted state.
        self._contributions.append((model.params, weight, contributor))
        self._metadata.update(metadata)
        self._skipped.update(skipped)
        self._params_type = model.params_type
        if model.metrics is None:
            self._all_metrics = False
        if self._all_metrics and model.metrics:
            metrics = filter_aggregatable_metrics(
                model.metrics,
                warn_skipped=lambda key, type_name: self.warning(
                    f"Metric '{key}' ({type_name}) skipped for aggregation."
                ),
                warned_metric_keys=self._warned_metrics,
            )
            if metrics:
                self._metrics.add(metrics, weight, contributor, self.fl_ctx.get_prop(AppConstants.CURRENT_ROUND))

    def _check_abort(self):
        signal = self.fl_ctx.get_run_abort_signal()
        if signal is not None and signal.triggered:
            raise RuntimeError("disk-backed FedAvg aggregation aborted")

    def _aggregate_key(self, key, base):
        helper = WeightedAggregationHelper()
        for params, weight, contributor in self._contributions:
            ref = params.get(key)
            if ref is None:
                continue
            self._check_abort()
            helper.add({key: ref}, weight, contributor, self.fl_ctx.get_prop(AppConstants.CURRENT_ROUND))
        tensor = helper.get_result()[key]
        if key in base:
            tensor = base[key].materialize() + tensor
        return tensor

    def _tensors(self, metadata, base):
        for key in metadata:
            self._check_abort()
            tensor = self._aggregate_key(key, base) if key in self._metadata else base[key].materialize()
            yield key, tensor
            del tensor

    def _stats(self):
        counts = Counter(key for params, _, _ in self._contributions for key in params if key in self._metadata)
        fully_matched = sum(count == len(self._contributions) for count in counts.values())
        return {
            AggregationStatsKey.ROUND: self.fl_ctx.get_prop(AppConstants.CURRENT_ROUND),
            AggregationStatsKey.ACCEPTED_CONTRIBUTIONS: len(self._contributions),
            AggregationStatsKey.CONTRIBUTORS: sorted({client for _, _, client in self._contributions}),
            AggregationStatsKey.KEYS_AGGREGATED: len(counts),
            AggregationStatsKey.KEYS_SEEN: len(counts) + len(self._skipped),
            AggregationStatsKey.FULLY_MATCHED_KEYS: fully_matched,
            AggregationStatsKey.PARTIALLY_MATCHED_KEYS: len(counts) - fully_matched,
            AggregationStatsKey.SKIPPED_KEYS: len(self._skipped),
        }

    def aggregate_model(self):
        current = _model_path(self.fl_ctx, CURRENT_MODEL)
        next_path = current + ".next"
        try:
            if self._failed:
                raise RuntimeError("disk-backed FedAvg round failed; refusing to publish a partial aggregate")
            if not self._contributions:
                raise RuntimeError("no accepted contributions to aggregate")
            base = {}
            if self._params_type == ParamsType.DIFF:
                model = self.fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
                base = model.get(ModelLearnableKey.WEIGHTS, {}) if model else {}
                if not base or not all(isinstance(ref, _LazyRef) for ref in base.values()):
                    raise TypeError("disk-backed DIFF aggregation requires a lazy global model")
                for key, item in self._metadata.items():
                    if key not in base or _weighted_metadata(base[key].get_metadata()) != _weighted_metadata(item):
                        raise ValueError(f"DIFF tensor '{key}' does not match the global model")
            metadata = {
                key: _weighted_metadata(self._metadata[key]) if key in self._metadata else base[key].get_metadata()
                for key in sorted(base if self._params_type == ParamsType.DIFF else self._metadata)
            }
            if not metadata:
                raise ValueError("no tensors remain after exclusions")
            write_safetensors(next_path, metadata, self._tensors(metadata, base))
            self._check_abort()
            os.replace(next_path, current)
            stats = self._stats()
            self.fl_ctx.set_prop(AppConstants.AGGREGATION_STATS, stats, private=True, sticky=False)
            info = make_fedavg_metrics_aggregation_info(
                weight_key="effective_fedavg_metric_weight",
                weight_formula="aggregation_weight * NUM_STEPS_CURRENT_ROUND",
                site_weights=[
                    {"name": client, "weight": weight, "weight_key": "effective_fedavg_metric_weight"}
                    for _, weight, client in self._contributions
                ],
            )
            return FLModel(
                params=safetensors_refs(current),
                params_type=ParamsType.FULL,
                metrics=(self._metrics.get_result() or None) if self._all_metrics else None,
                current_round=stats[AggregationStatsKey.ROUND],
                meta={AppConstants.METRICS_AGGREGATION_INFO: info},
            )
        finally:
            Path(next_path).unlink(missing_ok=True)
            self.reset_stats()


class DiskFedAvgPersistor(ModelPersistor):
    """Load saved/initial safetensors and move current output to the saved slot.

    This component owns storage only. The recipe selects the aggregator. Saved
    contains the latest model, or the controller's best model with early stopping.
    Model metadata and historical snapshots are not persisted.
    """

    def __init__(self, initial_model_path: str):
        super().__init__()
        self.initial_model_path = initial_model_path

    @staticmethod
    def _load_refs(path):
        path = os.path.realpath(path)
        if os.path.isdir(path):
            index = os.path.join(path, "model.safetensors.index.json")
            path = index if os.path.isfile(index) else os.path.join(path, "model.safetensors")
        if path.endswith(".json"):
            with open(path) as source:
                weight_map = json.load(source)["weight_map"]
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("safetensors index requires a non-empty weight_map")
            root = os.path.dirname(path)
            refs = {}
            for shard in set(weight_map.values()):
                shard_path = os.path.realpath(os.path.join(root, shard))
                if os.path.commonpath((root, shard_path)) != root:
                    raise ValueError(f"shard is outside the checkpoint directory: {shard}")
                refs.update(
                    {key: ref for key, ref in safetensors_refs(shard_path).items() if weight_map.get(key) == shard}
                )
            if set(refs) != set(weight_map):
                raise ValueError("safetensors index references missing tensors")
            return refs
        if not path.endswith(".safetensors"):
            raise ValueError("disk-backed FedAvg requires a safetensors checkpoint or index")
        refs = safetensors_refs(path)
        if not refs:
            raise ValueError("initial checkpoint contains no tensors")
        return refs

    def load_model(self, fl_ctx):
        fobs.register(TensorDecomposer)
        saved = _model_path(fl_ctx, SAVED_MODEL)
        source = self.initial_model_path
        if not os.path.isabs(source):
            source = os.path.join(fl_ctx.get_prop(FLContextKey.APP_ROOT), WorkspaceConstants.CUSTOM_FOLDER_NAME, source)
        return make_model_learnable(self._load_refs(saved if os.path.isfile(saved) else source), {})

    def save_model(self, model, fl_ctx):
        current = _model_path(fl_ctx, CURRENT_MODEL)
        saved = _model_path(fl_ctx, SAVED_MODEL)
        weights = model.get(ModelLearnableKey.WEIGHTS)
        if not weights or not all(
            isinstance(ref, _LazyRef) and ref.key == key and ref.file_path == current for key, ref in weights.items()
        ):
            raise ValueError("disk-backed persistence requires the current aggregation output")
        os.replace(current, saved)
        for ref in weights.values():
            ref.file_path = saved
