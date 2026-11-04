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

from abc import abstractmethod
from copy import deepcopy

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.utils.fl_model_utils import MODEL_ATTRS, FLModelUtils


class ModelFilter(DXOFilter):
    """Experimental task filter with the Client API's FLModel interface.

    Parameters are borrowed, not copied. Return a model to record processing, or
    None to skip without modifying the input. The callback cannot change the wire
    representation or round identity. Existing DXO filters can share the chain.
    """

    KINDS = [DataKind.WEIGHTS, DataKind.WEIGHT_DIFF, DataKind.METRICS, DataKind.FL_MODEL]
    ROUNDS = {
        "start_round": AppConstants.START_ROUND,
        "current_round": AppConstants.CURRENT_ROUND,
        "total_rounds": AppConstants.NUM_ROUNDS,
    }

    def __init__(self, data_kinds_to_filter=None):
        super().__init__(supported_data_kinds=list(self.KINDS), data_kinds_to_filter=data_kinds_to_filter)

    def process_dxo(self, dxo: DXO, shareable: Shareable, fl_ctx: FLContext) -> DXO | None:
        native = dxo.data_kind == DataKind.FL_MODEL
        # Decode this leaf, not an enclosing DXO collection or command envelope.
        model = FLModelUtils.from_dxo(dxo) if native else FLModelUtils.from_shareable(dxo.to_shareable())
        model.meta = deepcopy(model.meta)
        model.metrics = deepcopy(model.metrics)
        for attr, header in self.ROUNDS.items():
            envelope_round = shareable.get_header(header)
            # Legacy metadata can describe the preceding aggregation round. Task
            # headers are authoritative; metadata is only a compatibility fallback.
            values = (
                [envelope_round, getattr(model, attr)]
                if native
                else [envelope_round if envelope_round is not None else dxo.meta.get(header)]
            )
            values = [v for v in values if v is not None]
            if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
                raise ValueError(f"Invalid {attr}")
            if values and any(v != values[0] for v in values):
                raise ValueError(f"Conflicting {attr}")
            setattr(model, attr, values[0] if values else None)
        identity = tuple(getattr(model, a) for a in self.ROUNDS)
        params_type = model.params_type
        result = self.process_model(model, fl_ctx)
        if result is None:
            return None
        if not isinstance(result, FLModel):
            raise TypeError("process_model must return FLModel or None")
        if any(
            type(getattr(result, a)) is not type(v) or getattr(result, a) != v for a, v in zip(self.ROUNDS, identity)
        ):
            raise ValueError("ModelFilter cannot change round identity")
        if result.params_type != params_type:
            raise ValueError("ModelFilter cannot change parameter type")
        if native:
            data = dict(dxo.data)
            for attr in MODEL_ATTRS:
                data.pop(attr, None)
            data.update(FLModelUtils.to_dxo(result).data)
            dxo.data = data
        else:
            # Without a task header, this metadata supplies the next filter's
            # round identity. Guard it as well as the decoded model fields.
            for attr, header in self.ROUNDS.items():
                if shareable.get_header(header) is None:
                    original = dxo.meta.get(header)
                    updated = result.meta.get(header)
                    if type(updated) is not type(original) or updated != original:
                        raise ValueError("ModelFilter cannot change round identity in metadata")
            if result.optimizer_params is not None:
                raise ValueError("Legacy DXO encoding cannot carry optimizer parameters")
            if result.metrics is None:
                result.meta.pop(MetaKey.INITIAL_METRICS, None)
            converted = from_shareable(FLModelUtils.to_shareable(result))
            if converted.data_kind != dxo.data_kind:
                raise ValueError("ModelFilter cannot change DXO data kind")
            dxo.data, dxo.meta = converted.data, converted.meta
        return dxo

    @abstractmethod
    def process_model(self, model: FLModel, fl_ctx: FLContext) -> FLModel | None:
        """Process model fields. Returning None means an unmodified pass-through."""
