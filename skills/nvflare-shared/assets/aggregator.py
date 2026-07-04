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

"""Packaged custom-aggregation template: a bounded step-weighted aggregator.

Copy and adapt this into a generated ``aggregators.py`` when the conversion
needs custom aggregation. Wire it through the recipe ``aggregator=`` parameter
in ``job.py`` with the matching ``aggregator_data_kind`` and parameter transfer
settings. This uses NVFlare's ``ModelAggregator`` extension point and preserves
both parameters and scalar metrics from the ``FLModel`` exchange contract.

The limits below are deliberately conservative safety defaults. Tune them to
the expected model and federation size rather than removing validation.
``max_param_bytes`` bounds each input, promoted contribution, accumulator, and
result, but it is not a process-RSS ceiling: atomic out-of-place aggregation can
temporarily retain the input, prior accumulator, contribution, and replacement
at once. Enforce the validation sandbox's independent memory limit with enough
headroom for those copies.
"""

import math
import threading
from functools import lru_cache
from typing import Optional, Tuple

from nvflare.apis.dxo import MetaKey
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator

DEFAULT_MAX_STEP_WEIGHT = 1_000_000_000.0
DEFAULT_MAX_MODELS = 10_000
DEFAULT_MAX_PARAM_KEYS = 100_000
DEFAULT_MAX_PARAM_ELEMENTS = 250_000_000
DEFAULT_MAX_PARAM_BYTES = 1_073_741_824
DEFAULT_MAX_METRIC_KEYS = 1_024
DEFAULT_MAX_METRIC_BYTES = 1_048_576
DEFAULT_MAX_KEY_LENGTH = 1_024


def _default_trusted_lazy_types() -> Tuple[type, ...]:
    """Return NVFlare's disk-offload reference type when it is available."""

    try:
        # This is the type produced by enable_tensor_disk_offload. Keeping the
        # import here makes the template usable when the optional PT package is
        # absent, while avoiding duck-calling attacker-controlled materialize().
        from nvflare.app_opt.pt.lazy_tensor_dict import LazyTensorRef
    except ImportError:
        return ()
    return (LazyTensorRef,)


def _step_weight(model: FLModel, max_step_weight: float) -> float:
    """Return a safe positive step weight, falling back only for bad metadata."""

    value = (model.meta or {}).get(MetaKey.NUM_STEPS_CURRENT_ROUND)
    if type(value) not in (int, float) and not _is_numpy_scalar(value):
        return 1.0
    try:
        weight = float(value)
    except (TypeError, ValueError, OverflowError):
        return 1.0
    if not math.isfinite(weight) or weight <= 0:
        return 1.0
    if weight > max_step_weight:
        raise ValueError(f"step weight {weight!r} exceeds configured maximum {max_step_weight!r}")
    return weight


def _materialize(
    value,
    trusted_lazy_types: Tuple[type, ...],
    *,
    max_elements: int,
    max_bytes: int,
):
    """Materialize only the explicitly trusted NVFlare lazy-ref type(s)."""

    if trusted_lazy_types and type(value) in trusted_lazy_types:
        value_type = type(value)
        bounded_materialize = getattr(value_type, "materialize_bounded", None)
        if callable(bounded_materialize):
            return bounded_materialize(value, max_elements=max_elements, max_bytes=max_bytes)
        return value.materialize()
    return value


def _value_size(value) -> int:
    numel = getattr(value, "numel", None)
    if callable(numel):
        size = numel()
    else:
        size = getattr(value, "size", 1)
    if callable(size):
        size = size()
    if isinstance(size, tuple):
        result = 1
        for dimension in size:
            result *= int(dimension)
        return result
    try:
        return int(size)
    except (TypeError, ValueError, OverflowError):
        return 1


def _value_nbytes(value) -> int:
    if type(value) in (int, float, bool):
        return 8
    if _is_numpy_value(value):
        return int(value.nbytes)
    if _is_torch_value(value):
        return int(value.numel()) * int(value.element_size())
    return 0


def _value_schema(value):
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            shape = tuple(int(dimension) for dimension in shape)
        except (TypeError, ValueError, OverflowError):
            shape = repr(shape)
    dtype = getattr(value, "dtype", None)
    device = str(value.device) if _is_torch_value(value) else None
    layout = str(value.layout) if _is_torch_value(value) else None
    return (
        type(value).__module__,
        type(value).__qualname__,
        shape,
        str(dtype) if dtype is not None else None,
        device,
        layout,
    )


@lru_cache(maxsize=1)
def _numpy_scalar_types() -> frozenset[type]:
    try:
        import numpy as np
    except ImportError:
        return frozenset()
    return frozenset(
        scalar_type
        for scalar_type in np.sctypeDict.values()
        if isinstance(scalar_type, type) and issubclass(scalar_type, np.generic)
    )


def _is_numpy_value(value) -> bool:
    try:
        import numpy as np
    except ImportError:
        return False
    if type(value) is np.ndarray:
        return True
    return type(value) in _numpy_scalar_types()


def _is_numpy_scalar(value) -> bool:
    return _is_numpy_value(value) and _value_size(value) == 1 and type(value).__name__ != "ndarray"


def _is_supported_numpy_numeric(value) -> bool:
    try:
        import numpy as np
    except ImportError:
        return False
    # Exact NumPy types are required before inspecting dtype. This avoids
    # invoking subclass attribute/array hooks. Object, string, structured,
    # datetime, and timedelta dtypes are not aggregation-safe numeric payloads.
    if type(value) is np.ndarray or isinstance(value, np.generic):
        return not value.dtype.hasobject and value.dtype.kind in "biufc"
    return False


def _is_torch_value(value) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return type(value) is torch.Tensor


def _is_trusted_numeric(value) -> bool:
    # Use concrete classes, not spoofable module/class-name strings or a
    # duck-typed arithmetic protocol. Arithmetic on an arbitrary payload object
    # can execute attacker-controlled methods in the server process.
    return type(value) in (int, float, bool) or _is_numpy_value(value) or _is_torch_value(value)


def _all_finite(value) -> bool:
    if type(value) in (int, float, bool):
        return math.isfinite(float(value))

    if _is_torch_value(value):
        try:
            return bool(value.isfinite().all().item())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False
    if _is_numpy_value(value):
        try:
            import numpy as np

            return bool(np.isfinite(value).all())
        except (TypeError, ValueError):
            return False
    return False


def _normalize_numeric(value, *, label: str):
    if not _is_trusted_numeric(value):
        raise TypeError(f"{label} has unsupported numeric type {type(value).__name__!r}")
    if _is_numpy_value(value) and not _is_supported_numpy_numeric(value):
        if value.dtype.hasobject:
            raise TypeError(f"{label} has unsupported NumPy object dtype")
        raise TypeError(f"{label} has unsupported NumPy dtype {value.dtype!s}")
    if _is_torch_value(value) and value.requires_grad:
        # Client updates are values, not an autograd program. Detaching shares
        # storage, avoids a copy, and prevents accumulator graphs from retaining
        # every prior client tensor.
        value = value.detach()
    return value


def _check_finite(value, *, label: str) -> None:
    if not _all_finite(value):
        raise ValueError(f"{label} contains a non-finite value")


def _checked_size(value, *, label: str) -> Tuple[int, int]:
    elements = _value_size(value)
    num_bytes = _value_nbytes(value)
    if elements < 0 or num_bytes < 0:
        raise ValueError(f"{label} reported an invalid negative size")
    return elements, num_bytes


def _add_checked_size(
    value,
    *,
    label: str,
    current_elements: int,
    current_bytes: int,
    max_elements: int,
    max_bytes: int,
) -> Tuple[int, int]:
    elements, num_bytes = _checked_size(value, label=label)
    total_elements = current_elements + elements
    total_bytes = current_bytes + num_bytes
    if total_elements > max_elements:
        raise ValueError(f"{label} element count exceeds configured maximum {max_elements}")
    if total_bytes > max_bytes:
        raise ValueError(f"{label} byte size exceeds configured maximum {max_bytes}")
    return total_elements, total_bytes


def _predicted_binary_size(left, right) -> Optional[Tuple[int, int]]:
    """Predict a same-shape scalar arithmetic result without allocating it."""

    if type(left) in (int, float, bool):
        return 1, 8
    if _is_numpy_value(left):
        try:
            import numpy as np

            dtype = np.result_type(left, right)
            return _value_size(left), _value_size(left) * int(dtype.itemsize)
        except (TypeError, ValueError, OverflowError):
            return None
    if _is_torch_value(left):
        try:
            import torch

            dtype = torch.result_type(left, right)
            item_size = torch.empty((), dtype=dtype, device="meta").element_size()
            return _value_size(left), _value_size(left) * int(item_size)
        except (RuntimeError, TypeError, ValueError):
            return None
    return None


def _preflight_binary_size(
    left,
    right,
    *,
    label: str,
    current_elements: int,
    current_bytes: int,
    max_elements: int,
    max_bytes: int,
) -> None:
    predicted = _predicted_binary_size(left, right)
    if predicted is None:
        return
    elements, num_bytes = predicted
    if current_elements + elements > max_elements:
        raise ValueError(f"{label} element count exceeds configured maximum {max_elements}")
    if current_bytes + num_bytes > max_bytes:
        raise ValueError(f"{label} byte size exceeds configured maximum {max_bytes}")


class WeightedAggregator(ModelAggregator):
    """Average matching client updates and scalar metrics by local step count."""

    def __init__(
        self,
        *,
        max_step_weight: float = DEFAULT_MAX_STEP_WEIGHT,
        max_models: int = DEFAULT_MAX_MODELS,
        max_param_keys: int = DEFAULT_MAX_PARAM_KEYS,
        max_param_elements: int = DEFAULT_MAX_PARAM_ELEMENTS,
        max_param_bytes: int = DEFAULT_MAX_PARAM_BYTES,
        max_metric_keys: int = DEFAULT_MAX_METRIC_KEYS,
        max_metric_bytes: int = DEFAULT_MAX_METRIC_BYTES,
        max_key_length: int = DEFAULT_MAX_KEY_LENGTH,
        trusted_lazy_types: Optional[Tuple[type, ...]] = None,
    ):
        super().__init__()
        if not math.isfinite(max_step_weight) or max_step_weight <= 0:
            raise ValueError("max_step_weight must be finite and positive")
        if any(
            limit <= 0
            for limit in (
                max_models,
                max_param_keys,
                max_param_elements,
                max_param_bytes,
                max_metric_keys,
                max_metric_bytes,
                max_key_length,
            )
        ):
            raise ValueError("aggregation size limits must be positive")

        self.max_step_weight = float(max_step_weight)
        self.max_models = int(max_models)
        self.max_param_keys = int(max_param_keys)
        self.max_param_elements = int(max_param_elements)
        self.max_param_bytes = int(max_param_bytes)
        self.max_metric_keys = int(max_metric_keys)
        self.max_metric_bytes = int(max_metric_bytes)
        self.max_key_length = int(max_key_length)
        self._trusted_lazy_types = (
            _default_trusted_lazy_types() if trusted_lazy_types is None else tuple(trusted_lazy_types)
        )
        self._lock = threading.RLock()
        self.reset_stats()

    def reset_stats(self):
        with self._lock:
            self._weighted_sum = {}
            self._key_weight = {}
            self._metric_sum = {}
            self._metric_weight = {}
            self._metric_bytes = {}
            self._params_type = None
            self._param_schema = None
            self._all_metrics = True
            self._accepted_count = 0

    def _validated_params(self, model: FLModel):
        if not isinstance(model.params, dict) or not model.params:
            raise ValueError("model.params must be a non-empty dictionary")
        if len(model.params) > self.max_param_keys:
            raise ValueError(f"parameter key count exceeds configured maximum {self.max_param_keys}")
        if self._param_schema is not None and set(model.params) != set(self._param_schema):
            raise ValueError("client parameter schema does not match earlier contributions in this round")

        values = {}
        schema = {}
        total_elements = 0
        total_bytes = 0
        for key, raw_value in model.params.items():
            if not isinstance(key, str) or not key or len(key) > self.max_key_length:
                raise TypeError("parameter keys must be non-empty strings")
            value = _materialize(
                raw_value,
                self._trusted_lazy_types,
                max_elements=self.max_param_elements - total_elements,
                max_bytes=self.max_param_bytes - total_bytes,
            )
            value = _normalize_numeric(value, label=f"parameter {key!r}")
            value_schema = _value_schema(value)
            if self._param_schema is not None and value_schema != self._param_schema[key]:
                raise ValueError("client parameter schema does not match earlier contributions in this round")
            total_elements, total_bytes = _add_checked_size(
                value,
                label="parameter",
                current_elements=total_elements,
                current_bytes=total_bytes,
                max_elements=self.max_param_elements,
                max_bytes=self.max_param_bytes,
            )
            _check_finite(value, label=f"parameter {key!r}")
            values[key] = value
            schema[key] = value_schema
        return values, schema

    def _validated_metrics(self, metrics):
        if metrics is None:
            return None
        if not isinstance(metrics, dict):
            raise TypeError("model.metrics must be a dictionary or None")

        if len(metrics) > self.max_metric_keys:
            raise ValueError(f"metric key count exceeds configured maximum {self.max_metric_keys}")
        validated = {}
        total_elements = 0
        total_bytes = 0
        for key, value in metrics.items():
            if not isinstance(key, str) or not key or len(key) > self.max_key_length:
                raise TypeError("metric keys must be non-empty strings")
            value = _normalize_numeric(value, label=f"metric {key!r}")
            if _value_size(value) != 1:
                raise ValueError(f"metric {key!r} must be a scalar")
            total_elements, total_bytes = _add_checked_size(
                value,
                label="metric",
                current_elements=total_elements,
                current_bytes=total_bytes,
                max_elements=self.max_metric_keys,
                max_bytes=self.max_metric_bytes,
            )
            _check_finite(value, label=f"metric {key!r}")
            validated[key] = value
        return validated

    def accept_model(self, model: FLModel):
        with self._lock:
            if self._accepted_count >= self.max_models:
                raise ValueError(f"client model count exceeds configured maximum {self.max_models}")
            if self._params_type is not None and model.params_type != self._params_type:
                raise ValueError("client params_type does not match earlier contributions in this round")

            weight = _step_weight(model, self.max_step_weight)
            params, schema = self._validated_params(model)
            metrics = self._validated_metrics(model.metrics) if self._all_metrics else None

            # Build the next accumulator state before committing it. A failure
            # in one key therefore cannot leave a partially accepted model.
            next_sum = dict(self._weighted_sum)
            next_weight = dict(self._key_weight)
            contribution_elements = 0
            contribution_bytes = 0
            accumulator_elements = 0
            accumulator_bytes = 0
            for key, value in params.items():
                _preflight_binary_size(
                    value,
                    weight,
                    label="weighted parameter contribution",
                    current_elements=contribution_elements,
                    current_bytes=contribution_bytes,
                    max_elements=self.max_param_elements,
                    max_bytes=self.max_param_bytes,
                )
                contribution = value * weight
                contribution = _normalize_numeric(contribution, label=f"weighted parameter {key!r}")
                contribution_elements, contribution_bytes = _add_checked_size(
                    contribution,
                    label="weighted parameter contribution",
                    current_elements=contribution_elements,
                    current_bytes=contribution_bytes,
                    max_elements=self.max_param_elements,
                    max_bytes=self.max_param_bytes,
                )
                _check_finite(contribution, label=f"weighted parameter {key!r}")
                if key in next_sum:
                    _preflight_binary_size(
                        next_sum[key],
                        contribution,
                        label="parameter accumulator",
                        current_elements=accumulator_elements,
                        current_bytes=accumulator_bytes,
                        max_elements=self.max_param_elements,
                        max_bytes=self.max_param_bytes,
                    )
                    combined = next_sum[key] + contribution
                else:
                    combined = contribution
                combined = _normalize_numeric(combined, label=f"parameter accumulator {key!r}")
                accumulator_elements, accumulator_bytes = _add_checked_size(
                    combined,
                    label="parameter accumulator",
                    current_elements=accumulator_elements,
                    current_bytes=accumulator_bytes,
                    max_elements=self.max_param_elements,
                    max_bytes=self.max_param_bytes,
                )
                _check_finite(combined, label=f"parameter accumulator {key!r}")
                total_weight = next_weight.get(key, 0.0) + weight
                if not math.isfinite(total_weight):
                    raise ValueError("parameter weight total is non-finite")
                next_sum[key] = combined
                next_weight[key] = total_weight

            next_metric_sum = dict(self._metric_sum)
            next_metric_weight = dict(self._metric_weight)
            next_metric_bytes = dict(self._metric_bytes)
            if model.metrics is None:
                next_metric_sum = {}
                next_metric_weight = {}
                next_metric_bytes = {}
            elif self._all_metrics and metrics is not None:
                combined_keys = set(next_metric_sum).union(metrics)
                if len(combined_keys) > self.max_metric_keys:
                    raise ValueError(f"metric key union exceeds configured maximum {self.max_metric_keys}")
                metric_contribution_elements = 0
                metric_contribution_bytes = 0
                for key, value in metrics.items():
                    old_bytes = next_metric_bytes.get(key, 0)
                    _preflight_binary_size(
                        value,
                        weight,
                        label="weighted metric contribution",
                        current_elements=metric_contribution_elements,
                        current_bytes=metric_contribution_bytes,
                        max_elements=self.max_metric_keys,
                        max_bytes=self.max_metric_bytes,
                    )
                    contribution = value * weight
                    contribution = _normalize_numeric(contribution, label=f"weighted metric {key!r}")
                    if _value_size(contribution) != 1:
                        raise ValueError(f"weighted metric {key!r} must remain a scalar")
                    metric_contribution_elements, metric_contribution_bytes = _add_checked_size(
                        contribution,
                        label="weighted metric contribution",
                        current_elements=metric_contribution_elements,
                        current_bytes=metric_contribution_bytes,
                        max_elements=self.max_metric_keys,
                        max_bytes=self.max_metric_bytes,
                    )
                    _check_finite(contribution, label=f"weighted metric {key!r}")
                    if key in next_metric_sum:
                        _preflight_binary_size(
                            next_metric_sum[key],
                            contribution,
                            label="metric accumulator",
                            current_elements=len(next_metric_sum) - 1,
                            current_bytes=sum(next_metric_bytes.values()) - old_bytes,
                            max_elements=self.max_metric_keys,
                            max_bytes=self.max_metric_bytes,
                        )
                        combined = next_metric_sum[key] + contribution
                    else:
                        combined = contribution
                    combined = _normalize_numeric(combined, label=f"metric accumulator {key!r}")
                    combined_elements, combined_bytes = _checked_size(combined, label="metric accumulator")
                    if combined_elements != 1:
                        raise ValueError(f"metric accumulator {key!r} must remain a scalar")
                    if sum(next_metric_bytes.values()) - old_bytes + combined_bytes > self.max_metric_bytes:
                        raise ValueError(f"metric accumulator exceeds configured maximum {self.max_metric_bytes} bytes")
                    _check_finite(combined, label=f"metric accumulator {key!r}")
                    total_weight = next_metric_weight.get(key, 0.0) + weight
                    if not math.isfinite(total_weight):
                        raise ValueError("metric weight total is non-finite")
                    next_metric_sum[key] = combined
                    next_metric_weight[key] = total_weight
                    next_metric_bytes[key] = combined_bytes

            self._weighted_sum = next_sum
            self._key_weight = next_weight
            self._metric_sum = next_metric_sum
            self._metric_weight = next_metric_weight
            self._metric_bytes = next_metric_bytes
            self._all_metrics = self._all_metrics and metrics is not None
            self._params_type = model.params_type if self._params_type is None else self._params_type
            self._param_schema = schema if self._param_schema is None else self._param_schema
            self._accepted_count += 1

    def aggregate_model(self) -> FLModel:
        with self._lock:
            if not self._weighted_sum:
                raise RuntimeError("no client models accepted this round")

            averaged = {}
            result_elements = 0
            result_bytes = 0
            for key in self._weighted_sum:
                _preflight_binary_size(
                    self._weighted_sum[key],
                    self._key_weight[key],
                    label="aggregated parameter result",
                    current_elements=result_elements,
                    current_bytes=result_bytes,
                    max_elements=self.max_param_elements,
                    max_bytes=self.max_param_bytes,
                )
                value = self._weighted_sum[key] / self._key_weight[key]
                value = _normalize_numeric(value, label=f"aggregated parameter {key!r}")
                result_elements, result_bytes = _add_checked_size(
                    value,
                    label="aggregated parameter result",
                    current_elements=result_elements,
                    current_bytes=result_bytes,
                    max_elements=self.max_param_elements,
                    max_bytes=self.max_param_bytes,
                )
                _check_finite(value, label=f"aggregated parameter {key!r}")
                averaged[key] = value

            averaged_metrics = None
            if self._all_metrics and self._metric_sum:
                averaged_metrics = {}
                metric_result_elements = 0
                metric_result_bytes = 0
                for key in self._metric_sum:
                    _preflight_binary_size(
                        self._metric_sum[key],
                        self._metric_weight[key],
                        label="aggregated metric result",
                        current_elements=metric_result_elements,
                        current_bytes=metric_result_bytes,
                        max_elements=self.max_metric_keys,
                        max_bytes=self.max_metric_bytes,
                    )
                    value = self._metric_sum[key] / self._metric_weight[key]
                    value = _normalize_numeric(value, label=f"aggregated metric {key!r}")
                    if _value_size(value) != 1:
                        raise ValueError(f"aggregated metric {key!r} must remain a scalar")
                    metric_result_elements, metric_result_bytes = _add_checked_size(
                        value,
                        label="aggregated metric result",
                        current_elements=metric_result_elements,
                        current_bytes=metric_result_bytes,
                        max_elements=self.max_metric_keys,
                        max_bytes=self.max_metric_bytes,
                    )
                    _check_finite(value, label=f"aggregated metric {key!r}")
                    averaged_metrics[key] = value

            result = FLModel(params=averaged, params_type=self._params_type, metrics=averaged_metrics)
            self.reset_stats()
            return result
