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

"""NumPy model arithmetic for the standalone research smoke example."""

import numpy as np


class NumpyModelAdapter:
    """Average floating arrays; preserve non-floating buffers from the current model.

    Full states must have identical keys, shapes and dtypes. Deltas contain only
    floating entries. All methods return owned arrays and leave inputs untouched.
    """

    def snapshot(self, model):
        return {name: np.array(value, copy=True) for name, value in model.items()}

    def validate(self, model, reference):
        if model.keys() != reference.keys():
            raise ValueError("model keys do not match reference")
        for name, value in model.items():
            base = reference[name]
            if not isinstance(value, np.ndarray) or value.shape != base.shape or value.dtype != base.dtype:
                raise ValueError(f"incompatible model entry: {name}")
            if not np.isfinite(value).all():
                raise ValueError(f"non-finite model entry: {name}")

    def difference(self, model, base):
        return {
            name: np.asarray(value - base[name])
            for name, value in model.items()
            if np.issubdtype(value.dtype, np.floating)
        }

    def prepare_for_aggregation(self, delta):
        """Widen float16 deltas before weighting and running accumulation."""
        return {name: value.astype(np.float32) if value.dtype == np.float16 else value for name, value in delta.items()}

    def apply(self, model, delta, scale):
        result = self.snapshot(model)
        for name, value in delta.items():
            result[name] = np.asarray(result[name] + scale * value, dtype=result[name].dtype)
        self.validate(result, model)
        return result
