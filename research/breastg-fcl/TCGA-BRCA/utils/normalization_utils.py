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

"""Fit preprocessing on training rows and reuse the frozen statistics."""

import numpy as np


def fit_standardizer(values):
    """Return per-feature population statistics from an explicit training matrix."""
    training = np.asarray(values, dtype=np.float64)
    if training.ndim != 2 or not all(training.shape):
        raise ValueError("Standardization requires a nonempty 2D training matrix")
    if not np.isfinite(training).all():
        raise ValueError("Standardization training values must be finite")
    mean = training.mean(axis=0)
    scale = training.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return {"mean": mean, "scale": scale, "n_samples_seen": int(training.shape[0])}


def apply_standardizer(values, statistics):
    """Transform rows without fitting or updating any normalization parameters."""
    matrix = np.asarray(values, dtype=np.float64)
    mean = np.asarray(statistics["mean"], dtype=np.float64)
    scale = np.asarray(statistics["scale"], dtype=np.float64)
    if matrix.ndim != 2 or mean.ndim != 1 or scale.shape != mean.shape or matrix.shape[1] != len(mean):
        raise ValueError("Feature dimensions do not match the saved standardization statistics")
    if not np.isfinite(matrix).all() or not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Standardization values and statistics must be finite")
    if (scale <= 0).any():
        raise ValueError("Standardization scales must be positive")
    transformed = ((matrix - mean) / scale).astype(np.float32)
    if not np.isfinite(transformed).all():
        raise ValueError("Standardized values exceed the float32 range")
    return transformed
