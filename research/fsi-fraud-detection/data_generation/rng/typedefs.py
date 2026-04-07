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

"""Type aliases for RNG sample return values.

``SampleValueType`` is the scalar returned when ``size=1`` (the default).
``VectorSampleValueType`` is the array-like returned when ``size > 1``.
"""

import numpy as np

type SampleValueType = str | int | float | bool | None
"""Scalar value produced by a single RNG draw."""

type VectorSampleValueType = (list[SampleValueType] | tuple[SampleValueType, ...] | np.ndarray)
"""Array-like collection produced by a vectorised RNG draw (``size > 1``)."""
