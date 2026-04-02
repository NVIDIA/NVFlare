"""Type aliases for RNG sample return values.

``SampleValueType`` is the scalar returned when ``size=1`` (the default).
``VectorSampleValueType`` is the array-like returned when ``size > 1``.
"""

import numpy as np

type SampleValueType = str | int | float | bool | None
"""Scalar value produced by a single RNG draw."""

type VectorSampleValueType = (
    list[SampleValueType] | tuple[SampleValueType, ...] | np.ndarray
)
"""Array-like collection produced by a vectorised RNG draw (``size > 1``)."""
