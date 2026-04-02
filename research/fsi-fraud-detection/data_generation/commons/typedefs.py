"""Shared type aliases used across the data-generation package.

These describe the scalar and collection types that individual cell values
and multi-column outputs can hold.
"""

from typing import Sequence

type ColumnValueType = str | int | float | bool | None
"""Type of a single cell value in a generated dataset."""

type MultiColumnValueType = (list[ColumnValueType] | tuple[ColumnValueType, ...] | Sequence[ColumnValueType])
"""A sequence of cell values, typically representing one multi-column output."""
