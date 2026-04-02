"""Protocol defining the callable signature for attribute data providers.

Every function that generates column values for the synthetic payment dataset
must conform to ``AttributeDataProviderProtocol``.  This ensures a uniform
interface whether the function produces a single column (``pd.Series``) or
multiple columns (``pd.DataFrame``).
"""

from typing import Protocol

import pandas as pd
from data_generation.synthetic_data_provider.synthetic_data_provider import SyntheticDataProvider


class AttributeDataProviderProtocol[T: SyntheticDataProvider](Protocol):
    """Protocol for callables that generate column values for a batch of rows.

    Implementations receive the active data provider, the current partially-built
    DataFrame, the names of columns this attribute depends on, and return a
    Series (single attribute) or DataFrame (attribute group).

    The caller sets up `df` with the correct index so that `len(df)` gives the
    number of rows to generate.
    """

    def __call__(
        self,
        provider: T,
        df: pd.DataFrame,
        dependent_columns: list[str] | None = None,
        **kwargs,
    ) -> pd.Series | pd.DataFrame:
        ...
