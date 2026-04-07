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
