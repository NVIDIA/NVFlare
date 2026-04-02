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

"""Anomaly Type 4 — Unusually High Activity Events.

Inflates the debitor's 30-day activity event count well beyond normal ranges
for the account type to simulate automated transaction flooding.
"""

import numpy as np
import pandas as pd

# (low, high) ranges for anomalous event counts per account type.
_RANGES: dict[str, tuple[int, int]] = {
    "BUSINESS": (1_000_000, 5_000_001),
    "CHECKING": (525, 2_000),
    "SAVINGS": (75, 2_000),
}


def apply(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Apply type-4 anomaly to the given DataFrame rows (in-place).

    Sets ``DEBITOR_ACCOUNT_ACTIVITY_EVENTS_PAST_30D`` to an anomalously high
    value whose range depends on the debitor's account type.

    Args:
        df:   DataFrame slice containing the rows to mutate.
        seed: RNG seed for reproducibility.

    Returns:
        The mutated DataFrame (same object, modified in-place).
    """
    rng = np.random.default_rng(seed)
    acct_types = df["DEBITOR_ACCOUNT_TYPE"].to_numpy()
    result = df["DEBITOR_ACCOUNT_ACTIVITY_EVENTS_PAST_30D"].to_numpy(copy=True)

    for acct_type, (low, high) in _RANGES.items():
        mask = acct_types == acct_type
        count = int(mask.sum())
        if count > 0:
            result[mask] = rng.integers(low, high, size=count)

    df["DEBITOR_ACCOUNT_ACTIVITY_EVENTS_PAST_30D"] = result
    df["TYPE4_ANOMALY"] = True
    return df
