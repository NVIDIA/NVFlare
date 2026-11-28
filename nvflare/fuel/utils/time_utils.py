# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import re
import time
from datetime import datetime, timezone

_ISO_TIME = re.compile(
    r"([0-9]{4}-[0-9]{2}-[0-9]{2})(?:[T ]((?:[01][0-9]|2[0-3]):[0-5][0-9])(?::([0-5][0-9])(?:[.,]([0-9]+))?)?"
    r"(Z|[+-](?:[01][0-9]|2[0-3])(?::?[0-5][0-9])?)?)?"
)


def utc_time_string(value: str, *, assume_utc: bool = False) -> str:
    """Normalize a calendar date or ISO timestamp to a sortable UTC string."""
    match = _ISO_TIME.fullmatch(value) if isinstance(value, str) else None
    if not match:
        raise ValueError("expected a date or ISO timestamp with a known timezone")
    day, clock, seconds, fraction, offset = match.groups()
    if not offset and not assume_utc:
        raise ValueError("timestamp has no timezone")
    offset = "+0000" if offset in (None, "Z") else offset.replace(":", "")
    if len(offset) == 3:
        offset += "00"
    # Normalize whole seconds separately so fractional precision survives conversion.
    dt = datetime.fromisoformat(f"{day}T{clock or '00:00'}:{seconds or '00'}{offset[:3]}:{offset[3:]}")
    try:
        utc = dt.astimezone(timezone.utc).isoformat(timespec="seconds")
    except OverflowError as e:
        raise ValueError("timestamp is outside the UTC calendar range") from e
    fraction = (fraction or "").rstrip("0")
    return utc[:-6] + ("." + fraction if fraction else "") + "+00:00"


def job_start_bounds(since: str | None, until: str | None) -> dict[str, str]:
    bounds = {}
    for key, value in (("since", since), ("until", until)):
        if value is not None:
            try:
                bounds[key] = utc_time_string(value, assume_utc=True)
            except ValueError as e:
                raise ValueError(f"--{key} requires YYYY-MM-DD or a valid ISO timestamp") from e
    if len(bounds) == 2 and bounds["since"] >= bounds["until"]:
        raise ValueError("--since must be earlier than --until")
    return bounds


def time_to_string(t) -> str:
    """Convert time into a formatted string.

    Args:
        t: input time string in seconds since the Epoch

    Returns:
        formatted time string
    """
    if t is None:
        return "N/A"

    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
