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

"""Small stateless helpers shared by more than one probe and by the accumulator.

Kept out of any single probe module so cpu/memory and GPU probing do not need
to import from each other, and out of accumulator.py so the probes do not need
to import the accounting core.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation, localcontext
from typing import Optional

_MODEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._()+/@-]{0,127}$")
_ARCH_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")


def _normalize_model(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.encode("ascii", errors="ignore").decode("ascii")
    value = re.sub(r"[^A-Za-z0-9 ._()+/@-]+", " ", value)
    value = " ".join(value.split())[:128].rstrip()
    if value and _MODEL_PATTERN.fullmatch(value):
        return value
    return None


def _normalize_architecture(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    return value if _ARCH_PATTERN.fullmatch(value) else None


def _decimal(value: object, label: str, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a decimal value")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{label} must be a decimal value") from exc
    if not result.is_finite() or result < 0 or (positive and result <= 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be a finite {qualifier} decimal value")
    return result


def _decimal_text(value: Decimal) -> str:
    with localcontext() as context:
        context.prec = 100
        if value.as_tuple().exponent < -9:
            value = value.quantize(Decimal("0.000000001"), rounding=ROUND_HALF_EVEN)
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"
