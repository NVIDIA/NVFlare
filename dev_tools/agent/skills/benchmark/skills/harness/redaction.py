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

"""Redact credential-like material from derived logs and reports."""

from __future__ import annotations

import os
import re
from functools import lru_cache

SENSITIVE_NAME = re.compile(r"(?:TOKEN|API_KEY|SECRET|PASSWORD|CREDENTIAL|AUTHORIZATION)", re.IGNORECASE)
LABELED_SECRET = re.compile(
    r"(?i)(\b(?:authorization|api[_ -]?key|access[_ -]?token|refresh[_ -]?token|password|secret)"
    r"\b\s*[:=]\s*[\"']?(?:bearer\s+|basic\s+)?)[^\s,;\"'}]+"
)


@lru_cache(maxsize=1)
def environment_secret_values() -> tuple[str, ...]:
    values = {value for name, value in os.environ.items() if value and len(value) >= 8 and SENSITIVE_NAME.search(name)}
    return tuple(sorted(values, key=len, reverse=True))


def redact_text(value: object) -> str:
    text = str(value or "")
    for secret in environment_secret_values():
        text = text.replace(secret, "[REDACTED]")
    return LABELED_SECRET.sub(r"\1[REDACTED]", text)
