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

"""Small filesystem helpers for publishing runtime artifacts safely."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


def safe_path_slug(value: str, *, fallback: str, max_length: int = 96) -> str:
    """Return a bounded readable path component with a stable collision suffix."""

    if max_length < 18:
        raise ValueError("max_length must be at least 18")
    slug = re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", value)).strip("_") or fallback
    if len(slug) <= max_length:
        return slug
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    prefix = slug[: max_length - len(digest) - 1].rstrip("_") or fallback[: max_length - len(digest) - 1]
    return f"{prefix}_{digest}"


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Publish one JSON object without exposing a partially written target."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
