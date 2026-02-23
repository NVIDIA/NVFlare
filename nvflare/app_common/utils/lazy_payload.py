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

"""Helpers for lazy payload cleanup used by tensor disk offloading.

When tensor disk offload is enabled, payload trees can include lazy reference objects
instead of in-memory tensors. These lazy refs commonly carry a `_temp_ref` attribute:

- `_temp_ref` is a shared cleanup owner for temporary on-disk tensor data.
- Each lazy ref holds `_temp_ref` to keep temp files alive while refs are still in use.
- `_temp_ref.cleanup()` removes those temporary resources.

`cleanup_inplace()` traverses nested payload containers and calls `cleanup()` either on
the object itself or on its `_temp_ref` when present.
"""

from collections.abc import Mapping
from typing import Any


def _iter_children(x: Any):
    if isinstance(x, Mapping):
        return x.values()
    if isinstance(x, (list, tuple, set)):
        return x
    return ()


def _cleanup_obj(x: Any) -> bool:
    cleaned = False

    for target in (x, getattr(x, "_temp_ref", None)):
        cleanup_fn = getattr(target, "cleanup", None)
        if callable(cleanup_fn):
            cleanup_fn()
            cleaned = True

    return cleaned


def cleanup_inplace(data: Any) -> bool:
    """Cleanup lazy payload resources via cleanup() or _temp_ref.cleanup().

    Returns:
        bool: whether any cleanup path was invoked.
    """

    visited = set()
    cleaned = False
    stack = [data]

    while stack:
        x = stack.pop()
        if x is None:
            continue

        oid = id(x)
        if oid in visited:
            continue
        visited.add(oid)

        if _cleanup_obj(x):
            cleaned = True

        stack.extend(_iter_children(x))
    return cleaned
