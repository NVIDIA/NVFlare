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
from collections.abc import Mapping
from typing import Any


def _cleanup_obj(x: Any) -> bool:
    cleaned = False

    cleanup_fn = getattr(x, "cleanup", None)
    if callable(cleanup_fn):
        cleanup_fn()
        cleaned = True

    temp_ref = getattr(x, "_temp_ref", None)
    temp_cleanup = getattr(temp_ref, "cleanup", None)
    if callable(temp_cleanup):
        temp_cleanup()
        cleaned = True

    return cleaned


def _collect_cleanup_entities(
    x: Any,
    cleanup_objs: list[Any],
    cleanup_obj_ids: set[int],
    temp_refs: list[Any],
    temp_ref_ids: set[int],
) -> None:
    cleanup_fn = getattr(x, "cleanup", None)
    if callable(cleanup_fn):
        oid = id(x)
        if oid not in cleanup_obj_ids:
            cleanup_obj_ids.add(oid)
            cleanup_objs.append(x)

    temp_ref = getattr(x, "_temp_ref", None)
    temp_cleanup = getattr(temp_ref, "cleanup", None)
    if callable(temp_cleanup):
        tid = id(temp_ref)
        if tid not in temp_ref_ids:
            temp_ref_ids.add(tid)
            temp_refs.append(temp_ref)


def contains_lazy(data: Any) -> bool:
    """Return whether the payload contains any lazy refs (duck-typed via resolve())."""

    visited = set()

    def _contains(x: Any) -> bool:
        if x is None:
            return False

        oid = id(x)
        if oid in visited:
            return False
        visited.add(oid)

        if callable(getattr(x, "resolve", None)):
            return True

        if isinstance(x, Mapping):
            return any(_contains(v) for v in x.values())
        if isinstance(x, list):
            return any(_contains(v) for v in x)
        if isinstance(x, tuple):
            return any(_contains(v) for v in x)
        if isinstance(x, set):
            return any(_contains(v) for v in x)
        return False

    return _contains(data)


def resolve_inplace(data: Any, cleanup_resolved: bool = False) -> Any:
    """Resolve lazy refs in-place where possible and return resolved payload.

    Args:
        data: Payload that may include lazy refs.
        cleanup_resolved: If True, call cleanup hooks on each resolved lazy ref.
    """

    visited = set()
    cleanup_objs = []
    cleanup_obj_ids = set()
    temp_refs = []
    temp_ref_ids = set()

    def _resolve(x: Any) -> Any:
        if x is None:
            return x

        oid = id(x)
        if oid in visited:
            return x
        visited.add(oid)

        if callable(getattr(x, "resolve", None)):
            resolved = x.resolve()
            if cleanup_resolved:
                _collect_cleanup_entities(
                    x=x,
                    cleanup_objs=cleanup_objs,
                    cleanup_obj_ids=cleanup_obj_ids,
                    temp_refs=temp_refs,
                    temp_ref_ids=temp_ref_ids,
                )
            return resolved

        if isinstance(x, Mapping):
            for k, v in list(x.items()):
                x[k] = _resolve(v)
            return x
        if isinstance(x, list):
            for i, v in enumerate(list(x)):
                x[i] = _resolve(v)
            return x
        if isinstance(x, tuple):
            return tuple(_resolve(v) for v in x)
        if isinstance(x, set):
            resolved = {_resolve(v) for v in x}
            x.clear()
            x.update(resolved)
            return x
        return x

    try:
        return _resolve(data)
    finally:
        if cleanup_resolved:
            # Important: defer cleanup until all refs have been resolved.
            # Cleaning a shared temp dir early can break subsequent resolves.
            for obj in cleanup_objs:
                cleanup_fn = getattr(obj, "cleanup", None)
                if callable(cleanup_fn):
                    cleanup_fn()
            for temp_ref in temp_refs:
                temp_cleanup = getattr(temp_ref, "cleanup", None)
                if callable(temp_cleanup):
                    temp_cleanup()


def cleanup_inplace(data: Any) -> bool:
    """Cleanup lazy payload resources via cleanup() or _temp_ref.cleanup().

    Returns:
        bool: whether any cleanup path was invoked.
    """

    visited = set()
    cleaned = False

    def _cleanup_recursive(x: Any):
        nonlocal cleaned
        if x is None:
            return

        oid = id(x)
        if oid in visited:
            return
        visited.add(oid)

        if _cleanup_obj(x):
            cleaned = True

        if isinstance(x, Mapping):
            for v in x.values():
                _cleanup_recursive(v)
            return
        if isinstance(x, list):
            for v in x:
                _cleanup_recursive(v)
            return
        if isinstance(x, tuple):
            for v in x:
                _cleanup_recursive(v)
            return
        if isinstance(x, set):
            for v in x:
                _cleanup_recursive(v)
            return

    _cleanup_recursive(data)
    return cleaned
