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

"""Entry-point-based plugin registry for backends, strategies, and benchmarks.

External packages (e.g. `decent-vla-myvla`) register themselves under the
groups `decent_vla.backends`, `decent_vla.strategies`, `decent_vla.benchmarks`.
The CLI's `list` and `info` subcommands read this registry; `from_config`
factory functions look up plugins by name.

Failure mode is deliberate: a plugin that fails to import (e.g. because
its optional dep isn't installed) gets surfaced in `registry.errors` but
does NOT crash the whole CLI. `decent-vla doctor` shows the errors.
"""
from __future__ import annotations

import importlib
import importlib.metadata as ilmd
from dataclasses import dataclass, field
from typing import Any

BackendGroup = "decent_vla.backends"
StrategyGroup = "decent_vla.strategies"
BenchmarkGroup = "decent_vla.benchmarks"


@dataclass
class Registry:
    backends: dict[str, Any] = field(default_factory=dict)
    strategies: dict[str, Any] = field(default_factory=dict)
    benchmarks: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


def _load_group(group: str, registry: Registry, bucket: dict[str, Any]) -> None:
    try:
        eps = ilmd.entry_points(group=group)
    except TypeError:
        # Python 3.9 fallback (we require 3.10 but be defensive)
        eps = ilmd.entry_points().get(group, [])
    for ep in eps:
        try:
            obj = ep.load()
            bucket[ep.name] = obj
        except Exception as exc:  # noqa: BLE001 — surface any import error
            registry.errors[f"{group}:{ep.name}"] = f"{type(exc).__name__}: {exc}"


def discover() -> Registry:
    """Walk all three entry-point groups and return a populated registry."""
    reg = Registry()
    _load_group(BackendGroup, reg, reg.backends)
    _load_group(StrategyGroup, reg, reg.strategies)
    _load_group(BenchmarkGroup, reg, reg.benchmarks)
    return reg


# Module-level cache. CLI commands call discover() lazily; tests can
# monkeypatch this.
registry = discover()


def reload() -> Registry:
    """Force a fresh discovery (used by tests)."""
    global registry
    importlib.reload(ilmd)
    registry = discover()
    return registry
