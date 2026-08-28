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

"""Public NVFlare simulator environment selection."""

from __future__ import annotations

import os
from typing import Any

from nvflare.recipe import SimEnv

SIMULATOR_MODE_ENV = "AGENTICFL_SIMULATOR_MODE"
PUBLIC_SIMULATOR_MODE = "public"
IPV6_UNIX_SIMULATOR_MODE = "ipv6_unix"


def simulator_mode() -> str:
    """Return the explicit simulator mode, defaulting to NVFlare's public API."""

    mode = os.environ.get(SIMULATOR_MODE_ENV, PUBLIC_SIMULATOR_MODE).strip().lower().replace("-", "_")
    if mode in {"", "public", "nvflare"}:
        return PUBLIC_SIMULATOR_MODE
    if mode in {"ipv6_unix", "ipv6"}:
        return IPV6_UNIX_SIMULATOR_MODE
    raise ValueError(f"{SIMULATOR_MODE_ENV} must select 'public' or 'ipv6_unix', got {mode!r}")


def create_sim_env(**kwargs: Any) -> SimEnv:
    """Create public ``SimEnv`` unless the compatibility mode is explicit."""

    if simulator_mode() == PUBLIC_SIMULATOR_MODE:
        return SimEnv(**kwargs)
    try:
        from agenticfl.flare.simulator_compat import AgenticFLSimEnv, validate_private_simulator_contract
    except ImportError as exc:  # pragma: no cover - depends on the installed NVFlare revision
        raise RuntimeError(
            "AGENTICFL_SIMULATOR_MODE=ipv6_unix requires the NVFlare 2.9 simulator internals "
            "used by this research example"
        ) from exc
    validate_private_simulator_contract()
    return AgenticFLSimEnv(**kwargs)
