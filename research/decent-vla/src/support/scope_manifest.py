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

"""scope_manifest.json — provenance for the two scope axes of an FL run.

Written at run start next to ``partition_manifest.json``. Records the
configured patterns (``backend.trainable_params`` / ``backend.shared_params``),
the RESOLVED tensor names for the shared and personal scopes, and the
counts/bytes that price the federated payload (parameter and uplink
numbers come straight from here instead of being reverse-engineered from
checkpoints).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.support import wire


def write_scope_manifest(out_dir: str | Path, backend_cfg: Any, fl_cfg: Any, backend: Any) -> Path:
    """Resolve both scope axes against a built backend and write the manifest."""
    shared = backend.get_state("shared")
    personal = backend.get_state("personal")
    shared_numel = sum(int(v.numel()) for v in shared.values())
    personal_numel = sum(int(v.numel()) for v in personal.values())
    wire_dtype = getattr(fl_cfg, "wire_dtype", "fp32")
    # Uplink bytes per element: the delta direction of a scalar or per-direction spec.
    _, delta_bytes = wire.nbytes(wire_dtype)
    blob = {
        "trainable_params": backend_cfg.trainable_params,
        "shared_params": backend_cfg.shared_params,
        "wire_dtype": wire_dtype,
        "shared_names": sorted(shared),
        "personal_names": sorted(personal),
        "counts": {
            "shared_tensors": len(shared),
            "personal_tensors": len(personal),
            "trainable_tensors": len(shared) + len(personal),
            "shared_numel": shared_numel,
            "personal_numel": personal_numel,
            "trainable_numel": shared_numel + personal_numel,
        },
        "bytes": {
            "shared_fp32": 4 * shared_numel,
            "shared_wire": delta_bytes * shared_numel,
        },
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "scope_manifest.json"
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(blob, f, indent=2)
    tmp.rename(path)
    return path
