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

"""Parameter selection for local training and FLARE model exchange.

backend.trainable_params overrides policy freeze flags using fnmatch patterns.
backend.shared_params selects which trainable parameters are federated;
the remainder stay personal to each client. None preserves policy defaults
for training and shares all trainable parameters.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from fnmatch import fnmatchcase
from pathlib import Path

import torch


def resolve_shared_names(names: Iterable[str], patterns: Sequence[str] | None) -> set[str]:
    """Return the subset of ``names`` matching any pattern.

    ``patterns=None`` → all names are shared. A non-empty pattern list that
    matches nothing raises: a typo'd pattern would otherwise produce an FL run
    that silently never federates anything.
    """
    names = set(names)
    if patterns is None:
        return names
    if not patterns:
        raise ValueError(
            "backend.shared_params is an empty list — that is neither 'share "
            "everything' (omit the key) nor a valid subset. Fix the config."
        )
    shared = {n for n in names if any(fnmatchcase(n, p) for p in patterns)}
    if not shared:
        raise ValueError(
            f"backend.shared_params {list(patterns)!r} match no trainable "
            f"parameter names (e.g. {sorted(names)[:3]}); the federated scope "
            "would be empty. Fix the patterns."
        )
    return shared


def apply_trainable_patterns(model: torch.nn.Module, patterns: Sequence[str] | None) -> None:
    """Axis 1: set ``requires_grad`` from ``backend.trainable_params``.

    Matching params become trainable, the complement is frozen. ``None`` is a
    no-op (the policy's native flags decide). Patterns matching nothing raise:
    a typo would otherwise freeze the whole model and train nothing.
    """
    if patterns is None:
        return
    if not patterns:
        raise ValueError(
            "backend.trainable_params is an empty list — it would silently "
            "freeze the entire model. Omit the key or list real patterns."
        )
    # Two passes: validate BEFORE mutating, so a bad pattern list leaves the
    # model untouched instead of fully frozen.
    hits = {name: any(fnmatchcase(name, pat) for pat in patterns) for name, _ in model.named_parameters()}
    if not any(hits.values()):
        raise ValueError(
            f"backend.trainable_params {list(patterns)!r} match no parameter "
            "names; the model would have nothing to train. Fix the patterns."
        )
    for name, p in model.named_parameters():
        p.requires_grad_(hits[name])


def trainable_state_keys(model: torch.nn.Module) -> set[str]:
    """Return trainable state-dict keys without cloning model tensors."""
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k for k in model.state_dict() if k in names or any(k.startswith(n + ".") for n in names)}


def partial_load_warning(trainable: Iterable[str], loaded: Iterable[str], source: str) -> str | None:
    """Describe trainable tensors missing from a checkpoint, or return None.

    Server checkpoints contain only shared tensors. Standalone evaluation or
    export also needs client state when training uses personal parameters.
    """
    trainable = set(trainable)
    untouched = sorted(trainable - set(loaded))
    if not untouched:
        return None
    return (
        f"PARTIAL load from {source}: {len(untouched)} of {len(trainable)} trainable "
        f"tensors are absent from the checkpoint and keep their PRETRAINED values "
        f"(e.g. {untouched[:3]}). Expected mid-resume, where the client's personal "
        f"state is applied next. For an EXPORT or an EVAL it means the resulting "
        f"policy is NOT a deployed model — a partial-federation server checkpoint "
        f"holds only the shared subset, and the personal half has to be overlaid "
        f"from a client state (eval.personalized)."
    )


def trainable_state(model: torch.nn.Module, shared_patterns, scope: str = "shared") -> dict[str, torch.Tensor]:
    """Clone the trainable tensors selected for FL exchange or local persistence."""
    keys = trainable_state_keys(model)
    state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in keys}
    if scope == "all":
        return state
    shared = resolve_shared_names(state, shared_patterns)
    if scope == "personal":
        return {k: v for k, v in state.items() if k not in shared}
    if scope != "shared":
        raise ValueError(f"Unknown parameter scope: {scope!r}")
    return {k: v for k, v in state.items() if k in shared}


def set_model_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load an FL tensor subset with the model's dtype and device."""
    own = model.state_dict()
    own.update({k: v.to(device=own[k].device, dtype=own[k].dtype) for k, v in state.items() if k in own})
    model.load_state_dict(own, strict=False)


def load_model_checkpoint(model: torch.nn.Module, path: str | Path, device, label: str) -> None:
    """Load a native server or policy checkpoint, rejecting incompatible keys."""
    path = Path(path)
    blob = torch.load(path, map_location=device, weights_only=False)
    state = blob
    if isinstance(blob, dict):
        if isinstance(blob.get("state_dict"), dict):
            state = blob["state_dict"]
        elif isinstance(blob.get("state"), dict):
            state = blob["state"]
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint {path} does not contain a tensor state")
    own = set(model.state_dict())
    if not own.intersection(state):
        raise ValueError(f"checkpoint {path} shares no keys with the {label} state_dict")
    absent = set(state) - own
    if absent:
        raise ValueError(
            f"checkpoint {path} has {len(absent)} key(s) the {label} cannot receive "
            f"(e.g. {sorted(absent)[:3]}); use the matching model configuration"
        )
    warning = partial_load_warning(trainable_state_keys(model), state, str(path))
    if warning:
        import logging

        logging.getLogger(__name__).warning("%s", warning)
    set_model_state(model, state)
