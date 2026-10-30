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

"""The two orthogonal scope axes of a federated run (pattern-based).

Axis 1 — local training scope (`backend.trainable_params`):
    which parameters are trainable at all. `apply_trainable_patterns` is the
    generic, backend-agnostic form of the per-policy freeze flags: matching
    params get ``requires_grad=True``, the complement is frozen. Applied
    after model build (and after LoRA/native tune flags), so it is the final
    word on what the optimizer sees.

Axis 2 — federated scope (`backend.shared_params`):
    which TRAINABLE parameters are federated. `resolve_shared_names` splits
    the trainable names into the "shared" subset (delta, aggregation,
    broadcast) and its client-local "personal" complement (personalization).

Patterns are fnmatch globs over parameter names
(e.g. ``"model.paligemma_with_expert.paligemma.*"``). Either axis ``None``
preserves historical behavior. Core-safe: stdlib + torch only.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from fnmatch import fnmatchcase

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
    """State-dict keys belonging to trainable parameters — NAMES ONLY.

    The same filter the adapters' ``get_state`` applies, but without
    materializing anything. ``get_state("all")`` detaches and *clones* every
    trainable tensor (~12.6 GB at GR00T's 3.14B), which is the right thing when
    you want the weights and pure waste when you only want the key set — as the
    partial-load coverage check does. Sub-keys of a trainable module are
    included (``head.weight`` under a trainable ``head``); buffers are not,
    since they carry no ``requires_grad``.
    """
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k for k in model.state_dict() if k in names or any(k.startswith(n + ".") for n in names)}


def partial_load_warning(trainable: Iterable[str], loaded: Iterable[str], source: str) -> str | None:
    """Message describing an incomplete checkpoint load, or ``None`` if complete.

    A partial load is LEGAL — training resume overlays the client's personal
    half right afterwards — but it must never be silent. Under axis 2 a server
    checkpoint holds only the shared subset, so materializing it as a
    standalone policy leaves every personal parameter at its PRETRAINED value.
    A backbone-share server checkpoint exported that way scored 0.0% on all
    four LIBERO suites over 400 episodes with nothing in the log to explain it.

    Coverage is judged against the TRAINABLE set, never the shared scope: under
    ``backend.shared_params`` the checkpoint matches the shared scope exactly,
    which would call it complete. Extra keys in ``loaded`` are ignored here —
    the adapters reject those separately as an arm mismatch.
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
