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

"""Where the GR00T adapter attaches LoRA, and what it refuses to do quietly.

`_apply_lora` was written for exactly one arm — LoRA on the Qwen3-VL backbone
language model, action head left fully trainable — and hardcoded that path. A
second arm (backbone frozen, LoRA on the *action head*) needs the same code to
point somewhere else, so the attachment point becomes config.

These tests use small stand-in modules rather than the 3B checkpoint: the logic
under test is path resolution, re-binding, and target validation, none of which
depends on the real weights. `peft` is real here — its matching semantics
(suffix matching, including dotted suffixes like ``to_out.0``) are load-bearing
and a fake would only assert that the fake works.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from torch import nn  # noqa: E402

from decent_vla.adapters.groot.backend import GrootBackend  # noqa: E402
from decent_vla.configs.schemas import BackendConfig  # noqa: E402
from decent_vla.core.lora import LoRAConfig  # noqa: E402


class _Attn(nn.Module):
    """Qwen3-VL-shaped: q_proj/k_proj/v_proj leaves."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)


class _DiTBlock(nn.Module):
    """diffusers-shaped: to_q/to_k/to_v and a to_out ModuleList, like the GR00T
    action head's DiT (gr00t/model/modules/dit.py uses diffusers' Attention)."""

    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(8, 8)
        self.to_k = nn.Linear(8, 8)
        self.to_v = nn.Linear(8, 8)
        self.to_out = nn.ModuleList([nn.Linear(8, 8), nn.Dropout(0.0)])


def _fake_groot() -> nn.Module:
    """model.backbone.model.language_model + model.action_head, the two paths
    that matter, with the real nesting depth."""
    model = nn.Module()
    lm = nn.Module()
    lm.layers = nn.ModuleList([_Attn()])
    inner = nn.Module()
    inner.language_model = lm
    backbone = nn.Module()
    backbone.model = inner
    model.backbone = backbone
    head = nn.Module()
    head.transformer_blocks = nn.ModuleList([_DiTBlock()])
    model.action_head = head
    return model


def _backend() -> GrootBackend:
    return GrootBackend(BackendConfig(name="groot"), torch.device("cpu"))


def _adapter_names(module) -> list[str]:
    return [n for n, _ in module.named_modules() if n.endswith("lora_A.default")]


def test_default_attach_point_is_still_the_backbone_language_model():
    """No attach_to => the historical arm, byte-for-byte in behaviour."""
    model = _fake_groot()
    _backend()._apply_lora(model, LoRAConfig(r=4, alpha=8, target_modules=["q_proj", "v_proj"]))

    wrapped = model.backbone.model.language_model
    assert type(wrapped).__name__ == "PeftModel"
    assert len(_adapter_names(wrapped)) == 2
    # the action head must not have been touched
    assert _adapter_names(model.action_head) == []


def test_attach_to_points_lora_at_the_action_head_instead():
    model = _fake_groot()
    _backend()._apply_lora(
        model,
        LoRAConfig(r=4, alpha=8, attach_to="action_head", target_modules=["to_q", "to_out.0"]),
    )

    assert type(model.action_head).__name__ == "PeftModel"
    assert len(_adapter_names(model.action_head)) == 2, _adapter_names(model.action_head)
    # and the backbone LM is left exactly as it was
    assert type(model.backbone.model.language_model).__name__ != "PeftModel"


def test_attach_to_accepts_a_deep_dotted_path():
    model = _fake_groot()
    _backend()._apply_lora(
        model,
        LoRAConfig(r=4, alpha=8, attach_to="backbone.model.language_model",
                   target_modules=["q_proj"]),
    )
    assert type(model.backbone.model.language_model).__name__ == "PeftModel"


def test_unresolvable_attach_path_names_the_path_it_could_not_resolve():
    model = _fake_groot()
    with pytest.raises(AttributeError, match="action_hed"):
        _backend()._apply_lora(model, LoRAConfig(r=4, attach_to="action_hed"))


def test_a_target_module_matching_nothing_is_an_error_not_a_smaller_lora():
    """PEFT raises only when EVERY target misses. A single typo among several
    silently shrinks the adapted set -- which would silently shrink the
    parameter count this arm reports."""
    model = _fake_groot()
    with pytest.raises(ValueError, match="typo_proj"):
        _backend()._apply_lora(
            model,
            LoRAConfig(r=4, attach_to="action_head", target_modules=["to_q", "typo_proj"]),
        )
