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

import sys
from types import SimpleNamespace

import pytest
import torch

from tests.unit_test.examples.fedcore_test_utils import fedcore_import_context


def test_lora_loader_rejects_partial_adapter_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "partial.pt"
    torch.save({"model": {"model.lora_a": torch.ones(1)}}, checkpoint)
    model = SimpleNamespace(load_state_dict=lambda *args, **kwargs: pytest.fail("partial adapter was loaded"))
    helpers = SimpleNamespace(
        DEFAULT_LORA_TARGET_MODULES=["q_proj"],
        map_adapter_state_dict_for_peft_model=lambda _model, state: ({"mapped.lora_a": state["lora_a"]}, []),
        get_expected_peft_adapter_keys=lambda _model: {"mapped.lora_a", "mapped.lora_b"},
    )
    fake_peft = SimpleNamespace(
        LoraConfig=lambda **kwargs: kwargs,
        TaskType=SimpleNamespace(CAUSAL_LM="CAUSAL_LM"),
        get_peft_model=lambda base, _config: base,
    )
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    with fedcore_import_context():
        from src.qwen_backend import _load_nvflare_lora_checkpoint

        with pytest.raises(ValueError, match="missing 1 required adapter weights"):
            _load_nvflare_lora_checkpoint(model, checkpoint, helpers, lora_r=4, lora_alpha=8)
