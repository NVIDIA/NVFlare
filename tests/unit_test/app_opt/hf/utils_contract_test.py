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

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from nvflare.app_common.abstract.fl_model import FLModel  # noqa: E402
from nvflare.client.config import ExchangeFormat  # noqa: E402

from ._helpers import (  # noqa: E402
    import_hf_utils_module,
    install_fake_peft,
    make_fake_trainer_class,
    make_training_args,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 1)


def _fresh_utils(monkeypatch):
    utils = import_hf_utils_module(monkeypatch)
    transformers = pytest.importorskip("transformers")
    trainer_cls = make_fake_trainer_class(transformers)
    return utils, trainer_cls


def _make_trainer(trainer_cls, tmp_path, model=None, **arg_overrides):
    return trainer_cls(model or TinyModel(), make_training_args(tmp_path, **arg_overrides))


def _clone_state_dict(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def test_load_params_strips_server_key_prefix_before_strict_model_load(monkeypatch, tmp_path):
    utils, trainer_cls = _fresh_utils(monkeypatch)
    trainer = _make_trainer(trainer_cls, tmp_path)
    target_weight = torch.full_like(trainer.model.fc.weight, 3.0)
    target_bias = torch.full_like(trainer.model.fc.bias, -2.0)

    utils.load_params(
        trainer,
        FLModel(params={"model.fc.weight": target_weight, "model.fc.bias": target_bias}),
        params_scope="model",
        strict=True,
        server_key_prefix="model.",
    )

    assert torch.equal(trainer.model.fc.weight, target_weight)
    assert torch.equal(trainer.model.fc.bias, target_bias)


def test_load_params_reports_prefix_hint_when_no_keys_match(monkeypatch, tmp_path):
    utils, trainer_cls = _fresh_utils(monkeypatch)
    trainer = _make_trainer(trainer_cls, tmp_path)
    params = {
        "model.fc.weight": torch.ones_like(trainer.model.fc.weight),
        "model.fc.bias": torch.zeros_like(trainer.model.fc.bias),
    }

    with pytest.raises(RuntimeError, match=r"None of .* matched|stripping common prefix 'model\\.'"):
        utils.load_params(trainer, FLModel(params=params), params_scope="model", strict=True, server_key_prefix=None)


def test_load_params_rejects_shape_mismatch(monkeypatch, tmp_path):
    utils, trainer_cls = _fresh_utils(monkeypatch)
    trainer = _make_trainer(trainer_cls, tmp_path)

    with pytest.raises(RuntimeError, match=r"fc.weight.*expected.*got|shape mismatch"):
        utils.load_params(
            trainer,
            FLModel(params={"fc.weight": torch.ones(3, 2)}),
            params_scope="model",
            strict=False,
            server_key_prefix=None,
        )


def test_load_params_filters_unexpected_keys_only_in_non_strict_mode(monkeypatch, tmp_path, caplog):
    utils, trainer_cls = _fresh_utils(monkeypatch)
    trainer = _make_trainer(trainer_cls, tmp_path)
    original_state = _clone_state_dict(trainer.model)
    params = {
        "fc.weight": torch.full_like(trainer.model.fc.weight, 4.0),
        "model.fc.bias": torch.full_like(trainer.model.fc.bias, 9.0),
    }

    with pytest.raises(RuntimeError, match="unexpected|Rejecting"):
        utils.load_params(trainer, FLModel(params=params), params_scope="model", strict=True, server_key_prefix=None)

    trainer.model.load_state_dict(original_state)
    with caplog.at_level("WARNING"):
        utils.load_params(trainer, FLModel(params=params), params_scope="model", strict=False, server_key_prefix=None)

    assert "Ignoring" in caplog.text or "unexpected" in caplog.text
    assert torch.equal(trainer.model.fc.weight, params["fc.weight"])
    assert torch.equal(trainer.model.fc.bias, original_state["fc.bias"])


def test_apply_server_key_prefix_is_symmetric(monkeypatch):
    utils = import_hf_utils_module(monkeypatch)
    params = {"fc.weight": torch.ones(1), "fc.bias": torch.zeros(1)}

    prefixed = utils.apply_server_key_prefix(params, "model.")

    assert set(prefixed) == {"model.fc.weight", "model.fc.bias"}
    assert prefixed["model.fc.weight"] is params["fc.weight"]
    assert utils.strip_server_key_prefix(prefixed, "model.") == params


def test_prepare_out_params_casts_half_tensors_only_for_numpy_exchange(monkeypatch):
    utils = import_hf_utils_module(monkeypatch)
    source = {
        "bf16": torch.tensor([1.0], dtype=torch.bfloat16),
        "fp16": torch.tensor([2.0], dtype=torch.float16),
        "fp32": torch.tensor([3.0], dtype=torch.float32),
    }

    numpy_out = utils.prepare_out_params(source, ExchangeFormat.NUMPY)

    assert isinstance(numpy_out["bf16"], np.ndarray)
    assert isinstance(numpy_out["fp16"], np.ndarray)
    assert numpy_out["bf16"].dtype == np.float32
    assert numpy_out["fp16"].dtype == np.float32
    assert numpy_out["fp32"].dtype == np.float32

    tensor_out = utils.prepare_out_params(source, ExchangeFormat.PYTORCH)
    assert tensor_out["bf16"].dtype == torch.bfloat16
    assert tensor_out["fp16"].dtype == torch.float16
    assert tensor_out["fp32"].dtype == torch.float32


def test_resolve_params_scope_requires_peft_for_adapter_scope(monkeypatch, tmp_path):
    install_fake_peft(monkeypatch)
    utils, trainer_cls = _fresh_utils(monkeypatch)
    trainer = _make_trainer(trainer_cls, tmp_path)

    assert utils.resolve_params_scope(trainer, "auto") == "model"
    assert utils.resolve_params_scope(trainer, "model") == "model"
    with pytest.raises((RuntimeError, ValueError), match="adapter|PEFT|PeftModel"):
        utils.resolve_params_scope(trainer, "adapter")


def test_adapter_scope_extract_and_load_uses_peft_adapter_keyspace(monkeypatch, tmp_path):
    peft = install_fake_peft(monkeypatch)
    utils, trainer_cls = _fresh_utils(monkeypatch)

    class TinyPeftModel(TinyModel, peft.PeftModel):
        def __init__(self):
            super().__init__()
            self.adapter_weight = torch.nn.Parameter(torch.tensor([0.5, -0.5]))

        def get_adapter_state_dict(self):
            return {"adapter.weight": self.adapter_weight.detach().clone()}

        def load_adapter_state_dict(self, state_dict):
            self.adapter_weight.data.copy_(state_dict["adapter.weight"])
            self.loaded_adapter_keys = tuple(state_dict)
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    trainer = _make_trainer(trainer_cls, tmp_path, model=TinyPeftModel())

    assert utils.resolve_params_scope(trainer, "auto") == "adapter"
    assert set(utils.get_reference_state_dict(trainer, "adapter")) == {"adapter.weight"}
    assert set(utils.extract_params(trainer, "adapter")) == {"adapter.weight"}

    incoming_adapter = torch.tensor([9.0, 8.0])
    utils.load_params(
        trainer,
        FLModel(params={"adapter.weight": incoming_adapter}),
        params_scope="adapter",
        strict=True,
        server_key_prefix=None,
    )

    assert trainer.model.loaded_adapter_keys == ("adapter.weight",)
    assert torch.equal(trainer.model.adapter_weight, incoming_adapter)
