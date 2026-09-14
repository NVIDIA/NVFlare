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

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from nvflare.app_common.abstract.model import ModelLearnableKey, make_model_learnable
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor


def _load_persistor_module():
    example_dir = Path(__file__).parents[5] / "examples" / "advanced" / "bionemo" / "evo2"
    previous_path = sys.path[:]
    sys.path.insert(0, str(example_dir))
    try:
        module_path = example_dir / "evo2_persistor.py"
        spec = importlib.util.spec_from_file_location("evo2_persistor_test_target", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = previous_path


def test_load_model_clones_trainable_state_to_cpu_and_rejects_backbone_keys(monkeypatch):
    module = _load_persistor_module()
    source = torch.ones(3, requires_grad=True)
    model_learnable = {
        ModelLearnableKey.WEIGHTS: {"classification_head.bias": source},
        ModelLearnableKey.META: {},
    }
    monkeypatch.setattr(PTFileModelPersistor, "load_model", lambda _self, _fl_ctx: model_learnable)
    persistor = module.CPUTrainablePTFileModelPersistor()

    loaded = persistor.load_model(None)

    tensor = loaded[ModelLearnableKey.WEIGHTS]["classification_head.bias"]
    assert tensor.device.type == "cpu"
    assert tensor.dtype == torch.float32
    assert not tensor.requires_grad
    assert tensor.data_ptr() != source.data_ptr()

    model_learnable[ModelLearnableKey.WEIGHTS] = {"decoder.layers.0.weight": torch.ones(1)}
    with pytest.raises(ValueError, match="unsupported parameter names"):
        persistor.load_model(None)

    model_learnable[ModelLearnableKey.WEIGHTS] = {"classification_head.bias": torch.ones(1, dtype=torch.bfloat16)}
    with pytest.raises(ValueError, match="only float32 tensors"):
        persistor.load_model(None)


def test_source_checkpoint_is_loaded_directly_on_cpu(tmp_path, monkeypatch):
    module = _load_persistor_module()
    checkpoint_path = tmp_path / "initial.pt"
    state = {"classification_head.bias": torch.ones(3)}
    torch.save({"model": state, "meta_props": {"backend": "mock"}}, checkpoint_path)
    original_load = torch.load
    map_locations = []

    def recording_load(*args, **kwargs):
        map_locations.append(kwargs.get("map_location"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(module.torch, "load", recording_load)
    persistor = module.CPUTrainablePTFileModelPersistor(
        source_ckpt_file_full_name=str(checkpoint_path),
        allow_numpy_conversion=False,
    )

    loaded = persistor.load_model(None)

    assert map_locations == ["cpu"]
    assert loaded[ModelLearnableKey.WEIGHTS]["classification_head.bias"].device.type == "cpu"
    assert loaded[ModelLearnableKey.META] == {"backend": "mock"}


def test_save_model_file_round_trips_float32_and_rejects_bfloat16(tmp_path):
    module = _load_persistor_module()
    state = {"classification_head.bias": torch.ones(3, dtype=torch.float32)}
    persistor = module.CPUTrainablePTFileModelPersistor(allow_numpy_conversion=False)
    persistor.persistence_manager = module.PTModelPersistenceFormatManager(
        {"model": state, "meta_props": {"exchange_dtype": "float32"}},
        allow_numpy_conversion=False,
    )
    checkpoint = tmp_path / "global.pt"

    persistor.save_model_file(str(checkpoint))

    saved = module.adapter_checkpoint.load_nvflare_checkpoint(checkpoint)
    assert saved["classification_head.bias"].dtype == torch.float32
    assert torch.equal(saved["classification_head.bias"], state["classification_head.bias"])
    assert persistor.persistence_manager.var_dict["classification_head.bias"].device.type == "cpu"

    persistor.persistence_manager.var_dict["classification_head.bias"] = state["classification_head.bias"].to(
        torch.bfloat16
    )
    with pytest.raises(ValueError, match="only float32 tensors"):
        persistor.save_model_file(str(tmp_path / "invalid.pt"))


def test_get_model_validates_retrieved_exchange_tensors(monkeypatch):
    module = _load_persistor_module()
    persistor = module.CPUTrainablePTFileModelPersistor(allow_numpy_conversion=False)
    retrieved = make_model_learnable(
        weights={"classification_head.bias": torch.ones(3, dtype=torch.bfloat16)},
        meta_props={"exchange_dtype": "float32"},
    )
    monkeypatch.setattr(PTFileModelPersistor, "get_model", lambda _self, _model_file, _fl_ctx: retrieved)

    with pytest.raises(ValueError, match="only float32 tensors"):
        persistor.get_model("FL_global_model.pt", None)
