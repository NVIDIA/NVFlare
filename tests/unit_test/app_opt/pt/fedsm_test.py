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

import pytest

torch = pytest.importorskip("torch")

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.abstract.model import ModelLearnableKey
from nvflare.app_opt.pt.fedsm import FedSM, FedSMConstants, FedSMModelAggregator, PTFedSMHelper, PTFedSMModelPersistor


def _state(value):
    return {"weight": torch.tensor([value], dtype=torch.float32)}


def _result(client, global_diff, personal, selector_diff, steps=1):
    return FLModel(
        params={
            FedSMConstants.GLOBAL_MODEL: _state(global_diff),
            FedSMConstants.PERSONAL_MODEL: _state(personal),
            FedSMConstants.SELECTOR_MODEL: _state(selector_diff),
        },
        params_type=ParamsType.FULL,
        meta={"client_name": client, FLMetaKey.NUM_STEPS_CURRENT_ROUND: steps},
    )


def test_fedsm_aggregates_fedavg_and_softpull_models():
    aggregator = FedSMModelAggregator(soft_pull_lambda=0.75)
    aggregator.accept_model(_result("site-1", global_diff=2.0, personal=0.0, selector_diff=4.0))
    aggregator.accept_model(_result("site-2", global_diff=4.0, personal=10.0, selector_diff=8.0))

    result = aggregator.aggregate_model()

    assert result.params[FedSMConstants.GLOBAL_MODEL]["weight"].item() == pytest.approx(3.0)
    assert result.params[FedSMConstants.SELECTOR_MODEL]["weight"].item() == pytest.approx(6.0)
    personal = result.params[FedSMConstants.PERSONAL_MODELS]
    assert personal["site-1"]["weight"].item() == pytest.approx(2.5)
    assert personal["site-2"]["weight"].item() == pytest.approx(7.5)


def test_fedsm_uses_num_steps_for_global_and_selector_updates():
    aggregator = FedSMModelAggregator()
    aggregator.accept_model(_result("site-1", 0.0, 0.0, 0.0, steps=1))
    aggregator.accept_model(_result("site-2", 8.0, 10.0, 4.0, steps=3))

    result = aggregator.aggregate_model()

    assert result.params[FedSMConstants.GLOBAL_MODEL]["weight"].item() == pytest.approx(6.0)
    assert result.params[FedSMConstants.SELECTOR_MODEL]["weight"].item() == pytest.approx(3.0)


def test_fedsm_updates_mixed_semantics_bundle():
    bundle = {
        FedSMConstants.GLOBAL_MODEL: _state(10.0),
        FedSMConstants.PERSONAL_MODELS: {"site-1": _state(1.0), "site-2": _state(2.0)},
        FedSMConstants.SELECTOR_MODEL: _state(20.0),
        FedSMConstants.SELECTOR_OPTIMIZER: {},
    }
    update = {
        FedSMConstants.GLOBAL_MODEL: _state(2.0),
        FedSMConstants.PERSONAL_MODELS: {"site-1": _state(3.0), "site-2": _state(4.0)},
        FedSMConstants.SELECTOR_MODEL: _state(-5.0),
    }

    updated = FedSM._update_bundle(bundle, update)

    assert updated[FedSMConstants.GLOBAL_MODEL]["weight"].item() == pytest.approx(12.0)
    assert updated[FedSMConstants.SELECTOR_MODEL]["weight"].item() == pytest.approx(15.0)
    assert updated[FedSMConstants.PERSONAL_MODELS]["site-1"]["weight"].item() == pytest.approx(3.0)


def test_fedsm_requires_complete_result_bundle():
    aggregator = FedSMModelAggregator()
    result = FLModel(
        params={FedSMConstants.GLOBAL_MODEL: _state(1.0)},
        params_type=ParamsType.FULL,
        meta={"client_name": "site-1"},
    )

    with pytest.raises(ValueError, match="missing bundle entries"):
        aggregator.accept_model(result)


def test_fedsm_helper_loads_bundle_and_returns_mixed_update_types():
    global_model = torch.nn.Linear(1, 1, bias=False)
    personal_model = torch.nn.Linear(1, 1, bias=False)
    selector_model = torch.nn.Linear(1, 1, bias=False)
    helper = PTFedSMHelper(global_model, personal_model, selector_model)
    incoming = FLModel(
        params={
            FedSMConstants.GLOBAL_MODEL: {"weight": torch.tensor([[1.0]])},
            FedSMConstants.PERSONAL_MODEL: {"weight": torch.tensor([[2.0]])},
            FedSMConstants.SELECTOR_MODEL: {"weight": torch.tensor([[3.0]])},
            FedSMConstants.SELECTOR_OPTIMIZER: {},
        },
        params_type=ParamsType.FULL,
        meta={FedSMConstants.TARGET_ID: "site-1", FedSMConstants.SELECTOR_LABEL: 0},
    )

    assert helper.load_bundle(incoming, client_name="site-1") == 0
    global_model.weight.data.add_(4.0)
    personal_model.weight.data.add_(5.0)
    selector_model.weight.data.add_(6.0)
    result = helper.build_result(num_steps=7)

    assert result.params[FedSMConstants.GLOBAL_MODEL]["weight"].item() == pytest.approx(4.0)
    assert result.params[FedSMConstants.PERSONAL_MODEL]["weight"].item() == pytest.approx(7.0)
    assert result.params[FedSMConstants.SELECTOR_MODEL]["weight"].item() == pytest.approx(6.0)
    assert result.meta[FLMetaKey.NUM_STEPS_CURRENT_ROUND] == 7


def test_fedsm_persistor_round_trips_complete_bundle(tmp_path):
    model = torch.nn.Linear(1, 1, bias=False)
    selector = torch.nn.Linear(1, 2, bias=False)
    persistor = PTFedSMModelPersistor(model, selector, ["site-1", "site-2"])
    learnable = persistor.load_model(fl_ctx=None)
    checkpoint = tmp_path / "fedsm.pt"
    persistor._ckpt_save_path = str(checkpoint)
    persistor.save_model(learnable, fl_ctx=None)

    resumed = PTFedSMModelPersistor(
        model,
        selector,
        ["site-1", "site-2"],
        source_ckpt_file_full_name=str(checkpoint),
    ).load_model(fl_ctx=None)
    bundle = resumed[ModelLearnableKey.WEIGHTS]

    assert set(bundle[FedSMConstants.PERSONAL_MODELS]) == {"site-1", "site-2"}
    assert torch.equal(
        bundle[FedSMConstants.GLOBAL_MODEL]["weight"],
        learnable[ModelLearnableKey.WEIGHTS][FedSMConstants.GLOBAL_MODEL]["weight"],
    )
