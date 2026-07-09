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

from unittest.mock import Mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.abstract.model import ModelLearnableKey
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.pt.fedsm import (
    FedSM,
    FedSMConstants,
    FedSMModelAggregator,
    PTFedSMHelper,
    PTFedSMModelPersistor,
    _add_state_diff,
    _to_cpu_copy,
    _weighted_average,
)


class _CheckpointMeta:
    def __init__(self, value):
        self.value = value


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


def _initial_bundle():
    return FLModel(
        params={
            FedSMConstants.GLOBAL_MODEL: _state(10.0),
            FedSMConstants.PERSONAL_MODELS: {"site-1": _state(1.0), "site-2": _state(2.0)},
            FedSMConstants.SELECTOR_MODEL: _state(20.0),
            FedSMConstants.SELECTOR_OPTIMIZER: {},
        },
        params_type=ParamsType.FULL,
    )


def _persistor_fl_ctx(tmp_path, log_dir=None):
    fl_ctx = Mock()

    def get_prop(key):
        if key == FLContextKey.APP_ROOT:
            return str(tmp_path)
        if key == AppConstants.LOG_DIR:
            return log_dir
        return None

    fl_ctx.get_prop.side_effect = get_prop
    return fl_ctx


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
        FedSMConstants.SELECTOR_OPTIMIZER: {"step": 1},
    }

    updated = FedSM._update_bundle(bundle, update)

    assert updated[FedSMConstants.GLOBAL_MODEL]["weight"].item() == pytest.approx(12.0)
    assert updated[FedSMConstants.SELECTOR_MODEL]["weight"].item() == pytest.approx(15.0)
    assert updated[FedSMConstants.PERSONAL_MODELS]["site-1"]["weight"].item() == pytest.approx(3.0)
    assert updated[FedSMConstants.SELECTOR_OPTIMIZER] == {"step": 1}


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


def test_fedsm_helper_synchronizes_selector_optimizer_state():
    global_model = torch.nn.Linear(1, 1, bias=False)
    personal_model = torch.nn.Linear(1, 1, bias=False)
    selector_model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.Adam(selector_model.parameters(), lr=0.1)
    selector_model(torch.tensor([[1.0]])).sum().backward()
    optimizer.step()
    optimizer_state = optimizer.state_dict()["state"]
    helper = PTFedSMHelper(global_model, personal_model, selector_model, optimizer)
    incoming = FLModel(
        params={
            FedSMConstants.GLOBAL_MODEL: global_model.state_dict(),
            FedSMConstants.PERSONAL_MODEL: personal_model.state_dict(),
            FedSMConstants.SELECTOR_MODEL: selector_model.state_dict(),
            FedSMConstants.SELECTOR_OPTIMIZER: optimizer_state,
        },
        params_type=ParamsType.FULL,
        meta={FedSMConstants.TARGET_ID: "site-1", FedSMConstants.SELECTOR_LABEL: 0},
    )

    helper.load_bundle(incoming, client_name="site-1")
    result = helper.build_result(num_steps=1)

    assert result.params[FedSMConstants.SELECTOR_OPTIMIZER]


def test_fedsm_helper_validates_models_target_and_call_order():
    model = torch.nn.Linear(1, 1, bias=False)
    with pytest.raises(TypeError, match="personal_model must be"):
        PTFedSMHelper(model, object(), model)

    helper = PTFedSMHelper(model, torch.nn.Linear(1, 1, bias=False), torch.nn.Linear(1, 1, bias=False))
    with pytest.raises(RuntimeError, match=r"load_bundle\(\) must be called"):
        helper.build_result(num_steps=1)

    incoming = FLModel(
        params={
            FedSMConstants.GLOBAL_MODEL: model.state_dict(),
            FedSMConstants.PERSONAL_MODEL: model.state_dict(),
            FedSMConstants.SELECTOR_MODEL: model.state_dict(),
        },
        params_type=ParamsType.FULL,
        meta={FedSMConstants.TARGET_ID: "site-2", FedSMConstants.SELECTOR_LABEL: 1},
    )
    with pytest.raises(ValueError, match="targets 'site-2'"):
        helper.load_bundle(incoming, client_name="site-1")


def test_fedsm_helper_rejects_non_full_bundle():
    model = torch.nn.Linear(1, 1, bias=False)
    helper = PTFedSMHelper(model, torch.nn.Linear(1, 1, bias=False), torch.nn.Linear(1, 1, bias=False))
    incoming = FLModel(
        params={"weight": torch.tensor([[1.0]])},
        params_type=ParamsType.DIFF,
        meta={FedSMConstants.TARGET_ID: "site-1", FedSMConstants.SELECTOR_LABEL: 0},
    )

    with pytest.raises(ValueError, match="ParamsType.FULL"):
        helper.load_bundle(incoming, client_name="site-1")


def test_fedsm_helper_reports_missing_bundle_entries():
    model = torch.nn.Linear(1, 1, bias=False)
    helper = PTFedSMHelper(model, torch.nn.Linear(1, 1, bias=False), torch.nn.Linear(1, 1, bias=False))
    incoming = FLModel(
        params={FedSMConstants.GLOBAL_MODEL: {"weight": torch.tensor([[1.0]])}},
        params_type=ParamsType.FULL,
        meta={},
    )

    with pytest.raises(ValueError, match="missing parameter entries"):
        helper.load_bundle(incoming, client_name="site-1")


def test_fedsm_helper_reports_missing_bundle_metadata():
    model = torch.nn.Linear(1, 1, bias=False)
    helper = PTFedSMHelper(model, torch.nn.Linear(1, 1, bias=False), torch.nn.Linear(1, 1, bias=False))
    incoming = FLModel(
        params={
            FedSMConstants.GLOBAL_MODEL: {"weight": torch.tensor([[1.0]])},
            FedSMConstants.PERSONAL_MODEL: {"weight": torch.tensor([[2.0]])},
            FedSMConstants.SELECTOR_MODEL: {"weight": torch.tensor([[3.0]])},
        },
        params_type=ParamsType.FULL,
        meta={},
    )

    with pytest.raises(ValueError, match="missing metadata entries"):
        helper.load_bundle(incoming, client_name="site-1")


def test_fedsm_requires_exact_configured_client_set_each_round():
    controller = FedSM(
        num_clients=2,
        client_id_label_mapping={"site-1": 0, "site-2": 1},
    )

    controller._validate_sampled_clients(["site-1", "site-2"])
    with pytest.raises(RuntimeError, match=r"missing clients: \['site-2'\]"):
        controller._validate_sampled_clients(["site-1"])
    with pytest.raises(RuntimeError, match=r"unexpected clients: \['site-3'\]"):
        controller._validate_sampled_clients(["site-1", "site-3"])


def test_fedsm_requires_num_clients_to_match_label_mapping():
    with pytest.raises(ValueError, match="num_clients must equal"):
        FedSM(
            num_clients=1,
            client_id_label_mapping={"site-1": 0, "site-2": 1},
        )


def test_fedsm_weighted_average_supports_buffers_numpy_and_scalars():
    integer = _weighted_average(
        [torch.tensor([1], dtype=torch.int64), torch.tensor([4], dtype=torch.int64)],
        [0.5, 0.5],
    )
    array = _weighted_average([np.array([1.0]), np.array([3.0])], [0.25, 0.75])
    scalar = _weighted_average([2.0, 6.0], [0.5, 0.5])

    assert integer.item() == 2
    assert array.tolist() == pytest.approx([2.5])
    assert scalar == pytest.approx(4.0)
    with pytest.raises(TypeError, match="cannot average"):
        _weighted_average([object(), object()], [0.5, 0.5])


def test_fedsm_add_state_diff_ignores_unknown_parameters():
    updated = _add_state_diff(_state(2.0), {"missing": torch.tensor([10.0])})
    copied_meta = _to_cpu_copy(_CheckpointMeta(3))

    assert updated["weight"].item() == pytest.approx(2.0)
    assert copied_meta.value == 3


def test_fedsm_aggregator_validates_results_and_single_client_fallback():
    with pytest.raises(ValueError, match="soft_pull_lambda"):
        FedSMModelAggregator(soft_pull_lambda=1.1)

    aggregator = FedSMModelAggregator()
    with pytest.raises(ValueError, match="missing FLModel.meta"):
        aggregator.accept_model(
            FLModel(
                params={FedSMConstants.GLOBAL_MODEL: _state(1.0)},
                params_type=ParamsType.FULL,
            )
        )
    with pytest.raises(ValueError, match="ParamsType.FULL"):
        aggregator.accept_model(
            FLModel(
                params={"weight": torch.tensor([1.0])},
                params_type=ParamsType.DIFF,
                meta={"client_name": "site-1"},
            )
        )
    with pytest.raises(ValueError, match="empty result set"):
        aggregator.aggregate_model()

    result = _result("site-1", 2.0, 4.0, 6.0, steps=0)
    result.params[FedSMConstants.SELECTOR_OPTIMIZER] = {"step": torch.tensor(1.0)}
    aggregator.accept_model(result)
    with pytest.raises(ValueError, match="more than one result"):
        aggregator.accept_model(result)

    aggregated = aggregator.aggregate_model()

    assert aggregated.params[FedSMConstants.PERSONAL_MODELS]["site-1"]["weight"].item() == pytest.approx(4.0)
    assert aggregated.params[FedSMConstants.SELECTOR_OPTIMIZER]["step"].item() == pytest.approx(1.0)
    aggregator.reset_stats()
    assert aggregator._results == {}


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


def test_fedsm_persistor_initializes_from_event_and_resolves_model_configs(tmp_path):
    model = torch.nn.Linear(1, 1, bias=False)
    selector = torch.nn.Linear(1, 2, bias=False)
    fl_ctx = _persistor_fl_ctx(tmp_path, log_dir="models")
    persistor = PTFedSMModelPersistor(model, selector, ["site-1"])

    persistor.handle_event(EventType.START_RUN, fl_ctx)

    assert persistor._ckpt_save_path == str(tmp_path / "models" / "FL_fedsm_model.pt")
    configured = persistor._resolve_model(
        {"path": "torch.nn.Linear", "args": {"in_features": 1, "out_features": 1}},
        "model",
        fl_ctx,
    )
    assert isinstance(configured, torch.nn.Linear)

    fl_ctx.get_engine.return_value.get_component.return_value = model
    assert persistor._resolve_model("model_component", "model", fl_ctx) is model
    with pytest.raises(TypeError, match="must resolve to torch.nn.Module"):
        persistor._resolve_model(object(), "model", fl_ctx)


def test_fedsm_persistor_resolves_relative_paths_and_legacy_checkpoints(tmp_path):
    model = torch.nn.Linear(1, 1, bias=False)
    selector = torch.nn.Linear(1, 2, bias=False)
    fl_ctx = _persistor_fl_ctx(tmp_path)
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    checkpoint = custom_dir / "legacy.pt"
    torch.save({"model": _state(9.0)}, checkpoint)
    persistor = PTFedSMModelPersistor(
        model,
        selector,
        ["site-1"],
        source_ckpt_file_full_name="legacy.pt",
    )

    learnable = persistor.load_model(fl_ctx)

    assert learnable[ModelLearnableKey.WEIGHTS][FedSMConstants.GLOBAL_MODEL]["weight"].item() == pytest.approx(9.0)

    missing = PTFedSMModelPersistor(
        model,
        selector,
        ["site-1"],
        source_ckpt_file_full_name="missing.pt",
    )
    with pytest.raises(ValueError, match="source checkpoint not found"):
        missing.load_model(fl_ctx)


def test_fedsm_persistor_save_initializes_output_path(tmp_path):
    model = torch.nn.Linear(1, 1, bias=False)
    selector = torch.nn.Linear(1, 2, bias=False)
    persistor = PTFedSMModelPersistor(model, selector, ["site-1"])
    fl_ctx = _persistor_fl_ctx(tmp_path)
    learnable = persistor.load_model(fl_ctx)

    persistor.save_model(learnable, fl_ctx)

    assert (tmp_path / "FL_fedsm_model.pt").exists()


def test_fedsm_persistor_can_load_trusted_checkpoint_with_custom_metadata(tmp_path):
    model = torch.nn.Linear(1, 1, bias=False)
    selector = torch.nn.Linear(1, 2, bias=False)
    persistor = PTFedSMModelPersistor(model, selector, ["site-1"])
    learnable = persistor.load_model(fl_ctx=None)
    learnable[ModelLearnableKey.META]["custom"] = _CheckpointMeta(7)
    checkpoint = tmp_path / "fedsm-custom-meta.pt"
    persistor._ckpt_save_path = str(checkpoint)
    persistor.save_model(learnable, fl_ctx=None)

    resumed = PTFedSMModelPersistor(
        model,
        selector,
        ["site-1"],
        source_ckpt_file_full_name=str(checkpoint),
        load_weights_only=False,
    ).load_model(fl_ctx=None)

    assert resumed[ModelLearnableKey.META]["custom"].value == 7


def test_fedsm_controller_runs_complete_round():
    controller = FedSM(
        num_clients=2,
        num_rounds=1,
        client_id_label_mapping={"site-1": 0, "site-2": 1},
    )
    controller.fl_ctx = Mock()
    controller.abort_signal = Mock(triggered=False)
    controller.load_model = Mock(return_value=_initial_bundle())
    controller.sample_clients = Mock(return_value=["site-1", "site-2"])
    controller.get_num_standing_tasks = Mock(return_value=0)
    controller.event = Mock()
    controller.info = Mock()
    controller.save_model = Mock()
    controller._maybe_cleanup_memory = Mock()
    controller.aggregate = lambda results, aggregate_fn: aggregate_fn(results)

    def send_model(*, targets, callback, **kwargs):
        client = targets[0]
        callback(
            _result(
                client,
                global_diff=2.0 if client == "site-1" else 4.0,
                personal=3.0 if client == "site-1" else 5.0,
                selector_diff=1.0 if client == "site-1" else 3.0,
            )
        )

    controller.send_model = send_model

    controller.run()

    saved = controller.save_model.call_args.args[0]
    assert saved.params[FedSMConstants.GLOBAL_MODEL]["weight"].item() == pytest.approx(13.0)
    assert saved.params[FedSMConstants.SELECTOR_MODEL]["weight"].item() == pytest.approx(22.0)
    assert controller.save_model.call_count == 1
    assert controller.fl_ctx.set_prop.call_count == 3


def test_fedsm_controller_rejects_incomplete_results():
    controller = FedSM(
        num_clients=2,
        num_rounds=1,
        client_id_label_mapping={"site-1": 0, "site-2": 1},
    )
    controller.fl_ctx = Mock()
    controller.abort_signal = Mock(triggered=False)
    controller.load_model = Mock(return_value=_initial_bundle())
    controller.sample_clients = Mock(return_value=["site-1", "site-2"])
    controller.send_model = Mock()
    controller.get_num_standing_tasks = Mock(return_value=0)
    controller.event = Mock()
    controller.info = Mock()

    with pytest.raises(RuntimeError, match="received 0 of 2 expected"):
        controller.run()


def test_fedsm_controller_returns_when_aborted_with_standing_tasks():
    controller = FedSM(
        num_clients=2,
        num_rounds=1,
        client_id_label_mapping={"site-1": 0, "site-2": 1},
    )
    controller.fl_ctx = Mock()
    controller.abort_signal = Mock(triggered=True)
    controller.load_model = Mock(return_value=_initial_bundle())
    controller.sample_clients = Mock(return_value=["site-1", "site-2"])
    controller.send_model = Mock()
    controller.get_num_standing_tasks = Mock(return_value=1)
    controller.event = Mock()
    controller.info = Mock()
    controller.save_model = Mock()

    controller.run()

    controller.save_model.assert_not_called()


def test_fedsm_controller_waits_for_standing_tasks(monkeypatch):
    controller = FedSM(
        num_clients=2,
        num_rounds=1,
        client_id_label_mapping={"site-1": 0, "site-2": 1},
    )
    controller.fl_ctx = Mock()
    controller.abort_signal = Mock(triggered=False)
    controller.load_model = Mock(return_value=_initial_bundle())
    controller.sample_clients = Mock(return_value=["site-1", "site-2"])
    controller.send_model = Mock()
    controller.get_num_standing_tasks = Mock(side_effect=[1, 0])
    controller.event = Mock()
    controller.info = Mock()
    sleep = Mock()
    monkeypatch.setattr("nvflare.app_opt.pt.fedsm.time.sleep", sleep)

    with pytest.raises(RuntimeError, match="received 0 of 2 expected"):
        controller.run()

    sleep.assert_called_once_with(controller._task_check_period)
