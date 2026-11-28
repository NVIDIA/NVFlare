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
#
# Authors: Anbang Liu, Junhan Zhao, and Ziyue Xu

"""Exercise FLARE messages, aggregation, client delegation, and job export without launching clients."""

import copy
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import fabric_client
import pytest
import torch
from fabric_aggregator import FabricFedAvg
from fabric_common import SITES, load_core, local_seed, read_json, state_digest, write_json
from run_fabric import build_recipe

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.app_opt.pt.decomposers import TensorDecomposer
from nvflare.fuel.utils import fobs


def test_full_model_exchange_preserves_tensors_and_metadata(make_fold, client_update):
    case = make_fold()
    message = client_update(case, SITES[0], case.initial, 2)
    fobs.register(TensorDecomposer)
    received = FLModelUtils.from_shareable(fobs.loads(fobs.dumps(FLModelUtils.to_shareable(message))))
    assert received.params_type == ParamsType.FULL
    assert received.current_round == 2
    assert state_digest(received.params) == state_digest(message.params)
    for key, value in message.meta.items():
        assert received.meta[key] == value
    assert received.metrics == message.metrics


def test_aggregation_uses_sample_weights_and_fixed_client_order(make_fold, client_update):
    states = []
    for name, sites in (("forward", SITES), ("reverse", tuple(reversed(SITES)))):
        case = make_fold(name)
        aggregator = FabricFedAvg(str(case.path))
        for site in sites:
            aggregator.accept_model(client_update(case, site, case.initial, 0))
        result = aggregator.aggregate_model()
        # Counts 4, 6, 8 and local increments 1, 2, 3 give an increment of 40/18.
        torch.testing.assert_close(result.params["weight"], torch.tensor([1.0 + 40 / 18, -2.0 + 40 / 18]))
        assert torch.equal(result.params["buffer"], torch.tensor([4], dtype=torch.int64))
        assert result.metrics["train_loss"] == pytest.approx(1.25)
        audit = read_json(case.folder / "server_audit/round_1.json")
        assert audit["sites_in_aggregation_order"] == list(SITES)
        assert audit["patient_weights"] == [4, 6, 8]
        states.append(state_digest(result.params))
    assert states[0] == states[1]


@pytest.mark.parametrize("fault", ["site", "count", "seed", "epochs", "digest", "round", "diff", "dtype", "nan"])
def test_inconsistent_client_updates_are_rejected(make_fold, client_update, fault):
    case = make_fold()
    aggregator = FabricFedAvg(str(case.path))
    valid = client_update(case, SITES[0], case.initial, 0)
    update = copy.deepcopy(valid)
    if fault in ("site", "count", "seed", "epochs", "digest"):
        key = {
            "site": "site",
            "count": "train_patients",
            "seed": "seed",
            "epochs": "local_epochs",
            "digest": "input_state_sha256",
        }[fault]
        update.meta[key] = "invalid"
    elif fault == "round":
        update.current_round = 1
    elif fault == "diff":
        update.params_type = ParamsType.DIFF
    elif fault == "dtype":
        update.params["weight"] = update.params["weight"].double()
    else:
        update.params["weight"][0] = float("nan")
    with pytest.raises((ValueError, TypeError)):
        aggregator.accept_model(update)
    assert not aggregator.updates
    aggregator.accept_model(valid)
    with pytest.raises(ValueError, match="duplicate"):
        aggregator.accept_model(valid)
    with pytest.raises(RuntimeError, match="All three"):
        aggregator.aggregate_model()
    assert not (case.folder / "server_audit").exists()


@pytest.mark.parametrize("encoder", ["uni", "virchow2"])
@pytest.mark.parametrize("variant", ["pooling", "topk"])
@pytest.mark.parametrize("offset", [0, 3])
def test_client_preserves_training_settings_and_resume_seeds(make_fold, encoder, variant, offset, monkeypatch):
    case = make_fold(encoder=encoder, variant=variant)
    case.runtime.update(resume_offset=offset, resume_state_sha256=state_digest(case.initial))
    write_json(case.path, case.runtime)
    core = load_core()
    site = SITES[1]
    incoming = FLModel(params=case.initial, params_type=ParamsType.FULL, current_round=0, total_rounds=5 - offset)
    send = Mock()
    train = Mock(return_value=(case.initial, {"loss": 0.4, "seconds": 0.0, "train_patients": 6}))
    monkeypatch.setattr(sys, "argv", ["fabric_client.py", "--runtime", str(case.path)])
    monkeypatch.setattr(fabric_client, "check_environment", lambda root: None)
    monkeypatch.setattr(fabric_client, "install_feature_reader", lambda *args: None)
    for name in ("init", "shutdown"):
        monkeypatch.setattr(flare, name, lambda: None)
    monkeypatch.setattr(flare, "get_site_name", lambda: site)
    monkeypatch.setattr(flare, "receive", lambda: incoming)
    monkeypatch.setattr(flare, "send", send)
    # Only the CUDA entry guard is mocked; no tensor or model is moved to a GPU.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(core.client_training, "resolve_device", lambda args: torch.device("cpu"))
    monkeypatch.setattr(core.client_training, "train_local_client_final", train)
    fabric_client.main()
    train.assert_called_once()
    positional = train.call_args.args
    assert positional[3:5] == (case.setup["model_name"], case.setup["input_dim"])
    for key, value in case.setup["settings"].items():
        assert getattr(positional[5], key) == value
    assert train.call_args.kwargs["seed"] == local_seed(case.setup, 0, offset, site)
    if variant == "topk":
        assert positional[5].dtfd_top_k == 1
        assert positional[5].dtfd_pseudo_loss_weight == 1.0
    send.assert_called_once()
    outgoing = send.call_args.args[0]
    assert outgoing.params_type == ParamsType.FULL
    assert outgoing.current_round == 0
    assert outgoing.metrics == {"train_loss": 0.4}
    assert outgoing.meta["train_patients"] == 6
    assert outgoing.meta["local_log"]["round"] == offset + 1
    assert outgoing.meta["seed"] == train.call_args.kwargs["seed"]
    assert outgoing.meta["input_state_sha256"] == state_digest(case.initial)
    assert not any(
        text in json.dumps(outgoing.meta) for text in ("synthetic-", "patient_id", "feature_paths", "unused.h5")
    )


@pytest.mark.parametrize("variant", ["pooling", "topk"])
def test_exported_job_contains_three_clients_and_the_selected_model(make_fold, variant, monkeypatch):
    case = make_fold(variant=variant)
    monkeypatch.chdir(Path(fabric_client.__file__).parent)
    recipe = build_recipe(case.path, case.setup)
    recipe.export(str(case.folder / "exported_job"))
    job = case.folder / "exported_job" / recipe.name
    metadata = read_json(job / "meta.json")
    assert metadata["min_clients"] == 3
    assert {site for sites in metadata["deploy_map"].values() for site in sites} == {"server", *SITES}
    server = read_json(job / "app_server/config/config_fed_server.json")
    workflow = server["workflows"][0]["args"]
    assert workflow.get("num_rounds", 5) == 5
    assert workflow["aggregator"]["path"] == "fabric_aggregator.FabricFedAvg"
    assert "model_selector" not in {component["id"] for component in server["components"]}
    persistor = next(component for component in server["components"] if component["id"] == "persistor")
    assert persistor["args"]["allow_numpy_conversion"] is False
    assert persistor["args"]["model"]["args"]["model_name"] == case.setup["model_name"]
    for site in SITES:
        client = read_json(job / f"app_{site}/config/config_fed_client.json")
        launcher = next(component for component in client["components"] if component["id"] == "launcher")
        assert launcher["args"]["launch_once"] is False
        executor = client["executors"][0]["executor"]["args"]
        assert executor["server_expected_format"] == "pytorch"
        assert executor.get("params_transfer_type", "FULL") == "FULL"
        assert (job / f"app_{site}/custom/core/topk_mil.py").is_file()
