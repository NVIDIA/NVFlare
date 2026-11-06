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

import copy
from unittest.mock import Mock

import numpy as np
import pytest

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import TaskCompletionStatus
from nvflare.apis.fl_constant import FLMetaKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.wf_comm_server import WFCommServer
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.aggregators.weighted_aggregation_helper import AggregationShapeError, AggregationStatsKey
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.app_common.workflows.fedavg import FedAvg


def prepare_controller(monkeypatch):
    controller = FedAvg(
        num_clients=3,
        num_rounds=1,
        model=FLModel(params={"early": np.zeros(2), "w": np.zeros(2)}),
        aggregation_weights={"site-1": 2.0, "site-2": 50.0},
    )
    controller.fl_ctx = FLContext()
    controller.abort_signal = Signal()
    monkeypatch.setattr(controller, "sample_clients", lambda _: ["site-1", "site-2", "site-3"])
    monkeypatch.setattr(controller, "event", lambda _: None)
    comm = WFCommServer()
    comm.controller = controller
    comm._engine = Mock()
    comm._engine.new_context.side_effect = lambda: controller.fl_ctx
    controller.fl_ctx.put(ReservedKey.ENGINE, comm._engine, private=True, sticky=False)
    monkeypatch.setattr(controller, "get_num_standing_tasks", comm.get_num_standing_tasks)
    monkeypatch.setattr(controller, "cancel_all_tasks", comm.cancel_all_tasks)
    return controller, comm


def snapshot_round(controller):
    helpers = []
    for helper in (controller._aggr_helper, controller._aggr_metrics_helper):
        helpers.append(
            {
                "total": {k: v.tolist() for k, v in helper.total.items()},
                "counts": helper.counts,
                "history": helper.history,
                "stats": helper.get_aggregation_stats(),
            }
        )
    return copy.deepcopy(
        (
            helpers,
            controller._received_count,
            controller._params_type,
            controller._site_metric_weights,
            controller._all_metrics,
            controller._warned_metric_keys,
        )
    )


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("bad_component", ["params", "metrics"])
def test_shape_rejection_keeps_fedavg_open_for_valid_subset(monkeypatch, backend, bad_component):
    array = np.array if backend == "numpy" else pytest.importorskip("torch").tensor
    controller, comm = prepare_controller(monkeypatch)
    warnings, saved, accepted, tasks = Mock(), [], [], []
    monkeypatch.setattr(controller, "warning", warnings)
    monkeypatch.setattr(controller, "save_model", saved.append)

    def broadcast(task, **kwargs):
        tasks.append(task)
        comm.broadcast(task, **kwargs)
        contributions = [
            FLModel(
                params={"early": array([2.0, 4.0]), "w": array([1.0, 2.0])},
                metrics={"score": array([2.0, 4.0])},
                meta={FLMetaKey.NUM_STEPS_CURRENT_ROUND: 1},
            ),
            FLModel(
                params={
                    "early": array([100.0, 200.0]),
                    "rejected_only": array([99.0]),
                    "w": array([10.0]) if bad_component == "params" else array([10.0, 20.0]),
                },
                metrics=None if bad_component == "params" else {"score": array([100.0]), "report": {}},
                meta={FLMetaKey.NUM_STEPS_CURRENT_ROUND: 4},
            ),
            FLModel(
                params={"early": array([6.0, 8.0]), "w": array([5.0, 6.0])},
                metrics={"score": array([6.0, 8.0]), "report": {}},
                meta={FLMetaKey.NUM_STEPS_CURRENT_ROUND: 6},
            ),
        ]
        for i, model in enumerate(contributions, start=1):
            client = Client(f"site-{i}", f"token-{i}")
            task_name, task_id, task_data = comm.process_task_request(client, controller.fl_ctx)
            if i == 2:
                before = snapshot_round(controller)
            result = FLModelUtils.to_shareable(model)
            # Real clients echo the assignment cookie jar, including the attempt identity.
            for name, value in (task_data.get_cookie_jar() or {}).items():
                result.add_cookie(name, value)
            comm.process_submission(client, task_name, task_id, result, controller.fl_ctx)
            accepted.append(controller.fl_ctx.get_prop(AppConstants.AGGREGATION_ACCEPTED))
            assert task.completion_status is None
            if i == 2:
                assert snapshot_round(controller) == before
        comm.check_tasks()

    monkeypatch.setattr(controller, "broadcast", broadcast)
    controller.run()

    assert accepted == [True, False, True]
    assert tasks[0].completion_status == TaskCompletionStatus.OK
    assert len(saved) == 1
    np.testing.assert_allclose(saved[0].params["early"], [5.0, 7.0])
    np.testing.assert_allclose(saved[0].params["w"], [4.0, 5.0])
    np.testing.assert_allclose(saved[0].metrics["score"], [5.0, 7.0])
    assert "rejected_only" not in saved[0].params
    assert saved[0].meta["nr_aggregated"] == 2
    stats = controller.fl_ctx.get_prop(AppConstants.AGGREGATION_STATS)
    assert stats[AggregationStatsKey.CONTRIBUTORS] == ["site-1", "site-3"]
    assert stats[AggregationStatsKey.ACCEPTED_CONTRIBUTIONS] == 2
    assert stats[AggregationStatsKey.KEYS_SEEN] == 2
    sites = saved[0].meta[AppConstants.METRICS_AGGREGATION_INFO]["site_weights"]
    assert [(site["name"], site["weight"]) for site in sites] == [("site-1", 2.0), ("site-3", 6.0)]
    report_warnings = [call for call in warnings.call_args_list if "Metric 'report'" in call.args[0]]
    assert len(report_warnings) == 1


@pytest.mark.parametrize("empty_results", [False, True])
def test_no_accepted_contributions_prevents_update_and_save(monkeypatch, empty_results):
    controller, comm = prepare_controller(monkeypatch)
    update, save = Mock(), Mock()
    monkeypatch.setattr(controller, "update_model", update)
    monkeypatch.setattr(controller, "save_model", save)

    def broadcast(task, **kwargs):
        if empty_results:
            comm.broadcast(task, **kwargs)
            for i in range(1, 4):
                client = Client(f"site-{i}", f"token-{i}")
                task_name, task_id, task_data = comm.process_task_request(client, controller.fl_ctx)
                result = FLModelUtils.to_shareable(FLModel(params={}))
                for name, value in (task_data.get_cookie_jar() or {}).items():
                    result.add_cookie(name, value)
                comm.process_submission(client, task_name, task_id, result, controller.fl_ctx)
                assert controller.fl_ctx.get_prop(AppConstants.AGGREGATION_ACCEPTED) is False
            comm.check_tasks()

    monkeypatch.setattr(controller, "broadcast", broadcast)
    with pytest.raises(RuntimeError, match="no accepted contributions"):
        controller.run()
    update.assert_not_called()
    save.assert_not_called()
    assert controller._received_count == 0
    assert controller._params_type is None
    assert controller._site_metric_weights == {}


def test_shape_error_after_parameter_accumulation_is_fatal(monkeypatch):
    controller, comm = prepare_controller(monkeypatch)
    update, save = Mock(), Mock()
    monkeypatch.setattr(controller, "update_model", update)
    monkeypatch.setattr(controller, "save_model", save)

    def broadcast(task, **kwargs):
        monkeypatch.setattr(
            controller._aggr_metrics_helper, "add", Mock(side_effect=AggregationShapeError("late error"))
        )
        comm.broadcast(task, **kwargs)
        for i in range(1, 4):
            client = Client(f"site-{i}", f"token-{i}")
            task_name, task_id, task_data = comm.process_task_request(client, controller.fl_ctx)
            result = FLModelUtils.to_shareable(FLModel(params={"w": np.ones(2)}, metrics={"score": np.ones(2)}))
            for name, value in (task_data.get_cookie_jar() or {}).items():
                result.add_cookie(name, value)
            comm.process_submission(client, task_name, task_id, result, controller.fl_ctx)
            assert controller.fl_ctx.get_prop(AppConstants.AGGREGATION_ACCEPTED) is False
        comm.check_tasks()

    monkeypatch.setattr(controller, "broadcast", broadcast)
    with pytest.raises(RuntimeError, match="refusing to update or save"):
        controller.run()
    np.testing.assert_array_equal(controller._aggr_helper.total["w"], np.full(2, 2.0))
    update.assert_not_called()
    save.assert_not_called()


def test_metric_shape_rejection_does_not_materialize_lazy_parameters(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    from nvflare.app_opt.pt.lazy_tensor_dict import _LazyRef, safetensors_refs

    tensor_file = tmp_path / "params.safetensors"
    save_file({"early": torch.tensor([6.0, 8.0]), "w": torch.tensor([5.0, 6.0])}, str(tensor_file))
    params = safetensors_refs(str(tensor_file))
    reads, accepted, saved = [], [], []
    original_materialize = _LazyRef.materialize

    def materialize(ref):
        reads.append(ref.key)
        return original_materialize(ref)

    monkeypatch.setattr(_LazyRef, "materialize", materialize)
    controller, _ = prepare_controller(monkeypatch)
    monkeypatch.setattr(controller, "save_model", saved.append)

    def send_model(callback, **kwargs):
        assert (
            callback(
                FLModel(
                    params={"early": torch.tensor([2.0, 4.0]), "w": torch.tensor([1.0, 2.0])},
                    metrics={"score": torch.tensor([2.0, 4.0])},
                    meta={"client_name": "site-1"},
                )
            )
            is True
        )
        before = snapshot_round(controller)
        accepted.append(
            callback(FLModel(params=params, metrics={"score": torch.tensor([100.0])}, meta={"client_name": "site-2"}))
        )
        assert reads == []
        assert snapshot_round(controller) == before
        accepted.append(
            callback(
                FLModel(
                    params=params,
                    metrics={"score": torch.tensor([6.0, 8.0])},
                    meta={"client_name": "site-3", FLMetaKey.NUM_STEPS_CURRENT_ROUND: 6},
                )
            )
        )

    monkeypatch.setattr(controller, "send_model", send_model)
    controller.run()
    assert accepted == [False, True]
    assert reads == list(params)
    torch.testing.assert_close(saved[0].params["early"], torch.tensor([5.0, 7.0]))
    torch.testing.assert_close(saved[0].params["w"], torch.tensor([4.0, 5.0]))
