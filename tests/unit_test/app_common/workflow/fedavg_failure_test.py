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

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, current_thread
from unittest.mock import Mock

import pytest

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, TaskCompletionStatus
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.wf_comm_server import WFCommServer
from nvflare.apis.shareable import ReservedHeaderKey
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator
from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.app_common.workflows import fedavg as fedavg_module
from nvflare.app_common.workflows.fedavg import FedAvg


def deliver(controller, task, params, name="site"):
    client_task = ClientTask(Client(name, "test-token"), task)
    task.before_task_sent_cb(client_task, controller.fl_ctx)
    client_task.result = FLModelUtils.to_shareable(FLModel(params=params, metrics={"loss": 1.0}))
    task.result_received_cb(client_task, controller.fl_ctx)
    return controller.fl_ctx.get_prop(AppConstants.AGGREGATION_ACCEPTED)


def prepare_controller(monkeypatch, model, rounds=1, custom=False):
    aggregator = None
    if custom:
        helper = WeightedAggregationHelper()
        aggregator = Mock(spec=ModelAggregator, fl_ctx=None)
        aggregator.reset_stats.side_effect = helper.reset_stats
        aggregator.accept_model.side_effect = lambda result: helper.add(result.params, 1.0, "site", 0)
        aggregator.aggregate_model.side_effect = lambda: FLModel(params=helper.get_result())
    controller = FedAvg(num_clients=2, num_rounds=rounds, model=model, aggregator=aggregator)
    controller.fl_ctx = FLContext()
    controller.abort_signal = Signal()
    monkeypatch.setattr(controller, "sample_clients", lambda _: ["good", "bad"])
    monkeypatch.setattr(controller, "event", lambda event: None)
    monkeypatch.setattr(controller, "get_num_standing_tasks", lambda: 0)
    return controller


@pytest.mark.parametrize(
    "fault,outstanding", [("missing", False), ("missing", True), ("metrics", False), (None, False)]
)
@pytest.mark.parametrize("good_first", [True, False])
def test_failed_round_preserves_checkpoint(tmp_path, monkeypatch, fault, outstanding, good_first):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
    from nvflare.app_opt.pt.lazy_tensor_dict import LazyTensorDict

    initial = {"a": torch.tensor([0.0]), "b": torch.tensor([0.0])}
    controller = prepare_controller(monkeypatch, FLModel(params=initial))
    controller.fl_ctx.set_prop(FLContextKey.APP_ROOT, str(tmp_path))
    checkpoint = tmp_path / "initial.pt"
    torch.save(initial, checkpoint)
    persistor = PTFileModelPersistor(source_ckpt_file_full_name=str(checkpoint), allow_numpy_conversion=False)
    persistor._initialize(controller.fl_ctx)
    persistor.load(controller.fl_ctx)
    controller.persistor = persistor
    controller.save_model(FLModel(params=initial))
    saved_path = tmp_path / persistor.global_model_file_name
    before = saved_path.read_bytes()
    update = Mock(wraps=controller.update_model)
    monkeypatch.setattr(controller, "update_model", update)
    offload = tmp_path / "offload"
    offload.mkdir()
    first, second = offload / "a.safetensors", offload / "b.safetensors"
    save_file({"a": torch.tensor([100.0])}, str(first))
    save_file({"b": torch.tensor([200.0])}, str(second))
    lazy = LazyTensorDict({"a": (str(first), "a"), "b": (str(second), "b")}, str(offload))
    if fault == "missing":
        second.unlink()
    accepted = []

    def broadcast(task, **kwargs):
        if good_first:
            accepted.append(deliver(controller, task, {"a": torch.tensor([2.0]), "b": torch.tensor([4.0])}, "good"))
        if fault == "metrics":
            monkeypatch.setattr(controller._aggr_metrics_helper, "add", Mock(side_effect=ValueError("metric failure")))
        accepted.append(deliver(controller, task, {k: lazy.make_lazy_ref(k) for k in lazy.keys()}, "bad"))

    monkeypatch.setattr(controller, "broadcast", broadcast)
    cancel = Mock()
    if outstanding:
        monkeypatch.setattr(controller, "get_num_standing_tasks", lambda: 1)
        monkeypatch.setattr(controller, "cancel_all_tasks", cancel)
        monkeypatch.setattr(fedavg_module.time, "sleep", Mock(side_effect=AssertionError("waiting after failure")))
    if fault:
        with pytest.raises(RuntimeError, match="refusing to update or save"):
            controller.run()
        assert accepted == ([True, False] if good_first else [False])
        update.assert_not_called()
        assert saved_path.read_bytes() == before
        if outstanding:
            cancel.assert_called_once_with(TaskCompletionStatus.ERROR)
    else:
        controller.run()
        actual = torch.load(saved_path, weights_only=True)["model"]
        expected = (51.0, 102.0) if good_first else (100.0, 200.0)
        assert (actual["a"].item(), actual["b"].item()) == expected


@pytest.mark.parametrize("abort", [False, True])
@pytest.mark.parametrize("custom", [False, True])
def test_defensive_finalization_waits_for_failed_callback(monkeypatch, abort, custom):
    """Force early closure to test the guard, not normal communicator retirement ordering."""
    controller = prepare_controller(monkeypatch, FLModel(params={"a": 0.0, "b": 0.0}), custom=custom)
    if abort:
        controller.abort_signal.trigger(True)
        monkeypatch.setattr(controller, "get_num_standing_tasks", lambda: 1)
    entered, release, finalizing = Event(), Event(), Event()
    update, save = Mock(), Mock()
    monkeypatch.setattr(controller, "update_model", update)
    monkeypatch.setattr(controller, "save_model", save)
    original_get = controller._get_aggregated_result

    def get_result():
        # Deliberately request finalization while a callback is still blocked.
        finalizing.set()
        return original_get()

    monkeypatch.setattr(controller, "_get_aggregated_result", get_result)

    class ObservedLock:
        def __init__(self):
            self.lock = Lock()

        def __enter__(self):
            if current_thread().name.startswith("finalizer"):
                finalizing.set()
            self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()

    monkeypatch.setattr(fedavg_module, "Lock", ObservedLock, raising=False)

    class FailingTensor:
        def materialize(self):
            entered.set()
            assert release.wait(5), "test did not release callback"
            raise OSError("controlled late file failure")

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="consumer") as consumers:
        callbacks = []

        def broadcast(task, **kwargs):
            callbacks.append(task.get_prop(AppConstants.TASK_PROP_CALLBACK))
            consumers.submit(deliver, controller, task, {"a": 10.0, "b": FailingTensor()})
            assert entered.wait(5)

        monkeypatch.setattr(controller, "broadcast", broadcast)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="finalizer") as finalizers:
            future = finalizers.submit(controller.run)
            future.add_done_callback(lambda _: finalizing.set())
            try:
                assert finalizing.wait(5), "finalization never started"
                returned_before_release = future.done()
            finally:
                release.set()
            if abort:
                assert future.result(timeout=5) is None
                assert not returned_before_release, "abort returned while a callback was still running"
            else:
                with pytest.raises(RuntimeError, match="refusing to update or save"):
                    future.result(timeout=5)
        update.assert_not_called()
        save.assert_not_called()
        assert callbacks[0](FLModel(params={"a": 999.0})) is False


@pytest.mark.parametrize("custom", [False, True])
def test_closed_round_callback_cannot_change_next_round(monkeypatch, custom):
    controller = prepare_controller(monkeypatch, FLModel(params={"a": 0.0}), rounds=2, custom=custom)
    callbacks, saved = [], []
    monkeypatch.setattr(controller, "save_model", lambda model: saved.append(dict(model.params)))

    def broadcast(task, **kwargs):
        if callbacks:
            assert callbacks[0](FLModel(params={"a": 999.0})) is False
        callbacks.append(task.get_prop(AppConstants.TASK_PROP_CALLBACK))
        assert deliver(controller, task, {}) is False  # deliberate skip does not fail the round
        assert deliver(controller, task, {"a": 2.0}) is True

    monkeypatch.setattr(controller, "broadcast", broadcast)
    controller.run()
    assert saved == [{"a": 2.0}, {"a": 2.0}]


def test_custom_callback_failure_through_communicator_prevents_publication(monkeypatch):
    controller = prepare_controller(monkeypatch, FLModel(params={"a": 0.0}), custom=True)
    comm = WFCommServer()
    comm.controller = controller
    comm._engine = Mock()
    comm._engine.new_context.side_effect = lambda: controller.fl_ctx
    monkeypatch.setattr(controller, "get_num_standing_tasks", comm.get_num_standing_tasks)
    monkeypatch.setattr(controller, "cancel_all_tasks", comm.cancel_all_tasks)
    update, save = Mock(), Mock()
    monkeypatch.setattr(controller, "update_model", update)
    monkeypatch.setattr(controller, "save_model", save)
    accept = controller.aggregator.accept_model.side_effect

    def fail_after_mutation(result):
        accept(result)
        if result.meta["client_name"] == "bad":
            raise RuntimeError("aggregator failed after changing its state")

    controller.aggregator.accept_model.side_effect = fail_after_mutation

    def broadcast(task, **kwargs):
        comm.broadcast(task, controller.fl_ctx, targets=["good", "bad"])
        for name, value in (("good", 1.0), ("bad", 9.0)):
            client = Client(name, name)
            task_name, task_id, task_data = comm.process_task_request(client, controller.fl_ctx)
            result = FLModelUtils.to_shareable(FLModel(params={"a": value}))
            result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, task_data.get_task_attempt_id())
            comm.process_submission(client, task_name, task_id, result, controller.fl_ctx)
            assert controller.fl_ctx.get_prop(FLContextKey.TASK_RESULT_ACCEPTED) is (name == "good")
        # Actual task retirement follows callback completion under the communicator lock.
        comm.check_tasks()
        assert comm.get_num_standing_tasks() == 0

    monkeypatch.setattr(controller, "broadcast", broadcast)
    with pytest.raises(RuntimeError, match="refusing to update or save"):
        controller.run()
    assert controller.aggregator.aggregate_model.side_effect().params == {"a": 5.0}
    controller.aggregator.aggregate_model.assert_not_called()
    update.assert_not_called()
    save.assert_not_called()
