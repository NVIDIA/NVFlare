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
from nvflare.apis.controller_spec import ClientTask
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.fl_model import FLModel
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


def prepare_controller(monkeypatch, model, rounds=1):
    controller = FedAvg(num_clients=2, num_rounds=rounds, model=model)
    controller.fl_ctx = FLContext()
    controller.abort_signal = Signal()
    monkeypatch.setattr(controller, "sample_clients", lambda _: ["good", "bad"])
    monkeypatch.setattr(controller, "event", lambda event: None)
    monkeypatch.setattr(controller, "get_num_standing_tasks", lambda: 0)
    return controller


@pytest.mark.parametrize("fault", ["missing", "metrics", None])
@pytest.mark.parametrize("good_first", [True, False])
def test_failed_round_preserves_checkpoint(tmp_path, monkeypatch, fault, good_first):
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
    if fault:
        with pytest.raises(RuntimeError, match="refusing to update or save"):
            controller.run()
        assert accepted == ([True, False] if good_first else [False])
        update.assert_not_called()
        assert saved_path.read_bytes() == before
    else:
        controller.run()
        actual = torch.load(saved_path, weights_only=True)["model"]
        expected = (51.0, 102.0) if good_first else (100.0, 200.0)
        assert (actual["a"].item(), actual["b"].item()) == expected


def test_finalization_waits_for_failed_callback(monkeypatch):
    controller = prepare_controller(monkeypatch, FLModel(params={"a": 0.0, "b": 0.0}))
    entered, release, finalizing = Event(), Event(), Event()
    update, save = Mock(), Mock()
    monkeypatch.setattr(controller, "update_model", update)
    monkeypatch.setattr(controller, "save_model", save)
    original_get = controller._get_aggregated_result

    def get_result():
        # Baseline code has no round lock: expose its premature finalization deterministically.
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
            try:
                assert finalizing.wait(5), "finalization never started"
            finally:
                release.set()
            with pytest.raises(RuntimeError, match="refusing to update or save"):
                future.result(timeout=5)
        update.assert_not_called()
        save.assert_not_called()
        assert callbacks[0](FLModel(params={"a": 999.0})) is False


def test_closed_round_callback_cannot_change_next_round(monkeypatch):
    controller = prepare_controller(monkeypatch, FLModel(params={"a": 0.0}), rounds=2)
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
