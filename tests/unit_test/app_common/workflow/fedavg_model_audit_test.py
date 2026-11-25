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

import json

import pytest

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.abstract.model import ModelLearnableKey, make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.workflows.base_model_controller import BaseModelController
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.fuel.sec.audit import Auditor, AuditService


@pytest.fixture
def audit_path(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    writer = Auditor(str(path))
    monkeypatch.setattr(AuditService, "the_auditor", writer)
    yield path
    writer.close()


def _configure(controller, tmp_path, monkeypatch):
    controller.fl_ctx = FLContextManager(identity_name="server", job_id="job-1").new_context()
    controller.fl_ctx.set_prop(FLContextKey.EFFECTIVE_JOB_SCOPE_NAME, "training scope", private=True, sticky=False)
    monkeypatch.setattr(controller, "get_run_dir", lambda: str(tmp_path))
    return controller


def _observations(audit_path):
    records = [json.loads(line) for line in audit_path.read_text().splitlines()]
    observations = []
    for record in records:
        assert record["job"] == "job-1"
        assert record["scope"] == "training scope"
        assert record["user"] == "unknown"
        details = dict(part.split("=", 1) for part in record.get("message", "").split())
        observations.append((record["action"], details))
    return observations


def test_fedavg_initializes_without_a_file_then_saves_and_reads_one(audit_path, tmp_path, monkeypatch):
    filename = "private-model-filename"
    controller = _configure(FedAvg(persistor_id="", save_filename=filename), tmp_path, monkeypatch)
    assert controller.load_model().params == {}
    observed = _observations(audit_path)
    assert ("model.load.stage_returned", {"stage": "initialize", "reason": "file_missing"}) in observed
    assert ("model.load.stage_returned", {"stage": "previous_read"}) not in observed

    original = FLModel(params={"private-model-parameter": [11, 12]}, meta={"tag": "private-model-metadata"})
    assert controller.save_model(original) is None
    loaded = controller.load_model()
    assert loaded.params == original.params
    assert loaded.meta == original.meta

    observed = _observations(audit_path)
    assert observed.count(("model.load.started", {})) == 2
    assert observed.count(("model.save.started", {})) == 1
    assert observed.count(("model.save.stage_returned", {"stage": "write"})) == 1
    assert observed.count(("model.load.stage_returned", {"stage": "previous_read"})) == 1
    assert observed.count(("model.load.returned", {"status": "value_returned"})) == 2
    assert ("model.save.returned", {}) in observed
    for private in (filename, str(tmp_path), "private-model-parameter", "private-model-metadata"):
        assert private not in audit_path.read_text()


def test_run_initializes_supplied_model_without_claiming_save_when_aborted(audit_path, tmp_path, monkeypatch):
    source = FLModel(params={"weight": 13})
    controller = _configure(
        FedAvg(num_clients=1, num_rounds=1, model=source, save_filename=None), tmp_path, monkeypatch
    )
    controller.abort_signal = Signal()
    sent = []

    def send_and_abort(**kwargs):
        sent.append(kwargs["data"])
        controller.abort_signal.trigger(True)

    monkeypatch.setattr(controller, "sample_clients", lambda _: ["site-1"])
    monkeypatch.setattr(controller, "event", lambda _: None)
    monkeypatch.setattr(controller, "send_model", send_and_abort)
    monkeypatch.setattr(controller, "get_num_standing_tasks", lambda: 1)

    controller.run()

    assert len(sent) == 1
    assert sent[0] is source
    assert _observations(audit_path) == [
        ("model.load.started", {}),
        ("model.load.stage_returned", {"stage": "initialize", "reason": "configured_model"}),
        ("model.load.returned", {"status": "value_returned"}),
    ]


class _BaseController(BaseModelController):
    def run(self):
        raise AssertionError("training is not exercised by this test")


@pytest.mark.parametrize("use_fedavg", [False, True])
def test_controller_records_no_destination_without_claiming_file_activity(
    audit_path, tmp_path, monkeypatch, use_fedavg
):
    controller = FedAvg(persistor_id="", save_filename=None) if use_fedavg else _BaseController(persistor_id="")
    _configure(controller, tmp_path, monkeypatch)
    events = []
    monkeypatch.setattr(controller, "event", events.append)

    model = controller.load_model()
    assert model.params == {}
    assert controller.save_model(model) is None

    if use_fedavg:
        assert not events
    else:
        global_model = controller.fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
        assert global_model[ModelLearnableKey.WEIGHTS] is model.params
        assert events == [AppEventType.INITIAL_MODEL_LOADED]
    assert list(tmp_path.iterdir()) == [audit_path]
    observed = _observations(audit_path)
    assert ("model.load.stage_returned", {"stage": "initialize", "reason": "no_destination"}) in observed
    assert ("model.load.returned", {"status": "value_returned"}) in observed
    assert ("model.save.skipped", {"reason": "no_destination"}) in observed
    assert ("model.save.returned", {}) not in observed
    assert all(details.get("stage") not in ("previous_read", "write", "implementation") for _, details in observed)


class _TestPersistor(ModelPersistor):
    def __init__(self):
        super().__init__()
        self.loaded = make_model_learnable({"weight": 42}, {})
        self.saved = []

    def load_model(self, fl_ctx):
        return self.loaded

    def save_model(self, model, fl_ctx):
        self.saved.append(model)


def test_configured_persistor_owns_the_only_lifecycle(audit_path, tmp_path, monkeypatch):
    controller = _configure(FedAvg(save_filename="must-not-read-or-write"), tmp_path, monkeypatch)
    controller.persistor = _TestPersistor()
    events = []
    monkeypatch.setattr(controller, "event", events.append)

    model = controller.load_model()
    controller.save_model(model)

    assert model.params == {"weight": 42}
    assert [saved[ModelLearnableKey.WEIGHTS] for saved in controller.persistor.saved] == [model.params]
    assert events == [
        AppEventType.INITIAL_MODEL_LOADED,
        AppEventType.BEFORE_LEARNABLE_PERSIST,
        AppEventType.AFTER_LEARNABLE_PERSIST,
    ]
    observed = _observations(audit_path)
    assert observed.count(("model.load.started", {})) == 1
    assert observed.count(("model.save.started", {})) == 1
    assert list(tmp_path.iterdir()) == [audit_path]


def test_pre_read_logging_error_records_no_read(audit_path, tmp_path, monkeypatch):
    path = tmp_path / "private-checkpoint"
    path.write_bytes(b"must not be read")
    controller = _configure(FedAvg(persistor_id="", save_filename=path.name), tmp_path, monkeypatch)
    error = RuntimeError("private-logging-error")

    def fail_logging(message):
        raise error

    def fail_if_called(filepath):
        pytest.fail("file helper must not run after logging failed")

    monkeypatch.setattr(controller, "info", fail_logging)
    monkeypatch.setattr(controller, "load_model_file", fail_if_called)
    with pytest.raises(RuntimeError) as caught:
        controller.load_model()
    assert caught.value is error

    observed = _observations(audit_path)
    assert ("model.load.error", {"stage": "prepare"}) in observed
    assert all(details.get("stage") != "previous_read" for _, details in observed)
    assert "private-logging-error" not in audit_path.read_text()


@pytest.mark.parametrize("fail_metadata", [False, True])
def test_pytorch_records_parameter_write_before_metadata_result(audit_path, tmp_path, monkeypatch, fail_metadata):
    torch = pytest.importorskip("torch")
    from nvflare.app_opt.pt import fedavg

    path = tmp_path / "private-params.pt"
    sidecar = tmp_path / "private-params.pt.metadata"
    controller = _configure(fedavg.PTFedAvg(persistor_id="", save_filename=path.name), tmp_path, monkeypatch)
    params = {"private-tensor": torch.tensor([21, 22])}
    model = FLModel(params=params, meta={"private-metadata": "private-sidecar-value"})
    error = RuntimeError("private-sidecar-error")

    if fail_metadata:

        def fail_writing_metadata(value, filepath):
            assert value is model
            assert value.params == {}
            assert filepath == str(sidecar)
            assert torch.equal(torch.load(path, weights_only=True)["private-tensor"], params["private-tensor"])
            raise error

        monkeypatch.setattr(fedavg.fobs, "dumpf", fail_writing_metadata)
        with pytest.raises(RuntimeError) as caught:
            controller.save_model(model)
        assert caught.value is error
    else:
        assert controller.save_model(model) is None

    assert model.params is params
    assert torch.equal(torch.load(path, weights_only=True)["private-tensor"], params["private-tensor"])
    assert sidecar.exists() is not fail_metadata
    saved = _observations(audit_path)
    write = ("model.save.stage_returned", {"stage": "parameters_write"})
    metadata_outcome = "stage_error" if fail_metadata else "stage_returned"
    metadata = (f"model.save.{metadata_outcome}", {"stage": "metadata_write"})
    assert saved.index(write) < saved.index(metadata)
    assert saved.count(("model.save.started", {})) == 1
    assert (("model.save.returned", {}) in saved) is not fail_metadata
    assert (("model.save.error", {"stage": "implementation"}) in saved) is fail_metadata

    loaded = controller.load_model()
    assert torch.equal(loaded.params["private-tensor"], params["private-tensor"])
    assert loaded.meta == ({} if fail_metadata else model.meta)
    observed = _observations(audit_path)
    assert ("model.load.stage_returned", {"stage": "parameters_read"}) in observed
    assert (("model.load.stage_returned", {"stage": "metadata_read"}) in observed) is not fail_metadata

    audit_size = audit_path.stat().st_size
    controller.load_model_file(str(path))
    assert audit_path.stat().st_size == audit_size
    for private in (path.name, str(tmp_path), "private-tensor", "private-sidecar-value", "private-sidecar-error"):
        assert private not in audit_path.read_text()


def test_named_evaluation_read_inside_save_is_not_a_training_stage(audit_path, tmp_path, monkeypatch):
    filepath = str(tmp_path / "validation-model")
    controller = _configure(FedAvg(persistor_id=""), tmp_path, monkeypatch)
    controller.save_model_file(FLModel(params={"validation_weight": 42}), filepath)

    class EvaluatingPersistor(_TestPersistor):
        def get_model(self, model_file, fl_ctx):
            assert model_file == "validation"
            return controller.load_model_file(filepath)

        def save_model(self, model, fl_ctx):
            self.validation_model = self.get("validation", fl_ctx)
            super().save_model(model, fl_ctx)

    persistor = EvaluatingPersistor()
    persistor.save(persistor.loaded, controller.fl_ctx)

    assert persistor.validation_model.params == {"validation_weight": 42}
    assert persistor.saved == [persistor.loaded]
    assert _observations(audit_path) == [
        ("model.save.started", {}),
        ("model.retrieve.started", {}),
        ("model.retrieve.stage_returned", {"stage": "implementation"}),
        ("model.retrieve.returned", {"status": "value_returned"}),
        ("model.save.stage_returned", {"stage": "implementation"}),
        ("model.save.returned", {}),
    ]


def test_save_does_not_claim_read_even_for_the_same_controller(audit_path, tmp_path, monkeypatch):
    reference_file = str(tmp_path / "validation-model")
    writer = FedAvg(persistor_id="")
    writer.save_model_file(FLModel(params={"validation_weight": 42}), reference_file)

    class CheckingFedAvg(FedAvg):
        def save_model_file(self, model, filepath):
            self.validation_model = self.load_model_file(reference_file)
            super().save_model_file(model, filepath)

    controller = _configure(CheckingFedAvg(persistor_id="", save_filename="training-model"), tmp_path, monkeypatch)
    source = FLModel(params={"weight": 24})
    controller.save_model(source)

    assert controller.validation_model.params == {"validation_weight": 42}
    assert controller.load_model_file(str(tmp_path / "training-model")).params == source.params
    assert _observations(audit_path) == [
        ("model.save.started", {}),
        ("model.save.stage_returned", {"stage": "prepare"}),
        ("model.save.stage_returned", {"stage": "write"}),
        ("model.save.stage_returned", {"stage": "implementation"}),
        ("model.save.returned", {}),
    ]


def test_load_claims_only_its_own_file_read(audit_path, tmp_path, monkeypatch):
    other = FedAvg(persistor_id="")
    other_file = str(tmp_path / "other-model")
    main_file = tmp_path / "training-model"
    other.save_model_file(FLModel(params={"other_weight": 42}), other_file)
    other.save_model_file(FLModel(params={"weight": 24}), str(main_file))

    class CheckingFedAvg(FedAvg):
        def load_model_file(self, filepath):
            self.other_model = other.load_model_file(other_file)
            return super().load_model_file(filepath)

    controller = _configure(CheckingFedAvg(persistor_id="", save_filename=main_file.name), tmp_path, monkeypatch)

    assert controller.load_model().params == {"weight": 24}
    assert controller.other_model.params == {"other_weight": 42}
    assert _observations(audit_path).count(("model.load.stage_returned", {"stage": "previous_read"})) == 1
