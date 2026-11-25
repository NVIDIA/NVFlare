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

"""Unit tests for numpy model persistors."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.workspace import Workspace
from nvflare.app_common.abstract.model import ModelLearnableKey, make_model_learnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.ccwf.comps.np_file_model_persistor import NPFileModelPersistor
from nvflare.app_common.ccwf.comps.np_trainer import NPTrainer as CCWFNPTrainer
from nvflare.app_common.np.constants import NPConstants
from nvflare.app_common.np.np_model_persistor import NPModelPersistor
from nvflare.app_common.np.np_trainer import NPTrainer
from nvflare.fuel.sec.audit import Auditor, AuditService


class _Engine:
    def __init__(self, workspace):
        self._workspace = workspace

    def get_workspace(self):
        return self._workspace

    def fire_event(self, event_type, fl_ctx):
        pass


def _make_fl_ctx(tmp_path):
    (tmp_path / "startup").mkdir()
    (tmp_path / "local").mkdir()
    workspace = Workspace(root_dir=str(tmp_path), site_name="site-1")
    fl_ctx = FLContext()
    fl_ctx.put(key=ReservedKey.ENGINE, value=_Engine(workspace), private=True, sticky=False)
    fl_ctx.put(key=ReservedKey.RUN_NUM, value="job-1", private=False, sticky=True)
    fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace, private=True, sticky=False)
    fl_ctx.set_prop(FLContextKey.CURRENT_RUN, "job-1", private=False, sticky=True)
    return fl_ctx


def _model_learnable():
    return make_model_learnable(
        weights={NPConstants.NUMPY_KEY: np.array([[1, 2], [3, 4]], dtype=np.float32)}, meta_props={}
    )


class TestNPModelPersistorInit:
    """Tests for NPModelPersistor initialization with source_ckpt_file_full_name."""

    def test_init_without_source_ckpt(self):
        """Init without source_ckpt should work."""
        persistor = NPModelPersistor()

        assert persistor.source_ckpt_file_full_name is None

    def test_init_with_source_ckpt_stores_path(self):
        """Init should store the source_ckpt path."""
        persistor = NPModelPersistor(
            source_ckpt_file_full_name="/data/pretrained/model.npy",
        )

        assert persistor.source_ckpt_file_full_name == "/data/pretrained/model.npy"

    def test_init_with_model_and_ckpt(self):
        """Init with both model and source_ckpt should work."""
        persistor = NPModelPersistor(
            model=[[1, 2], [3, 4]],
            source_ckpt_file_full_name="/data/pretrained/model.npy",
        )

        assert persistor.model == [[1, 2], [3, 4]]
        assert persistor.source_ckpt_file_full_name == "/data/pretrained/model.npy"

    def test_save_model_accepts_relative_model_path(self, tmp_path):
        fl_ctx = _make_fl_ctx(tmp_path)
        persistor = NPModelPersistor(model_dir="models", model_name="server.npy")
        persistor.save_model(_model_learnable(), fl_ctx)

        result_root = fl_ctx.get_workspace().get_result_root(fl_ctx.get_job_id())
        assert os.path.isfile(os.path.join(result_root, "models", "server.npy"))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"model_dir": "../outside"},
            {"model_name": "../../outside.npy"},
            {"model_dir": "/tmp/outside"},
        ],
    )
    def test_save_model_rejects_escaping_model_path(self, tmp_path, kwargs):
        fl_ctx = _make_fl_ctx(tmp_path)
        persistor = NPModelPersistor(**kwargs)

        with pytest.raises(ValueError, match="must (be relative|stay inside)"):
            persistor.save_model(_model_learnable(), fl_ctx)


class TestNPFileModelPersistorInit:
    """Tests for NPFileModelPersistor initialization with source_ckpt_file_full_name."""

    def test_init_without_source_ckpt(self):
        """Init without source_ckpt should work."""
        persistor = NPFileModelPersistor()

        assert persistor.source_ckpt_file_full_name is None

    def test_init_with_source_ckpt_stores_path(self):
        """Init should store the source_ckpt path."""
        persistor = NPFileModelPersistor(
            source_ckpt_file_full_name="/data/pretrained/model.npy",
        )

        assert persistor.source_ckpt_file_full_name == "/data/pretrained/model.npy"

    def test_save_model_accepts_relative_model_path(self, tmp_path):
        fl_ctx = _make_fl_ctx(tmp_path)
        persistor = NPFileModelPersistor(model_dir="models", last_global_model_file_name="last.npy")
        persistor.save_model(_model_learnable(), fl_ctx)

        run_dir = fl_ctx.get_workspace().get_run_dir(fl_ctx.get_job_id())
        assert os.path.isfile(os.path.join(run_dir, "models", "last.npy"))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"model_dir": "../outside"},
            {"last_global_model_file_name": "../../outside.npy"},
            {"best_global_model_file_name": "/tmp/outside.npy"},
        ],
    )
    def test_save_model_rejects_escaping_model_path(self, tmp_path, kwargs):
        fl_ctx = _make_fl_ctx(tmp_path)
        persistor = NPFileModelPersistor(**kwargs)
        model = _model_learnable()

        with pytest.raises(ValueError, match="must (be relative|stay inside)"):
            if "best_global_model_file_name" in kwargs:
                persistor._save(fl_ctx, model, persistor.best_global_model_file_name)
            else:
                persistor.save_model(model, fl_ctx)


@pytest.mark.parametrize(
    "trainer_cls",
    [NPTrainer, CCWFNPTrainer],
)
def test_np_trainers_accept_relative_model_path(tmp_path, trainer_cls):
    fl_ctx = _make_fl_ctx(tmp_path)
    trainer = trainer_cls(model_dir="models", model_name="local.npy")
    weights = _model_learnable()[ModelLearnableKey.WEIGHTS][NPConstants.NUMPY_KEY]

    trainer._save_local_model(fl_ctx, {NPConstants.NUMPY_KEY: weights})

    expected_root = fl_ctx.get_workspace().get_result_root(fl_ctx.get_job_id())
    if trainer_cls is CCWFNPTrainer:
        expected_root = fl_ctx.get_workspace().get_run_dir(fl_ctx.get_job_id())
    assert os.path.isfile(os.path.join(expected_root, "models", "local.npy"))


@pytest.mark.parametrize(
    "trainer_cls",
    [NPTrainer, CCWFNPTrainer],
)
def test_np_trainers_reject_escaping_model_path(tmp_path, trainer_cls):
    fl_ctx = _make_fl_ctx(tmp_path)
    trainer = trainer_cls(model_dir="../outside")
    weights = _model_learnable()[ModelLearnableKey.WEIGHTS][NPConstants.NUMPY_KEY]

    with pytest.raises(ValueError, match="must stay inside"):
        trainer._save_local_model(fl_ctx, {NPConstants.NUMPY_KEY: weights})


@pytest.fixture
def model_audit_records(tmp_path, monkeypatch):
    path = tmp_path / "model-audit.jsonl"
    auditor = Auditor(str(path))
    monkeypatch.setattr(AuditService, "the_auditor", auditor)
    try:
        yield lambda: [json.loads(line) for line in path.read_text().splitlines()]
    finally:
        auditor.close()


def _outcomes(records):
    return [(record["action"], record.get("message", "").split()) for record in records]


@pytest.mark.parametrize("scenario", ["missing_source", "corrupt_source", "fallback"])
def test_np_load_audits_actual_read_and_fallback(tmp_path, model_audit_records, scenario):
    fl_ctx = _make_fl_ctx(tmp_path)
    source = tmp_path / "private-source.npy"
    if scenario == "corrupt_source":
        source.write_bytes(b"not an array")
    persistor = NPModelPersistor(
        model=[[593.0]], source_ckpt_file_full_name=str(source) if scenario != "fallback" else None
    )

    if scenario == "fallback":
        result = persistor.load(fl_ctx)
        np.testing.assert_array_equal(result[ModelLearnableKey.WEIGHTS][NPConstants.NUMPY_KEY], [[593.0]])
    else:
        with pytest.raises(ValueError, match="[Ss]ource checkpoint"):
            persistor.load(fl_ctx)

    records = model_audit_records()
    outcomes = _outcomes(records)
    reads = [(action, stage) for action, details in outcomes for stage in details if stage.endswith("_read")]
    if scenario == "missing_source":
        assert reads == []
    elif scenario == "corrupt_source":
        assert reads == [("model.load.stage_error", "stage=source_read")]
    else:
        assert reads == [("model.load.stage_error", "stage=previous_read")]
    assert any(
        action == "model.load.stage_returned" and "stage=initialize" in details for action, details in outcomes
    ) == (scenario == "fallback")
    assert records[-1]["action"] == ("model.load.returned" if scenario == "fallback" else "model.load.error")
    assert all(
        "private-source" not in json.dumps(record) and "593" not in record.get("message", "") for record in records
    )


@pytest.mark.parametrize("explicit_source", [False, True])
def test_np_logging_failure_does_not_claim_read_error(tmp_path, monkeypatch, model_audit_records, explicit_source):
    fl_ctx = _make_fl_ctx(tmp_path)
    path = Path(fl_ctx.get_workspace().get_run_dir(fl_ctx.get_job_id())) / "models" / "server.npy"
    path.parent.mkdir(parents=True)
    np.save(path, [[23.0]])
    persistor = NPModelPersistor(source_ckpt_file_full_name=str(path) if explicit_source else None)
    failure = RuntimeError("private logging error")

    def fail_loading_message(_ctx, message, **_kwargs):
        if message.startswith(("Loading model from source checkpoint:", "Loaded model from ")):
            raise failure

    monkeypatch.setattr(persistor, "log_info", fail_loading_message)
    if explicit_source:
        with pytest.raises(ValueError) as caught:
            persistor.load(fl_ctx)
        assert caught.value.__cause__ is failure
    else:
        result = persistor.load(fl_ctx)
        np.testing.assert_array_equal(result[ModelLearnableKey.WEIGHTS][NPConstants.NUMPY_KEY], [[23.0]])

    outcomes = _outcomes(model_audit_records())
    reads = [(action, stage) for action, details in outcomes for stage in details if stage.endswith("_read")]
    assert reads == ([] if explicit_source else [("model.load.stage_returned", "stage=previous_read")])
    assert not any("stage=initialize" in details for _, details in outcomes)


def test_np_regular_and_best_save_write_separately(tmp_path, model_audit_records):
    fl_ctx = _make_fl_ctx(tmp_path)
    model = _model_learnable()
    persistor = NPFileModelPersistor()
    fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, model, private=True, sticky=False)

    persistor.save(model, fl_ctx)
    persistor.handle_event(AppEventType.GLOBAL_BEST_MODEL_AVAILABLE, fl_ctx)

    directory = Path(fl_ctx.get_workspace().get_run_dir(fl_ctx.get_job_id())) / persistor.model_dir
    for filename in (persistor.last_global_model_file_name, persistor.best_global_model_file_name):
        saved = np.load(directory / filename, allow_pickle=False)
        np.testing.assert_array_equal(saved, model[ModelLearnableKey.WEIGHTS][NPConstants.NUMPY_KEY])
    outcomes = _outcomes(model_audit_records())
    assert [action for action, _ in outcomes if action.endswith(".started")] == [
        "model.save.started",
        "model.best_save.started",
    ]
    assert [action for action, details in outcomes if "stage=write" in details] == [
        "model.save.stage_returned",
        "model.best_save.stage_returned",
    ]


def test_np_best_save_write_error_keeps_original_exception(tmp_path, monkeypatch, model_audit_records):
    fl_ctx = _make_fl_ctx(tmp_path)
    fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, _model_learnable(), private=True, sticky=False)
    failure = OSError("private write error")

    def fail_write(*_args):
        raise failure

    monkeypatch.setattr(np, "save", fail_write)
    with pytest.raises(OSError) as caught:
        NPFileModelPersistor().handle_event(AppEventType.GLOBAL_BEST_MODEL_AVAILABLE, fl_ctx)

    assert caught.value is failure
    records = model_audit_records()
    outcomes = _outcomes(records)
    assert [action for action, details in outcomes if "stage=write" in details] == ["model.best_save.stage_error"]
    assert records[-1]["action"] == "model.best_save.error"
    assert all("private write error" not in json.dumps(record) for record in records)


@pytest.mark.parametrize("scenario", ["read", "missing", "corrupt"])
def test_numpy_named_retrieval_observes_only_the_selected_file(tmp_path, model_audit_records, scenario):
    fl_ctx = _make_fl_ctx(tmp_path)
    persistor = NPFileModelPersistor(last_global_model_file_name="private-selected.npy")
    run_dir = fl_ctx.get_workspace().get_run_dir(fl_ctx.get_job_id())
    saved = Path(run_dir) / persistor.model_dir / persistor.last_global_model_file_name
    saved.parent.mkdir(parents=True)
    if scenario == "corrupt":
        saved.write_bytes(b"private-corrupt-checkpoint")
    else:
        np.save(saved, [[621.0]])
    name = "private-missing" if scenario == "missing" else saved.stem

    result = persistor.get(name, fl_ctx)

    if scenario == "read":
        np.testing.assert_array_equal(result[ModelLearnableKey.WEIGHTS][NPConstants.NUMPY_KEY], [[621.0]])
    else:
        assert result is None
    expected_reads = {
        "read": ["model.retrieve.stage_returned"],
        "missing": [],
        "corrupt": ["model.retrieve.stage_error"],
    }
    records = model_audit_records()
    file_reads = [action for action, details in _outcomes(records) if "stage=file_read" in details]
    assert file_reads == expected_reads[scenario]
    assert records[-1]["action"] == "model.retrieve.returned"
    assert ("status=value_returned" if scenario == "read" else "status=no_value") in records[-1]["message"].split()
    serialized_records = json.dumps(records)
    assert "private-" not in serialized_records
    assert str(tmp_path) not in serialized_records
    assert all("621" not in record.get("message", "") for record in records)
