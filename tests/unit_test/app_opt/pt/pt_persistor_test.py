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

"""Unit tests for PTFileModelPersistor."""

import errno
import json
import os
import stat
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import AppConstants
from nvflare.fuel.sec.audit import Auditor, AuditService


class TestPTFileModelPersistorInit:
    """Tests for PTFileModelPersistor initialization with source_ckpt_file_full_name."""

    def test_init_without_source_ckpt(self):
        """Init without source_ckpt should work."""
        try:
            import torch.nn as nn

            from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

            class SimpleNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(10, 2)

            model = SimpleNet()
            persistor = PTFileModelPersistor(model=model)

            assert persistor.source_ckpt_file_full_name is None
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_init_with_source_ckpt_no_existence_check(self):
        """Init with non-existent source_ckpt should NOT raise error (deferred to runtime)."""
        try:
            import torch.nn as nn

            from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

            class SimpleNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(10, 2)

            model = SimpleNet()
            # This path doesn't exist, but should NOT raise error at init time
            persistor = PTFileModelPersistor(
                model=model,
                source_ckpt_file_full_name="/nonexistent/path/model.pt",
            )

            assert persistor.source_ckpt_file_full_name == "/nonexistent/path/model.pt"
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_init_with_source_ckpt_stores_path(self):
        """Init should store the source_ckpt path."""
        try:
            import torch.nn as nn

            from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

            class SimpleNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(10, 2)

            model = SimpleNet()
            persistor = PTFileModelPersistor(
                model=model,
                source_ckpt_file_full_name="/data/pretrained/model.pt",
            )

            assert persistor.source_ckpt_file_full_name == "/data/pretrained/model.pt"
        except ImportError:
            pytest.skip("PyTorch not installed")

    @pytest.mark.parametrize(
        ("persistor_kwargs", "log_dir", "expected_error"),
        [
            ({}, "{outside}", "checkpoint directory"),
            ({}, "../outside", "checkpoint directory"),
            ({"global_model_file_name": "{outside}"}, None, "global model file name"),
            ({"global_model_file_name": "../outside.pt"}, None, "global model file name"),
            ({"global_model_file_name": "../checkpoints/model.pt"}, "models", "global model file name"),
            ({"best_global_model_file_name": "{outside}"}, None, "best global model file name"),
            ({"best_global_model_file_name": "../outside.pt"}, None, "best global model file name"),
            ({"best_global_model_file_name": "../checkpoints/model.pt"}, "models", "best global model file name"),
        ],
    )
    def test_initialize_rejects_unsafe_checkpoint_write_paths(
        self, tmp_path, persistor_kwargs, log_dir, expected_error
    ):
        from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

        app_root = tmp_path / "app"
        app_root.mkdir()
        outside = str(tmp_path / "outside")
        persistor_kwargs = {name: value.format(outside=outside) for name, value in persistor_kwargs.items()}
        fl_ctx = FLContext()
        fl_ctx.set_prop(FLContextKey.APP_ROOT, str(app_root), private=True, sticky=False)
        if log_dir:
            fl_ctx.set_prop(
                AppConstants.LOG_DIR,
                log_dir.format(outside=outside),
                private=True,
                sticky=False,
            )

        persistor = PTFileModelPersistor(**persistor_kwargs)
        persistor.system_panic = Mock()

        persistor._initialize(fl_ctx)

        persistor.system_panic.assert_called_once()
        call = persistor.system_panic.call_args
        assert expected_error in call.kwargs["reason"]
        assert call.kwargs["fl_ctx"] is fl_ctx
        assert not hasattr(persistor, "_ckpt_save_path")
        assert not hasattr(persistor, "_best_ckpt_save_path")

    def test_initialize_accepts_checkpoint_write_paths_under_app_root(self, tmp_path):
        from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

        app_root = tmp_path / "app"
        app_root.mkdir()
        fl_ctx = FLContext()
        fl_ctx.set_prop(FLContextKey.APP_ROOT, str(app_root), private=True, sticky=False)
        fl_ctx.set_prop(AppConstants.LOG_DIR, "models", private=True, sticky=False)

        persistor = PTFileModelPersistor(
            global_model_file_name="current/model.pt",
            best_global_model_file_name="best/model.pt",
        )
        persistor._initialize(fl_ctx)

        assert persistor.log_dir == str(app_root / "models")
        assert persistor._ckpt_save_path == str(app_root / "models" / "current" / "model.pt")
        assert persistor._best_ckpt_save_path == str(app_root / "models" / "best" / "model.pt")

    @pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink support required")
    @pytest.mark.parametrize(
        ("file_name_arg", "save_path_attr"),
        [
            ("global_model_file_name", "_ckpt_save_path"),
            ("best_global_model_file_name", "_best_ckpt_save_path"),
        ],
    )
    def test_initialize_allows_checkpoint_link_elsewhere_in_app_root(self, tmp_path, file_name_arg, save_path_attr):
        import torch

        from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

        app_root = tmp_path / "app"
        models = app_root / "models"
        checkpoints = app_root / "checkpoints"
        models.mkdir(parents=True)
        checkpoints.mkdir()
        alias = models / "current.pt"
        target = checkpoints / "current.pt"
        alias.symlink_to("../checkpoints/current.pt")
        fl_ctx = FLContext()
        fl_ctx.set_prop(FLContextKey.APP_ROOT, str(app_root), private=True, sticky=False)
        fl_ctx.set_prop(AppConstants.LOG_DIR, "models", private=True, sticky=False)
        persistor = PTFileModelPersistor(**{file_name_arg: alias.name})
        persistor.system_panic = Mock()

        persistor._initialize(fl_ctx)

        persistor.system_panic.assert_not_called()
        assert getattr(persistor, save_path_attr) == str(target)
        persistor.persistence_manager = _PersistenceManager({"model": {"weight": torch.tensor([1.0])}})
        persistor.persistence_manager.get_persist_model_format = Mock(return_value="PT")
        persistor.save_model_file(getattr(persistor, save_path_attr))
        assert target.is_file()
        assert persistor.get_model_inventory(fl_ctx)[alias.name].location == str(alias)

    @pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink support required")
    @pytest.mark.parametrize(
        ("file_name_arg", "save_path_attr"),
        [
            ("global_model_file_name", "_ckpt_save_path"),
            ("best_global_model_file_name", "_best_ckpt_save_path"),
        ],
    )
    def test_checkpoint_symlink_then_parent_saves_to_inventory_path(self, tmp_path, file_name_arg, save_path_attr):
        import torch

        from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

        app_root = tmp_path / "app"
        models = app_root / "models"
        checkpoints = app_root / "checkpoints"
        models.mkdir(parents=True)
        (checkpoints / "cache").mkdir(parents=True)
        (models / "latest").symlink_to("../checkpoints/cache", target_is_directory=True)
        file_name = "latest/../current.pt"
        target = checkpoints / "current.pt"
        fl_ctx = FLContext()
        fl_ctx.set_prop(FLContextKey.APP_ROOT, str(app_root), private=True, sticky=False)
        fl_ctx.set_prop(AppConstants.LOG_DIR, "models", private=True, sticky=False)
        persistor = PTFileModelPersistor(**{file_name_arg: file_name})
        persistor.system_panic = Mock()

        persistor._initialize(fl_ctx)

        persistor.system_panic.assert_not_called()
        assert getattr(persistor, save_path_attr) == str(target)
        persistor.persistence_manager = _PersistenceManager({"model": {"weight": torch.tensor([1.0])}})
        persistor.persistence_manager.get_persist_model_format = Mock(return_value="PT")
        persistor.save_model_file(getattr(persistor, save_path_attr))
        assert target.is_file()
        assert not (models / "current.pt").exists()
        inventory_path = persistor.get_model_inventory(fl_ctx)["current.pt"].location
        assert os.path.samefile(inventory_path, target)

    @pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink support required")
    @pytest.mark.parametrize("file_name_arg", ["global_model_file_name", "best_global_model_file_name"])
    def test_checkpoint_symlink_then_parent_rejects_escape(self, tmp_path, file_name_arg):
        from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

        app_root = tmp_path / "app"
        models = app_root / "models"
        models.mkdir(parents=True)
        outside = tmp_path / "outside"
        (outside / "cache").mkdir(parents=True)
        (models / "latest").symlink_to("../../outside/cache", target_is_directory=True)
        fl_ctx = FLContext()
        fl_ctx.set_prop(FLContextKey.APP_ROOT, str(app_root), private=True, sticky=False)
        fl_ctx.set_prop(AppConstants.LOG_DIR, "models", private=True, sticky=False)
        persistor = PTFileModelPersistor(**{file_name_arg: "latest/../current.pt"})
        persistor.system_panic = Mock()

        persistor._initialize(fl_ctx)

        persistor.system_panic.assert_called_once()
        assert file_name_arg.replace("_", " ") in persistor.system_panic.call_args.kwargs["reason"]
        assert not hasattr(persistor, "_ckpt_save_path")
        assert not hasattr(persistor, "_best_ckpt_save_path")
        assert not (outside / "current.pt").exists()

    @pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink support required")
    @pytest.mark.parametrize("file_name_arg", ["global_model_file_name", "best_global_model_file_name"])
    def test_initialize_rejects_checkpoint_link_outside_app_root(self, tmp_path, file_name_arg):
        from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

        app_root = tmp_path / "app"
        models = app_root / "models"
        models.mkdir(parents=True)
        outside = tmp_path / "outside.pt"
        alias = models / "current.pt"
        alias.symlink_to(outside)
        fl_ctx = FLContext()
        fl_ctx.set_prop(FLContextKey.APP_ROOT, str(app_root), private=True, sticky=False)
        fl_ctx.set_prop(AppConstants.LOG_DIR, "models", private=True, sticky=False)
        persistor = PTFileModelPersistor(**{file_name_arg: alias.name})
        persistor.system_panic = Mock()

        persistor._initialize(fl_ctx)

        assert file_name_arg.replace("_", " ") in persistor.system_panic.call_args.kwargs["reason"]
        assert not hasattr(persistor, "_ckpt_save_path")
        assert not outside.exists()


class _PersistenceManager:
    def __init__(self, persistence_dict):
        self.persistence_dict = persistence_dict
        self.updated_model = None

    def update(self, model):
        self.updated_model = model

    def to_persistence_dict(self):
        return self.persistence_dict


class TestPTFileModelPersistorSave:
    def test_partial_save_failures_preserve_checkpoint_and_clean_unique_temp_files(self, tmp_path, monkeypatch):
        import torch

        from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

        final_path = tmp_path / "model.pt"
        old_checkpoint = {"model": {"weight": torch.tensor([1.0])}}
        torch.save(old_checkpoint, final_path)
        old_bytes = final_path.read_bytes()

        persistor = PTFileModelPersistor()
        persistor.persistence_manager = _PersistenceManager({"model": {"weight": torch.tensor([2.0])}})
        temp_paths = []

        def partial_save(_checkpoint, temp_file):
            temp_file.write(b"partial checkpoint")
            temp_file.flush()
            temp_paths.append(temp_file.name)
            if len(temp_paths) == 1:
                raise OSError(errno.ENOSPC, "no space left on device")
            raise KeyboardInterrupt

        monkeypatch.setattr(torch, "save", partial_save)

        with pytest.raises(OSError, match="no space left on device"):
            persistor.save_model_file(str(final_path))
        with pytest.raises(KeyboardInterrupt):
            persistor.save_model_file(str(final_path))

        assert len(set(temp_paths)) == 2
        assert all(os.path.dirname(path) == str(tmp_path) for path in temp_paths)
        assert all(not os.path.exists(path) for path in temp_paths)
        assert final_path.read_bytes() == old_bytes
        loaded = torch.load(final_path, weights_only=True)
        assert torch.equal(loaded["model"]["weight"], old_checkpoint["model"]["weight"])

    def test_fsync_failure_preserves_checkpoint_and_cleans_temp_file(self, tmp_path, monkeypatch):
        import torch

        from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

        final_path = tmp_path / "model.pt"
        old_checkpoint = {"model": {"weight": torch.tensor([1.0])}}
        torch.save(old_checkpoint, final_path)
        old_bytes = final_path.read_bytes()

        persistor = PTFileModelPersistor()
        persistor.persistence_manager = _PersistenceManager({"model": {"weight": torch.tensor([2.0])}})

        def fail_fsync(_fd):
            raise OSError(errno.ENOSPC, "no space left on device")

        monkeypatch.setattr(os, "fsync", fail_fsync)

        with pytest.raises(OSError, match="no space left on device"):
            persistor.save_model_file(str(final_path))

        assert final_path.read_bytes() == old_bytes
        loaded = torch.load(final_path, weights_only=True)
        assert torch.equal(loaded["model"]["weight"], old_checkpoint["model"]["weight"])
        assert list(tmp_path.glob(f".{final_path.name}.*.tmp")) == []

    @pytest.mark.parametrize("checkpoint_kind", ["current", "best"])
    def test_successfully_replaces_current_and_best_checkpoints(self, checkpoint_kind, tmp_path, monkeypatch):
        import torch

        from nvflare.app_common.app_constant import AppConstants
        from nvflare.app_common.app_event_type import AppEventType
        from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

        final_path = tmp_path / f"{checkpoint_kind}.pt"
        torch.save({"model": {"weight": torch.tensor([1.0])}}, final_path)
        if os.name != "nt":
            final_path.chmod(0o640)

        new_checkpoint = {
            "model": {"weight": torch.tensor([2.0])},
            "train_conf": {"train": {"model": "SimpleNet"}},
        }
        model = object()
        persistence_manager = _PersistenceManager(new_checkpoint)
        persistor = PTFileModelPersistor()
        persistor.persistence_manager = persistence_manager
        persistor._ckpt_save_path = str(final_path)
        persistor._best_ckpt_save_path = str(final_path)

        real_fsync = os.fsync
        fsync_calls = []

        def recording_fsync(fd):
            fsync_calls.append(fd)
            return real_fsync(fd)

        monkeypatch.setattr(os, "fsync", recording_fsync)

        if checkpoint_kind == "current":
            persistor.save_model(model, fl_ctx=None)
        else:

            class _FLContext:
                @staticmethod
                def get_prop(key):
                    assert key == AppConstants.GLOBAL_MODEL
                    return model

            persistor.handle_event(AppEventType.GLOBAL_BEST_MODEL_AVAILABLE, _FLContext())

        loaded = torch.load(final_path, weights_only=True)
        assert torch.equal(loaded["model"]["weight"], new_checkpoint["model"]["weight"])
        assert loaded["train_conf"] == new_checkpoint["train_conf"]
        assert persistence_manager.updated_model is model
        assert len(fsync_calls) >= (1 if os.name == "nt" else 2)
        if os.name != "nt":
            assert stat.S_IMODE(final_path.stat().st_mode) == 0o640
        assert list(tmp_path.glob(f".{final_path.name}.*.tmp")) == []


@pytest.fixture
def model_audit_records(tmp_path, monkeypatch):
    path = tmp_path / "model-audit.jsonl"
    auditor = Auditor(str(path))
    monkeypatch.setattr(AuditService, "the_auditor", auditor)
    try:
        yield lambda: [json.loads(line) for line in path.read_text().splitlines()]
    finally:
        auditor.close()


def _stage_outcomes(records, stage):
    return [record["action"] for record in records if f"stage={stage}" in record.get("message", "").split()]


@pytest.mark.parametrize("scenario", ["missing", "corrupt", "invalid_device", "read"])
def test_pt_load_distinguishes_read_from_preparation(tmp_path, monkeypatch, model_audit_records, scenario):
    torch = pytest.importorskip("torch")
    from nvflare.app_common.abstract.model import ModelLearnableKey
    from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

    source = tmp_path / "private-source.pt"
    if scenario == "corrupt":
        source.write_bytes(b"not a checkpoint")
    elif scenario != "missing":
        torch.save({"weight": torch.tensor([42.0])}, source)
    persistor = PTFileModelPersistor(
        source_ckpt_file_full_name=str(source),
        load_device="invalid-device" if scenario == "invalid_device" else "cpu",
    )
    panic = Mock()
    monkeypatch.setattr(persistor, "system_panic", panic)

    result = persistor.load(FLContext())

    records = model_audit_records()
    if scenario == "read":
        panic.assert_not_called()
        assert result[ModelLearnableKey.WEIGHTS]["weight"].tolist() == [42.0]
        assert _stage_outcomes(records, "source_read") == ["model.load.stage_returned"]
    else:
        panic.assert_called_once()
        assert result is None
        expected = ["model.load.stage_error"] if scenario == "corrupt" else []
        assert _stage_outcomes(records, "source_read") == expected
        assert "status=no_value" in records[-1]["message"].split()
    assert records[-1]["action"] == "model.load.returned"
    assert all("private-source" not in json.dumps(record) for record in records)


@pytest.mark.parametrize("kind", ["save", "best_save"])
def test_pt_save_lazy_initialization_has_one_separate_load(tmp_path, model_audit_records, kind):
    torch = pytest.importorskip("torch")
    from nvflare.app_common.abstract.model import make_model_learnable
    from nvflare.app_common.app_event_type import AppEventType
    from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

    persistor = PTFileModelPersistor()
    destination = tmp_path / "private-checkpoint.pt"
    fl_ctx = FLContext()
    persistor._ckpt_save_path = str(destination)
    persistor._best_ckpt_save_path = str(destination)
    learned = make_model_learnable({"weight": torch.tensor([19.0])}, {})

    if kind == "best_save":
        fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, learned, private=True, sticky=False)
        persistor.handle_event(AppEventType.GLOBAL_BEST_MODEL_AVAILABLE, fl_ctx)
    else:
        persistor.save(learned, fl_ctx)

    loaded = torch.load(destination, weights_only=True)
    assert torch.equal(loaded["model"]["weight"], torch.tensor([19.0]))
    records = model_audit_records()
    assert [record["action"] for record in records if record["action"].endswith(".started")] == [
        f"model.{kind}.started",
        "model.load.started",
    ]
    assert _stage_outcomes(records, "initialize") == ["model.load.stage_returned"]
    assert _stage_outcomes(records, "replace") == [f"model.{kind}.stage_returned"]
    assert records[-1]["action"] == f"model.{kind}.returned"


@pytest.mark.skipif(os.name == "nt", reason="PyTorch checkpoint code skips directory synchronization on Windows")
def test_pt_replacement_stays_recorded_after_directory_sync_error(tmp_path, monkeypatch, model_audit_records):
    torch = pytest.importorskip("torch")
    from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

    destination = tmp_path / "private-checkpoint.pt"
    torch.save({"model": {"weight": torch.tensor([1.0])}}, destination)
    persistor = PTFileModelPersistor()
    persistor.persistence_manager = _PersistenceManager({"model": {"weight": torch.tensor([2.0])}})
    persistor._ckpt_save_path = str(destination)
    failure = OSError(errno.EIO, "private directory sync failure")
    real_fsync = os.fsync

    def sync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise failure
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", sync)
    model = object()
    with pytest.raises(OSError) as caught:
        persistor.save(model, FLContext())

    assert caught.value is failure
    assert persistor.persistence_manager.updated_model is model
    assert torch.equal(torch.load(destination, weights_only=True)["model"]["weight"], torch.tensor([2.0]))
    assert not list(tmp_path.glob(".*.tmp"))
    records = model_audit_records()
    stages = [record for record in records if record["action"].startswith("model.save.stage_")]
    replace = next(record for record in stages if "stage=replace" in record.get("message", "").split())
    failed_sync = next(record for record in stages if "stage=directory_sync" in record.get("message", "").split())
    assert replace["action"] == "model.save.stage_returned"
    assert failed_sync["action"] == "model.save.stage_error"
    assert records.index(replace) < records.index(failed_sync) < len(records) - 1
    assert records[-1]["action"] == "model.save.error"
    assert all("private directory sync failure" not in json.dumps(record) for record in records)


def test_pt_named_retrieval_inside_custom_save_does_not_audit_inventory_load(tmp_path, model_audit_records):
    torch = pytest.importorskip("torch")
    from nvflare.app_common.abstract.model import ModelLearnableKey
    from nvflare.app_common.abstract.model_persistor import ModelPersistor
    from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

    source = tmp_path / "private-inventory-model.pt"
    torch.save({"weight": torch.tensor([19.0])}, source)
    retrieval = PTFileModelPersistor(source_ckpt_file_full_name=str(source), load_device="cpu")
    retrieval.log_dir = str(tmp_path)

    class _SavingPersistor(ModelPersistor):
        def load_model(self, fl_ctx):
            raise NotImplementedError

        def save_model(self, model, fl_ctx):
            self.retrieved = retrieval.get_model(source.name, fl_ctx)

    assert retrieval.persistence_manager is None
    custom = _SavingPersistor()
    custom.save(None, FLContext())

    assert retrieval.persistence_manager is not None
    assert custom.retrieved[ModelLearnableKey.WEIGHTS]["weight"].tolist() == [19.0]
    assert [record["action"] for record in model_audit_records()] == [
        "model.save.started",
        "model.save.stage_returned",
        "model.save.returned",
    ]


@pytest.mark.parametrize("scenario", ["read", "missing", "corrupt", "decode_error"])
def test_pt_named_retrieval_observes_only_the_selected_file(tmp_path, model_audit_records, scenario):
    torch = pytest.importorskip("torch")
    from nvflare.app_common.abstract.model import ModelLearnableKey
    from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

    source = tmp_path / "private-source.pt"
    torch.save({"source_weight": torch.tensor([17.0])}, source)
    saved = tmp_path / "private-selected.pt"
    if scenario == "corrupt":
        saved.write_bytes(b"private-corrupt-checkpoint")
    elif scenario == "decode_error":
        torch.save(torch.tensor([21.0]), saved)
    else:
        torch.save(
            {"model": {"private-weight": torch.tensor([621.0])}, "meta_props": {"tag": "private-metadata"}}, saved
        )
    persistor = PTFileModelPersistor(
        source_ckpt_file_full_name=str(source), global_model_file_name=saved.name, load_device="cpu"
    )
    persistor.log_dir = str(tmp_path)
    name = "private-missing.pt" if scenario == "missing" else saved.name

    result = persistor.get(name, FLContext())

    assert persistor.persistence_manager is not None
    if scenario == "read":
        assert result[ModelLearnableKey.WEIGHTS]["private-weight"].tolist() == [621.0]
        assert result[ModelLearnableKey.META] == {"tag": "private-metadata"}
    else:
        assert result is None
    expected_reads = {
        "read": ["model.retrieve.stage_returned"],
        "missing": [],
        "corrupt": ["model.retrieve.stage_error"],
        "decode_error": ["model.retrieve.stage_returned"],
    }
    records = model_audit_records()
    assert _stage_outcomes(records, "file_read") == expected_reads[scenario]
    assert all(record["action"].startswith("model.retrieve.") for record in records)
    assert records[-1]["action"] == "model.retrieve.returned"
    assert ("status=value_returned" if scenario == "read" else "status=no_value") in records[-1]["message"].split()
    serialized_records = json.dumps(records)
    assert "private-" not in serialized_records
    assert str(tmp_path) not in serialized_records
    assert all("621" not in record.get("message", "") for record in records)
