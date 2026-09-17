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
import os
import stat

import pytest


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
