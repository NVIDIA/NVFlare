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

import os

import numpy as np
import pytest

from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.workspace import Workspace
from nvflare.app_common.abstract.model import ModelLearnableKey, make_model_learnable
from nvflare.app_common.ccwf.comps.np_trainer import NPTrainer as CCWFNPTrainer
from nvflare.app_common.np.constants import NPConstants
from nvflare.app_common.np.np_trainer import NPTrainer


class _Engine:
    def __init__(self, workspace):
        self._workspace = workspace

    def get_workspace(self):
        return self._workspace


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
        from nvflare.app_common.np.np_model_persistor import NPModelPersistor

        persistor = NPModelPersistor()

        assert persistor.source_ckpt_file_full_name is None

    def test_init_with_source_ckpt_stores_path(self):
        """Init should store the source_ckpt path."""
        from nvflare.app_common.np.np_model_persistor import NPModelPersistor

        persistor = NPModelPersistor(
            source_ckpt_file_full_name="/data/pretrained/model.npy",
        )

        assert persistor.source_ckpt_file_full_name == "/data/pretrained/model.npy"

    def test_init_with_model_and_ckpt(self):
        """Init with both model and source_ckpt should work."""
        from nvflare.app_common.np.np_model_persistor import NPModelPersistor

        persistor = NPModelPersistor(
            model=[[1, 2], [3, 4]],
            source_ckpt_file_full_name="/data/pretrained/model.npy",
        )

        assert persistor.model == [[1, 2], [3, 4]]
        assert persistor.source_ckpt_file_full_name == "/data/pretrained/model.npy"

    def test_save_model_accepts_relative_model_path(self, tmp_path):
        from nvflare.app_common.np.np_model_persistor import NPModelPersistor

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
        from nvflare.app_common.np.np_model_persistor import NPModelPersistor

        fl_ctx = _make_fl_ctx(tmp_path)
        persistor = NPModelPersistor(**kwargs)

        with pytest.raises(ValueError, match="must (be relative|stay inside)"):
            persistor.save_model(_model_learnable(), fl_ctx)


class TestNPFileModelPersistorInit:
    """Tests for NPFileModelPersistor initialization with source_ckpt_file_full_name."""

    def test_init_without_source_ckpt(self):
        """Init without source_ckpt should work."""
        from nvflare.app_common.ccwf.comps.np_file_model_persistor import NPFileModelPersistor

        persistor = NPFileModelPersistor()

        assert persistor.source_ckpt_file_full_name is None

    def test_init_with_source_ckpt_stores_path(self):
        """Init should store the source_ckpt path."""
        from nvflare.app_common.ccwf.comps.np_file_model_persistor import NPFileModelPersistor

        persistor = NPFileModelPersistor(
            source_ckpt_file_full_name="/data/pretrained/model.npy",
        )

        assert persistor.source_ckpt_file_full_name == "/data/pretrained/model.npy"

    def test_save_model_accepts_relative_model_path(self, tmp_path):
        from nvflare.app_common.ccwf.comps.np_file_model_persistor import NPFileModelPersistor

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
        from nvflare.app_common.ccwf.comps.np_file_model_persistor import NPFileModelPersistor

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
