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

"""Unit tests for JoblibModelParamPersistor and model_path support."""

import pytest

from nvflare.apis.app_validation import AppValidationKey
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnableKey


class TestJoblibModelParamPersistorInit:
    """Tests for JoblibModelParamPersistor initialization with model_path."""

    def test_init_without_model_path(self):
        """Init without model_path should work."""
        from nvflare.app_opt.sklearn.joblib_model_param_persistor import JoblibModelParamPersistor

        persistor = JoblibModelParamPersistor(initial_params={"n_clusters": 3})

        assert persistor.model_path is None

    def test_init_with_model_path_stores_path(self):
        """Init should store the model_path."""
        from nvflare.app_opt.sklearn.joblib_model_param_persistor import JoblibModelParamPersistor

        persistor = JoblibModelParamPersistor(
            initial_params={"n_clusters": 3},
            model_path="/data/pretrained/model.joblib",
        )

        assert persistor.model_path == "/data/pretrained/model.joblib"

    def test_init_with_model_path_only(self):
        """Init with only model_path should work (initial_params optional)."""
        from nvflare.app_opt.sklearn.joblib_model_param_persistor import JoblibModelParamPersistor

        persistor = JoblibModelParamPersistor(
            model_path="/data/pretrained/model.joblib",
        )

        assert persistor.initial_params == {}
        assert persistor.model_path == "/data/pretrained/model.joblib"


class TestJoblibModelParamPersistorLoadModel:
    """Tests for JoblibModelParamPersistor load_model behavior."""

    def test_load_model_no_params_raises_error(self, tmp_path):
        """load_model should raise ValueError when no params are available."""
        from nvflare.app_opt.sklearn.joblib_model_param_persistor import JoblibModelParamPersistor

        # Create persistor with no initial_params and no model_path
        persistor = JoblibModelParamPersistor(initial_params=None)

        # Manually set up save_path (normally done in _initialize)
        persistor.log_dir = str(tmp_path)
        persistor.save_path = str(tmp_path / "model_param.joblib")

        with pytest.raises(ValueError, match="No model parameters available"):
            persistor.load_model(FLContext())


@pytest.mark.parametrize("checkpoint_source", ["implicit", "explicit", "legacy"])
def test_pickle_checkpoint_load_requires_trusted_job_context(tmp_path, checkpoint_source):
    from joblib import dump

    from nvflare.app_opt.sklearn.joblib_model_param_persistor import JoblibModelParamPersistor

    checkpoint = tmp_path / "model_param.joblib"
    dump({"checkpoint": True}, checkpoint)
    params = {"n_clusters": 3}
    kwargs = {}
    if checkpoint_source == "explicit":
        kwargs["model_path"] = str(checkpoint)
    if checkpoint_source == "legacy":
        params["model_path"] = str(checkpoint)
    persistor = JoblibModelParamPersistor(initial_params=params, **kwargs)
    fl_ctx = FLContext()
    fl_ctx.set_prop(FLContextKey.APP_ROOT, str(tmp_path))
    persistor.handle_event(EventType.START_RUN, fl_ctx)

    if checkpoint_source == "implicit":
        assert persistor.load_model(fl_ctx)[ModelLearnableKey.WEIGHTS] == params
    else:
        with pytest.raises(PermissionError, match="requires BYOC authorization"):
            persistor.load_model(fl_ctx)

    fl_ctx.set_prop(FLContextKey.JOB_META, {AppValidationKey.BYOC: True})
    assert persistor.load_model(fl_ctx)[ModelLearnableKey.WEIGHTS] == {"checkpoint": True}
