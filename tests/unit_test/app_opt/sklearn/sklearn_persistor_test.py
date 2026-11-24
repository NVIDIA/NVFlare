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
        import pytest

        from nvflare.app_opt.sklearn.joblib_model_param_persistor import JoblibModelParamPersistor

        # Create persistor with no initial_params and no model_path
        persistor = JoblibModelParamPersistor(initial_params=None)

        # Manually set up save_path (normally done in _initialize)
        persistor.log_dir = str(tmp_path)
        persistor.save_path = str(tmp_path / "model_param.joblib")

        # Mock FLContext
        class MockFLContext:
            pass

        fl_ctx = MockFLContext()

        # load_model should raise ValueError when no params are available
        with pytest.raises(ValueError, match="No model parameters available"):
            persistor.load_model(fl_ctx)
