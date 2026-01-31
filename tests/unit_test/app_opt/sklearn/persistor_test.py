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

"""Unit tests for JoblibModelParamPersistor source_ckpt_file_full_name support."""


class TestJoblibModelParamPersistorInit:
    """Tests for JoblibModelParamPersistor initialization with source_ckpt_file_full_name."""

    def test_init_without_source_ckpt(self):
        """Init without source_ckpt should work."""
        from nvflare.app_opt.sklearn.joblib_model_param_persistor import JoblibModelParamPersistor

        persistor = JoblibModelParamPersistor(initial_params={"n_clusters": 3})

        assert persistor.source_ckpt_file_full_name is None

    def test_init_with_source_ckpt_stores_path(self):
        """Init should store the source_ckpt path."""
        from nvflare.app_opt.sklearn.joblib_model_param_persistor import JoblibModelParamPersistor

        persistor = JoblibModelParamPersistor(
            initial_params={"n_clusters": 3},
            source_ckpt_file_full_name="/data/pretrained/model.joblib",
        )

        assert persistor.source_ckpt_file_full_name == "/data/pretrained/model.joblib"

    def test_init_without_initial_params(self):
        """Init without initial_params should work (params now optional)."""
        from nvflare.app_opt.sklearn.joblib_model_param_persistor import JoblibModelParamPersistor

        persistor = JoblibModelParamPersistor(
            source_ckpt_file_full_name="/data/pretrained/model.joblib",
        )

        assert persistor.initial_params is None
        assert persistor.source_ckpt_file_full_name == "/data/pretrained/model.joblib"
