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

import os

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext

xgboost = pytest.importorskip("xgboost")

from nvflare.app_opt.xgboost.histogram_based.executor import FedXGBHistogramExecutor  # noqa: E402
from nvflare.app_opt.xgboost.histogram_based_v2.runners.xgb_client_runner import XGBClientRunner  # noqa: E402
from nvflare.app_opt.xgboost.tree_based.model_persistor import XGBModelPersistor  # noqa: E402


def test_histogram_executor_accepts_relative_model_file_name(tmp_path):
    executor = FedXGBHistogramExecutor(
        num_rounds=1,
        early_stopping_rounds=1,
        xgb_params={},
        data_loader_id="data_loader",
        model_file_name="models/model.json",
    )

    assert executor._get_model_path(str(tmp_path)) == os.path.realpath(tmp_path / "models" / "model.json")


@pytest.mark.parametrize("model_file_name", ["/tmp/model.json", "../model.json"])
def test_histogram_executor_rejects_escaping_model_file_name(tmp_path, model_file_name):
    executor = FedXGBHistogramExecutor(
        num_rounds=1,
        early_stopping_rounds=1,
        xgb_params={},
        data_loader_id="data_loader",
        model_file_name=model_file_name,
    )

    with pytest.raises(ValueError, match="must (be relative|stay inside)"):
        executor._get_model_path(str(tmp_path))


def test_histogram_v2_runner_accepts_relative_model_file_name(tmp_path):
    runner = XGBClientRunner(data_loader_id="data_loader", model_file_name="models/model.json")
    runner._model_dir = str(tmp_path)

    assert runner._get_model_path() == os.path.realpath(tmp_path / "models" / "model.json")


@pytest.mark.parametrize("model_file_name", ["/tmp/model.json", "../model.json"])
def test_histogram_v2_runner_rejects_escaping_model_file_name(tmp_path, model_file_name):
    runner = XGBClientRunner(data_loader_id="data_loader", model_file_name=model_file_name)
    runner._model_dir = str(tmp_path)

    with pytest.raises(ValueError, match="must (be relative|stay inside)"):
        runner._get_model_path()


def test_tree_based_xgb_persistor_accepts_relative_save_name(tmp_path):
    fl_ctx = FLContext()
    fl_ctx.set_prop(FLContextKey.APP_ROOT, str(tmp_path / "app"), private=True, sticky=False)
    persistor = XGBModelPersistor(save_name="models/xgboost_model.json")

    persistor.handle_event(EventType.START_RUN, fl_ctx)

    assert persistor.save_path == os.path.realpath(tmp_path / "app" / "models" / "xgboost_model.json")


@pytest.mark.parametrize("save_name", ["/tmp/xgboost_model.json", "../xgboost_model.json"])
def test_tree_based_xgb_persistor_rejects_escaping_save_name(tmp_path, save_name):
    fl_ctx = FLContext()
    fl_ctx.set_prop(FLContextKey.APP_ROOT, str(tmp_path / "app"), private=True, sticky=False)
    persistor = XGBModelPersistor(save_name=save_name)

    with pytest.raises(ValueError, match="must (be relative|stay inside)"):
        persistor.handle_event(EventType.START_RUN, fl_ctx)
