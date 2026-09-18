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

from unittest.mock import MagicMock, patch

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.xgboost.tree_based.executor import FedXGBTreeExecutor


def test_cyclic_xgboost_records_evaluation_metric():
    executor = FedXGBTreeExecutor(training_mode="cyclic", lr_scale=1.0, data_loader_id="data")
    executor.client_id = "site-1"
    executor.train_data = MagicMock()
    executor.val_data = MagicMock()
    executor.bst = MagicMock()
    executor.bst.num_boosted_rounds.return_value = 2
    executor.bst.eval_set.return_value = "[2]\ttrain-auc:0.92000\tvalid-auc:0.85000"

    executor._local_boost_cyclic(FLContext())

    executor.bst.eval_set.assert_called_once()
    assert executor._last_metrics == {}
    assert executor._progress_metrics == {"auc": 0.85}


def test_bagging_xgboost_records_metric_after_local_training():
    executor = FedXGBTreeExecutor(training_mode="bagging", lr_scale=1.0, data_loader_id="data")
    executor.train_data = MagicMock()
    executor.val_data = MagicMock()
    executor.bst = MagicMock()
    executor.bst.num_boosted_rounds.return_value = 2
    executor.bst.eval_set.side_effect = [
        "[1]\ttrain-auc:0.90000\tvalid-auc:0.80000",
        "[2]\ttrain-auc:0.92000\tvalid-auc:0.85000",
    ]
    executor.bst.__getitem__.return_value = MagicMock()

    executor._local_boost_bagging(FLContext())

    executor.bst.update.assert_called_once()
    assert executor._last_metrics == {"auc": 0.8}
    assert executor._progress_metrics == {"auc": 0.85}


def test_first_bagging_invocation_evaluates_received_model_before_training():
    executor = FedXGBTreeExecutor(training_mode="bagging", lr_scale=1.0, data_loader_id="data")
    executor.client_id = "site-1"
    executor.train_data = MagicMock()
    executor.val_data = MagicMock()

    incoming_bst = MagicMock()
    incoming_bst.num_boosted_rounds.return_value = 2
    incoming_bst.eval_set.return_value = "[1]\ttrain-auc:0.90000\tvalid-auc:0.80000"

    trained_bst = MagicMock()
    trained_bst.num_boosted_rounds.return_value = 3
    trained_bst.save_config.return_value = "{}"
    trained_bst.save_raw.return_value = b"{}"

    def train_with_metrics(*_args, evals_result, **_kwargs):
        evals_result["validate"] = {"auc": [0.85]}
        return trained_bst

    model_data = b"existing model"
    request = DXO(data_kind=DataKind.WEIGHTS, data={"model_data": model_data}).to_shareable()
    with (
        patch("nvflare.app_opt.xgboost.tree_based.executor.xgb.Booster", return_value=incoming_bst),
        patch("nvflare.app_opt.xgboost.tree_based.executor.xgb.train", side_effect=train_with_metrics),
        patch("nvflare.app_opt.xgboost.tree_based.executor.mask_sum_hessian", return_value=0),
    ):
        result = executor.train(request, FLContext(), Signal())

    incoming_bst.load_model.assert_called_once_with(bytearray(model_data))
    result_dxo = from_shareable(result)
    assert result_dxo.get_meta_prop(MetaKey.INITIAL_METRICS) == {"auc": 0.8}
    assert result_dxo.get_meta_prop(AppConstants.PROGRESS_METRICS) == {"auc": 0.85}


def test_first_cyclic_invocation_reports_only_post_training_progress_metric():
    executor = FedXGBTreeExecutor(training_mode="cyclic", lr_scale=1.0, data_loader_id="data")
    executor.client_id = "site-1"
    executor.train_data = MagicMock()
    executor.val_data = MagicMock()

    trained_bst = MagicMock()
    trained_bst.num_boosted_rounds.return_value = 3
    trained_bst.save_config.return_value = "{}"
    trained_bst.save_raw.return_value = b"{}"

    def train_with_metrics(*_args, evals_result, **_kwargs):
        evals_result["validate"] = {"auc": [0.85]}
        return trained_bst

    model_data = b"existing model"
    request = DXO(data_kind=DataKind.WEIGHTS, data={"model_data": model_data}).to_shareable()
    with (
        patch("nvflare.app_opt.xgboost.tree_based.executor.xgb.Booster") as booster,
        patch("nvflare.app_opt.xgboost.tree_based.executor.xgb.train", side_effect=train_with_metrics),
        patch("nvflare.app_opt.xgboost.tree_based.executor.mask_sum_hessian", return_value=0),
    ):
        result = executor.train(request, FLContext(), Signal())

    booster.assert_not_called()
    result_dxo = from_shareable(result)
    assert result_dxo.get_meta_prop(MetaKey.INITIAL_METRICS) is None
    assert result_dxo.get_meta_prop(AppConstants.PROGRESS_METRICS) == {"auc": 0.85}


def test_scratch_training_reports_only_post_training_progress_metric():
    executor = FedXGBTreeExecutor(training_mode="cyclic", lr_scale=1.0, data_loader_id="data")
    executor.client_id = "site-1"
    executor.train_data = MagicMock()
    executor.val_data = MagicMock()

    trained_bst = MagicMock()
    trained_bst.num_boosted_rounds.return_value = 1
    trained_bst.save_config.return_value = "{}"
    trained_bst.save_raw.return_value = b"{}"

    def train_with_metrics(*_args, evals_result, **_kwargs):
        evals_result["validate"] = {"auc": [0.75]}
        return trained_bst

    request = DXO(data_kind=DataKind.WEIGHTS, data={}).to_shareable()
    with (
        patch("nvflare.app_opt.xgboost.tree_based.executor.xgb.train", side_effect=train_with_metrics),
        patch("nvflare.app_opt.xgboost.tree_based.executor.mask_sum_hessian", return_value=0),
    ):
        result = executor.train(request, FLContext(), Signal())

    result_dxo = from_shareable(result)
    assert result_dxo.get_meta_prop(MetaKey.INITIAL_METRICS) is None
    assert result_dxo.get_meta_prop(AppConstants.PROGRESS_METRICS) == {"auc": 0.75}
