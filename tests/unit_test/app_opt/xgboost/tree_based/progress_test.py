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

from unittest.mock import MagicMock

from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.xgboost.tree_based.executor import FedXGBTreeExecutor


def test_cyclic_xgboost_records_evaluation_metric():
    executor = FedXGBTreeExecutor(training_mode="cyclic", lr_scale=1.0, data_loader_id="data")
    executor.client_id = "site-1"
    executor.train_data = MagicMock()
    executor.val_data = MagicMock()
    executor.bst = MagicMock()
    executor.bst.num_boosted_rounds.return_value = 2
    executor.bst.eval_set.return_value = "[1]\ttrain-auc:0.90000\tvalid-auc:0.80000"

    executor._local_boost_cyclic(FLContext())

    assert executor._last_metrics == {"auc": 0.8}


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
