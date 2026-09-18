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

from nvflare.app_opt.xgboost.histogram_based_v2.runners.xgb_client_runner import _ProgressCallback


def test_collective_xgboost_rank_zero_reports_round_metrics():
    callback = _ProgressCallback(logger=MagicMock(), rank=0, num_rounds=3)

    with patch("nvflare.app_opt.xgboost.histogram_based_v2.runners.xgb_client_runner.log_progress_round") as progress:
        should_stop = callback.after_iteration(
            model=MagicMock(),
            epoch=1,
            evals_log={"eval": {"auc": [0.7, 0.8]}, "train": {"auc": [0.75, 0.85]}},
        )

    assert should_stop is False
    assert progress.call_args.args[:4] == (callback.logger, 2, 3, "XGBoost training")
    assert progress.call_args.kwargs["rows"] == [("eval", {"auc": 0.8}), ("train", {"auc": 0.85})]
    assert progress.call_args.kwargs["label"] == "Dataset"


def test_collective_xgboost_nonzero_rank_does_not_duplicate_progress():
    callback = _ProgressCallback(logger=MagicMock(), rank=1, num_rounds=3)

    with patch("nvflare.app_opt.xgboost.histogram_based_v2.runners.xgb_client_runner.log_progress_round") as progress:
        callback.after_iteration(model=MagicMock(), epoch=0, evals_log={"eval": {"auc": [0.8]}})

    progress.assert_not_called()
